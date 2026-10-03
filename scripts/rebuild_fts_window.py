#!/usr/bin/env python
"""Rebuild a fitted FTS window from its report alone, and check it reproduces.

The report written by ``fit_fts_window.py`` is meant to be sufficient: the
spectrum and its sha256, the window, the grid, the profile, the species, the
instrument, the zenith angle and every fitted parameter. If it is, this script
can reconstruct the model flux without the driver.

**It deliberately shares no code with the driver.** An independent
reconstruction is the evidence; calling one function twice is not. That is the
same rule ``rebuild_arcturus_page.py`` follows, and the reason both exist.

    UV_CACHE_DIR=.uv-cache uv run python scripts/rebuild_fts_window.py \\
        --report docs/solar_fts_window_fit.json --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tellurix import (
    AER_MOLECULE_IDS, AERLineDatabase, BoxcarFTSInstrumentProfile, ExoJAXOpacityBackend,
    MTCKDWaterContinuum, StellarSpectrum, TelluricModel, TelluricParameters,
    constant_velocity_grid, load_atmosphere_csv, prepare_stellar_source, trim_wavenumber_grid,
)
from tellurix_fts import fts_spectral_order, read_fts_spectrum

# Every molecule AER ships, from the package rather than a local copy: keeping
# a second list here is what let O3 be 'unknown' after OCS had been added, and
# what let OCS be unreachable for as long as it was. Which species are worth
# fitting is a per-window question, answered by --species and by `at_bound`.
MOLECULE_IDS = AER_MOLECULE_IDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=Path("docs/solar_fts_window_fit.json"))
    parser.add_argument("--npz", type=Path, default=None,
                        help="the driver's diagnostic npz; --check compares against its model_flux")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / args.report).read_text())
    spec, grid_spec, physics, stellar_spec = (
        report["spectrum"], report["grid"], report["physics"], report["stellar"])
    fitted = report["parameters"]

    spectrum = read_fts_spectrum(spec["path"])
    digest = hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise SystemExit(
            f"{spec['path']} has changed since the fit: {digest[:12]} against {spec['sha256'][:12]}")
    v1, v2 = spec["window_cm1"]
    window = spectrum.select(v1, v2)
    profile = load_atmosphere_csv(root / physics["profile"])

    grid = constant_velocity_grid(
        1.0e7 / v2, 1.0e7 / v1,
        resolving_power=grid_spec["resolving_power"],
        samples_per_resolution=grid_spec["samples_per_resolution"],
        margin_cm1=grid_spec["margin_cm1"],
    )
    grid = trim_wavenumber_grid(grid, v1, v2, grid_spec["grid_margin_cm1"])
    if grid.size != grid_spec["points"]:
        raise SystemExit(f"grid rebuilt to {grid.size} points, report says {grid_spec['points']}")

    line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    databases = {}
    for species in physics["species"]:
        name = f"{MOLECULE_IDS[species]:02d}_{species}"
        databases[species] = AERLineDatabase(
            line_root / name / name, species, (v1, v2), margin_cm1=grid_spec["margin_cm1"])
        counted = int(databases[species].nu_lines.size)
        if counted != physics["lines_per_species"][species]:
            raise SystemExit(f"{species}: {counted} lines against the report's "
                             f"{physics['lines_per_species'][species]}")

    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse",
        temperature_range_k=(float(np.min(profile.temperature_k)),
                             float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        vectorize_layers=True, mixed_precision=physics["mixed_precision"],
        pressure_shift=physics["pressure_shift"],
    )
    continuum = MTCKDWaterContinuum.from_netcdf(
        root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc", grid)
    instrument = (None if physics["mopd_cm"] is None else BoxcarFTSInstrumentProfile(
        mopd_cm=physics["mopd_cm"], wavenumber_center_cm1=float(0.5 * (v1 + v2)),
        max_residual_sigma_kms=4.0))
    model = TelluricModel(
        profile, grid, opacity, continuum=continuum, accuracy_mode=physics["accuracy_mode"],
        max_lsf_sigma_kms=4.0, pixel_integration=physics["pixel_integration"],
        instrument=instrument)

    if stellar_spec["source"] == "flat":
        source = prepare_stellar_source(StellarSpectrum.flat(grid), model)
    else:
        source = prepare_stellar_source(
            StellarSpectrum.from_npz(stellar_spec["source"]), model,
            vsini_kms=stellar_spec["vsini_kms"],
            macroturbulence_kms=stellar_spec["macroturbulence_kms"],
            normalize=stellar_spec["normalized"])
    order = fts_spectral_order(window, zenith_angle_deg=physics["zenith_angle_deg"],
                               source_flux_model_grid=source)

    parameters = TelluricParameters(
        log_column_scales={name: fitted[name] for name in physics["species"]},
        velocity_kms=fitted["velocity_kms"],
        wavelength_stretch=0.0,
        lsf_sigma_kms=fitted["lsf_sigma_kms"],
        continuum_coeffs=np.asarray(fitted["continuum_coeffs"], dtype=float),
        log_jitter=fitted["log_jitter"],
        stellar_velocity_kms=fitted["stellar_velocity_kms"],
    )
    # Freezing at the fitted parameters is exact there, and it is what the
    # driver evaluated its own products through.
    rebuilt = np.asarray(model.precompute_opacity(parameters).predict(order, parameters))

    print(f"rebuilt {spec['name']} {v1}-{v2} cm-1: {rebuilt.size} pixels, "
          f"{grid.size} grid points, zenith {physics['zenith_angle_deg']:.3f} deg")
    if not args.check:
        return

    npz_path = args.npz or (root / "benchmarks/results/solar_fts_window.npz")
    stored = np.load(npz_path)["model_flux"]
    if stored.shape != rebuilt.shape:
        raise SystemExit(f"shape {rebuilt.shape} against the stored {stored.shape}")
    difference = float(np.max(np.abs(stored - rebuilt)))
    noise = spec["uncertainty"]
    print(f"max |rebuilt - stored| = {difference:.3e}  ({difference / noise:.2e} of the noise)")
    if difference > args.tolerance:
        raise SystemExit(f"reconstruction differs by {difference:.3e}, over {args.tolerance:g}")
    print("reconstruction agrees")


if __name__ == "__main__":
    main()
