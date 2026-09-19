#!/usr/bin/env python
"""Rebuild one page's arrays from the run record, and check them.

This reconstructs the forward model from the record alone -- the fitted
parameters, the configuration, and the input files the record identifies by
hash -- and recomputes the transmission, the model, the stellar model, the
continuum and the corrected spectrum.

It deliberately does **not** share code with the fitting driver. Calling the
same construction twice would prove only that a function is deterministic. An
independent reconstruction that lands on the same numbers is what shows the
record is sufficient, which is the whole claim being tested. If the two ever
drift apart, ``--check`` says so.

    uv run python scripts/rebuild_arcturus_page.py --page ab5000_ --epoch summer --check
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MOLECULE_IDS = {"H2O": 1, "CO2": 2, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
ARRAYS = ("transmission", "model_flux", "stellar_only", "continuum", "corrected")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path,
                        default=root / "data/corrected/atlas/arcturus_atlas.h5")
    parser.add_argument("--page", required=True)
    parser.add_argument("--epoch", choices=("summer", "winter"), default="summer")
    parser.add_argument("--atlas-root", type=Path, default=None,
                        help="overrides the path the record names, not its hash")
    parser.add_argument("--cached", type=Path, default=None,
                        help="the .npz to compare against; defaults beside the record")
    parser.add_argument("--check", action="store_true", help="compare and exit non-zero on drift")
    parser.add_argument("--tolerance", type=float, default=1.0e-5,
                        help="maximum absolute difference allowed by --check")
    parser.add_argument("--out", type=Path, default=None, help="write the rebuilt arrays here")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()

    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import numpy as np

    from jax_telluric import (
        AERLineDatabase, ArrayOpacityBackend, BoxcarFTSInstrumentProfile,
        ExoJAXOpacityBackend, MTCKDWaterContinuum, StellarSpectrum, TelluricModel,
        arcturus_spectral_order, chebyshev_continuum, file_sha256,
        igrins_wavenumber_grid, ils_fingerprint, load_atmosphere_csv,
        parameters_from_row, prepare_stellar_source, read_arcturus_page, read_record,
        select_significant_lines, trim_wavenumber_grid,
    )
    from jax_telluric.record import text

    record = read_record(args.record)
    row = record.row(args.page, args.epoch)
    config, physics, inputs = record.config, record.physics, record.inputs
    species = [text(s) for s in
               (record.pages.dtype.names and [n[11:] for n in record.pages.dtype.names
                                              if n.startswith("log_column_")])]

    # --- the inputs, checked against the identities the record carries ---
    def verified(key: str, path: Path) -> Path:
        stored = inputs.get(f"{key}_sha256")
        if stored and path.exists():
            actual = file_sha256(path)
            if actual != stored:
                raise SystemExit(
                    f"{key} has changed since the run: {path}\n"
                    f"  recorded {stored}\n  now      {actual}"
                )
        return path

    atlas_root = Path(args.atlas_root or inputs["atlas_root"])
    profile = load_atmosphere_csv(verified("profile", Path(inputs["profile"])))
    stellar_path = verified("stellar", Path(inputs["stellar"]))

    v1, v2 = float(row["v1"]), float(row["v2"])
    grid = trim_wavenumber_grid(
        igrins_wavenumber_grid(1.0e7 / v2, 1.0e7 / v1,
                               resolving_power=float(config["resolving_power"]),
                               samples_per_resolution=float(config["samples_per_resolution"]),
                               margin_cm1=float(config["margin_cm1"])),
        v1, v2, float(config["grid_margin_cm1"]))
    if grid.size != int(row["grid_points"]):
        raise SystemExit(f"grid rebuilt to {grid.size} samples, record says {row['grid_points']}")

    line_root = Path(inputs["aer_line_root"])
    databases = {}
    for name, molecule_id in sorted(MOLECULE_IDS.items()):
        stem = f"{molecule_id:02d}_{name}"
        try:
            databases[name] = AERLineDatabase(
                verified(f"aer_{name}", line_root / stem / stem), name, (v1, v2),
                margin_cm1=float(config["margin_cm1"]))
        except ValueError as exc:
            if not str(exc).startswith(f"no {name} lines found"):
                raise
    budget = float(config.get("line_budget", 0.0))
    if budget > 0.0:
        databases = {n: select_significant_lines(d, profile, n, budget,
                                                 maximum_column_scale=float(np.exp(2.0)))
                     for n, d in databases.items()}

    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods=physics["opacity_method"],
        temperature_range_k=(float(np.min(profile.temperature_k)),
                             float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        vectorize_layers=bool(physics["vectorize_layers"]),
        mixed_precision=bool(physics["mixed_precision"]),
        pressure_shift=bool(physics["pressure_shift"]))
    continuum = MTCKDWaterContinuum.from_netcdf(verified("mt_ckd", Path(inputs["mt_ckd"])), grid)
    instrument = BoxcarFTSInstrumentProfile(
        mopd_cm=float(row["mopd_cm"]), wavenumber_center_cm1=float(0.5 * (v1 + v2)),
        max_residual_sigma_kms=float(physics["instrument_residual_sigma_kms"]))
    model = TelluricModel(profile, grid, opacity, continuum=continuum,
                          accuracy_mode=physics["accuracy_mode"],
                          max_lsf_sigma_kms=float(physics["max_lsf_sigma_kms"]),
                          pixel_integration=physics["pixel_integration"], instrument=instrument)

    source = prepare_stellar_source(
        StellarSpectrum.from_npz(stellar_path), model,
        vsini_kms=float(physics["vsini_kms"]),
        limb_darkening=float(physics.get("limb_darkening", 0.6)),
        macroturbulence_kms=float(physics["macroturbulence_kms"]),
        normalize=bool(physics.get("normalize_stellar_source", True)))

    page = read_arcturus_page(atlas_root / args.page, args.epoch).select(v1, v2)
    if page.sha256 != text(row["page_sha256"]):
        raise SystemExit(f"the atlas page has changed since the run: {args.page} {args.epoch}")
    order = arcturus_spectral_order(
        page, source_flux_model_grid=source, column=config["column"],
        saturation_floor=float(config["saturation_floor"]),
        telluric_ceiling=float(config["telluric_ceiling"]),
        zenith_angle_deg=float(config["zenith_angle_deg"]))

    parameters = parameters_from_row(row, [s for s in species if s in model.species])

    # Refreezing at the fitted parameters is exact there, and reuses one
    # compilation for all four evaluations below.
    exact = model.precompute_opacity(parameters)
    order_idx = np.argsort(1.0e7 / np.asarray(order.wavelength_vacuum_nm))
    rebuilt = {
        "wavenumber_cm1": (1.0e7 / np.asarray(order.wavelength_vacuum_nm))[order_idx],
        "model_flux": np.asarray(exact.predict(order, parameters))[order_idx],
        "transmission": np.interp(
            (1.0e7 / np.asarray(order.wavelength_vacuum_nm))[order_idx],
            np.asarray(model.wavenumber_cm1),
            np.asarray(exact.transmission(parameters, order.zenith_angle_deg))),
        "continuum": np.asarray(chebyshev_continuum(
            parameters.continuum_coeffs,
            np.linspace(-1.0, 1.0, len(order.wavelength_vacuum_nm))))[order_idx],
    }
    star_only = TelluricModel(
        profile, grid,
        ArrayOpacityBackend({s: np.zeros((len(profile.temperature_k), grid.size))
                             for s in model.species}),
        accuracy_mode="fast", max_lsf_sigma_kms=float(physics["max_lsf_sigma_kms"]),
        pixel_integration=physics["pixel_integration"], instrument=instrument)
    rebuilt["stellar_only"] = np.asarray(star_only.predict(order, parameters))[order_idx]
    observed = np.asarray(order.flux)[order_idx]
    rebuilt["corrected"] = (
        observed / np.maximum(rebuilt["model_flux"], 1e-6)) * rebuilt["stellar_only"]

    fingerprint = ils_fingerprint(instrument, float(parameters.lsf_sigma_kms),
                                  model.velocity_step_kms)[1]
    index = int(np.flatnonzero((record.pages["page"] == args.page.encode())
                               & (record.pages["epoch"] == args.epoch.encode()))[0])
    ils_drift = float(np.max(np.abs(fingerprint - record.ils_profile[index])))

    if args.out:
        np.savez_compressed(args.out, **rebuilt)
        print(f"wrote {args.out}")

    print(f"{args.page} {args.epoch}: {grid.size} grid samples, "
          f"{len(order.wavelength_vacuum_nm)} pixels")
    print(f"instrument fingerprint drift: {ils_drift:.2e}")

    if not args.check:
        return
    cached = args.cached or (args.record.parent / f"{args.page}_{args.epoch}.npz")
    if not cached.exists():
        raise SystemExit(f"nothing to compare against: {cached}")
    stored = np.load(cached)
    mask = np.asarray(stored["mask"]).astype(bool)
    worst = 0.0
    print(f"\n{'array':16s} {'max |diff|':>12} {'rms':>12}")
    for name in ARRAYS:
        a, b = rebuilt[name][mask], np.asarray(stored[name])[mask]
        finite = np.isfinite(a) & np.isfinite(b)
        difference = np.abs(a[finite] - b[finite])
        worst = max(worst, float(difference.max()))
        print(f"{name:16s} {difference.max():12.3e} "
              f"{np.sqrt(np.mean(difference**2)):12.3e}")
    print(f"\nworst {worst:.3e} against a tolerance of {args.tolerance:.1e}")
    if worst > args.tolerance:
        raise SystemExit("the rebuild does not reproduce the cache: something is unrecorded")
    print("the record reproduces the cached arrays")


if __name__ == "__main__":
    main()
