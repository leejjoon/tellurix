#!/usr/bin/env python
"""What the choice of layering costs a fitted correction.

Builds the default 12 layers from one ERA5 column both ways
``site_profile.LAYERINGS`` offers, and a 150-layer reference from the same
levels, then fits each 12-layer profile to the reference -- H2O, CO2 and CH4
column scales and a linear continuum, at R=45,000 -- the way a real correction
would absorb the difference. Two more variants swap only the layer pressure,
to show how much of the difference is pressure and how much is temperature and
water. Writes ``docs/layering_comparison.json``.

    UV_CACHE_DIR=.uv-cache uv run python scripts/compare_layering.py

Needs the AER line files (``bootstrap_lblrtm.sh``) and network access to
ARCO-ERA5; about 20 minutes on one GPU.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import tellurix  # noqa: F401  (x64 before exojax)
import jax.numpy as jnp
from scipy.optimize import least_squares

from tellurix import (
    AERLineDatabase, AtmosphereProfile, DataPaths, ExoJAXOpacityBackend, constant_velocity_grid,
    trim_wavenumber_grid,
)
from tellurix.era5 import build_era5_profile, fetch_column
from tellurix.site_profile import DEFAULT_EDGES_KM, EPOCH_DRY_VMR

# Two nights from the IGRINS work, dry and wet: ERA5 hour, position, and the
# station pressure the frames' headers gave.
CASES = {
    "dct_2018": {"era5_time": "2018-12-21T05:00", "latitude": 34.7444, "longitude": -111.4223,
                 "site_altitude_km": 2.360, "station_pressure_hpa": 768.5},
    "mcdonald_2017": {"era5_time": "2017-04-21T08:00", "latitude": 30.6717, "longitude": -104.0225,
                      "site_altitude_km": 2.077, "station_pressure_hpa": 792.4},
}
# IGRINS windows: CH4 and water at 2.32 um, water and CO2 at 2.0 um, water in
# H, CO2 in H.
WINDOWS_CM1 = ((4300.0, 4330.0), (5000.0, 5030.0), (5600.0, 5630.0), (6300.0, 6330.0))
SPECIES = ("H2O", "CO2", "CH4")
MOLECULE_IDS = {"H2O": 1, "CO2": 2, "CH4": 6}
RESOLVING_POWER = 45_000.0
SAMPLES_PER_RESOLUTION = 4.0
AIRMASS = 1.5
REFERENCE_LAYERS = 150


def as_profile(layers: dict, mean_pressure_bar="keep") -> AtmosphereProfile:
    """A builder's arrays as a profile; ``mean_pressure_bar`` may replace its pressure."""

    edges = np.concatenate([layers["pressure_top_bar"][:1], layers["pressure_bottom_bar"]])
    if isinstance(mean_pressure_bar, str):
        mean_pressure_bar = layers.get("pressure_bar")
    return AtmosphereProfile(edges, layers["temperature_k"], layers["altitude_km"],
                             {name: layers[name] for name in SPECIES},
                             layers["mean_molecular_weight_g_mol"], layers["gravity_m_s2"],
                             mean_pressure_bar=mean_pressure_bar)


def optical_depths(backend, profile: AtmosphereProfile) -> dict:
    pressure = jnp.asarray(profile.pressure_layer_bar)
    partial = {s: pressure * jnp.asarray(profile.vmr[s]) for s in backend.species}
    xs = backend.cross_sections(jnp.asarray(profile.temperature_k), pressure, partial)
    return {s: np.asarray(jnp.sum(
        xs[s] * jnp.asarray(profile.air_column_cm2 * profile.vmr[s])[:, None], axis=0))
        for s in backend.species}


def convolve(transmission: np.ndarray) -> np.ndarray:
    sigma = SAMPLES_PER_RESOLUTION / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    offsets = np.arange(-int(6 * sigma) - 1, int(6 * sigma) + 2)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return np.convolve(transmission, kernel / kernel.sum(), "same")


def fit_to_reference(taus: dict, reference: np.ndarray, keep: np.ndarray, x: np.ndarray) -> dict:
    species = list(taus)

    def model(p):
        tau = sum(np.exp(p[i]) * taus[s] for i, s in enumerate(species))
        return convolve(np.exp(-AIRMASS * tau)) * (1.0 + p[-2] + p[-1] * x)

    start = np.zeros(len(species) + 2)
    before = np.abs(model(start) - reference)[keep]
    fit = least_squares(lambda p: (model(p) - reference)[keep], start)
    after = np.abs(fit.fun)
    return {"unfitted_p99": float(np.percentile(before, 99)), "unfitted_max": float(before.max()),
            "fitted_median": float(np.median(after)), "fitted_p99": float(np.percentile(after, 99)),
            "fitted_max": float(after.max()),
            "scales": {s: float(np.exp(fit.x[i])) for i, s in enumerate(species)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("docs/layering_comparison.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = DataPaths.bootstrapped(root)
    dry_vmr = EPOCH_DRY_VMR["2020"]
    reference_edges = tuple(np.linspace(0.0, DEFAULT_EDGES_KM[-1], REFERENCE_LAYERS + 1))
    results = {}
    for name, case in CASES.items():
        when = datetime.fromisoformat(case["era5_time"]).replace(tzinfo=timezone.utc)
        column = fetch_column(case["latitude"], case["longitude"], when)

        def build(edges, layering):
            return build_era5_profile(column, case["station_pressure_hpa"], case["site_altitude_km"],
                                      dry_vmr, edges, layering=layering)

        centre, weighted = build(DEFAULT_EDGES_KM, "centre"), build(DEFAULT_EDGES_KM, "weighted")
        geometric = np.sqrt(weighted["pressure_top_bar"] * weighted["pressure_bottom_bar"])
        profiles = {
            "centre": as_profile(centre),
            "weighted": as_profile(weighted),
            "weighted_geometric_pressure": as_profile(weighted, geometric),
            "centre_weighted_pressure": as_profile(centre, weighted["pressure_bar"]),
            "reference": as_profile(build(reference_edges, "weighted")),
        }
        result = {"water_column_cm2": {k: float(np.sum(p.air_column_cm2 * p.vmr["H2O"]))
                                       for k, p in profiles.items()}}
        print(name, result["water_column_cm2"], flush=True)
        for v1, v2 in WINDOWS_CM1:
            wide = constant_velocity_grid(1.0e7 / v2, 1.0e7 / v1, resolving_power=RESOLVING_POWER,
                                          samples_per_resolution=SAMPLES_PER_RESOLUTION,
                                          margin_cm1=25.0)
            grid = trim_wavenumber_grid(wide, v1, v2, 5.0)
            databases = {}
            for species in SPECIES:
                try:
                    databases[species] = AERLineDatabase(
                        paths.line_file(species, MOLECULE_IDS[species]), species, (v1, v2),
                        margin_cm1=25.0)
                except ValueError as exc:
                    if "lines found" not in str(exc):
                        raise
            backend = ExoJAXOpacityBackend.prepare(
                databases, grid, methods="direct_sparse", temperature_range_k=(150.0, 320.0),
                maximum_pressure_bar=1.0, vectorize_layers=True, pressure_shift=True,
                # The reference's 150 layers at once do not fit on a 32 GB card.
                layer_chunk_size=25)
            taus = {k: optical_depths(backend, p) for k, p in profiles.items()}
            inner = (grid >= v1) & (grid <= v2)
            reference = convolve(np.exp(-AIRMASS * sum(taus.pop("reference").values())))
            keep = inner & (reference > 0.05)
            x = (grid - 0.5 * (v1 + v2)) / (0.5 * (v2 - v1))
            window = {"min_reference_transmission": float(reference[inner].min())}
            for k, tau in taus.items():
                window[k] = fit_to_reference(tau, reference, keep, x)
            print(f"  {v1:.0f}-{v2:.0f}", {k: round(v["fitted_p99"], 5) for k, v in window.items()
                                           if isinstance(v, dict)}, flush=True)
            result[f"{v1:.0f}-{v2:.0f}"] = window
        results[name] = result

    report = {
        "description": __doc__.split("\n\n")[1].replace("\n", " "),
        "generated_by": "scripts/compare_layering.py",
        "measured": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "era5": "ARCO-ERA5 37-level via tellurix.era5.fetch_column; layers by build_era5_profile",
        "dry_vmr": "EPOCH_DRY_VMR['2020']",
        "edges_km_above_site": list(DEFAULT_EDGES_KM),
        "reference_layers": REFERENCE_LAYERS,
        "resolving_power": RESOLVING_POWER, "samples_per_resolution": SAMPLES_PER_RESOLUTION,
        "airmass": AIRMASS,
        "metric": "absolute transmission error over the window where the reference exceeds 0.05",
        "opacity": "AER 3.9 lines, ExoJAX direct_sparse with pressure shifts, 25 cm-1 line margin",
        "variants": {
            "centre": "values at the layer centre, geometric-mean pressure: profiles before 2026-10",
            "weighted": "air-weighted pressure and temperature, integrated amounts (the default)",
            "weighted_geometric_pressure": "weighted temperature and amounts, geometric-mean pressure",
            "centre_weighted_pressure": "centre temperature and amounts, air-weighted pressure"},
        "cases": CASES,
        "results": results,
    }
    output = root / args.output
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
