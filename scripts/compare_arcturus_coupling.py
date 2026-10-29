"""What AER first-order line coupling does to the Arcturus atlas fit, page by page.

Reads two runs of ``fit_arcturus_batch.py`` over the same pages -- one with
``--line-coupling``, one without as the control -- and the committed
full-coverage record, and writes ``docs/arcturus_line_coupling.json``:

* that the control reproduces the record (so every difference is coupling);
* the residual change per page and per CO2 band, and the column shifts;
* which page-epochs the coupled run could not fit, and why: for each failed
  window, the most negative coupled optical depth per layer and vertically,
  against the uncoupled minimum (``--negatives``, one opacity build per
  species per window);
* the cost, stage by stage, from the runs' own timings.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/compare_arcturus_coupling.py --negatives
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BANDS = [("4.8 um", 1850.0, 2140.0), ("4.3 um", 2140.0, 2450.0), ("2.7 um", 3300.0, 3800.0),
         ("2.0 um", 4750.0, 5150.0), ("1.6 um", 6150.0, 6400.0)]


def band(v1: float, v2: float) -> str:
    centre = 0.5 * (v1 + v2)
    for label, lo, hi in BANDS:
        if lo <= centre < hi:
            return label
    return "other"


def text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def negative_optical_depth(root: Path, v1: float, v2: float, profile_path: Path) -> dict:
    """The most negative coupled optical depth per species, as the run built it."""
    import tellurix  # noqa: F401  (x64 before ExoJAX)
    from tellurix import (AER_MOLECULE_IDS, AERLineDatabase, DataPaths, ExoJAXOpacityBackend,
                          constant_velocity_grid, load_atmosphere_csv, trim_wavenumber_grid)

    profile = load_atmosphere_csv(profile_path)
    paths = DataPaths.bootstrapped(root)
    grid = trim_wavenumber_grid(constant_velocity_grid(1.0e7 / v2, 1.0e7 / v1, resolving_power=1.0e5,
                                                       samples_per_resolution=4.0, margin_cm1=25.0),
                                v1, v2, 5.0)
    out = {}
    for species in ("CO2", "CH4", "O2"):
        try:
            database = AERLineDatabase(paths.line_file(species, AER_MOLECULE_IDS[species]), species,
                                       (v1, v2), margin_cm1=25.0, line_coupling=paths.line_coupling)
        except ValueError:
            continue
        if not database.has_line_coupling:
            continue
        column = profile.vmr[species] * profile.air_column_cm2
        tau = {}
        for coupled in (False, True):
            backend = ExoJAXOpacityBackend.prepare(
                {species: database}, grid, methods="direct_sparse",
                temperature_range_k=(float(profile.temperature_k.min()), float(profile.temperature_k.max())),
                maximum_pressure_bar=float(profile.pressure_layer_bar.max()),
                vectorize_layers=True, mixed_precision=True, pressure_shift=True, line_coupling=coupled)
            sigma = backend.cross_sections(profile.temperature_k, profile.pressure_layer_bar,
                                           {species: profile.pressure_layer_bar * profile.vmr[species]})
            tau[coupled] = np.asarray(sigma[species]) * column[:, None]
        layer = tau[True]
        out[species] = {
            "negative_layer_samples": int(np.sum(layer < 0.0)),
            "min_layer_tau": float(layer.min()),
            "min_vertical_tau_coupled": float(layer.sum(0).min()),
            "min_vertical_tau_uncoupled": float(tau[False].sum(0).min()),
            "max_vertical_tau_uncoupled": float(tau[False].sum(0).max()),
        }
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coupled", type=Path, default=root / "data/corrected/atlas_coupling_test/cpl/summary.json")
    parser.add_argument("--control", type=Path, default=root / "data/corrected/atlas_coupling_test/nocpl/summary.json")
    parser.add_argument("--record", type=Path, default=root / "data/corrected/superseded_pre_solar_era/atlas/arcturus_atlas.h5")
    parser.add_argument("--negatives", action="store_true",
                        help="rebuild each failed window's opacity to measure how negative it goes")
    parser.add_argument("--output", type=Path, default=root / "docs/arcturus_line_coupling.json")
    args = parser.parse_args()

    from tellurix import read_record

    def load(path):
        results = json.loads(path.read_text())["results"]
        return ({(r["page"], r["epoch"]): r for r in results if "error" not in r},
                [r for r in results if "error" in r])

    coupled, failed = load(args.coupled)
    control, control_failed = load(args.control)
    if control_failed:
        raise SystemExit("the control run has failures; it must fit every page")
    record = read_record(args.record)
    published = {(text(r["page"]), text(r["epoch"])): r for r in record.pages}

    # The control is the current driver without coupling; it must reproduce the record.
    reproduce = np.array([control[k]["residual_rms_over_noise"] /
                          float(published[k]["residual_rms_over_noise"]) - 1 for k in control])

    keys = sorted(set(coupled) & set(control))
    pages = []
    for k in keys:
        c, n = coupled[k], control[k]
        v1, v2 = n["wavenumber_cm1"]
        pages.append({
            "page": k[0], "epoch": k[1], "v1": v1, "v2": v2, "band": band(v1, v2),
            "rms_over_noise_control": round(n["residual_rms_over_noise"], 4),
            "rms_over_noise_coupled": round(c["residual_rms_over_noise"], 4),
            "change": round(c["residual_rms_over_noise"] / n["residual_rms_over_noise"] - 1, 5),
            "log_column_co2_control": round(n["parameters"]["CO2"], 5),
            "log_column_co2_coupled": round(c["parameters"]["CO2"], 5),
        })

    bands = {}
    for label in [b[0] for b in BANDS] + ["other"]:
        rows = [p for p in pages if p["band"] == label]
        if not rows:
            continue
        change = np.array([p["change"] for p in rows])
        shift = np.array([p["log_column_co2_coupled"] - p["log_column_co2_control"] for p in rows])
        bands[label] = {
            "page_epochs": len(rows),
            "median_rms_over_noise_control": round(float(np.median([p["rms_over_noise_control"] for p in rows])), 3),
            "median_rms_over_noise_coupled": round(float(np.median([p["rms_over_noise_coupled"] for p in rows])), 3),
            "median_change": round(float(np.median(change)), 5),
            "better_by_2pct": int(np.sum(change < -0.02)),
            "worse_by_2pct": int(np.sum(change > 0.02)),
            "median_co2_shift": round(float(np.median(shift)), 5),
        }

    failures = []
    windows = {}
    for r in failed:
        n = control[(r["page"], r["epoch"])]
        v1, v2 = n["wavenumber_cm1"]
        failures.append({"page": r["page"], "epoch": r["epoch"], "v1": v1, "v2": v2, "error": r["error"]})
        windows[(v1, v2)] = r["page"]
    negatives = {}
    if args.negatives:
        profile = Path(json.loads(args.control.read_text())["profile"])
        for (v1, v2), page in sorted(windows.items()):
            negatives[page] = {"v1": v1, "v2": v2, **negative_optical_depth(root, v1, v2, root / profile)}
            print(page, json.dumps(negatives[page]))

    def total(row):
        return sum(v for v in row["timing"].values() if isinstance(v, (int, float)))

    stages = list(control[keys[0]]["timing"])
    timing = {stage: {"coupled_s": round(sum(coupled[k]["timing"].get(stage, 0.0) for k in keys), 1),
                      "control_s": round(sum(control[k]["timing"].get(stage, 0.0) for k in keys), 1)}
              for stage in stages}
    report = {
        "about": "AER first-order line coupling on the Arcturus atlas pages where CO2 is fitted, "
                 "against a control run without it. Written by scripts/compare_arcturus_coupling.py.",
        "runs": {"coupled": str(args.coupled), "control": str(args.control), "record": str(args.record)},
        "control_reproduces_record": {
            "page_epochs": len(reproduce),
            "rms_over_noise_change_min": round(float(reproduce.min()), 5),
            "rms_over_noise_change_max": round(float(reproduce.max()), 5),
        },
        "page_epochs_compared": len(keys),
        "bands": bands,
        "failures": failures,
        "negative_optical_depth": negatives,
        "timing": {
            "page_epochs": len(keys),
            "total_coupled_s": round(sum(total(coupled[k]) for k in keys), 1),
            "total_control_s": round(sum(total(control[k]) for k in keys), 1),
            "median_page_ratio": round(float(np.median([total(coupled[k]) / total(control[k]) for k in keys])), 3),
            "stages": timing,
        },
        "pages": pages,
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
