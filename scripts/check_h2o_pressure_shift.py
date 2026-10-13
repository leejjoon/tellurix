#!/usr/bin/env python
"""AER's H2O pressure shifts against HITRAN's, judged on niratl's own telluric column.

niratl (Wallace, Hinkle & Livingston 1993) ships the atmospheric spectrum it
divided out, separated empirically and so independent of any line list or solar
model. Fitting it with continuum x transmission alone isolates the line list:
the same window is fitted twice, once with AER v3.9's air pressure shifts and
once with HITRAN's (``tellurix.hitran``), everything else identical.

Two measures, per source:

- the window's residual at telluric pixels (T < 0.98);
- each isolated, unblended H2O line's position error (data minus fit, by a
  shift scan over +-0.2 cm-1), regressed on the shift coefficient the fit used,
  with one velocity per window. A pressure shift that is right leaves no slope;
  AER's lines with a positive coefficient are reported on their own, because
  HITRAN gives all of them a negative one.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/check_h2o_pressure_shift.py --shard 0/4
    ...
    uv run python scripts/check_h2o_pressure_shift.py --report   # -> docs/h2o_pressure_shift_check.json

Per-window results go to data/corrected/solar/h2o_shift_check/ (gitignored).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/corrected/solar/niratl/niratl_summary.json"
PROFILE = Path("data/profiles/kitt_peak_19830626_era5_afgl.csv")
WORK = ROOT / "data/corrected/solar/h2o_shift_check"
OUTPUT = ROOT / "docs/h2o_pressure_shift_check.json"
SOURCES = ("aer", "hitran")
# Isolated-line selection, as in the per-line strength checks of §4q: a water
# line 2-50% deep, nothing but water within 0.2 cm-1, and no other line within
# 0.15 cm-1 carrying a tenth of its strength.
HALF_WIDTH_CM1, MIN_DEPTH, MAX_DEPTH = 0.2, 0.02, 0.5


def windows():
    rows = json.loads(SUMMARY.read_text())["results"]
    return [r for r in rows if "H2O" in (r.get("species") or [])
            and not r.get("negligible_telluric") and not r.get("opaque")]


def fit_one(row, source, paths):
    from tellurix_fts.nso import niratl_as_fts_spectrum, read_niratl_page
    from tellurix_fts.window import WindowSettings, fit_window

    page = read_niratl_page(row["file"])
    atmospheric = np.where(page.atmospheric == -1.0, np.nan, page.atmospheric)
    if np.isfinite(atmospheric).sum() < 1000:
        return None
    spectrum = dataclasses.replace(niratl_as_fts_spectrum(page), flux=atmospheric)
    settings = WindowSettings(
        spectrum=Path(row["file"]), profile=PROFILE, v1=row["v1"], v2=row["v2"],
        fwhm_cm1=0.01859, species=",".join(row["species"]), stellar="flat",
        continuum_degree=3, stages="continuum,velocity,columns",
        zenith_angle_deg=row["zenith_angle_deg"], accuracy_mode="mt_ckd", o2_cia=True,
        h2o_pressure_shift=source)
    report, arrays = fit_window(settings, paths, stellar_directory=ROOT / "data/stellar",
                                base=ROOT, spectrum=spectrum)
    return report, arrays


def isolated_lines(arrays, database):
    """Position error of each isolated H2O line, with the coefficient the fit used."""

    nu = np.asarray(arrays["wavenumber_cm1"])
    order = np.argsort(nu)
    nu = nu[order]
    take = lambda key: np.asarray(arrays[key])[order]
    continuum = take("continuum")
    data = np.where(take("mask").astype(bool), take("flux"), np.nan) / continuum
    model = take("model_flux") / continuum
    water = take("species_transmission_H2O")
    others = take("transmission_pixels") / water
    step = float(np.median(np.diff(nu)))
    half = int(HALF_WIDTH_CM1 / step)
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(1.0 - model, height=MIN_DEPTH, distance=int(0.1 / step))
    strength = np.asarray(database.line_strength_ref_original)
    lines = np.asarray(database.nu_lines)
    shifts = np.linspace(-0.012, 0.012, 481)
    out = []
    for i in peaks:
        if i < half or i > nu.size - half - 1 or 1.0 - model[i] > MAX_DEPTH:
            continue
        window = slice(i - half, i + half)
        if np.isnan(data[window]).any() or np.ptp(others[window]) > 0.003:
            continue
        near = np.flatnonzero(np.abs(lines - nu[i]) < 0.04)
        if near.size == 0:
            continue
        k = near[np.argmax(strength[near])]
        wide = np.abs(lines - nu[i]) < 0.15
        if strength[wide].sum() - strength[k] > 0.1 * strength[k]:
            continue
        x, y = nu[window], data[window]
        misfit = [np.sum((y - np.interp(x, nu + s, model)) ** 2) for s in shifts]
        out.append({"nu_cm1": float(lines[k]), "position_error_cm1": float(shifts[int(np.argmin(misfit))]),
                    "delta_air": float(database.delta_air[k]), "depth": float(1.0 - model[i])})
    return out


def run_shard(shard, count):
    import tellurix
    from tellurix import AERLineDatabase, DataPaths

    paths = DataPaths.bootstrapped(ROOT)
    WORK.mkdir(parents=True, exist_ok=True)
    for row in windows()[shard::count]:
        name = Path(row["file"]).name
        target = WORK / f"{name}.json"
        if target.exists():
            continue
        result = {"page": name, "v1": row["v1"], "v2": row["v2"]}
        for source in SOURCES:
            fitted = fit_one(row, source, paths)
            if fitted is None:
                result = None
                break
            report, arrays = fitted
            database = AERLineDatabase(
                paths.line_file("H2O", 1), "H2O", (row["v1"], row["v2"]), margin_cm1=25.0,
                pressure_shift_table=paths.pressure_shift_table("H2O", source))
            mask = np.asarray(arrays["mask"]).astype(bool)
            telluric = mask & (np.asarray(arrays["transmission_pixels"]) < 0.98)
            residual = (np.asarray(arrays["flux"]) - np.asarray(arrays["model_flux"])) / np.asarray(arrays["continuum"])
            result[source] = {
                "telluric_rms": float(np.sqrt(np.mean(residual[telluric] ** 2))) if telluric.any() else None,
                "telluric_pixels": int(telluric.sum()),
                "parameters": {k: report["parameters"][k] for k in ("H2O", "velocity_kms", "lsf_sigma_kms")},
                "lines": isolated_lines(arrays, database),
                "provenance": {k: v for k, v in report["physics"].items() if k.startswith("h2o_pressure")},
            }
        if result is None:
            print(f"{name}: no atmospheric column", flush=True)
            continue
        target.write_text(json.dumps(result))
        print(f"{name}: telluric rms {result['aer']['telluric_rms']} (AER) "
              f"{result['hitran']['telluric_rms']} (HITRAN)", flush=True)


def regression(results, source):
    """Position error on the coefficient the fit used, with one velocity per window."""

    rows = [(i, line) for i, r in enumerate(results) for line in r[source]["lines"]]
    window = np.array([i for i, _ in rows])
    error = np.array([line["position_error_cm1"] for _, line in rows])
    delta = np.array([line["delta_air"] for _, line in rows])
    nu = np.array([line["nu_cm1"] for _, line in rows])
    used = sorted(set(window))
    design = np.zeros((error.size, len(used) + 1))
    for column, w in enumerate(used):
        design[window == w, column] = nu[window == w] / 299792.458
    design[:, -1] = delta
    coefficients, *_ = np.linalg.lstsq(design, error, rcond=None)
    residual = error - design @ coefficients
    return coefficients[-1], residual, nu, delta


def report():
    results = [json.loads(p.read_text()) for p in sorted(WORK.glob("ph*.json"))]
    if not results:
        raise SystemExit(f"nothing in {WORK}; run the shards first")
    summary = {"windows": len(results), "sources": {}}
    aer_lines = {round(l["nu_cm1"], 4): l["delta_air"] for r in results for l in r["aer"]["lines"]}
    for source in SOURCES:
        slope, residual, nu, delta = regression(results, source)
        positive_in_aer = np.array([aer_lines.get(round(v, 4), 0.0) > 0.0 for v in nu])
        rms = np.array([r[source]["telluric_rms"] for r in results if r[source]["telluric_rms"] is not None])
        summary["sources"][source] = {
            "isolated_lines": int(nu.size),
            "position_error_slope_on_delta_atm": float(slope),
            "position_scatter_mK": float(1e3 * np.std(residual)),
            "lines_positive_in_aer": int(positive_in_aer.sum()),
            "median_position_error_positive_in_aer_mK": float(1e3 * np.median(residual[positive_in_aer])),
            "median_position_error_rest_mK": float(1e3 * np.median(residual[~positive_in_aer])),
            "median_telluric_rms": float(np.median(rms)),
            "provenance": results[0][source]["provenance"],
        }
    pairs = np.array([(r["aer"]["telluric_rms"], r["hitran"]["telluric_rms"]) for r in results
                      if r["aer"]["telluric_rms"] is not None])
    ratio = pairs[:, 1] / pairs[:, 0]
    summary["telluric_rms_hitran_over_aer"] = {
        "median": float(np.median(ratio)), "better_by_over_2pct": int((ratio < 0.98).sum()),
        "worse_by_over_2pct": int((ratio > 1.02).sum()), "windows": int(ratio.size)}
    summary["per_window"] = {r["page"]: {s: r[s]["telluric_rms"] for s in SOURCES} for r in results}
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_window"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard", default="0/1", help="I/N: fit every Nth window from the Ith")
    parser.add_argument("--report", action="store_true", help="combine the windows into the report")
    args = parser.parse_args()
    if args.report:
        report()
    else:
        shard, count = (int(v) for v in args.shard.split("/"))
        run_shard(shard, count)


if __name__ == "__main__":
    main()
