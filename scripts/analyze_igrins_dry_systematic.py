#!/usr/bin/env python
"""Attribute the per-night systematic in the IGRINS well-mixed columns (GitHub #1).

Three nights give CO2 and CH4 airmass slopes that disagree in sign, and two of
them a dependence on time of night that neither gas can have. This reproduces
that table from the published run records alone, then asks what the thing that
moves the columns looks like, frame by frame, and rules out what it is not.

It reads only ``record.h5`` files, which are what ``results_archive.py fetch``
restores -- not the per-frame ``*_summary.json``. Those are not published, and
in every K run the 2026-09-30 refit left the pre-fix shard summaries
(``*.s0_summary.json``) beside the merged ones, so a glob over a directory now
counts each K order twice, half of it at the defective zenith angle.

One fixed set of orders per species, rather than a per-frame optical-depth cut:
the record does not carry the optical depth, and the orders that pass 0.15 are
the same on every night anyway (checked against the summaries), so a fixed set
also keeps nights comparable. The ``.npz`` fit caches are used for one test, the
residual against transmission, only where they exist locally.

    uv run python scripts/analyze_igrins_dry_systematic.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from tellurix.io import load_atmosphere_csv
from tellurix.record import file_sha256, read_record

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/corrected/igrins"

# (label, H run, K run, profile kind). The analytic runs are the ones the
# three-night table in docs/igrins_a0v.md was built from; the ERA5 runs are
# the same frames against a reanalysis profile, and Gemini South has only that.
NIGHTS = (
    ("dct2018", "ladder_a0v", "ladder_k_a0v", "analytic"),
    ("mcd2017", "mcdonald_h", "mcdonald_k", "analytic"),
    ("dct2016", "dct2016_h", "dct2016_k", "analytic"),
    ("dct2018", "era2018_h", "era2018_k", "era5"),
    ("mcd2017", "era2017_h", "era2017_k", "era5"),
    ("dct2016", "era2016_h", "era2016_k", "era5"),
    ("gemini2021", "gemini2021_h", "gemini2021_k", "era5"),
    # The analytic runs again with each frame's airmass taken over its whole
    # exposure sequence (igrins_pointing.py --sequence), when present.
    ("dct2018", "seq_dct2018_h", "seq_dct2018_k", "sequence"),
    ("mcd2017", "seq_mcd2017_h", "seq_mcd2017_k", "sequence"),
    ("dct2016", "seq_dct2016_h", "seq_dct2016_k", "sequence"),
)

# The orders where each species reaches 0.15 of vertical optical depth on
# every frame of all three nights. H110 (CO2) and H103 (CH4) pass on some
# nights only and are left out so every night measures the same thing.
ORDERS = {
    "CO2": ("H108", "H109", *(f"H{n}" for n in range(111, 118)),
            *(f"K{n}" for n in range(85, 93))),
    "CH4": (*(f"H{n}" for n in range(104, 111)), *(f"K{n}" for n in range(73, 84))),
}
COMPONENTS = tuple(f"{species}{band}" for species in ORDERS for band in "HK")
WATER_MOLAR_MASS = 18.015


def text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def robust(values):
    """Median and its error from the spread between orders, as the ladder does."""

    values = np.asarray(values, float)
    centre = float(np.median(values))
    spread = 1.4826 * float(np.median(np.abs(values - centre)))
    return centre, spread / np.sqrt(max(values.size, 1))


def line(x, y, sigma):
    """Weighted straight line, its slope error inflated by the scatter."""

    x, y, w = np.asarray(x, float), np.asarray(y, float), 1.0 / np.asarray(sigma, float) ** 2
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    sxx = float((w * (x - mx) ** 2).sum())
    slope = float((w * (x - mx) * (y - my)).sum() / sxx)
    chi2_nu = float((w * (y - my - slope * (x - mx)) ** 2).sum() / max(x.size - 2, 1))
    return slope, float(np.sqrt(max(chi2_nu, 1.0) / sxx))


def precipitable_water_mm(profile_path: Path) -> float:
    profile = load_atmosphere_csv(profile_path)
    dp_pa = np.diff(np.asarray(profile.pressure_edges_bar)) * 1e5
    vmr = np.asarray(profile.vmr["H2O"])
    mass_fraction = vmr * WATER_MOLAR_MASS / np.asarray(profile.mean_molecular_weight_g_mol)
    return float((mass_fraction * dp_pa / np.asarray(profile.gravity_m_s2)).sum())


def load_night(h_run: str, k_run: str):
    """Per frame: geometry, the header weather, and each species' column per band."""

    frames: dict[str, dict] = {}
    records = [read_record(RUNS / run / "record.h5") for run in (h_run, k_run)]
    for record in records:
        for row in record.pages:
            band, order = text(row["band"]), f"{text(row['band'])}{row['order_number']}"
            key = text(row["frame"]).split("_", 1)[1][:13]
            frame = frames.setdefault(key, {"orders": {}, "water": [], "continuum": {}, "rows": {}})
            frame["orders"][order] = {s: float(row[f"log_column_{s}"]) for s in (*ORDERS, "H2O")}
            frame["rows"][order] = {k: float(row[k]) for k in (
                "log_column_H2O", "residual_rms_over_noise", "lsf_sigma_kms")}
            frame["water"].append(float(row["log_column_H2O"]))
            frame["continuum"].setdefault(band, []).append(float(row["continuum_level"]))
            if band == "K" or "airmass" not in frame:
                frame.update(airmass=float(row["airmass"]), mjd=float(row["mjd"]),
                             object=text(row["object"]).strip(),
                             surface_temperature_k=float(row["surface_temperature_k"]),
                             surface_pressure_hpa=float(row["surface_pressure_hpa"]))
    start = min(f["mjd"] for f in frames.values())
    for frame in frames.values():
        frame["hours"] = (frame["mjd"] - start) * 24.0
        frame["log_water"] = float(np.median(frame["water"]))
        frame["continuum_counts"] = float(np.median(frame["continuum"].get("K", [np.nan])))
        for species, orders in ORDERS.items():
            for band in "HK" + "*":
                values = [frame["orders"][o][species] for o in orders
                          if o in frame["orders"] and (band == "*" or o[0] == band)]
                frame[f"{species}{band if band != '*' else ''}"] = robust(values)
    # Runs made in a worktree record that worktree's absolute path; the profile
    # is found by name here and held to the hash the run recorded.
    profile = ROOT / "data/profiles" / Path(text(records[1].inputs["profile"])).name
    if file_sha256(profile) != text(records[1].inputs["profile_sha256"]):
        raise SystemExit(f"{profile} is not the profile {k_run} was fitted with")
    return dict(sorted(frames.items(), key=lambda kv: kv[1]["mjd"])), profile


def dry_factor(frames):
    """The common part: each band-species median about its night mean, averaged.

    Four estimates per frame from two detectors and two molecules; where they
    agree, the thing moving them belongs to the exposure, not to a band.
    """

    means = {c: np.mean([f[c][0] for f in frames.values()]) for c in COMPONENTS}
    for f in frames.values():
        f["dry"] = float(np.mean([f[c][0] - means[c] for c in COMPONENTS]))
        f["dry_components"] = {c: round(f[c][0] - means[c], 4) for c in COMPONENTS}
    return means


def order_response(frames):
    """Each order's column against the frame's dry factor: slope and r.

    A pure scaling of the dry optical depth moves every order alike -- slope 1
    at weak and saturated lines, in both bands. Anything acting on line shape
    (LSF, blaze, fringe, veiling) would move orders by their depth and band.
    """

    names = list(frames)
    dry = np.array([frames[n]["dry"] for n in names])
    out = {}
    for species, orders in ORDERS.items():
        for order in orders:
            y = np.array([frames[n]["orders"].get(order, {}).get(species, np.nan) for n in names])
            ok = np.isfinite(y)
            if ok.sum() < 5:
                continue
            slope = float(np.polyfit(dry[ok], y[ok] - y[ok].mean(), 1)[0])
            out[f"{species} {order}"] = {"slope": round(slope, 3),
                                         "r": round(float(np.corrcoef(dry[ok], y[ok])[0, 1]), 3)}
    return out


def residual_curves(k_run, frames, chosen, bins):
    """Mean residual over the stellar-continuum model, binned by transmission.

    From the local .npz caches; None where they are absent (they are not in the
    published archive).
    """

    curves = {}
    for species, orders in ORDERS.items():
        for label, names in chosen.items():
            total, count = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
            for name in names:
                for order in orders:
                    if order[0] != "K":
                        continue
                    found = glob.glob(str(RUNS / k_run / f"SDCK_{name}_{order}.npz"))
                    if not found:
                        continue
                    cache = np.load(found[0])
                    transmission = cache["transmission"]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        relative = cache["residual"] * transmission / cache["model_flux"]
                    index = np.digitize(transmission, bins) - 1
                    for k in range(len(bins) - 1):
                        keep = cache["reliable"] & (index == k) & np.isfinite(relative)
                        total[k] += relative[keep].sum()
                        count[k] += keep.sum()
            if count.sum() == 0:
                return None
            curves[f"{species}K {label}"] = [round(float(v), 5) for v in total / np.maximum(count, 1)]
    return curves


def veiling_signature(epsilon, bins):
    """What an additive offset epsilon leaves once the column has absorbed it.

    Observed (T + eps)/(1 + eps) fitted by T**a: weak lines fix a = 1/(1 + eps),
    so the leftover at transmission T is (T + eps)/(1 + eps) - T**a.
    """

    centres = 0.5 * (bins[1:] + bins[:-1])
    exponent = 1.0 / (1.0 + epsilon)
    return [round(float((t + epsilon) / (1.0 + epsilon) - t**exponent), 5) for t in centres]


def chi2_common(values, errors):
    values, w = np.asarray(values), 1.0 / np.asarray(errors) ** 2
    mean = float((w * values).sum() / w.sum())
    return float((w * (values - mean) ** 2).sum()), mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/igrins_dry_systematic.json")
    args = parser.parse_args()

    bins = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.92, 0.97, 1.01])
    report = {"orders": {k: list(v) for k, v in ORDERS.items()}, "runs": {}}
    loaded = {}
    for night, h_run, k_run, kind in NIGHTS:
        if not all((RUNS / run / "record.h5").exists() for run in (h_run, k_run)):
            continue
        frames, profile_path = load_night(h_run, k_run)
        loaded[(night, kind)] = frames
        means = dry_factor(frames)
        names = list(frames)
        x_am = np.array([frames[n]["airmass"] for n in names])
        x_hr = np.array([frames[n]["hours"] for n in names])
        slopes = {}
        for species in (*ORDERS, *COMPONENTS):
            y = np.array([frames[n][species][0] for n in names])
            e = np.maximum([frames[n][species][1] for n in names], 1e-3)
            design = np.vstack([np.ones_like(x_am), x_am, x_hr]).T
            joint = np.linalg.lstsq(design, y, rcond=None)[0]
            slopes[species] = {"per_airmass": line(x_am, y, e), "per_hour": line(x_hr, y, e),
                               "joint_per_airmass": float(joint[1]),
                               "joint_per_hour": float(joint[2]),
                               "frame_scatter": float(np.std(y)),
                               "median_error": float(np.median(e))}
        dry = np.array([frames[n]["dry"] for n in names])
        component = {c: np.array([frames[n][c][0] for n in names]) for c in COMPONENTS}
        pwv = precipitable_water_mm(profile_path) * float(
            np.exp(np.median([frames[n]["log_water"] for n in names])))
        entry = {
            "night": night, "profile": kind, "runs": [h_run, k_run], "frames": len(names),
            "airmass_range": [float(x_am.min()), float(x_am.max())],
            "precipitable_water_mm": round(pwv, 2),
            "slopes": slopes,
            "dry_factor": {
                "scatter": float(np.std(dry)),
                "agreement": {
                    "CO2 H vs K": float(np.corrcoef(component["CO2H"], component["CO2K"])[0, 1]),
                    "CH4 H vs K": float(np.corrcoef(component["CH4H"], component["CH4K"])[0, 1]),
                    "CO2 vs CH4": float(np.corrcoef(
                        [frames[n]["CO2"][0] for n in names],
                        [frames[n]["CH4"][0] for n in names])[0, 1]),
                },
                "correlation_with": {
                    key: float(np.corrcoef(dry, [frames[n][key] for n in names])[0, 1])
                    for key in ("airmass", "hours", "surface_temperature_k", "log_water",
                                "continuum_counts")
                    if np.std([frames[n][key] for n in names]) > 0},
            },
            "per_frame": [{"frame": n, "object": frames[n]["object"],
                           "airmass": round(frames[n]["airmass"], 3),
                           "hours": round(frames[n]["hours"], 3),
                           "log_water": round(frames[n]["log_water"], 4),
                           "continuum_counts": round(frames[n]["continuum_counts"], 1),
                           "surface_temperature_k": round(frames[n]["surface_temperature_k"], 2),
                           "dry": round(frames[n]["dry"], 4),
                           "components": frames[n]["dry_components"]} for n in names],
        }
        # Water's departure from its own trend in time against the factor's: a
        # path-like factor shared by every gas would put them at slope 1.
        water = np.array([frames[n]["log_water"] for n in names])
        entry["water_anomaly"] = {}
        for degree in (1, 2, 3):
            w = water - np.polyval(np.polyfit(x_hr, water, degree), x_hr)
            d = dry - np.polyval(np.polyfit(x_hr, dry, degree), x_hr)
            entry["water_anomaly"][f"trend_degree_{degree}"] = {
                "r": round(float(np.corrcoef(w, d)[0, 1]), 3),
                "slope": round(float(np.polyfit(d, w, 1)[0]), 2)}
        if kind == "analytic" or night == "gemini2021":
            entry["order_response"] = order_response(frames)
        if kind == "analytic" and np.std(dry) > 0.01:
            ranked = sorted(names, key=lambda n: frames[n]["dry"])
            curves = residual_curves(k_run, frames, {"most dry": ranked[-2:],
                                                     "least dry": ranked[:2]}, bins)
            if curves is not None:
                spread = float(frames[ranked[-1]]["dry"] - frames[ranked[0]]["dry"])
                entry["residual_vs_transmission"] = {
                    "bins": bins.tolist(), "frames": {"most dry": ranked[-2:],
                                                      "least dry": ranked[:2]},
                    "curves": curves,
                    # What veiling large enough to make this spread would leave.
                    "veiling_signature": {"epsilon": round(spread / 2, 4),
                                          "plus": veiling_signature(spread / 2, bins),
                                          "minus": veiling_signature(-spread / 2, bins)}}
        report["runs"][f"{night}_{kind}"] = entry

    # The three-night consistency test, as docs/igrins_a0v.md quotes it.
    report["between_nights"] = {}
    for kind, nights in (("analytic", ("dct2018", "mcd2017", "dct2016")),
                         ("era5", ("dct2018", "mcd2017", "dct2016")),
                         ("sequence", ("dct2018", "mcd2017", "dct2016"))):
        if not all(f"{n}_{kind}" in report["runs"] for n in nights):
            continue
        for species in ORDERS:
            values = [report["runs"][f"{n}_{kind}"]["slopes"][species]["per_airmass"] for n in nights]
            chi2, mean = chi2_common([v[0] for v in values], [v[1] for v in values])
            report["between_nights"][f"{species} {kind}"] = {
                "slopes": [round(v[0], 4) for v in values],
                "errors": [round(v[1], 4) for v in values],
                "chi2_common_2dof": round(chi2, 2), "weighted_mean": round(mean, 4),
                "scatter": round(float(np.std([v[0] for v in values], ddof=1)), 4)}

    # A whole reanalysis profile against the analytic one moves each frame's
    # columns by this much; a per-frame profile changes far less than that.
    report["profile_sensitivity"] = {}
    for night in ("dct2018", "mcd2017", "dct2016"):
        a, b = loaded[(night, "analytic")], loaded[(night, "era5")]
        shared = [n for n in a if n in b]
        shift = {c: np.array([b[n][c][0] - a[n][c][0] for n in shared]) for c in COMPONENTS}
        temperatures = [a[n]["surface_temperature_k"] for n in shared]
        report["profile_sensitivity"][night] = {
            "mean_shift": {c: round(float(v.mean()), 4) for c, v in shift.items()},
            "per_frame_scatter_of_shift": {c: round(float(v.std()), 4) for c, v in shift.items()},
            "dry_factor_scatter": round(float(np.std([a[n]["dry"] for n in shared])), 4),
            "surface_temperature_range_k": round(float(np.ptp(temperatures)), 2)}

    # Same star, minutes apart: if the detector's response were nonlinear, the
    # dry factor would follow the count level between the two.
    pairs = []
    for (night, kind), frames in loaded.items():
        if kind != "analytic":
            continue
        names = list(frames)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                fl, fr = frames[left], frames[right]
                if fl["object"] == fr["object"] and abs(fr["mjd"] - fl["mjd"]) * 1440 < 30:
                    pairs.append({"night": night, "frames": [left, right], "object": fl["object"],
                                  "minutes": round(abs(fr["mjd"] - fl["mjd"]) * 1440, 1),
                                  "count_ratio": round(fr["continuum_counts"] / fl["continuum_counts"], 2),
                                  "dry_difference": round(fr["dry"] - fl["dry"], 4)})
    report["repeat_pairs"] = pairs

    # The cause: each frame combines a sequence of exposures whose airmass the
    # header does not describe. Predicted dry factor, ln(sequence airmass /
    # airmass the fit used), from the pointing.json igrins_pointing.py
    # --sequence writes beside each frame, against the measured one.
    report["sequence_prediction"] = {}
    for night, date in (("dct2018", "20181220"), ("mcd2017", "20170420"),
                        ("dct2016", "20161208")):
        frames = loaded.get((night, "analytic"))
        sidecars = {n: ROOT / "data/igrins" / f"{date}_{n.split('_')[1]}" / "pointing.json"
                    for n in frames} if frames else {}
        if not sidecars or not all(p.exists() for p in sidecars.values()):
            continue
        names = list(frames)
        pointing = {n: json.loads(sidecars[n].read_text()) for n in names}
        if any(pointing[n].get("source") != "sequence" for n in names):
            continue
        measured = np.array([frames[n]["dry"] for n in names])
        predicted = np.array([np.log(1.0 / np.cos(np.radians(pointing[n]["zenith_angle_deg"]))
                                     / frames[n]["airmass"]) for n in names])
        left = (measured - measured.mean()) - (predicted - predicted.mean())
        report["sequence_prediction"][night] = {
            "frames": names,
            "exposures": [pointing[n]["sequence"]["exposures"] for n in names],
            "sequence_minutes": [round(pointing[n]["sequence"]["duration_s"] / 60, 1) for n in names],
            "predicted": [round(float(v), 4) for v in predicted],
            "measured": [round(float(v), 4) for v in measured],
            "r": round(float(np.corrcoef(predicted, measured)[0, 1]), 3),
            "slope": round(float(np.polyfit(predicted, measured, 1)[0]), 2),
            "measured_scatter": round(float(measured.std()), 4),
            "left_after_prediction": round(float(left.std()), 4)}

    # Two diagnostic refits of McDonald 2017, when present locally (they are
    # not published): the response pattern off, and CO2/CH4 pinned at each
    # order's night median (fit_igrins_standard.py --fix-columns-from).
    refits = {"no_pattern": ("diag_mcd2017_nopattern_h", "diag_mcd2017_nopattern_k"),
              "fixed_dry": ("diag_mcd2017_fixdry_h", "diag_mcd2017_fixdry_k")}
    if all((RUNS / run / "record.h5").exists() for pair in refits.values() for run in pair):
        base = loaded[("mcd2017", "analytic")]
        names = list(base)
        old = np.array([base[n]["dry"] for n in names])
        report["refits_mcd2017"] = {}
        free, _ = load_night(*refits["no_pattern"])
        dry_factor(free)
        new = np.array([free[n]["dry"] for n in names])
        report["refits_mcd2017"]["no_pattern"] = {
            "dry_factor_scatter": round(float(new.std()), 4),
            "with_pattern": round(float(old.std()), 4),
            "r_with_pattern": round(float(np.corrcoef(old, new)[0, 1]), 3)}
        pinned, _ = load_night(*refits["fixed_dry"])
        changes = {}
        for band in "HK":
            for key in ("H2O", "residual_rms_over_noise", "lsf_sigma_kms"):
                per_frame = []
                for n in names:
                    rows = [(pinned[n]["rows"][o], base[n]["rows"][o])
                            for o in pinned[n]["rows"] if o[0] == band and o in base[n]["rows"]]
                    if key == "H2O":
                        per_frame.append(np.median([a["log_column_H2O"] - b["log_column_H2O"]
                                                    for a, b in rows]))
                    else:
                        per_frame.append(np.median([a[key] / b[key] - 1.0 for a, b in rows]))
                per_frame = np.array(per_frame)
                changes[f"{key} {band}"] = {
                    "per_frame": [round(float(v), 4) for v in per_frame],
                    "slope_vs_dry_factor": round(float(np.polyfit(old, per_frame, 1)[0]), 3),
                    "r": round(float(np.corrcoef(old, per_frame)[0, 1]), 3)}
        report["refits_mcd2017"]["fixed_dry"] = {
            "frames": names, "dry_factor": [round(float(v), 4) for v in old],
            "note": "H2O is the change in log column; the others are fractional changes",
            "changes": changes}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))

    print("per-night slopes of the vertical log column (fixed orders, both bands)")
    print(f"  {'run':20s} {'PWV':>5s} {'species':7s} {'per airmass':>18s} {'per hour':>18s} {'joint/AM':>9s}")
    for name, entry in report["runs"].items():
        for species in ORDERS:
            s = entry["slopes"][species]
            print(f"  {name:20s} {entry['precipitable_water_mm']:5.1f} {species:7s} "
                  f"{s['per_airmass'][0]:+.4f}+-{s['per_airmass'][1]:.4f} "
                  f"{s['per_hour'][0]:+.4f}+-{s['per_hour'][1]:.4f} {s['joint_per_airmass']:+9.4f}")
    print("\nbetween nights (2 dof)")
    for name, b in report["between_nights"].items():
        print(f"  {name:13s} slopes {b['slopes']}  chi2 {b['chi2_common_2dof']:.1f}  scatter {b['scatter']:.4f}")
    print("\nthe per-frame dry factor")
    for name, entry in report["runs"].items():
        d = entry["dry_factor"]
        print(f"  {name:20s} PWV {entry['precipitable_water_mm']:5.1f} mm  scatter {d['scatter']:.4f}  "
              + "  ".join(f"{k} r={v:+.2f}" for k, v in d["agreement"].items()))
    for name, entry in report["runs"].items():
        if "residual_vs_transmission" in entry:
            r = entry["residual_vs_transmission"]
            print(f"\n  {name}: residual vs transmission, bins {r['bins']}")
            for k, v in r["curves"].items():
                print(f"    {k:22s} {v}")
            print(f"    veiling +{r['veiling_signature']['epsilon']:.3f} would leave "
                  f"{r['veiling_signature']['plus']}")
    print("\nprofile sensitivity (analytic -> ERA5), per-frame scatter of the shift")
    for night, s in report["profile_sensitivity"].items():
        print(f"  {night}: {s['per_frame_scatter_of_shift']}  vs dry scatter {s['dry_factor_scatter']}, "
              f"surface T range {s['surface_temperature_range_k']} K")
    print("\nrepeat pairs")
    for p in pairs:
        print(f"  {p}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
