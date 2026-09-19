#!/usr/bin/env python
"""Test the atmosphere model against a night's worth of standards at many airmasses.

The forward model already divides the optical depth by ``cos z`` using the
zenith angle measured in the header, so a fitted ``log_column_scale`` is a
scaling of the **vertical** column, with the slant path already taken out. If
the atmosphere model is right, that number does not depend on where the
telescope was pointing.

That makes two different tests out of one night:

* **Water** genuinely varies — precipitable water changes through a night — so a
  trend in H2O is weather as much as it is model error, and only the size of the
  scatter is diagnostic. Worse, airmass and time of night are strongly
  correlated in any real sequence (targets rise), so a water trend fitted
  against airmass alone will pick up the drying sky. Every trend here is
  therefore fitted against airmass, against time, and against both at once.
* **CO2, CH4 and O2 do not vary.** Their abundances are known and fixed, so any
  dependence of their fitted column on airmass is a direct measurement of error
  in the slant-path treatment or in the assumed profile shape. Nothing in the
  Arcturus atlas could measure this: its zenith angle was pinned at zero.

A third check comes free wherever the same star was observed twice close
together: the difference is repeatability, with no real atmospheric change in
between.

    uv run python scripts/analyze_igrins_ladder.py --directory data/corrected/igrins/ladder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# An order only measures a species if that species actually absorbs there. The
# driver frees anything above 0.02 of optical depth; for a column measurement
# demand rather more, or orders with a trace of absorption dominate the count
# while carrying no information.
MINIMUM_OPTICAL_DEPTH = 0.15
WELL_MIXED = ("CO2", "CH4", "O2", "N2O", "CO")


def robust(values):
    """Median and its uncertainty from the scatter, not from the formal errors.

    `FitResult.covariance` is formal and measured 3-9x too small on the atlas,
    because it assumes independent pixel errors. The spread between orders that
    each see the same sky is the honest error bar.
    """

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), 0
    centre = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - centre)))
    return centre, spread / max(np.sqrt(values.size), 1.0), int(values.size)


def weighted_line(x, y, sigma):
    """Least squares straight line with its slope uncertainty."""

    x, y = np.asarray(x, float), np.asarray(y, float)
    w = 1.0 / np.maximum(np.asarray(sigma, float), 1e-6) ** 2
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[keep], y[keep], w[keep]
    if x.size < 3 or np.ptp(x) <= 0:
        return {"slope": float("nan"), "slope_error": float("nan"),
                "intercept": float("nan"), "points": int(x.size)}
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * y).sum() / sw
    sxx = (w * (x - mx) ** 2).sum()
    slope = (w * (x - mx) * (y - my)).sum() / sxx
    intercept = my - slope * mx
    residual = y - (intercept + slope * x)
    # Inflate by the goodness of fit, so a scatter larger than the input errors
    # widens the slope error instead of being ignored.
    chi2_nu = float((w * residual**2).sum() / max(x.size - 2, 1))
    return {"slope": float(slope), "slope_error": float(np.sqrt(max(chi2_nu, 1.0) / sxx)),
            "intercept": float(intercept), "points": int(x.size),
            "scatter": float(np.std(residual))}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, default=root / "data/corrected/igrins/ladder")
    parser.add_argument("--output", type=Path, default=root / "docs/igrins_airmass_ladder.json")
    parser.add_argument("--min-optical-depth", type=float, default=MINIMUM_OPTICAL_DEPTH)
    args = parser.parse_args()

    summaries = sorted(args.directory.glob("*_summary.json"))
    if not summaries:
        raise SystemExit(f"no summaries in {args.directory}")

    frames = []
    for path in summaries:
        summary = json.loads(path.read_text())
        observation = summary["observation"]
        zenith = float(observation["zenith_angle_deg"])
        airmass = 1.0 / np.cos(np.radians(zenith))
        rows = summary["results"]
        if not rows:
            continue
        species_columns = {}
        for species in ("H2O",) + WELL_MIXED:
            usable = [
                r["log_column_scales"][species] for r in rows
                if species in r["free_species"]
                and r["max_optical_depth"].get(species, 0.0) >= args.min_optical_depth
                and species not in r.get("at_bound", [])
            ]
            centre, error, count = robust(usable)
            if count:
                species_columns[species] = {"log_column": centre, "error": error,
                                            "orders": count}
        ratio = np.array([r["residual_rms_over_noise"] for r in rows])
        frames.append({
            "frame": path.name.replace("_summary.json", ""),
            "object": observation["object"],
            "ut": observation["date_obs"][11:19],
            "mjd": observation["mjd"],
            "zenith_angle_deg": zenith,
            "airmass": float(airmass),
            "orders": len(rows),
            "median_rms_over_noise": float(np.median(ratio)),
            "surface": observation["surface"],
            "columns": species_columns,
        })
    frames.sort(key=lambda f: f["airmass"])

    start = min(f["mjd"] for f in frames)
    for f in frames:
        f["hours_from_start"] = round((f["mjd"] - start) * 24.0, 4)
    airmass_time_correlation = float(np.corrcoef(
        [f["airmass"] for f in frames], [f["hours_from_start"] for f in frames])[0, 1])

    trends = {}
    for species in ("H2O",) + WELL_MIXED:
        have = [f for f in frames if species in f["columns"]]
        if len(have) < 3:
            continue
        y = np.array([f["columns"][species]["log_column"] for f in have])
        x_airmass = np.array([f["airmass"] for f in have])
        x_hours = np.array([f["hours_from_start"] for f in have])
        # Both at once, so a real airmass dependence is not confused with the
        # sky drying and the other way round.
        design = np.vstack([np.ones_like(x_airmass), x_airmass, x_hours]).T
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        trends[species] = {
            **weighted_line(x_airmass, y, [f["columns"][species]["error"] for f in have]),
            "vs_time": weighted_line(x_hours, y,
                                     [f["columns"][species]["error"] for f in have]),
            "joint_slope_per_airmass": float(coefficients[1]),
            "joint_slope_per_hour": float(coefficients[2]),
            "joint_scatter": float(np.std(y - design @ coefficients)),
            "mean_log_column": float(np.mean([f["columns"][species]["log_column"] for f in have])),
            "scatter_between_frames": float(
                np.std([f["columns"][species]["log_column"] for f in have])),
            "median_orders_per_frame": float(
                np.median([f["columns"][species]["orders"] for f in have])),
        }

    # Repeats of one star close in time: the difference is repeatability, since
    # nothing about the sky had time to change.
    repeats = []
    for index, left in enumerate(frames):
        for right in frames[index + 1:]:
            if left["object"] != right["object"]:
                continue
            minutes = abs(left["mjd"] - right["mjd"]) * 1440.0
            shared = sorted(set(left["columns"]) & set(right["columns"]))
            repeats.append({
                "object": left["object"],
                "frames": [left["frame"], right["frame"]],
                "minutes_apart": round(minutes, 1),
                "airmass": [round(left["airmass"], 3), round(right["airmass"], 3)],
                "delta_log_column": {
                    s: round(right["columns"][s]["log_column"] - left["columns"][s]["log_column"], 4)
                    for s in shared},
            })
    repeats.sort(key=lambda r: r["minutes_apart"])

    quality = weighted_line([f["airmass"] for f in frames],
                            [f["median_rms_over_noise"] for f in frames],
                            [0.1] * len(frames))

    report = {
        "night": frames[0]["frame"].split("_")[1] if frames else None,
        "telescope": json.loads(summaries[0].read_text())["observation"]["telescope"],
        "band": json.loads(summaries[0].read_text())["observation"]["band"],
        "frames": len(frames),
        "airmass_range": [frames[0]["airmass"], frames[-1]["airmass"]],
        "minimum_optical_depth": args.min_optical_depth,
        "airmass_time_correlation": airmass_time_correlation,
        "note": (
            "log_column_scale scales the VERTICAL column; the model already "
            "divides optical depth by cos(z) with the measured zenith angle. A "
            "slope against airmass is therefore a model error, not a geometry "
            "term. For the well-mixed species it cannot be anything else, "
            "because their abundances do not vary."
        ),
        "column_vs_airmass": trends,
        "quality_vs_airmass": quality,
        "repeat_observations": repeats,
        "per_frame": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"{len(frames)} frames, airmass {frames[0]['airmass']:.3f}-{frames[-1]['airmass']:.3f}, "
          f"{report['telescope']} {report['band']} band\n")
    print(f"{'frame':22s} {'object':14s} {'UT':9s} {'AM':>6} {'rms/sig':>8}  columns")
    for f in frames:
        cols = "  ".join(f"{s} {v['log_column']:+.3f}({v['orders']})"
                         for s, v in sorted(f["columns"].items()))
        print(f"{f['frame']:22s} {f['object'][:14]:14s} {f['ut']:9s} {f['airmass']:6.3f} "
              f"{f['median_rms_over_noise']:8.2f}  {cols}")

    print(f"\nvertical column scale against airmass (expected slope: zero)")
    print(f"airmass and time of night correlate at {airmass_time_correlation:+.3f} on this "
          f"night, so both are fitted and then both together")
    print(f"  {'':6s} {'per airmass':>22s} {'per hour':>22s} {'joint':>18s}")
    for species, t in trends.items():
        sig = abs(t["slope"] / t["slope_error"]) if t["slope_error"] > 0 else float("nan")
        v = t["vs_time"]
        sig_time = abs(v["slope"] / v["slope_error"]) if v["slope_error"] > 0 else float("nan")
        print(f"  {species:6s} {t['slope']:+8.4f}+-{t['slope_error']:.4f} ({sig:4.1f}s) "
              f"{v['slope']:+8.4f}+-{v['slope_error']:.4f} ({sig_time:4.1f}s) "
              f"{t['joint_slope_per_airmass']:+8.4f}/AM {t['joint_slope_per_hour']:+7.4f}/hr")

    if repeats:
        print("\nrepeat observations of one star")
        for r in repeats[:6]:
            deltas = " ".join(f"{s} {v:+.3f}" for s, v in r["delta_log_column"].items())
            print(f"  {r['object'][:16]:16s} {r['minutes_apart']:6.1f} min apart, "
                  f"AM {r['airmass'][0]:.2f}->{r['airmass'][1]:.2f}:  {deltas}")

    print(f"\nresidual against airmass: {quality['slope']:+.3f} +- {quality['slope_error']:.3f} "
          f"sigma per unit airmass")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
