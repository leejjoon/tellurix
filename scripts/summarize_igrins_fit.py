#!/usr/bin/env python
"""Turn one standard's per-order fits into the committed quality report.

The number this exists to produce is the residual **against transmission**.
On the Arcturus atlas that curve was flat, which said the error was in the
stellar model and the telluric model was not the thing being measured. An A0V
with the hydrogen series masked has no stellar model to be wrong, so the same
curve here is a property of the atmosphere model and the reduction.

    uv run python scripts/summarize_igrins_fit.py \\
        --summary data/corrected/igrins/SDCH_20180402_0104_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Bins in effective transmission. Uneven on purpose: nearly two thirds of the
# pixels in an H-band standard sit above 0.97, so equal-width bins would put
# almost everything in one of them.
TRANSMISSION_BINS = (0.15, 0.40, 0.60, 0.80, 0.90, 0.97, 0.995, 1.01)


def pooled(summary_path: Path):
    """Every reliable pixel of every fitted order, with its model quantities.

    The transmission returned is the **effective** one, ``model_flux /
    stellar_only`` -- the operator the correction actually applies. The saved
    ``transmission`` array is unconvolved and differs from it by up to 0.48 in
    a deep core, so comparing anything against that one is a mistake.
    """

    summary = json.loads(summary_path.read_text())
    directory = summary_path.parent
    stem = summary_path.name.replace("_summary.json", "")
    columns = {name: [] for name in
               ("z", "observed_t", "model_t", "plp_t", "residual_over_continuum")}
    for row in summary["results"]:
        arrays = np.load(directory / f"{stem}_{row['name']}.npz")
        keep = arrays["reliable"] & (arrays["stellar_only"] > 1.0e-9)
        if not keep.any():
            continue
        columns["z"].append((arrays["residual"] / arrays["uncertainty"])[keep])
        columns["residual_over_continuum"].append((arrays["residual"] / arrays["continuum"])[keep])
        columns["observed_t"].append((arrays["observed"] / arrays["stellar_only"])[keep])
        columns["model_t"].append((arrays["model_flux"] / arrays["stellar_only"])[keep])
        columns["plp_t"].append(
            arrays["plp_telluric"][keep] if "plp_telluric" in arrays
            else np.full(int(keep.sum()), np.nan))
    return summary, {name: np.concatenate(values) for name, values in columns.items()}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=root / "docs/igrins_a0v_results.json")
    args = parser.parse_args()

    summary, data = pooled(args.summary)
    rows = summary["results"]
    z, observed_t, model_t, plp_t = (data["z"], data["observed_t"],
                                     data["model_t"], data["plp_t"])

    profile = []
    for low, high in zip(TRANSMISSION_BINS[:-1], TRANSMISSION_BINS[1:]):
        keep = (model_t >= low) & (model_t < high)
        if int(keep.sum()) < 200:
            continue
        profile.append({
            "transmission_low": low, "transmission_high": high,
            "pixels": int(keep.sum()),
            "rms_over_noise": float(np.sqrt(np.mean(z[keep] ** 2))),
            "median_abs_z": float(np.median(np.abs(z[keep]))),
            # In continuum units, so a bias can be read as a fraction rather
            # than in sigma, which varies across the band.
            "median_bias_over_continuum": float(
                np.median(data["residual_over_continuum"][keep])),
        })

    def against(model, name):
        entry = {"model": name, "pixels": int(np.isfinite(model).sum()), "bins": []}
        finite = np.isfinite(model)
        entry["median_difference"] = float(np.median((model - observed_t)[finite]))
        entry["rms_difference"] = float(np.sqrt(np.mean(((model - observed_t)[finite]) ** 2)))
        for low, high in ((0.15, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 0.97), (0.97, 1.02)):
            keep = finite & (observed_t >= low) & (observed_t < high)
            if int(keep.sum()) < 200:
                continue
            entry["bins"].append({
                "transmission_low": low, "transmission_high": high,
                "pixels": int(keep.sum()),
                "median_difference": float(np.median((model - observed_t)[keep])),
                "rms_difference": float(np.sqrt(np.mean(((model - observed_t)[keep]) ** 2))),
            })
        return entry

    ratio = np.array([r["residual_rms_over_noise"] for r in rows])
    velocity = np.array([r["velocity_kms"] for r in rows])
    centre = np.array([0.5 * (r["v1"] + r["v2"]) for r in rows])
    # Order 26 sits at the band edge and is the one order whose velocity does
    # not join the trend; clip rather than hand-remove it.
    steady = np.abs(velocity - np.median(velocity)) < 1.0
    slope, intercept = np.polyfit(centre[steady], velocity[steady], 1)
    water = np.array([r["log_column_scales"]["H2O"] for r in rows])

    report = {
        "observation": summary["observation"],
        "settings": summary["settings"],
        "physics": summary["physics"],
        "order_rule": summary["order_rule"],
        "orders_fitted": len(rows),
        "orders_skipped": len(summary["failures"]),
        "pixels": int(z.size),
        "quality": {
            "median_rms_over_noise": float(np.median(ratio)),
            "min_rms_over_noise": float(ratio.min()),
            "max_rms_over_noise": float(ratio.max()),
            "median_rms_over_noise_absorbing": float(
                np.median(ratio[np.array([r["median_transmission"] < 0.99 for r in rows])])),
            "median_rms_over_noise_clear": float(
                np.median(ratio[np.array([r["median_transmission"] >= 0.99 for r in rows])])),
        },
        # The headline curve. A flat profile means the telluric model is not
        # what limits the fit; a rising one means it is.
        "residual_vs_transmission": profile,
        "transmission_against_the_data": [
            against(model_t, "jax-telluric (this fit)"),
            against(plp_t, "IGRINS PLP MODEL_TELTRANS"),
        ],
        "wavelength_solution": {
            "mean_velocity_kms": float(velocity[steady].mean()),
            "scatter_kms": float(velocity[steady].std()),
            "trend_kms_per_1000_cm1": float(slope * 1000.0),
            "scatter_about_trend_kms": float(
                np.std(velocity[steady] - np.polyval([slope, intercept], centre[steady]))),
            "orders_used": int(steady.sum()),
        },
        "water": {
            "mean_log_column_scale": float(water.mean()),
            "scatter_log_column_scale": float(water.std()),
            "column_factor_on_the_seed": float(np.exp(water.mean())),
        },
        "per_order": [
            {k: r[k] for k in ("name", "v1", "v2", "kept", "reliable", "median_transmission",
                               "residual_rms_over_noise", "resolving_power_fitted",
                               "velocity_kms", "free_species", "all_stages_converged")
             if k in r}
            for r in rows
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"{report['orders_fitted']} orders, {report['pixels']} reliable pixels, "
          f"median rms/sigma {report['quality']['median_rms_over_noise']:.2f}")
    print("\n  effective T        n    rms/sigma   median|z|   bias/continuum")
    for entry in profile:
        print(f"  {entry['transmission_low']:.3f}-{entry['transmission_high']:.3f} "
              f"{entry['pixels']:8d}   {entry['rms_over_noise']:8.2f}  {entry['median_abs_z']:9.2f}   "
              f"{entry['median_bias_over_continuum']:+14.4f}")
    print("\n  transmission against the data")
    for entry in report["transmission_against_the_data"]:
        print(f"    {entry['model']:28s} median {entry['median_difference']:+.4f}  "
              f"rms {entry['rms_difference']:.4f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
