#!/usr/bin/env python
"""Is the O2 excess a width or a strength? Line by line, against line depth.

A window's O2 scale mixes two regimes: a weak line absorbs as strength x column
and is blind to its width; a saturated line absorbs through its wings, as
strength x width x column. A width error therefore shows as an excess that
grows with depth, a strength or column error as one that does not
(docs/solar_fit_plan.md §4n).

Each O2 window (1.06 um, A, B, gamma) of each run in FILES is read from its own
fit (the window's `.npz` cache, O2 free). Its O2 transmission is split into one
segment per line, at the transmission maxima between neighbouring minima, and
each segment gets its own O2 scale as one linear step from the window's, by
least squares on the unmasked pixels,

    ln s = ln s_window + sum(r D) / sum(D^2),  r = observed - model,  D = model ln T_O2,

D being d(model)/d(ln s) for a line resolved by the instrument and, to first
order, for one that is not. Stepping from the fitted optimum rather than from
the profile column matters: refitted with O2 pinned at 1, the continuum and
the other parameters absorb ~40% of the deficit before a line sees it, and not
equally for broad wings and narrow lines. A line is flagged blended when more
than a fifth of its leverage (sum D^2) falls on pixels where the rest of the
model -- water, the Sun -- takes more than 3%.

    uv run python scripts/o2_curve_of_growth.py [OUT.json]
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/o2_curve_of_growth.json"
# Run name -> its summary. The IAG atlas (scripts/fit_iag_o2.py) is the independent
# check: a different FTS, site and decade, and an air-mass average, so only its
# B/A ratio of weak-line scales is comparable.
FILES = {"ftsspec_830626_2": "ftsspec_830626_2_summary.json",
         "ftsspec_830626_3": "ftsspec_830626_3_summary.json",
         "iag_vis": "iag_o2/iag_vis_o2_summary.json"}
BANDS = {"a1Delta": (9300.0, 9440.0), "A": (12960.0, 13160.0), "B": (14370.0, 14540.0), "gamma": (15720.0, 15920.0)}
MIN_DEPTH = 0.02
BLEND = 0.03
BLENDED_LEVERAGE = 0.2
DEPTH_BINS = ((0.02, 0.1), (0.1, 0.3), (0.3, 0.85), (0.85, 1.01))


def scale(r, D):
    dd = float(np.sum(D * D))
    return (float(np.exp(np.sum(r * D) / dd)), dd) if dd > 0 else (float("nan"), 0.0)


def segments(t_o2):
    """Index ranges, one per O2 line, split at the maxima between minima."""
    minima = [i for i in range(2, len(t_o2) - 2)
              if t_o2[i] <= t_o2[i - 2:i + 3].min() and 1 - t_o2[i] > MIN_DEPTH]
    # Saturated cores are flat at zero; keep one minimum per core.
    merged = []
    for i in minima:
        if merged and np.max(t_o2[merged[-1]:i + 1]) < 0.15:
            continue
        merged.append(i)
    bounds = [0] + [a + int(np.argmax(t_o2[a:b + 1])) for a, b in zip(merged, merged[1:])] + [len(t_o2)]
    return [(m, bounds[k], bounds[k + 1]) for k, m in enumerate(merged)]


def main():
    lines, windows = [], []
    for name in FILES:
        blob = json.load(open(ROOT / "data/corrected/solar" / FILES[name]))
        for row in blob["results"]:
            band = next((b for b, (lo, hi) in BANDS.items() if lo <= row["v1"] <= hi), None)
            if band is None or "O2" not in row.get("species", []) or not row.get("parameters"):
                continue
            a = dict(np.load(ROOT / "data/corrected/solar" / row["npz"]))
            s_fit = float(np.exp(row["parameters"]["O2"]))
            o = np.argsort(a["wavenumber_cm1"])  # stored descending
            nu, ok = a["wavenumber_cm1"][o], a["mask"][o]
            flux, model, t_o2 = a["flux"][o], a["model_flux"][o], a["species_transmission_O2"][o]
            continuum = a["continuum"][o]
            # Everything in the model that is not O2: water, the Sun, the continuum shape.
            # Undefined where O2 is opaque; those pixels are masked and carry no leverage anyway.
            rest = np.where(t_o2 > 0.05, model / np.clip(t_o2, 0.05, None) / continuum, 1.0)
            other = rest / np.median(rest[ok]) < 1 - BLEND
            r = flux - model
            D = model * np.log(np.clip(t_o2, 1e-12, None))
            noise = float(np.std(r[ok]))
            s_window = s_fit * scale(r[ok], D[ok])[0]
            windows.append({"file": name, "band": band, "v1": row["v1"], "airmass": row["airmass"],
                            "fitted_scale": s_fit, "estimator_scale": s_window,
                            "residual_rms": noise})
            print(f"{name} {row['v1']:.0f}: fitted {windows[-1]['fitted_scale']:.4f} estimator {s_window:.4f}", flush=True)
            for m, lo, hi in segments(t_o2):
                use = ok[lo:hi]
                if use.sum() < 5:
                    continue
                step, dd = scale(r[lo:hi][use], D[lo:hi][use]); s = s_fit * step
                lines.append({
                    "file": name, "band": band, "airmass": row["airmass"], "v1": row["v1"],
                    "nu_cm1": float(nu[m]), "depth": float(1 - t_o2[m]), "saturated": bool(t_o2[m] < 0.15),
                    "masked_fraction": float(1 - use.mean()),
                    "scale": s, "sigma_ln_scale": noise / np.sqrt(dd),
                    "blended_leverage": float(np.sum((D[lo:hi][use] ** 2)[other[lo:hi][use]]) / dd),
                    "blended": bool(np.sum((D[lo:hi][use] ** 2)[other[lo:hi][use]]) > BLENDED_LEVERAGE * dd),
                })
    # Below 0.1 a line's scale scatters by tens of percent and its bin by several; the
    # bootstrap error, not the formal one, because solar residue dominates both.
    rng = np.random.default_rng(1)
    summary = {}
    for band in BANDS:
        for name in FILES:
            for lo, hi in DEPTH_BINS:
                sel = [x for x in lines if x["band"] == band and x["file"] == name and lo <= x["depth"] < hi
                       and not x["blended"] and np.isfinite(x["scale"])]
                if len(sel) < 3:
                    continue
                y = np.log([x["scale"] for x in sel]); w = 1 / np.array([x["sigma_ln_scale"] for x in sel]) ** 2
                boots = [np.sum(w[i] * y[i]) / w[i].sum() for i in rng.integers(0, len(sel), (2000, len(sel)))]
                summary[f"{band} {name} depth {lo}-{hi}"] = {
                    "lines": len(sel), "weighted_mean_scale": float(np.exp(np.sum(w * y) / w.sum())),
                    "bootstrap_sigma": float(np.std(boots)), "median_scale": float(np.exp(np.median(y)))}
    OUTPUT.write_text(json.dumps({"about": __doc__.split("\n\n")[0] + " scripts/o2_curve_of_growth.py.",
                                  "summary": summary, "windows": windows, "lines": lines}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
