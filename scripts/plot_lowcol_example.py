#!/usr/bin/env python
"""One water window of ftsspec_901218_{4,5}, fitted before and after the lowest kilometre got its own water.

Old: the run published 2026-10-25, one water scale for the whole ERA5 column. New: the
2026-11-09 run, H2O_LOW pinned per file (docs/solar_fit_plan.md, "Done 2026-11-09: files
4 and 5 refitted with their own boundary-layer water"). Reads the cached window arrays of
both runs; the old ones are in data/corrected/superseded_pre_lowcol/, which the results
archive does not carry, so this reruns only in a checkout that kept them.

    uv run python scripts/plot_lowcol_example.py   # -> docs/kitt_peak_lowcol_example.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHECKOUT = Path(__file__).resolve().parents[1]
ROOT = CHECKOUT / "data/corrected"
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--output", type=Path, default=CHECKOUT / "docs/kitt_peak_lowcol_example.png")
OUT = parser.parse_args().output
V1, ZOOM = 6766, (6770.6, 6773.45)
INK, OLD, NEW, NOISE = "#222222", "#c4572d", "#2766b5", "#e6e9eb"
AIRMASS = {"4": 4.73, "5": 1.985}


def load(path):
    z = np.load(path)
    w = z["wavenumber_cm1"]
    c = z["continuum"]
    return dict(w=w, obs=z["flux"] / c, model=z["model_flux"] / c,
                resid=(z["flux"] - z["model_flux"]) / c * 100, sigma=z["uncertainty"] / c * 100,
                mask=z["mask"].astype(bool))


def pixel_axis(ax, w):
    # npz arrays run from high to low wavenumber; pixel index i <-> w[0] - i * dw
    dw = (w[0] - w[-1]) / (len(w) - 1)
    sec = ax.secondary_xaxis("top", functions=(lambda x: (w[0] - x) / dw, lambda p: w[0] - p * dw))
    sec.tick_params(labelsize=7, colors="#777777", length=2)
    sec.set_xlabel("pixel", fontsize=7, color="#777777", labelpad=2)


def rms(r, m, sel):
    k = m & sel
    return np.sqrt(np.mean(r[k] ** 2))


plt.rcParams.update({"font.size": 9, "axes.spines.right": False, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555"})
fig, axes = plt.subplots(4, 2, figsize=(13, 11), gridspec_kw=dict(height_ratios=[1.1, 0.8, 1.3, 0.9]))
for col, f in enumerate(("4", "5")):
    old = load(ROOT / f"superseded_pre_lowcol/solar/ftsspec_901218_{f}_{V1}.npz")
    new = load(ROOT / f"solar/ftsspec_901218_{f}_{V1}.npz")
    w, m = new["w"], new["mask"] & old["mask"]
    zoom = (w >= ZOOM[0]) & (w <= ZOOM[1])
    allpix = np.ones_like(m)

    ax = axes[0, col]
    ax.plot(w, np.where(m, new["obs"], np.nan), color=INK, lw=0.6, label="observed")
    ax.plot(w, new["model"], color=NEW, lw=0.7, label="new fit")
    ax.axvspan(*ZOOM, color="#f3e7b3", alpha=0.6, lw=0, zorder=0)
    ax.set_xlim(w[-1], w[0]); ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("flux / continuum")
    ax.set_title(f"file {f} · air mass {AIRMASS[f]} · {V1}–{V1 + 30} cm⁻¹ (1.48 µm, 1.4 µm band edge)",
                 fontsize=10, loc="left", pad=22)
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    ax.text(np.mean(ZOOM), 1.05, "zoom below", ha="center", va="top", fontsize=7, color="#8a7a30")
    pixel_axis(ax, w)

    ax = axes[1, col]
    ax.plot(w, np.where(m, old["resid"], np.nan), color=OLD, lw=0.6, label=f"old fit  rms {rms(old['resid'], m, allpix):.2f}%")
    ax.plot(w, np.where(m, new["resid"], np.nan), color=NEW, lw=0.6, label=f"new fit  rms {rms(new['resid'], m, allpix):.2f}%")
    ax.axhline(0, color="#999999", lw=0.5)
    ax.axvspan(*ZOOM, color="#f3e7b3", alpha=0.6, lw=0, zorder=0)
    ax.set_xlim(w[-1], w[0]); ax.set_ylim(-6, 6)
    ax.set_ylabel("observed − model\n(% of continuum)")
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    pixel_axis(ax, w)

    ax = axes[2, col]
    k = zoom & m
    ax.plot(w[k], new["obs"][k], "o", ms=2.2, color=INK, label="observed", zorder=3)
    ax.plot(w[zoom], old["model"][zoom], color=OLD, lw=1.3, ls=(0, (4, 2)), label="old fit (one water scale)")
    ax.plot(w[zoom], new["model"][zoom], color=NEW, lw=1.3, label="new fit (lowest 1 km scaled separately)")
    ax.set_xlim(*ZOOM); ax.set_ylim(min(0.0, new["obs"][k].min() - 0.03), 1.06)
    ax.set_ylabel("flux / continuum")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    pixel_axis(ax, w)

    ax = axes[3, col]
    ax.fill_between(w[zoom], -new["sigma"][zoom], new["sigma"][zoom], color=NOISE, lw=0, label="±1σ pixel noise")
    ax.plot(w[k], old["resid"][k], color=OLD, lw=1.0, ls=(0, (4, 2)), label=f"old fit  rms {rms(old['resid'], m, zoom):.2f}%")
    ax.plot(w[k], new["resid"][k], color=NEW, lw=1.0, label=f"new fit  rms {rms(new['resid'], m, zoom):.2f}%")
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xlim(*ZOOM); ax.set_ylim(-7, 4)
    ax.set_ylabel("observed − model\n(% of continuum)")
    ax.set_xlabel("wavenumber (cm⁻¹)")
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=3)
    pixel_axis(ax, w)

fig.suptitle("Water lines too shallow in the old fit; the new fit gives the lowest kilometre its own water scale",
             fontsize=12, x=0.01, ha="left")
fig.text(0.01, 0.955, "Kitt Peak FTS 1990-12-18, disc centre. Old: published 2026-10-25. New: 2026-11-09, "
         "H2O_LOW pinned at −1.57 (file 4) and −0.47 (file 5). Negative residual at a line = model too shallow.",
         fontsize=8.5, color="#555555")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
