#!/usr/bin/env python
"""Paper figure: one IGRINS A0V standard fitted on the dry night (DCT 2018-12-20).

Reads the sequence-airmass standards run (``data/corrected/igrins/seq_dct2018_{h,k}``)
-- its per-frame summaries and cached order arrays -- and refits nothing. The
frame and the two orders are chosen by rule, not by eye:

* the order: among the orders with strong telluric absorption on this night
  (median over frames of the fraction of fitted pixels with transmission
  < 0.8 at least ``STRONG_FRACTION``), the one whose median per-pixel residual
  z over the night's frames is closest to the band's median over every
  order-frame. One per band.
* the frame: the frame whose median z over all its H and K orders is the
  night's median frame (lower middle of an even count).

The metric is ``residual_z_rms`` -- rms over the reliable pixels of
(observed - model) / pipeline sigma -- which the repository uses to compare fits
(packages/tellurix-igrins/CLAUDE.md). The two order-frames the docs call
pathological (H107 of frame 108, H104 of frame 84) are left out of the
percentiles.

    UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_igrins_standard.py

Writes PDF and PNG (200 dpi) to ``paper/figures/output/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/figures/output"
RUNS = {"H": ROOT / "data/corrected/igrins/seq_dct2018_h",
        "K": ROOT / "data/corrected/igrins/seq_dct2018_k"}
PATHOLOGICAL = {("SDCH_20181220_0108", "H107"), ("SDCH_20181220_0084", "H104")}
STRONG_FRACTION = 0.25
ZOOM_PIXELS = 200

OBSERVED = "black"
MODEL = "#D55E00"       # Okabe-Ito vermillion
STAR = "#0072B2"        # Okabe-Ito blue
TELLURIC = "#009E73"    # Okabe-Ito bluish green
SHADE = "#E8E8E8"


def style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": False, "ytick.right": True,
        "lines.linewidth": 0.7, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.dpi": 200,
    })


def read_rows(run: Path) -> list[dict]:
    from tellurix_igrins import summary_paths

    rows = []
    for path in summary_paths(run, "*summary.json"):
        blob = json.loads(path.read_text())
        stem = Path(blob["observation"]["path"]).name.split(".")[0]
        for row in blob["results"]:
            rows.append({**row, "stem": stem, "frame": stem.split("_")[-1],
                         "object": blob["observation"]["object"],
                         "spec": blob["observation"]["path"]})
    return rows


def load(run: Path, stem: str, name: str) -> dict:
    with np.load(run / f"{stem}_{name}.npz") as stored:
        return {k: np.asarray(stored[k]) for k in stored.files}


def strong_fraction(run: Path, row: dict) -> float:
    d = load(run, row["stem"], row["name"])
    mask = d["mask"].astype(bool)
    return float(np.mean(d["transmission"][mask] < 0.8))


def pixel_axis(spec: str, number: int, nu: np.ndarray):
    """Polynomial maps wavelength (um) <-> detector column for the top axis.

    The cache runs in ascending wavenumber and IGRINSOrder in ascending
    wavelength, so the two are matched by wavenumber, never by position."""
    from tellurix_igrins import read_igrins_observation

    order = read_igrins_observation(ROOT / spec).order(number)
    order_nu = 1.0e7 / np.asarray(order.wavelength_vacuum_nm)
    axis = np.argsort(order_nu)
    if order_nu.size != nu.size or not np.allclose(order_nu[axis], nu, atol=1e-6, rtol=0):
        raise SystemExit(f"{spec} order {number}: cached pixels are not the order's pixels")
    columns = np.asarray(order.pixel)[axis].astype(float)
    wavelength = 1.0e4 / nu
    to_pixel = np.polynomial.Polynomial.fit(wavelength, columns, 5)
    to_wavelength = np.polynomial.Polynomial.fit(columns, wavelength, 5)
    residual = np.max(np.abs(to_pixel(wavelength) - columns))
    if residual > 0.05:
        raise SystemExit(f"pixel polynomial misses by {residual:.3f} px")
    return to_pixel, to_wavelength, columns


def add_pixel_axis(ax, forward, inverse, label=True):
    top = ax.secondary_xaxis("top", functions=(forward, inverse))
    top.tick_params(direction="in", labelsize=6.5, pad=1.5)
    if label:
        top.set_xlabel("Detector column (pixel)", fontsize=7, labelpad=2)
    return top


def zoom_window(transmission: np.ndarray, reliable: np.ndarray, width: int) -> slice:
    """The ``width`` pixels with the most telluric absorption the fit used.

    Pixels the fit set aside (saturated cores, the mask) count against a
    window, so it lands on lines that were fitted rather than on a gap."""
    depth = np.where(reliable, 1.0 - np.nan_to_num(transmission, nan=1.0), -0.5)
    score = np.convolve(depth, np.ones(width), mode="valid")
    start = int(np.argmax(score))
    return slice(start, start + width)


def percentile_of(value: float, population: np.ndarray) -> float:
    return float(100.0 * np.mean(population <= value))


def main() -> None:
    style()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    rows = {band: [r for r in read_rows(run) if (r["stem"], r["name"]) not in PATHOLOGICAL]
            for band, run in RUNS.items()}

    # The frame: median over all its order z of each frame, the median frame.
    by_frame = {}
    for band in rows:
        for r in rows[band]:
            by_frame.setdefault(r["frame"], []).append(r["residual_z_rms"])
    frame_median = {f: float(np.median(v)) for f, v in by_frame.items()}
    ranked = sorted(frame_median, key=frame_median.get)
    frame = ranked[(len(ranked) - 1) // 2]

    chosen = {}
    for band, band_rows in rows.items():
        population = np.array([r["residual_z_rms"] for r in band_rows])
        names = sorted({r["name"] for r in band_rows}, key=lambda n: int(n[1:]))
        candidates = []
        for name in names:
            order_rows = [r for r in band_rows if r["name"] == name]
            fraction = np.median([strong_fraction(RUNS[band], r) for r in order_rows])
            if fraction >= STRONG_FRACTION and any(r["frame"] == frame for r in order_rows):
                z = float(np.median([r["residual_z_rms"] for r in order_rows]))
                candidates.append((abs(z - np.median(population)), name, fraction, z))
        _, name, fraction, z_order = min(candidates)
        row = next(r for r in band_rows if r["name"] == name and r["frame"] == frame)
        chosen[band] = {"row": row, "fraction": fraction, "z_order_median": z_order,
                        "z_band_median": float(np.median(population)),
                        "percentile": percentile_of(row["residual_z_rms"], population),
                        "count": int(population.size),
                        "candidates": [c[1] for c in sorted(candidates, key=lambda c: c[1])]}

    fig = plt.figure(figsize=(7.1, 8.6))
    outer = GridSpec(2, 2, figure=fig, height_ratios=[4.0, 2.75], hspace=0.30, wspace=0.22,
                     left=0.075, right=0.985, top=0.925, bottom=0.055)
    report = {"frame": frame, "frame_median_z": frame_median[frame],
              "frame_rank": ranked.index(frame) + 1, "frames": len(ranked), "orders": {}}
    for column, band in enumerate(("H", "K")):
        info = chosen[band]
        row = info["row"]
        d = load(RUNS[band], row["stem"], row["name"])
        nu = d["wavenumber_cm1"]
        wl = 1.0e4 / nu
        mask = d["mask"].astype(bool)
        reliable = d["reliable"].astype(bool)
        continuum = d["continuum"]
        observed = np.where(mask, d["observed"] / continuum, np.nan)
        model = np.where(mask, d["model_flux"] / continuum, np.nan)
        star = np.where(mask, d["stellar_only"] / continuum, np.nan)
        # What is divided out is the convolved effective transmission,
        # model / (continuum x star), not the unconvolved line-by-line one.
        effective = np.where(mask, d["model_flux"] / d["stellar_only"], np.nan)
        corrected = np.where(reliable, d["corrected"] / continuum, np.nan)
        z = np.where(reliable, d["residual"] / d["uncertainty"], np.nan)
        z_rms = float(np.sqrt(np.nanmean(z ** 2)))
        forward, inverse, columns = pixel_axis(row["spec"], row["order"], nu)
        zoom = zoom_window(effective, reliable, ZOOM_PIXELS)
        lo, hi = np.sort(wl[zoom][[0, -1]])
        span = (np.nanmin(wl[mask]), np.nanmax(wl[mask]))
        pad = 0.005 * (span[1] - span[0])

        top = outer[0, column].subgridspec(4, 1, height_ratios=[1.35, 0.8, 1.0, 0.7], hspace=0.0)
        bottom = outer[1, column].subgridspec(3, 1, height_ratios=[1.3, 1.0, 0.7], hspace=0.0)
        axes_full = [fig.add_subplot(top[i]) for i in range(4)]
        for ax in axes_full[1:]:
            ax.sharex(axes_full[0])
        axes_zoom = [fig.add_subplot(bottom[i]) for i in range(3)]
        for ax in axes_zoom[1:]:
            ax.sharex(axes_zoom[0])

        for axes, window in ((axes_full, slice(None)), (axes_zoom, zoom)):
            w = wl[window]
            flux_ax = axes[0]
            flux_ax.plot(w, observed[window], color=OBSERVED, lw=0.6 if window == zoom else 0.4,
                         label="observed")
            flux_ax.plot(w, model[window], color=MODEL, lw=0.6 if window == zoom else 0.45,
                         label="model")
            flux_ax.plot(w, star[window], color=STAR, lw=0.7, ls="--",
                         label="A0V model $\\times$ continuum")
            if window != zoom:
                axes[1].plot(w, effective[window], color=TELLURIC, lw=0.45,
                             label="telluric transmission $T_{\\rm eff}$")
                axes[1].set_ylabel("$T_{\\rm eff}$")
                axes[1].set_ylim(-0.05, 1.08)
                corrected_ax, z_ax = axes[2], axes[3]
            else:
                corrected_ax, z_ax = axes[1], axes[2]
            corrected_ax.plot(w, corrected[window], color=OBSERVED, lw=0.5 if window == zoom else 0.35,
                              label="corrected")
            corrected_ax.plot(w, star[window], color=STAR, lw=0.8, ls="--", label="A0V model")
            z_ax.axhspan(-1, 1, color=SHADE, lw=0)
            z_ax.axhline(0, color="0.5", lw=0.4)
            z_ax.plot(w, z[window], color=OBSERVED, lw=0.45 if window == zoom else 0.3)
            z_ax.set_ylim(-7.5, 7.5)
            z_ax.set_ylabel("$(O-M)/\\sigma$")
            flux_ax.set_ylabel("$F / C$")
            corrected_ax.set_ylabel("$F_{\\rm corr} / C$")
            values = corrected[window][np.isfinite(corrected[window])]
            if values.size:
                q1, q2 = np.percentile(values, [0.5, 99.5])
                corrected_ax.set_ylim(q1 - 0.15 * (q2 - q1), q2 + 0.25 * (q2 - q1))
            values = observed[window][np.isfinite(observed[window])]
            flux_ax.set_ylim(min(-0.03, np.nanmin(model[window]) - 0.03),
                             max(1.12, np.percentile(values, 99.8) + 0.06))
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)
            axes[-1].set_xlabel("Vacuum wavelength ($\\mu$m)")
            for ax in axes:
                ax.yaxis.set_label_coords(-0.105, 0.5)
            if window == zoom:
                axes[0].set_xlim(lo - 0.002 * (hi - lo), hi + 0.002 * (hi - lo))
            else:
                axes[0].set_xlim(span[0] - pad, span[1] + pad)
                for ax in axes:
                    ax.axvspan(lo, hi, color="#FFE9B3", lw=0, zorder=0)
            add_pixel_axis(axes[0], forward, inverse)

        axes_full[0].set_title(
            f"{row['name']}   frame {frame} ({row['object'].split(' V ')[0].strip()}), "
            f"airmass {row['airmass']:.2f}", pad=3)
        axes_full[3].text(0.01, 0.94, f"$z_{{\\rm rms}}$ = {z_rms:.2f} "
                          f"({info['percentile']:.0f}th pct of {info['count']} {band} order-frames)",
                          transform=axes_full[3].transAxes, va="top", fontsize=6.5,
                          bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
        axes_zoom[0].set_title(f"{row['name']} zoom (shaded above): {ZOOM_PIXELS} pixels", pad=3)
        for ax in axes_full + axes_zoom:
            ax.xaxis.set_major_locator(MaxNLocator(6))
        if column == 0:
            handles = [h for ax in axes_full[:3] for h in ax.get_lines() if not
                       h.get_label().startswith("_")]
            handles = [handles[i] for i in (0, 1, 2, 3)]
            legend = fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
                                ncol=4, bbox_to_anchor=(0.5, 0.997), handlelength=2.0,
                                columnspacing=1.4)
            for line in legend.get_lines():
                line.set_linewidth(1.2)
        report["orders"][band] = {
            "order": row["name"], "stem": row["stem"], "object": row["object"],
            "airmass": row["airmass"], "residual_z_rms": row["residual_z_rms"],
            "z_rms_recomputed": z_rms,
            "residual_rms_over_noise": row["residual_rms_over_noise"],
            "percentile_in_band": info["percentile"], "band_order_frames": info["count"],
            "band_median_z": info["z_band_median"], "order_median_z": info["z_order_median"],
            "strong_fraction_T_lt_0.8": info["fraction"], "candidates": info["candidates"],
            "free_species": row["free_species"],
            "log_column_scales": row["log_column_scales"],
            "fitted_pixels": int(reliable.sum()),
            "min_effective_transmission": float(np.nanmin(effective[reliable])),
            "zoom_um": [float(lo), float(hi)],
            "zoom_columns": [float(columns[zoom][0]), float(columns[zoom][-1])],
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"fig_igrins_standard.{suffix}")
    (OUTPUT / "fig_igrins_standard.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
