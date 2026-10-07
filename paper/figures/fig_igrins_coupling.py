#!/usr/bin/env python
"""Paper figure: the 2.0 um CO2 band order K89 with and without CO2 line coupling.

Reads the paired DCT 2018-12-20 runs in ``data/corrected/igrins/coupling_test/``
(``base_k`` and ``cpl_k``: the same ten A0V standards, ERA5 profile, A0V
source and settings, ``--line-coupling`` the only difference; see ``run.sh``
there and ``docs/igrins_a0v.md``, "Line coupling on a real night"). Nothing is
refitted.

The frame shown is the one whose improvement in ``residual_z_rms`` (rms of
(observed - model) / pipeline sigma over the reliable pixels) is the median of
the ten (lower middle). The bottom-left panel averages z over all ten frames
pixel by pixel -- within a night every frame of an order has the same
wavelengths (docs: spread 0.0000 cm-1) -- which shows the part of the residual
that repeats from frame to frame, i.e. the model's. The zoom is the
``ZOOM_PIXELS`` where that mean residual changes most between the two runs.

Note: these runs predate the sequence-airmass fix and use first-exposure
airmasses; both runs share that, so the comparison is paired.

    UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_igrins_coupling.py

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
RUNS = {"base": ROOT / "data/corrected/igrins/coupling_test/base_k",
        "cpl": ROOT / "data/corrected/igrins/coupling_test/cpl_k"}
ORDER = "K89"
ZOOM_PIXELS = 220
BIN_PIXELS = 32

OBSERVED = "black"
UNCOUPLED = "#D55E00"   # Okabe-Ito vermillion, as in fig_lblrtm_comparison.py
COUPLED = "#0072B2"     # Okabe-Ito blue
SHADE = "#E8E8E8"
LABELS = {"base": "no line coupling", "cpl": "first-order line coupling"}
COLORS = {"base": UNCOUPLED, "cpl": COUPLED}


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


def read_rows(run: Path) -> dict:
    from tellurix_igrins import summary_paths

    rows = {}
    for path in summary_paths(run, "*summary.json"):
        blob = json.loads(path.read_text())
        stem = Path(blob["observation"]["path"]).name.split(".")[0]
        for row in blob["results"]:
            if row["name"] == ORDER:
                rows[stem] = {**row, "stem": stem, "object": blob["observation"]["object"],
                              "spec": blob["observation"]["path"],
                              "line_coupling": blob["settings"].get("line_coupling")}
    return rows


def load(run: Path, stem: str) -> dict:
    with np.load(run / f"{stem}_{ORDER}.npz") as stored:
        return {k: np.asarray(stored[k]) for k in stored.files}


def pixel_axis(spec: str, number: int, nu: np.ndarray):
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
    if np.max(np.abs(to_pixel(wavelength) - columns)) > 0.05:
        raise SystemExit("pixel polynomial misses by more than 0.05 px")
    return to_pixel, to_wavelength


def main() -> None:
    style()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    rows = {k: read_rows(run) for k, run in RUNS.items()}
    if not rows["cpl"] or not all(r["line_coupling"] for r in rows["cpl"].values()):
        raise SystemExit("cpl run is not line-coupled")
    if any(r["line_coupling"] for r in rows["base"].values()):
        raise SystemExit("base run is line-coupled")
    stems = sorted(set(rows["base"]) & set(rows["cpl"]))
    gain = {s: rows["base"][s]["residual_z_rms"] - rows["cpl"][s]["residual_z_rms"] for s in stems}
    ranked = sorted(stems, key=gain.get)
    stem = ranked[(len(ranked) - 1) // 2]

    data = {k: {s: load(RUNS[k], s) for s in stems} for k in RUNS}
    nu = data["base"][stem]["wavenumber_cm1"]
    for k in RUNS:
        for s in stems:
            if not np.array_equal(data[k][s]["wavenumber_cm1"], nu):
                raise SystemExit(f"{k} {s}: not the same pixels as {stem}")
    wl = 1.0e4 / nu
    forward, inverse = pixel_axis(rows["base"][stem]["spec"], int(ORDER[1:]), nu)

    def z_of(d):
        return np.where(d["reliable"].astype(bool), d["residual"] / d["uncertainty"], np.nan)

    # Pixels reliable in every frame of both runs, so the means compare like with like.
    common = np.logical_and.reduce([data[k][s]["reliable"].astype(bool)
                                    for k in RUNS for s in stems])
    mean_z = {k: np.where(common, np.nanmean([z_of(data[k][s]) for s in stems], axis=0), np.nan)
              for k in RUNS}
    change = np.nan_to_num(np.abs(mean_z["base"]) - np.abs(mean_z["cpl"]), nan=0.0)
    start = int(np.argmax(np.convolve(change, np.ones(ZOOM_PIXELS), mode="valid")))
    zoom = slice(start, start + ZOOM_PIXELS)
    lo, hi = np.sort(wl[zoom][[0, -1]])

    fig = plt.figure(figsize=(7.1, 6.6))
    outer = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
                     hspace=0.38, wspace=0.22, left=0.075, right=0.985, top=0.93, bottom=0.07)
    full = outer[0, :].subgridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.0)
    ax_flux = fig.add_subplot(full[0])
    ax_z = fig.add_subplot(full[1], sharex=ax_flux)
    ax_mean = fig.add_subplot(outer[1, 0])
    zoom_grid = outer[1, 1].subgridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.0)
    ax_zflux = fig.add_subplot(zoom_grid[0])
    ax_zz = fig.add_subplot(zoom_grid[1], sharex=ax_zflux)

    d0 = data["base"][stem]
    mask = d0["mask"].astype(bool)
    observed = np.where(mask, d0["observed"] / d0["continuum"], np.nan)
    for axes, window in (((ax_flux, ax_z), slice(None)), ((ax_zflux, ax_zz), zoom)):
        fine = window == zoom
        a_flux, a_z = axes
        a_flux.plot(wl[window], observed[window], color=OBSERVED, lw=0.6 if fine else 0.35,
                    label="observed")
        a_z.axhspan(-1, 1, color=SHADE, lw=0)
        a_z.axhline(0, color="0.5", lw=0.4)
        for k in ("base", "cpl"):
            d = data[k][stem]
            model = np.where(mask, d["model_flux"] / d["continuum"], np.nan)
            a_flux.plot(wl[window], model[window], color=COLORS[k], lw=0.7 if fine else 0.4,
                        label=f"model, {LABELS[k]}")
            a_z.plot(wl[window], z_of(d)[window], color=COLORS[k], lw=0.6 if fine else 0.3,
                     alpha=0.9)
        a_flux.set_ylim(-0.05, 1.15 if fine else 1.32)
        a_z.set_ylim(-8, 8)
        a_flux.set_ylabel("$F/C$")
        a_z.set_ylabel("$(O-M)/\\sigma$")
        a_flux.tick_params(labelbottom=False)
        a_z.set_xlabel("Vacuum wavelength ($\\mu$m)")
        for a in axes:
            a.xaxis.set_major_locator(MaxNLocator(5 if fine else 8))
            a.yaxis.set_label_coords(-0.1 if fine else -0.045, 0.5)
        top = a_flux.secondary_xaxis("top", functions=(forward, inverse))
        top.tick_params(direction="in", labelsize=6.5, pad=1.5)
        top.set_xlabel("Detector column (pixel)", fontsize=7, labelpad=2)
        if fine:
            a_flux.set_xlim(lo, hi)
        else:
            a_flux.set_xlim(wl[mask].min(), wl[mask].max())
            for a in axes:
                a.axvspan(lo, hi, color="#FFE9B3", lw=0, zorder=0)
    r_base, r_cpl = rows["base"][stem], rows["cpl"][stem]
    ax_flux.set_title(
        f"(a) {ORDER}, DCT 2018-12-20 frame {stem.split('_')[-1]} "
        f"({r_base['object'].split(' V ')[0].strip()}, airmass {r_base['airmass']:.2f}):  "
        f"$z_{{\\rm rms}}$ {r_base['residual_z_rms']:.2f} without coupling, "
        f"{r_cpl['residual_z_rms']:.2f} with", loc="left", pad=3)
    legend = ax_flux.legend(loc="upper right", ncol=3, handlelength=1.6, borderaxespad=0.2)
    for line in legend.get_lines():
        line.set_linewidth(1.2)

    # The ten-frame mean, as an rms in bins of BIN_PIXELS: the pixel-level
    # trace is too dense to read at this size, and the rms is what changes.
    edges = np.arange(0, nu.size + BIN_PIXELS, BIN_PIXELS)
    binned = {}
    for k in ("base", "cpl"):
        values = mean_z[k]
        rms = float(np.sqrt(np.nanmean(values ** 2)))
        per_bin = []
        for a, b in zip(edges[:-1], edges[1:]):
            v = values[a:b]
            v = v[np.isfinite(v)]
            per_bin.append(np.sqrt(np.mean(v ** 2)) if v.size >= BIN_PIXELS // 4 else np.nan)
        binned[k] = np.array(per_bin)
        centres = 0.5 * (wl[edges[:-1]] + wl[np.minimum(edges[1:], nu.size) - 1])
        ax_mean.plot(centres, binned[k], color=COLORS[k], lw=1.0, marker="o", ms=2.2,
                     label=f"{LABELS[k]}: rms {rms:.2f}")
    ax_mean.axvspan(lo, hi, color="#FFE9B3", lw=0, zorder=0)
    ax_mean.set_xlim(wl[mask].min(), wl[mask].max())
    ax_mean.set_ylim(0, np.nanmax(np.concatenate(list(binned.values()))) * 1.35)
    ax_mean.set_ylabel(f"rms of 10-frame mean $z$\n({BIN_PIXELS}-pixel bins)")
    ax_mean.set_xlabel("Vacuum wavelength ($\\mu$m)")
    ax_mean.xaxis.set_major_locator(MaxNLocator(5))
    ax_mean.legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
    top = ax_mean.secondary_xaxis("top", functions=(forward, inverse))
    top.tick_params(direction="in", labelsize=6.5, pad=1.5)
    top.set_xlabel("Detector column (pixel)", fontsize=7, labelpad=2)
    better = sum(gain[s] > 0 for s in stems)
    ax_mean.set_title(f"(b) all {len(stems)} frames; $z_{{\\rm rms}}$ better in "
                      f"{better} of {len(stems)}", loc="left", pad=3)
    ax_zflux.set_title(f"(c) zoom (shaded): largest change in mean residual", loc="left",
                       pad=3)

    report = {
        "order": ORDER, "frame": stem, "object": r_base["object"], "airmass": r_base["airmass"],
        "z_base": r_base["residual_z_rms"], "z_coupled": r_cpl["residual_z_rms"],
        "per_frame": {s: {"base": rows["base"][s]["residual_z_rms"],
                          "coupled": rows["cpl"][s]["residual_z_rms"]} for s in stems},
        "frames_improved": int(better), "frames": len(stems),
        "median_z_base": float(np.median([rows["base"][s]["residual_z_rms"] for s in stems])),
        "median_z_coupled": float(np.median([rows["cpl"][s]["residual_z_rms"] for s in stems])),
        "mean_z_rms": {k: float(np.sqrt(np.nanmean(mean_z[k] ** 2))) for k in RUNS},
        "common_pixels": int(common.sum()),
        "co2_log_scale": {k: rows[k][stem]["log_column_scales"].get("CO2") for k in RUNS},
        "zoom_um": [float(lo), float(hi)],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"fig_igrins_coupling.{suffix}")
    (OUTPUT / "fig_igrins_coupling.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
