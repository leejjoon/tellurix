#!/usr/bin/env python
"""Paper figure: the best and worst IGRINS standard fits of one order, dry night against wet.

Reads the sequence-airmass standards runs of DCT 2018-12-20 (~2 mm precipitable
water) and McDonald 2017-04-20 (~10 mm) -- summaries and cached order arrays --
and refits nothing. Nothing is picked by eye:

* the order is chosen by the rule of ``fig_igrins_standard.py``: of the K
  orders with strong absorption on the dry night (median fraction of fitted
  pixels with transmission < 0.8 at least ``STRONG_FRACTION``) and fitted in
  every frame of both nights, the one whose dry-night median z is closest to
  the dry K band's median;
* on each night the best and the worst frame of that order are the lowest and
  highest ``residual_z_rms`` (rms of (observed - model) / pipeline sigma over
  the reliable pixels);
* the zoom is the ``ZOOM_PIXELS`` of strongest fitted absorption in the dry
  night's best frame, and every panel shows the same wavelengths.

Percentiles are quoted against every order-frame of the night, H and K pooled,
less the two the docs call pathological (DCT 2018 H107 of frame 108 and H104
of frame 84, at 115 and 33), which are drawn off scale in the distribution.

    UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_igrins_good_bad.py

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
CORRECTED = ROOT / "data/corrected/igrins"
NIGHTS = {
    "dry": {"label": "DCT 2018-12-20 (dry, ~2 mm PWV)", "short": "DCT 2018 (dry)",
            "runs": {"H": CORRECTED / "seq_dct2018_h", "K": CORRECTED / "seq_dct2018_k"}},
    "wet": {"label": "McDonald 2017-04-20 (wet, ~10 mm PWV)", "short": "McDonald 2017 (wet)",
            "runs": {"H": CORRECTED / "seq_mcd2017_h", "K": CORRECTED / "seq_mcd2017_k"}},
}
PATHOLOGICAL = {("SDCH_20181220_0108", "H107"), ("SDCH_20181220_0084", "H104")}
STRONG_FRACTION = 0.25
ZOOM_PIXELS = 220

OBSERVED = "black"
MODEL = "#D55E00"       # Okabe-Ito vermillion
DRY = "#0072B2"         # Okabe-Ito blue
WET = "#E69F00"         # Okabe-Ito orange
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


def read_rows(run: Path, band: str) -> list[dict]:
    from tellurix_igrins import summary_paths

    rows = []
    for path in summary_paths(run, "*summary.json"):
        blob = json.loads(path.read_text())
        stem = Path(blob["observation"]["path"]).name.split(".")[0]
        for row in blob["results"]:
            if (stem, row["name"]) in PATHOLOGICAL:
                continue
            if not (run / f"{stem}_{row['name']}.npz").exists():
                continue
            rows.append({**row, "stem": stem, "frame": stem.split("_")[-1], "run": run,
                         "band": band, "object": blob["observation"]["object"],
                         "spec": blob["observation"]["path"]})
    return rows


def load(row: dict) -> dict:
    with np.load(row["run"] / f"{row['stem']}_{row['name']}.npz") as stored:
        return {k: np.asarray(stored[k]) for k in stored.files}


def strong_fraction(row: dict) -> float:
    d = load(row)
    mask = d["mask"].astype(bool)
    return float(np.mean(d["transmission"][mask] < 0.8))


def pixel_axis(spec: str, number: int, nu: np.ndarray):
    """Polynomial maps wavelength (um) <-> detector column for the top axis,
    matched by wavenumber since the cache and IGRINSOrder run opposite ways."""
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


def zoom_window(transmission, reliable, width) -> slice:
    depth = np.where(reliable, 1.0 - np.nan_to_num(transmission, nan=1.0), -0.5)
    start = int(np.argmax(np.convolve(depth, np.ones(width), mode="valid")))
    return slice(start, start + width)


def percentile_of(value, population) -> float:
    return float(100.0 * np.mean(np.asarray(population) <= value))


def main() -> None:
    style()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    rows = {night: read_rows(spec["runs"]["H"], "H") + read_rows(spec["runs"]["K"], "K")
            for night, spec in NIGHTS.items()}
    pooled = {night: np.array([r["residual_z_rms"] for r in rows[night]]) for night in rows}

    # The order.
    dry_k = [r for r in rows["dry"] if r["band"] == "K"]
    k_median = float(np.median([r["residual_z_rms"] for r in dry_k]))
    frames = {n: {r["frame"] for r in rows[n]} for n in rows}
    candidates = []
    for name in sorted({r["name"] for r in dry_k}, key=lambda n: int(n[1:])):
        everywhere = all({r["frame"] for r in rows[n] if r["name"] == name} == frames[n]
                         for n in rows)
        if not everywhere:
            continue
        order_rows = [r for r in dry_k if r["name"] == name]
        fraction = float(np.median([strong_fraction(r) for r in order_rows]))
        if fraction >= STRONG_FRACTION:
            z = float(np.median([r["residual_z_rms"] for r in order_rows]))
            candidates.append((abs(z - k_median), name, fraction))
    _, order_name, order_fraction = min(candidates)

    picks = {}
    for night in rows:
        order_rows = sorted((r for r in rows[night] if r["name"] == order_name),
                            key=lambda r: r["residual_z_rms"])
        picks[night] = {"best": order_rows[0], "worst": order_rows[-1], "all": order_rows}

    fig = plt.figure(figsize=(7.1, 8.2))
    outer = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.55, 1.55], hspace=0.45, wspace=0.2,
                     left=0.075, right=0.985, top=0.97, bottom=0.055)

    # (a) every order-frame of both nights.
    ax = fig.add_subplot(outer[0, 0])
    top_z = 9.0
    for night, color in (("dry", DRY), ("wet", WET)):
        z = np.sort(pooled[night])
        ax.step(z, np.arange(1, z.size + 1) / z.size, where="post", color=color, lw=1.0,
                label=f"{NIGHTS[night]['short']}: N = {z.size}, median {np.median(z):.2f}")
        for kind, marker in (("best", "o"), ("worst", "s")):
            value = picks[night][kind]["residual_z_rms"]
            ax.plot(value, percentile_of(value, pooled[night]) / 100, marker=marker, ms=4.5,
                    mfc="white", mec=color, mew=1.0, ls="none")
    ax.set_xlim(0.5, top_z)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("$z_{\\rm rms}$ of an order-frame")
    ax.set_ylabel("cumulative fraction")
    ax.legend(loc="lower right", handlelength=1.4, borderaxespad=0.3,
              title=f"open circle/square: best/worst {order_name}", title_fontsize=6.5)
    ax.text(0.98, 0.62, "not shown: DCT H107 (115) and\nH104 (33), unexplained outliers",
            transform=ax.transAxes, ha="right", va="center", fontsize=6, color="0.35")
    ax.set_title("(a) every fitted order-frame, H and K pooled", fontsize=7.5, loc="left")

    ax = fig.add_subplot(outer[0, 1])
    for night, color in (("dry", DRY), ("wet", WET)):
        rs = picks[night]["all"]
        ax.plot([r["airmass"] for r in rs], [r["residual_z_rms"] for r in rs], "o", ms=3.5,
                color=color, label=NIGHTS[night]["short"])
        for kind, marker in (("best", "o"), ("worst", "s")):
            r = picks[night][kind]
            ax.plot(r["airmass"], r["residual_z_rms"], marker=marker, ms=7, mfc="none",
                    mec=color, mew=1.0)
    ax.set_xlabel("airmass (exposure sequence)")
    ax.set_ylabel(f"{order_name} $z_{{\\rm rms}}$")
    ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.3)
    ax.set_ylim(bottom=0)
    ax.set_title(f"(b) {order_name} in every standard frame", fontsize=7.5, loc="left")

    # (c-f) the four order-frames, same wavelengths.
    reference = load(picks["dry"]["best"])
    zoom = zoom_window(reference["model_flux"] / reference["stellar_only"],
                       reference["reliable"].astype(bool), ZOOM_PIXELS)
    lo, hi = np.sort(1.0e4 / reference["wavenumber_cm1"][zoom][[0, -1]])
    letters = {("dry", "best"): "c", ("wet", "best"): "d", ("dry", "worst"): "e",
               ("wet", "worst"): "f"}
    report = {"order": order_name, "strong_fraction_dry": order_fraction,
              "order_candidates": sorted(c[1] for c in candidates),
              "zoom_um": [float(lo), float(hi)], "nights": {}}
    for column, night in enumerate(("dry", "wet")):
        report["nights"][night] = {
            "order_frames": int(pooled[night].size),
            "median_z": float(np.median(pooled[night])),
            "p5_p95_z": [float(v) for v in np.percentile(pooled[night], [5, 95])],
            "order_median_z": float(np.median([r["residual_z_rms"] for r in picks[night]["all"]])),
            "order_frames_of_order": len(picks[night]["all"]),
        }
        for row_index, kind in enumerate(("best", "worst")):
            r = picks[night][kind]
            d = load(r)
            nu = d["wavenumber_cm1"]
            wl = 1.0e4 / nu
            mask = d["mask"].astype(bool)
            reliable = d["reliable"].astype(bool)
            continuum = d["continuum"]
            inside = (wl >= lo) & (wl <= hi)
            observed = np.where(mask, d["observed"] / continuum, np.nan)
            model = np.where(mask, d["model_flux"] / continuum, np.nan)
            z = np.where(reliable, d["residual"] / d["uncertainty"], np.nan)
            forward, inverse = pixel_axis(r["spec"], r["order"], nu)
            cell = outer[1 + row_index, column].subgridspec(2, 1, height_ratios=[1.6, 1.0],
                                                            hspace=0.0)
            flux_ax = fig.add_subplot(cell[0])
            z_ax = fig.add_subplot(cell[1], sharex=flux_ax)
            flux_ax.plot(wl[inside], observed[inside], color=OBSERVED, lw=0.6, label="observed")
            flux_ax.plot(wl[inside], model[inside], color=MODEL, lw=0.7, label="model")
            z_ax.axhspan(-1, 1, color=SHADE, lw=0)
            z_ax.axhline(0, color="0.5", lw=0.4)
            z_ax.plot(wl[inside], z[inside], color=OBSERVED, lw=0.5)
            z_ax.set_ylim(-9, 9)
            flux_ax.set_ylim(-0.04, 1.2)
            flux_ax.set_xlim(lo, hi)
            flux_ax.tick_params(labelbottom=False)
            flux_ax.set_ylabel("$F/C$")
            z_ax.set_ylabel("$(O-M)/\\sigma$")
            for a in (flux_ax, z_ax):
                a.yaxis.set_label_coords(-0.09, 0.5)
                a.xaxis.set_major_locator(MaxNLocator(5))
            if row_index == 1:
                z_ax.set_xlabel("Vacuum wavelength ($\\mu$m)")
            top = flux_ax.secondary_xaxis("top", functions=(forward, inverse))
            top.tick_params(direction="in", labelsize=6.5, pad=1.5)
            top.set_xlabel("Detector column (pixel)", fontsize=7, labelpad=2)
            star = r["object"].split(" V ")[0].strip()
            pct = percentile_of(r["residual_z_rms"], pooled[night])
            in_window = z[inside]
            flux_ax.set_title(
                f"({letters[night, kind]}) {NIGHTS[night]['short']}, {kind}: "
                f"{r['frame']}, {star}, $X$ = {r['airmass']:.2f}\n"
                f"$z_{{\\rm rms}}$ = {r['residual_z_rms']:.2f} ({pct:.0f}th pct of night); "
                f"{np.sqrt(np.nanmean(in_window**2)):.2f} in window",
                fontsize=7, loc="left", pad=3)
            if column == 0 and row_index == 0:
                flux_ax.legend(loc="upper right", ncol=2, handlelength=1.5, borderaxespad=0.2)
            report["nights"][night][kind] = {
                "stem": r["stem"], "object": r["object"], "airmass": r["airmass"],
                "residual_z_rms": r["residual_z_rms"],
                "residual_rms_over_noise": r["residual_rms_over_noise"],
                "percentile_in_night": pct,
                "z_rms_in_zoom": float(np.sqrt(np.nanmean(in_window ** 2))),
                "log_column_H2O": r["log_column_scales"].get("H2O"),
                "median_transmission": r["median_transmission"],
                "fitted_pixels": int(reliable.sum()),
                "pixel_sigma": r["pixel_sigma"],
            }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"fig_igrins_good_bad.{suffix}")
    (OUTPUT / "fig_igrins_good_bad.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
