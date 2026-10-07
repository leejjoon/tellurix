#!/usr/bin/env python
"""Paper figure: a DCT 2018-12-20 science target corrected with the night calibration.

The target's telluric model is the night's (``data/calibration/igrins_dct2018_{h,k}.h5``,
ten A0V standards) with one velocity shift and one water scale fitted on the
frame, the median over orders, and nothing else per order but the continuum
(``fit_igrins_science.py --clip-sigma 3``; products in
``data/corrected/igrins/science_dct2018``). Beside it is the IGRINS pipeline's
own product for the same frame, the target divided by an observed A0V standard
and multiplied by a Vega model (``SPEC_DIVIDE_A0V`` of ``*.spec_a0v.fits``;
checked here to equal TGT_SPEC / A0V_SPEC x VEGA_SPEC to 3e-8), which shares
none of this machinery. Nothing is refitted here.

Not ``SPEC_DIVIDE_CONT``: in these files it is the target over the standard's
*continuum* only -- it correlates with our telluric transmission at r = +0.99
on K88 -- so it still carries every telluric line.

Chosen by rule, not by eye:

* the target: of the six science targets, the one whose H-band and K-band
  water scales disagree by the median amount (upper middle of six) -- the
  disagreement is the check ``docs/igrins_science.md`` uses for a target,
  where the residual is the star's own lines;
* the orders: the strong-absorption H and K orders that
  ``fig_igrins_standard.py`` picks by its rule (H122, K88), so the standard and
  the target can be compared order for order;
* the zoom: the ``ZOOM_PIXELS`` of strongest absorption among fitted pixels.

Two numbers per order and product, both over the order's reliable pixels:
the slope of corrected flux against telluric depth 1 - T_eff (pixels with
0.3 < T_eff < 0.97; a perfect correction leaves none, an under-correction a
negative slope), and the point-to-point noise, 1.4826 MAD of first
differences / sqrt 2, over the least-absorbed third of the fitted pixels. The PLP spectrum is put on our
continuum by a quadratic fit of the ratio, which removes a slope and nothing
narrower.

Note: the science runs and the night calibration predate the sequence-airmass
fix (docs/igrins_a0v.md, "What moves the well-mixed columns") and use the
header's first-exposure airmass; the frame's water scale absorbs that.

    UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_igrins_science.py

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
RUN = ROOT / "data/corrected/igrins/science_dct2018"
ORDERS = {"H": 122, "K": 88}
ZOOM_PIXELS = 200
CLEAN_FRACTION = 1 / 3

OBSERVED = "black"
TELLURIX = "black"
PLP = "#0072B2"         # Okabe-Ito blue
TELLURIC = "#009E73"    # Okabe-Ito bluish green
DIFFERENCE = "#D55E00"  # Okabe-Ito vermillion


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


def read_targets() -> dict:
    """frame number -> {band: summary}, targets only."""
    targets: dict[str, dict] = {}
    for path in sorted(RUN.glob("*_science_summary.json")):
        blob = json.loads(path.read_text())
        if blob["observation"].get("object_type") != "TAR":
            continue
        stem = Path(blob["observation"]["path"]).name.split(".")[0]
        targets.setdefault(stem.split("_")[-1], {})[blob["observation"]["band"]] = blob
    return {k: v for k, v in targets.items() if set(v) == {"H", "K"}}


def water(blob: dict) -> float:
    shifts = blob["shifts"]
    return float(shifts[shifts.get("final", "frame")]["log_column_H2O"])


def order_columns(spec: Path, number: int, nu: np.ndarray):
    from tellurix_igrins import read_igrins_observation

    order = read_igrins_observation(spec).order(number)
    order_nu = 1.0e7 / np.asarray(order.wavelength_vacuum_nm)
    axis = np.argsort(order_nu)
    if order_nu.size != nu.size or not np.allclose(order_nu[axis], nu, atol=1e-6, rtol=0):
        raise SystemExit(f"{spec} order {number}: cached pixels are not the order's pixels")
    return np.asarray(order.pixel)[axis].astype(int)


def pixel_functions(wavelength, columns):
    to_pixel = np.polynomial.Polynomial.fit(wavelength, columns.astype(float), 5)
    to_wavelength = np.polynomial.Polynomial.fit(columns.astype(float), wavelength, 5)
    if np.max(np.abs(to_pixel(wavelength) - columns)) > 0.05:
        raise SystemExit("pixel polynomial misses by more than 0.05 px")
    return to_pixel, to_wavelength


def plp_normalized(spec: Path, columns: np.ndarray, wavelength_um: np.ndarray):
    """The PLP's A0V-divided flux (Vega restored) on our pixels, and the
    standard it used. Its arrays are indexed by detector column; the row is
    the one whose wavelengths are ours."""
    from astropy.io import fits

    path = spec.with_name(spec.name.replace(".spec.fits", ".spec_a0v.fits"))
    with fits.open(path) as hdus:
        wl = np.asarray(hdus["WAVELENGTH"].data, dtype=float)
        rows = [r for r in range(wl.shape[0])
                if np.allclose(wl[r, columns], wavelength_um, atol=1e-9, rtol=0)]
        if len(rows) != 1:
            raise SystemExit(f"{path.name}: no unique PLP row for these pixels")
        header = hdus["A0V_SPEC"].header
        values = np.asarray(hdus["SPEC_DIVIDE_A0V"].data, dtype=float)[rows[0], columns]
        check = (np.asarray(hdus["TGT_SPEC"].data, dtype=float)[rows[0], columns]
                 / np.asarray(hdus["A0V_SPEC"].data, dtype=float)[rows[0], columns]
                 * np.asarray(hdus["VEGA_SPEC"].data, dtype=float)[rows[0], columns]) / values
        if np.nanmax(np.abs(check[np.isfinite(check)] - 1.0)) > 1e-5:
            raise SystemExit(f"{path.name}: SPEC_DIVIDE_A0V is not TGT / A0V x VEGA")
        standard = {"object": str(header.get("OBJECT", "")).strip(),
                    "date_obs": str(header.get("DATE-OBS", "")), "file": path.name}
    return values, standard


def zoom_window(transmission, reliable, width) -> slice:
    depth = np.where(reliable, 1.0 - np.nan_to_num(transmission, nan=1.0), -0.5)
    start = int(np.argmax(np.convolve(depth, np.ones(width), mode="valid")))
    return slice(start, start + width)


def imprint_slope(flux, effective, good) -> float:
    use = good & np.isfinite(flux) & (effective > 0.3) & (effective < 0.97)
    if use.sum() < 20:
        return float("nan")
    return float(np.polyfit(1.0 - effective[use], flux[use], 1)[0])


def point_noise(flux, effective, good) -> float:
    # The least-absorbed third of the fitted pixels: H122 has almost none
    # above 0.97, so a fixed threshold would leave it without a number.
    clean = good & np.isfinite(flux)
    clean &= effective >= np.percentile(effective[clean], 100 * (1 - CLEAN_FRACTION))
    # First differences only between neighbouring pixels that are both clean.
    pairs = clean[1:] & clean[:-1]
    diff = (flux[1:] - flux[:-1])[pairs]
    return float(1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2.0))


def main() -> None:
    style()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    targets = read_targets()
    disagreement = {f: abs(water(b["H"]) - water(b["K"])) for f, b in targets.items()}
    ranked = sorted(disagreement, key=disagreement.get)
    frame = ranked[len(ranked) // 2]
    blobs = targets[frame]
    obs = blobs["H"]["observation"]

    fig = plt.figure(figsize=(7.1, 8.4))
    outer = GridSpec(2, 2, figure=fig, height_ratios=[2.3, 3.0], hspace=0.3, wspace=0.22,
                     left=0.08, right=0.985, top=0.925, bottom=0.05)
    report = {"frame": frame, "object": obs["object"], "airmass": obs["airmass"],
              "zenith_source": obs.get("zenith_source"),
              "water_shift": {b: water(blobs[b]) for b in "HK"},
              "velocity_shift_kms": {b: float(blobs[b]["shifts"][blobs[b]["shifts"].get(
                  "final", "frame")]["velocity_kms"]) for b in "HK"},
              "band_disagreement_all_targets": {f: disagreement[f] for f in ranked},
              "orders": {}}
    for column, band in enumerate(("H", "K")):
        blob = blobs[band]
        number = ORDERS[band]
        row = next(r for r in blob["results"] if int(r["order"]) == number)
        spec = ROOT / blob["observation"]["path"]
        stem = spec.name.split(".")[0]
        with np.load(RUN / f"{stem}_{row['name']}.npz") as stored:
            d = {k: np.asarray(stored[k]) for k in stored.files}
        nu = d["wavenumber_cm1"]
        wl = 1.0e4 / nu
        mask = d["mask"].astype(bool)
        reliable = d["reliable"].astype(bool)
        continuum = d["continuum"]
        effective = d["effective_transmission"]
        observed = np.where(mask, d["observed"] / continuum, np.nan)
        ours = np.where(reliable, d["corrected"] / continuum, np.nan)
        columns = order_columns(spec, number, nu)
        forward, inverse = pixel_functions(wl, columns)
        plp_raw, standard = plp_normalized(spec, columns, wl)
        plp = np.where(reliable & np.isfinite(plp_raw), plp_raw, np.nan)
        # Put the PLP on our continuum: a quadratic in the ratio, robustly.
        good = np.isfinite(plp) & np.isfinite(ours) & (effective > 0.9)
        x = (wl - wl.mean()) / np.ptp(wl)
        keep = good.copy()
        for _ in range(3):
            coeffs = np.polyfit(x[keep], (plp / ours)[keep], 2)
            ratio = np.polyval(coeffs, x)
            dev = plp / ours - ratio
            keep = good & (np.abs(dev) < 4 * 1.4826 * np.nanmedian(np.abs(dev[good])))
        plp = plp / ratio
        zoom = zoom_window(effective, reliable, ZOOM_PIXELS)
        lo, hi = np.sort(wl[zoom][[0, -1]])
        span = (np.nanmin(wl[mask]), np.nanmax(wl[mask]))

        top = outer[0, column].subgridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.0)
        bottom = outer[1, column].subgridspec(3, 1, height_ratios=[1.0, 1.25, 0.75], hspace=0.0)
        full = [fig.add_subplot(top[i]) for i in range(2)]
        full[1].sharex(full[0])
        zoomed = [fig.add_subplot(bottom[i]) for i in range(3)]
        for ax in zoomed[1:]:
            ax.sharex(zoomed[0])

        stats = {}
        for name, flux in (("tellurix", ours), ("plp", plp)):
            stats[name] = {
                "imprint_slope": imprint_slope(flux, effective, reliable),
                "point_noise": point_noise(flux, effective, reliable),
            }

        for axes, window, offset in ((full, slice(None), 0.3), (zoomed, zoom, 0.0)):
            w = wl[window]
            fine = window == zoom
            axes[0].plot(w, observed[window], color=OBSERVED, lw=0.6 if fine else 0.35,
                         label="observed $F/C$")
            axes[0].plot(w, effective[window], color=TELLURIC, lw=0.7 if fine else 0.4,
                         label="telluric $T_{\\rm eff}$ (night calibration)")
            axes[0].set_ylim(-0.05, 1.18)
            axes[0].set_ylabel("$F/C$, $T_{\\rm eff}$")
            axes[1].plot(w, ours[window], color=TELLURIX, lw=0.6 if fine else 0.35,
                         label="tellurix: $F / T_{\\rm eff}$")
            axes[1].plot(w, plp[window] - offset, color=PLP, lw=0.6 if fine else 0.35,
                         label="PLP: $F$ / A0V star" + (f" $-$ {offset}" if offset else ""))
            values = np.concatenate([ours[window][np.isfinite(ours[window])],
                                     plp[window][np.isfinite(plp[window])] - offset])
            q1, q2 = np.percentile(values, [0.3, 99.7])
            axes[1].set_ylim(q1 - 0.1 * (q2 - q1), q2 + 0.12 * (q2 - q1))
            axes[1].set_ylabel("corrected / $C$")
            if fine:
                axes[2].axhline(0, color="0.5", lw=0.4)
                axes[2].plot(w, (plp - ours)[window], color=DIFFERENCE, lw=0.6)
                axes[2].set_ylabel("PLP $-$ tellurix")
                lim = np.nanpercentile(np.abs((plp - ours)[window]), 99) * 1.25
                axes[2].set_ylim(-lim, lim)
                axes[0].set_xlim(lo, hi)
            else:
                axes[0].set_xlim(span[0] - 0.005 * np.ptp(span), span[1] + 0.005 * np.ptp(span))
                for ax in axes:
                    ax.axvspan(lo, hi, color="#FFE9B3", lw=0, zorder=0)
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)
            axes[-1].set_xlabel("Vacuum wavelength ($\\mu$m)")
            for ax in axes:
                ax.yaxis.set_label_coords(-0.115, 0.5)
                ax.xaxis.set_major_locator(MaxNLocator(5 if fine else 6))
            secondary = axes[0].secondary_xaxis("top", functions=(forward, inverse))
            secondary.tick_params(direction="in", labelsize=6.5, pad=1.5)
            secondary.set_xlabel("Detector column (pixel)", fontsize=7, labelpad=2)

        full[0].set_title(f"{row['name']}: {obs['object'].strip()}, frame {frame}, "
                          f"airmass {blob['observation']['airmass']:.2f}", pad=3)
        text = (f"$1-T_{{\\rm eff}}$ slope {stats['tellurix']['imprint_slope']:+.3f} "
                f"vs PLP {stats['plp']['imprint_slope']:+.3f}; noise "
                f"{100 * stats['tellurix']['point_noise']:.2f}% "
                f"vs {100 * stats['plp']['point_noise']:.2f}%")
        zoomed[0].set_title(f"{row['name']} zoom (shaded above). Whole order:\n" + text, pad=3,
                            fontsize=6.8, loc="left")
        if column == 0:
            handles = full[0].get_lines()[:2] + full[1].get_lines()[:2]
            labels = [h.get_label() for h in handles]
            labels[3] = "PLP: $F$ / A0V star (offset in full order)"
            legend = fig.legend(handles, labels, loc="upper center", ncol=4,
                                bbox_to_anchor=(0.5, 0.997), handlelength=1.8, columnspacing=1.2)
            for line in legend.get_lines():
                line.set_linewidth(1.2)
        report["orders"][band] = {
            "order": row["name"], "standard_used_by_plp": standard,
            "residual_rms_over_noise": row["residual_rms_over_noise"],
            "median_transmission": row["median_transmission"],
            "clipped_fraction": row.get("clipped_fraction"),
            "reliable_pixels": int(reliable.sum()),
            "plp_continuum_quadratic": [float(c) for c in coeffs],
            "zoom_um": [float(lo), float(hi)],
            "zoom_columns": [int(columns[zoom][0]), int(columns[zoom][-1])],
            "stats": stats,
            "rms_plp_minus_tellurix_reliable": float(np.sqrt(np.nanmean((plp - ours)[reliable] ** 2))),
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"fig_igrins_science.{suffix}")
    (OUTPUT / "fig_igrins_science.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
