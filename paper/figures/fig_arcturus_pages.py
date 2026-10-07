"""Publication figures: telluric correction of the Arcturus atlas, page by page.

Reads only saved products -- the per-page arrays in ``data/corrected/atlas/*.npz``
and the fitted parameters in ``data/corrected/arcturus_spectra.h5`` -- so it needs
no GPU, no line list and no refit.

Pages are chosen by rule, not by eye:

* telluric-rich = more than 10% of reliable pixels have an effective (convolved)
  transmission below 0.9;
* figure 1 (typical): the telluric-rich page inside the 2.0-2.1 um CO2 bands
  (4750-5300 cm-1) whose residual/noise is closest to the median of the whole
  atlas;
* figure 2 (good/bad): the telluric-rich pages closest to the 10th and 90th
  percentile of residual/noise among telluric-rich pages;
* figure 3: our effective transmission against the atlas's own telluric column,
  on the figure-1 page and atlas-wide.

Zoom panels are placed by rule too: on the 5 cm-1 (or quarter-page) interval
with the most telluric absorption, except on the bad page, where the zoom goes
to the interval with the largest residual so the failure is what is shown.

Run:  UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_arcturus_pages.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPECTRA = ROOT / "data/corrected/arcturus_spectra.h5"
ARRAYS = ROOT / "data/corrected/atlas"
OUT = Path(__file__).resolve().parent / "output"

# Okabe & Ito; observed data is always black.
C_DATA = "black"
C_MODEL = "#D55E00"
C_TRANS = "#0072B2"
C_STAR = "#E69F00"
C_ALT = "#009E73"
C_GREY = "0.6"
C_CORR = "#882255"  # Tol muted wine: the corrected spectrum, distinct from the observed

TELLURIC_DEPTH = 0.9
RICH_FRACTION = 0.10
CO2_BAND = (4750.0, 5300.0)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 0.7, "font.family": "serif",
    "mathtext.fontset": "dejavuserif", "xtick.direction": "in", "ytick.direction": "in",
    "savefig.dpi": 200,
})


def load_table():
    with h5py.File(SPECTRA) as f:
        P = f["parameters"][:]
        eff = f["effective_transmission"][:]
        rel = f["reliable"][:]
    frac = np.array([np.mean(eff[i][rel[i]] < TELLURIC_DEPTH) for i in range(len(P))])
    return P, frac


def load_page(row):
    page, epoch = row["page"].decode(), row["epoch"].decode()
    a = dict(np.load(ARRAYS / f"{page}_{epoch}.npz"))
    a["eff"] = a["model_flux"] / np.maximum(a["stellar_only"], 1e-12)
    a["sigma"] = float(row["pixel_sigma"])
    a["name"] = f"{page.rstrip('_')} {epoch}"
    a["row"] = row
    return a


def percentile_of(values, x):
    return 100.0 * np.mean(values <= x)


def nearest(values, candidates, target):
    idx = np.where(candidates)[0]
    return int(idx[np.argmin(np.abs(values[idx] - target))])


def zoom_interval(a, width, by="telluric"):
    nu = a["wavenumber_cm1"]
    m = a["mask"]
    if by == "telluric":
        # partially absorbed pixels, where the correction is applied and matters
        score = (a["reliable"] & (a["eff"] > 0.2) & (a["eff"] < TELLURIC_DEPTH)).astype(float)
    else:
        score = np.where(m, (a["residual"] / a["sigma"]) ** 2, 0.0)
    step = abs(nu[1] - nu[0])
    n = max(int(round(width / step)), 8)
    c = np.convolve(score, np.ones(n), mode="valid")
    j = int(np.argmax(c))
    return float(nu[j]), float(nu[min(j + n - 1, nu.size - 1)])


def add_top_axes(ax, nu, show_wavelength=True, pixel_offset=19, wl_bins=6):
    """Wavelength (um) on the top edge and the array index above it.

    The pixel index is the index into the page's saved arrays; the FTS grid is
    uniform in wavenumber, so it is a linear function of the primary axis.
    """
    step = nu[1] - nu[0]
    to_pix = lambda x: (x - nu[0]) / step  # noqa: E731
    to_nu = lambda p: nu[0] + p * step  # noqa: E731
    if show_wavelength:
        wl = ax.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
        wl.set_xlabel(r"vacuum wavelength ($\mu$m)", labelpad=2)
        wl.tick_params(direction="in", pad=1.5)
        wl.xaxis.set_major_locator(MaxNLocator(wl_bins))
    px = ax.secondary_xaxis("top", functions=(to_pix, to_nu))
    px.spines["top"].set_position(("outward", pixel_offset if show_wavelength else 0))
    px.tick_params(direction="out", pad=1.5, labelsize=6.5)
    px.set_xlabel("pixel index", labelpad=2, fontsize=7)
    return px


def plot_rows(axes, a, lo, hi, legend=True, star_label="Payne Zero model"):
    """Fill four stacked axes: data+model, transmission, corrected vs star, residual."""
    nu = a["wavenumber_cm1"]
    sel = (nu >= lo) & (nu <= hi)
    m = a["mask"]
    obs = np.where(m, a["observed"], np.nan)
    rel = a["reliable"]
    cont = a["continuum"]
    ax0, ax1, ax2, ax3 = axes

    ax0.plot(nu[sel], obs[sel], color=C_DATA, lw=0.7, label="observed")
    ax0.plot(nu[sel], a["model_flux"][sel], color=C_MODEL, lw=0.7, ls="-", alpha=0.9,
             label="fitted model")

    ax1.plot(nu[sel], a["eff"][sel], color=C_TRANS, lw=0.7)
    ax1.axhline(0.15, color=C_GREY, lw=0.5, ls=":")

    corr = np.where(rel, a["corrected"] / cont, np.nan)
    ax2.plot(nu[sel], (a["stellar_only"] / cont)[sel], color=C_STAR, lw=1.0,
             label=star_label)
    ax2.plot(nu[sel], corr[sel], color=C_CORR, lw=0.7, label="corrected")

    z = np.where(m, a["residual"] / a["sigma"], np.nan)
    ax3.plot(nu[sel], z[sel], color=C_DATA, lw=0.5)
    ax3.axhline(0, color=C_MODEL, lw=0.5)
    for ax in axes:
        unrel = ~rel
        # shade where the correction is not trusted (transmission < 0.15)
        y0, y1 = 0, 1
        ax.fill_between(nu[sel], y0, y1, where=unrel[sel], color="0.88", lw=0,
                        transform=ax.get_xaxis_transform(), zorder=0)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    if legend:
        ax0.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.4)
        ax2.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.4)


def finish_y(axes, a, lo, hi, ylabel=True):
    nu = a["wavenumber_cm1"]
    sel = (nu >= lo) & (nu <= hi) & a["mask"]
    z = a["residual"][sel] / a["sigma"]
    lim = max(5.0, float(np.nanpercentile(np.abs(z), 99.5)) * 1.1)
    axes[3].set_ylim(-lim, lim)
    axes[1].set_ylim(-0.03, 1.05)
    top = max(np.nanmax(a["observed"][sel]), np.nanmax(a["model_flux"][sel]))
    axes[0].set_ylim(min(-0.02, np.nanmin(a["observed"][sel]) - 0.05), top * 1.06)
    s = (a["stellar_only"] / a["continuum"])[sel]
    axes[2].set_ylim(max(0.0, float(np.nanmin(s)) - 0.08), 1.08)
    if ylabel:
        axes[0].set_ylabel("flux")
        axes[1].set_ylabel(r"$T_{\rm eff}$")
        axes[2].set_ylabel("flux / continuum")
        axes[3].set_ylabel(r"resid. / $\sigma$")


def page_stats(a):
    m, rel = a["mask"], a["reliable"]
    tel = rel & (a["eff"] < TELLURIC_DEPTH)
    clean = rel & (a["eff"] > 0.95)
    z = a["residual"] / a["sigma"]
    out = {
        "page": a["name"],
        "v1_cm1": float(a["row"]["v1"]), "v2_cm1": float(a["row"]["v2"]),
        "residual_rms_over_noise": float(a["row"]["residual_rms_over_noise"]),
        "pixel_sigma": a["sigma"],
        "residual_rms": float(a["row"]["residual_rms"]),
        "rms_over_noise_telluric_pixels": float(np.sqrt(np.mean(z[tel] ** 2))) if tel.sum() else None,
        "rms_over_noise_clean_pixels": float(np.sqrt(np.mean(z[clean] ** 2))) if clean.sum() else None,
        "telluric_pixels": int(tel.sum()), "clean_pixels": int(clean.sum()),
        "reliable_pixels": int(rel.sum()), "fitted_pixels": int(m.sum()),
        "min_effective_transmission": float(np.nanmin(a["eff"][rel])),
        "free_species": a["row"]["free_species"].decode(),
        "at_bound": a["row"]["at_bound"].decode(),
        "converged": bool(a["row"]["all_stages_converged"]),
        "lsf_sigma_kms": float(a["row"]["lsf_sigma_kms"]),
        "condition_number": float(a["row"]["condition_number"]),
    }
    # rms of (corrected - star model)/continuum split by telluric depth: a large
    # value in clean pixels is the stellar model, not the atmosphere.
    d = (a["corrected"] - a["stellar_only"]) / a["continuum"]
    out["corrected_minus_model_rms_clean"] = float(np.sqrt(np.nanmean(d[clean] ** 2))) if clean.sum() else None
    out["corrected_minus_model_rms_telluric"] = float(np.sqrt(np.nanmean(d[tel] ** 2))) if tel.sum() else None
    return out


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    plt.close(fig)


def figure_single(a, stem, title):
    nu = a["wavenumber_cm1"]
    v1, v2 = nu[a["mask"]].min(), nu[a["mask"]].max()
    width = min(5.0, 0.25 * (v2 - v1))
    z1, z2 = zoom_interval(a, width)
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 6.4), sharex="col",
                             gridspec_kw=dict(width_ratios=[2.3, 1], height_ratios=[1.3, 1, 1.3, 0.8],
                                              hspace=0.08, wspace=0.16, left=0.085, right=0.985,
                                              top=0.87, bottom=0.075))
    plot_rows(axes[:, 0], a, v1, v2)
    plot_rows(axes[:, 1], a, z1, z2, legend=False)
    finish_y(axes[:, 0], a, v1, v2)
    finish_y(axes[:, 1], a, z1, z2, ylabel=False)
    for ax in axes[:, 0]:
        ax.axvspan(z1, z2, color=C_TRANS, alpha=0.08, lw=0, zorder=0)
    add_top_axes(axes[0, 0], nu)
    add_top_axes(axes[0, 1], nu, wl_bins=3)
    for ax in axes[-1]:
        ax.set_xlabel(r"vacuum wavenumber (cm$^{-1}$)")
    fig.text(0.01, 0.995, title, va="top", ha="left", fontsize=8)
    for ax, lab in zip(axes[:, 0], "aceg"):
        ax.text(0.005, 0.96, f"({lab})", transform=ax.transAxes, va="top", fontsize=7)
    for ax, lab in zip(axes[:, 1], "bdfh"):
        ax.text(0.015, 0.96, f"({lab})", transform=ax.transAxes, va="top", fontsize=7)
    save(fig, stem)


def figure_pair(good, bad, stem):
    fig = plt.figure(figsize=(7.1, 7.6))
    outer = fig.add_gridspec(1, 2, wspace=0.2, left=0.075, right=0.972, top=0.905, bottom=0.06)
    zooms = []
    for k, (a, label, by) in enumerate(((good, "good", "telluric"), (bad, "bad", "residual"))):
        nu = a["wavenumber_cm1"]
        v1, v2 = nu[a["mask"]].min(), nu[a["mask"]].max()
        width = min(5.0, 0.25 * (v2 - v1))
        z1, z2 = zoom_interval(a, width, by=by)
        zooms.append((z1, z2))
        g = outer[k].subgridspec(2, 1, height_ratios=[1.25, 3.3], hspace=0.42)
        gf = g[0].subgridspec(2, 1, height_ratios=[1.3, 0.7], hspace=0.06)
        gz = g[1].subgridspec(4, 1, height_ratios=[1.3, 1, 1.3, 0.8], hspace=0.06)
        f0 = fig.add_subplot(gf[0])
        f1 = fig.add_subplot(gf[1], sharex=f0)
        zax = [fig.add_subplot(gz[0])]
        zax += [fig.add_subplot(gz[i], sharex=zax[0]) for i in (1, 2, 3)]

        m = a["mask"]
        f0.plot(nu, np.where(m, a["observed"], np.nan), color=C_DATA, lw=0.4, label="observed")
        f0.plot(nu, a["model_flux"], color=C_MODEL, lw=0.4, alpha=0.9, label="model")
        f1.plot(nu, np.where(m, a["residual"] / a["sigma"], np.nan), color=C_DATA, lw=0.3)
        f1.axhline(0, color=C_MODEL, lw=0.5)
        for ax in (f0, f1):
            ax.set_xlim(v1, v2)
            ax.axvspan(z1, z2, color=C_TRANS, alpha=0.12, lw=0, zorder=0)
        zr = np.abs(a["residual"][m] / a["sigma"])
        lim = max(5.0, float(np.nanpercentile(zr, 99.5)) * 1.1)
        f1.set_ylim(-lim, lim)
        f0.set_ylim(min(-0.02, np.nanmin(a["observed"][m]) - 0.05), 1.08 * np.nanmax(a["observed"][m]))
        plt.setp(f0.get_xticklabels(), visible=False)
        add_top_axes(f0, nu, pixel_offset=17)
        f1.set_xlabel(r"wavenumber (cm$^{-1}$)", labelpad=1)
        plot_rows(zax, a, z1, z2, legend=(k == 0))
        finish_y(zax, a, z1, z2, ylabel=(k == 0))
        add_top_axes(zax[0], nu, pixel_offset=17, wl_bins=5)
        for ax in zax[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        zax[-1].set_xlabel(r"vacuum wavenumber (cm$^{-1}$)")
        if k == 0:
            f0.set_ylabel("flux")
            f1.set_ylabel(r"res./$\sigma$")
        r = a["row"]
        head = (f"{label}: {a['name']}, residual {r['residual_rms_over_noise']:.2f}$\\sigma$ "
                f"($\\sigma$ = {a['sigma']:.4f})")
        fig.text(0.075 + k * 0.5, 0.995, head, va="top", ha="left", fontsize=8)
        for ax, lab in zip([f0, f1] + zax, "abcdef"):
            ax.text(0.01, 0.95, f"({lab}{k + 1})", transform=ax.transAxes, va="top", fontsize=6.5)
    save(fig, stem)
    return zooms


def figure_atlas_telluric(a, stem):
    nu = a["wavenumber_cm1"]
    at = a["atlas_telluric"]
    eff = a["eff"]
    comparable = a["reliable"] & np.isfinite(at) & (at > 0.2) & (at <= 1.0)
    deep = comparable & (eff < 0.8)
    k_page = float(np.median(np.log(np.maximum(at[deep], 1e-6)) / np.log(np.maximum(eff[deep], 1e-6))))

    rows = []
    for f in sorted(ARRAYS.glob("ab*_*.npz")):
        b = np.load(f)
        bat = b["atlas_telluric"]
        beff = b["model_flux"] / np.maximum(b["stellar_only"], 1e-12)
        g = b["reliable"] & np.isfinite(bat) & (bat > 0.2) & (bat <= 1.0)
        if g.sum() < 50:
            continue
        dp = g & (beff < 0.8)
        kk = (float(np.median(np.log(np.maximum(bat[dp], 1e-6)) / np.log(np.maximum(beff[dp], 1e-6))))
              if dp.sum() >= 30 else np.nan)
        rows.append((f.stem.split("_")[-1], float(np.mean(b["wavenumber_cm1"][g])),
                     float(np.median(np.abs(beff[g] - bat[g]))), kk))
    ep = np.array([r[0] for r in rows])
    nuc = np.array([r[1] for r in rows])
    med = np.array([r[2] for r in rows])
    kc = np.array([r[3] for r in rows])

    v1, v2 = nu[a["mask"]].min(), nu[a["mask"]].max()
    z1, z2 = zoom_interval(a, 5.0)
    fig = plt.figure(figsize=(7.1, 6.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.2], hspace=0.22,
                             left=0.085, right=0.985, top=0.87, bottom=0.075)
    gs = outer[0].subgridspec(2, 2, height_ratios=[1.3, 0.7], width_ratios=[2.3, 1],
                              hspace=0.06, wspace=0.17)
    ax_full = fig.add_subplot(gs[0, 0])
    ax_dfull = fig.add_subplot(gs[1, 0], sharex=ax_full)
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_dzoom = fig.add_subplot(gs[1, 1], sharex=ax_zoom)
    bottom = outer[1].subgridspec(1, 2, wspace=0.28)
    ax_nu = fig.add_subplot(bottom[0])
    ax_k = fig.add_subplot(bottom[1])

    shown = np.where(comparable, at, np.nan)
    diff = np.where(comparable, eff - at, np.nan)
    for ax, ad, lo, hi, leg in ((ax_full, ax_dfull, v1, v2, True), (ax_zoom, ax_dzoom, z1, z2, False)):
        s = (nu >= lo) & (nu <= hi)
        ax.plot(nu[s], np.where(comparable, np.nan, at)[s], color=C_GREY, lw=0.5,
                label="atlas, outside (0.2, 1]")
        ax.plot(nu[s], shown[s], color=C_DATA, lw=0.6, label="atlas telluric")
        ax.plot(nu[s], eff[s], color=C_TRANS, lw=0.8, ls="--", label=r"this work, $T_{\rm eff}$")
        ad.plot(nu[s], diff[s], color=C_DATA, lw=0.5)
        ad.axhline(0, color=C_MODEL, lw=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0.1, 1.05)
        ad.set_ylim(-0.15, 0.15)
        ax.xaxis.set_major_locator(MaxNLocator(5 if not leg else 8))
        plt.setp(ax.get_xticklabels(), visible=False)
        ad.set_xlabel(r"vacuum wavenumber (cm$^{-1}$)", labelpad=1)
        add_top_axes(ax, nu, wl_bins=6 if leg else 3)
        if leg:
            ax.legend(loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=3, frameon=False,
                      bbox_transform=fig.transFigure, handlelength=1.6)
    ax_full.axvspan(z1, z2, color=C_TRANS, alpha=0.08, lw=0)
    ax_dfull.axvspan(z1, z2, color=C_TRANS, alpha=0.08, lw=0)
    ax_full.set_ylabel("transmission")
    ax_dfull.set_ylabel("ours $-$ atlas")

    for e, c, mk in (("summer", C_MODEL, "o"), ("winter", C_TRANS, "s")):
        mm = ep == e
        ax_nu.scatter(nuc[mm], med[mm], s=4, marker=mk, color=c, alpha=0.7, lw=0,
                      label=f"{e}, median {np.median(med[mm]):.4f}")
        kk = kc[mm & np.isfinite(kc)]
        ax_k.hist(kk, bins=40, range=(0.3, 1.6), histtype="step", color=c, lw=1.0,
                  ls="-" if e == "summer" else "--",
                  label=f"{e}: median {np.median(kk):.2f}")
    ax_nu.set_yscale("log")
    ax_nu.set_xlabel(r"page mean wavenumber (cm$^{-1}$)")
    ax_nu.set_ylabel(r"median $|T_{\rm eff} - T_{\rm atlas}|$")
    ax_nu.set_ylim(2e-4, 20.0)
    ax_nu.legend(frameon=False, loc="upper right", markerscale=2)
    ax_k.axvline(1.0, color=C_GREY, lw=0.7, ls=":")
    ax_k.axvline(k_page, color=C_DATA, lw=0.8)
    ax_k.set_xlabel(r"$k$ = median $\ln T_{\rm atlas} / \ln T_{\rm eff}$ per page")
    ax_k.set_ylabel("pages")
    ax_k.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.04, 1.0))
    fig.text(0.01, 0.995, f"Arcturus {a['name']}",
             va="top", fontsize=8)
    for ax, lab in zip((ax_full, ax_zoom, ax_dfull, ax_dzoom, ax_nu, ax_k), "abcdef"):
        ax.text(0.01, 0.95, f"({lab})", transform=ax.transAxes, va="top", fontsize=7)
    save(fig, stem)
    return {
        "page": a["name"], "comparable_pixels": int(comparable.sum()),
        "page_median_abs_diff": float(np.nanmedian(np.abs(diff))),
        "page_k": k_page, "pages_compared": len(rows),
        "atlas_median_abs_diff": float(np.median(med)),
        "atlas_median_abs_diff_p90": float(np.percentile(med, 90)),
        "k_median_summer": float(np.nanmedian(kc[ep == "summer"])),
        "k_median_winter": float(np.nanmedian(kc[ep == "winter"])),
    }


def main():
    P, frac = load_table()
    r = P["residual_rms_over_noise"]
    rich = frac > RICH_FRACTION
    nu_mid = 0.5 * (P["v1"] + P["v2"])
    atlas_median = float(np.median(r))
    in_band = rich & (P["v1"] >= CO2_BAND[0]) & (P["v2"] <= CO2_BAND[1])
    i_typ = nearest(r, in_band, atlas_median)
    p10, p90 = np.percentile(r[rich], [10, 90])
    i_good = nearest(r, rich, p10)
    i_bad = nearest(r, rich, p90)

    report = {
        "rule": {
            "telluric_rich": f"fraction of reliable pixels with T_eff < {TELLURIC_DEPTH} > {RICH_FRACTION}",
            "typical": f"telluric-rich page in {CO2_BAND} cm-1 nearest the atlas median residual/noise",
            "good_bad": "telluric-rich pages nearest the 10th / 90th percentile of residual/noise",
        },
        "atlas_pages": int(len(P)), "telluric_rich_pages": int(rich.sum()),
        "atlas_median_residual_over_noise": atlas_median,
        "rich_p10": float(p10), "rich_p50": float(np.median(r[rich])), "rich_p90": float(p90),
    }
    a = load_page(P[i_typ])
    st = page_stats(a)
    st["percentile_all"] = percentile_of(r, r[i_typ])
    st["percentile_rich"] = percentile_of(r[rich], r[i_typ])
    report["typical"] = st
    figure_single(a, "fig_arcturus_typical",
                  f"Arcturus {a['name']} ({st['v1_cm1']:.0f}$-${st['v2_cm1']:.0f} cm$^{{-1}}$): "
                  f"residual {st['residual_rms_over_noise']:.2f}$\\sigma$, "
                  f"{st['percentile_all']:.0f}th percentile of the atlas")
    report["atlas_telluric"] = figure_atlas_telluric(a, "fig_arcturus_atlas_telluric")

    good, bad = load_page(P[i_good]), load_page(P[i_bad])
    for key, i, b in (("good", i_good, good), ("bad", i_bad, bad)):
        s = page_stats(b)
        s["percentile_all"] = percentile_of(r, r[i])
        s["percentile_rich"] = percentile_of(r[rich], r[i])
        report[key] = s
    zooms = figure_pair(good, bad, "fig_arcturus_good_bad")
    report["good"]["zoom_cm1"], report["bad"]["zoom_cm1"] = zooms
    report["typical"]["zoom_cm1"] = zoom_interval(a, min(5.0, 0.25 * (st["v2_cm1"] - st["v1_cm1"])))
    (OUT / "fig_arcturus_pages.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
