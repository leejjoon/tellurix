"""Publication figures: telluric correction of the NSO solar atlases, window by window.

Reads only saved products: the per-window arrays of the ftsspec_901218_5 fit
(``data/corrected/solar/ftsspec_901218_5_*.npz``, which is what photatl's
product is built from) and of the niratl fit (``data/corrected/solar/niratl/``),
their records, ``photatl_corrected.h5`` for photatl's per-page gain and offset,
and the atlases' own pages (read with ``tellurix_fts.nso``) for the authors'
telluric-free ``solar`` column.

Windows are chosen by rule, not by eye. The fit's residual/noise is not used
to rank solar windows: the noise differs by 100x between windows (and niratl's
is 5x below file 5's), so it ranks windows by their signal-to-noise and by the
solar model, not by the atmosphere. The ranking metric instead is

    rms of (observed - model) / fitted continuum over reliable pixels whose
    effective transmission is below 0.9,

the residual where the correction is actually applied, in continuum units.

* telluric-rich = at least 25% of reliable pixels below T_eff = 0.9;
* the typical windows (file 5, niratl) are the telluric-rich windows nearest
  that atlas's median of the metric;
* the O2 window is the niratl window with the most pixels in which O2 alone
  absorbs (the A band head region);
* the worst window is the largest metric over both atlases among windows with at
  least 200 telluric pixels (fewer and the rms is a handful of stellar lines).

Run:  UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_solar_pages.py
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

from tellurix_fts.nso import read_niratl_page, read_photatl_page  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOLAR = ROOT / "data/corrected/solar"
F5_RECORD = SOLAR / "ftsspec_901218_5.h5"
NIR_RECORD = SOLAR / "niratl/niratl.h5"
PHOTATL_PRODUCT = ROOT / "data/corrected/photatl_corrected.h5"
OUT = Path(__file__).resolve().parent / "output"
PHOTATL_PAGE_SCALES: dict[str, float] = {}

C_DATA = "black"
C_MODEL = "#D55E00"
C_TRANS = "#0072B2"
C_STAR = "#E69F00"
C_ATLAS = "#009E73"
C_GREY = "0.6"
C_CORR = "#882255"  # Tol muted wine: the corrected spectrum, distinct from the observed

TELLURIC_DEPTH = 0.9
RICH_FRACTION = 0.25
MIN_TELLURIC_PIXELS = 200

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 0.7, "font.family": "serif",
    "mathtext.fontset": "dejavuserif", "xtick.direction": "in", "ytick.direction": "in",
    "savefig.dpi": 200,
})


# ---------------------------------------------------------------- loading
def window_path(atlas, window):
    w0 = window.split("-")[0]
    if atlas == "file5":
        return SOLAR / f"ftsspec_901218_5_{w0}.npz"
    hits = sorted((SOLAR / "niratl").glob(f"ph*_{w0}.npz"))
    if len(hits) != 1:
        raise FileNotFoundError(f"niratl window {window}: {hits}")
    return hits[0]


def load_window(atlas, row):
    window = row["window"].decode()
    a = dict(np.load(window_path(atlas, window), allow_pickle=True))
    a["eff"] = a["model_flux"] / np.maximum(a["stellar_only_pixels"], 1e-300)
    a["sigma"] = float(row["pixel_sigma"])
    with np.errstate(divide="ignore", invalid="ignore"):
        a["scale"] = float(np.nanmedian(a["flux"] / a["observed_raw"]))
    a["row"] = row
    a["atlas"] = atlas
    a["name"] = f"{'photatl / file 5' if atlas == 'file5' else 'niratl'} {window} cm$^{{-1}}$"
    a["window"] = window
    return a


def window_metrics(atlas, row):
    a = load_window(atlas, row)
    rel = a["reliable"] & a["mask"]
    tel = rel & (a["eff"] < TELLURIC_DEPTH)
    res = a["residual"] / a["continuum"]
    o2_only = np.zeros_like(rel)
    if "species_transmission_O2" in a:
        others = np.ones_like(a["eff"])
        for k in a:
            if k.startswith("species_transmission_") and k not in ("species_transmission_O2",):
                others = others * a[k]
        o2_only = rel & (a["species_transmission_O2"] < TELLURIC_DEPTH) & (others > 0.98)
    return {
        "atlas": atlas, "window": row["window"].decode(),
        "telluric_rms": float(np.sqrt(np.mean(res[tel] ** 2))) if tel.sum() else np.nan,
        "telluric_pixels": int(tel.sum()), "reliable_pixels": int(rel.sum()),
        "telluric_fraction": float(tel.sum() / max(rel.sum(), 1)),
        "residual_rms_over_noise": float(row["residual_rms_over_noise"]),
        "o2_only_pixels": int(o2_only.sum()),
    }


def records():
    out = {}
    for atlas, path in (("file5", F5_RECORD), ("niratl", NIR_RECORD)):
        with h5py.File(path) as f:
            out[atlas] = f["pages"][:]
    return out


# ---------------------------------------------------- the atlases' own solar column
def photatl_solar(a):
    """photatl's ``solar`` column on this window's samples, in the window's flux units.

    photatl = gain * ftsspec_5 + offset per page (gain, offset in the product),
    so the inverse map puts the authors' solar column on file 5's scale. NaN
    where the authors interpolated it (atmospheric < dg) -- that is not a
    measurement.
    """
    with h5py.File(PHOTATL_PRODUCT) as f:
        pages = [p.decode() for p in f["page"][:]]
        gain, offset = f["gain"][:], f["offset"][:]
        atlas_dir = Path(f.attrs["photatl"])
    nu = a["wavenumber_cm1"]
    solar = np.full(nu.shape, np.nan)
    total = np.full(nu.shape, np.nan)
    for k, name in enumerate(pages):
        start = float(name[2:])
        if start > nu.max() or start + 27.5 < nu.min():
            continue
        page = read_photatl_page(atlas_dir / name)
        pn = page.wavenumber_vacuum_cm1
        inside = (nu >= pn[0]) & (nu <= pn[-1]) & np.isnan(solar)
        if not inside.any():
            continue
        # Each page's solar column carries its own scale, and on wn6700 that
        # scale is 5.5x off its own total; tie it to the total where the sky
        # is transparent, which is all the comparison needs.
        clear = (page.atmospheric > 0.98) & ~page.interpolated & (page.solar > 0)
        page_scale = float(np.median(page.total[clear] / page.solar[clear]))
        PHOTATL_PAGE_SCALES[name] = page_scale
        s = np.where(page.interpolated, np.nan, (page.solar * page_scale - offset[k]) / gain[k])
        t = (page.total - offset[k]) / gain[k]
        solar[inside] = np.interp(nu[inside], pn, s)
        total[inside] = np.interp(nu[inside], pn, t)
    # the check that the two share samples and scale: photatl's total mapped back
    # must reproduce the observed column
    m = a["mask"] & np.isfinite(total)
    check = float(np.nanmedian(np.abs(total[m] - a["observed_raw"][m])) / a["sigma"] * a["scale"])
    return solar * a["scale"], check


def niratl_solar(a):
    atlas_dir = Path(read_attr_photatl()).parent / "niratl"
    page_name = window_path("niratl", a["window"]).stem.split("_")[0]
    page = read_niratl_page(atlas_dir / page_name)
    # Where the authors' own sky is nearly opaque their solar column is not a
    # measurement (it dips to the telluric cores); drop it there, at the same
    # 0.15 our reliable mask uses.
    s = np.where(page.filled | (page.atmospheric < 0.15), np.nan, page.solar)
    nu = a["wavenumber_cm1"]
    solar = np.interp(nu, page.wavenumber_vacuum_cm1, s, left=np.nan, right=np.nan)
    obs = np.interp(nu, page.wavenumber_vacuum_cm1, page.observed, left=np.nan, right=np.nan)
    m = a["mask"] & np.isfinite(obs)
    check = float(np.nanmedian(np.abs(obs[m] - a["observed_raw"][m])) / a["sigma"] * a["scale"])
    return solar * a["scale"], check


def read_attr_photatl():
    with h5py.File(PHOTATL_PRODUCT) as f:
        return f.attrs["photatl"]


# ---------------------------------------------------------------- plotting
def zoom_interval(a, width, by="telluric"):
    nu = a["wavenumber_cm1"]
    rel = a["reliable"] & a["mask"]
    if by == "telluric":
        score = (rel & (a["eff"] > 0.2) & (a["eff"] < TELLURIC_DEPTH)).astype(float)
    else:
        score = np.where(rel & (a["eff"] < TELLURIC_DEPTH), (a["residual"] / a["continuum"]) ** 2, 0.0)
    step = abs(nu[1] - nu[0])
    n = max(int(round(width / step)), 8)
    c = np.convolve(score, np.ones(n), mode="valid")
    j = int(np.argmax(c))
    lo, hi = sorted((float(nu[j]), float(nu[min(j + n - 1, nu.size - 1)])))
    return lo, hi


def add_top_axes(ax, nu, pixel_offset=19, wl_bins=6):
    """Wavelength (um) on the top edge and the window's array index above it.

    The solar arrays descend in wavenumber, so the index runs right to left.
    """
    step = nu[1] - nu[0]
    to_pix = lambda x: (x - nu[0]) / step  # noqa: E731
    to_nu = lambda p: nu[0] + p * step  # noqa: E731
    wl = ax.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
    wl.set_xlabel(r"vacuum wavelength ($\mu$m)", labelpad=2)
    wl.tick_params(direction="in", pad=1.5)
    wl.xaxis.set_major_locator(MaxNLocator(wl_bins))
    px = ax.secondary_xaxis("top", functions=(to_pix, to_nu))
    px.spines["top"].set_position(("outward", pixel_offset))
    px.tick_params(direction="out", pad=1.5, labelsize=6.5)
    px.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    px.set_xlabel("pixel index", labelpad=2, fontsize=7)


def plot_rows(axes, a, lo, hi, atlas_solar, legend=True, atlas_label="atlas solar column",
              scale_in_label=True):
    nu = a["wavenumber_cm1"]
    sel = (nu >= lo) & (nu <= hi)
    m = a["mask"]
    rel = a["reliable"] & m
    cont = a["continuum"]
    ax0, ax1, ax2, ax3 = axes

    ax0.plot(nu[sel], np.where(m, a["flux"], np.nan)[sel], color=C_DATA, lw=0.7, label="observed")
    ax0.plot(nu[sel], a["model_flux"][sel], color=C_MODEL, lw=0.7, alpha=0.9, label="fitted model")

    ax1.plot(nu[sel], a["eff"][sel], color=C_TRANS, lw=0.7)
    ax1.axhline(0.15, color=C_GREY, lw=0.5, ls=":")

    corr = np.where(rel, a["flux"] / a["eff"] / cont, np.nan)
    ax2.plot(nu[sel], (a["stellar_only_pixels"] / cont)[sel], color=C_STAR, lw=1.0,
             label="Payne Zero model")
    if atlas_solar is not None:
        # compared on the pixels we correct, so both curves have the same gaps
        shown = np.where(rel, a["atlas_scale"] * atlas_solar / cont, np.nan)
        ax2.plot(nu[sel], shown[sel], color=C_ATLAS, lw=0.8,
                 ls="--", label=(f"{atlas_label} $\\times$ {a['atlas_scale']:.3f}" if scale_in_label
                        else f"{atlas_label} (scaled)"))
    ax2.plot(nu[sel], corr[sel], color=C_CORR, lw=0.7, label="corrected (this work)")

    res = np.where(m, a["residual"] / cont, np.nan) * 100.0
    sig = a["sigma"] / np.nanmedian(cont) * 100.0
    ax3.fill_between(nu[sel], -sig, sig, color=C_TRANS, alpha=0.25, lw=0, label=r"$\pm1\sigma$")
    ax3.plot(nu[sel], res[sel], color=C_DATA, lw=0.5)
    ax3.axhline(0, color=C_MODEL, lw=0.5)
    for ax in axes:
        ax.fill_between(nu[sel], 0, 1, where=(~rel)[sel], color="0.88", lw=0,
                        transform=ax.get_xaxis_transform(), zorder=0)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    if legend:
        handles, labels = [], []
        for ax in (ax0, ax2, ax3):
            h, lab = ax.get_legend_handles_labels()
            handles += h
            labels += lab
        ax0.figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                          handlelength=1.8, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))


def finish_y(axes, a, lo, hi, ylabel=True):
    nu = a["wavenumber_cm1"]
    sel = (nu >= lo) & (nu <= hi)
    m = sel & a["mask"]
    res = a["residual"][m] / a["continuum"][m] * 100.0
    lim = max(3 * a["sigma"] / np.nanmedian(a["continuum"]) * 100.0,
              float(np.nanpercentile(np.abs(res), 99.5)) * 1.15)
    axes[3].set_ylim(-lim, lim)
    axes[1].set_ylim(-0.03, 1.05)
    top = max(np.nanmax(a["flux"][m]), np.nanmax(a["model_flux"][m]))
    axes[0].set_ylim(min(-0.02, np.nanmin(a["flux"][m]) - 0.05), top * 1.06)
    s = (a["stellar_only_pixels"] / a["continuum"])[sel]
    c = np.where(a["reliable"] & a["mask"], a["flux"] / a["eff"] / a["continuum"], np.nan)[sel]
    low = np.nanmin(np.concatenate([s, np.nanpercentile(c, [1]) if np.isfinite(c).any() else s]))
    axes[2].set_ylim(max(0.0, float(low)) - 0.05, 1.12)
    if ylabel:
        axes[0].set_ylabel("flux")
        axes[1].set_ylabel(r"$T_{\rm eff}$")
        axes[2].set_ylabel("flux / continuum")
        axes[3].set_ylabel("residual (%)")


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    plt.close(fig)


def figure_single(a, atlas_solar, stem, title, atlas_label):
    nu = a["wavenumber_cm1"]
    v1, v2 = np.sort(nu[a["mask"]])[[0, -1]]
    z1, z2 = zoom_interval(a, 5.0)
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 6.6), sharex="col",
                             gridspec_kw=dict(width_ratios=[2.3, 1], height_ratios=[1.3, 1, 1.4, 0.8],
                                              hspace=0.08, wspace=0.17, left=0.085, right=0.985,
                                              top=0.87, bottom=0.105))
    plot_rows(axes[:, 0], a, v1, v2, atlas_solar, atlas_label=atlas_label)
    plot_rows(axes[:, 1], a, z1, z2, atlas_solar, legend=False)
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
    return [z1, z2]


def figure_pair(panels, stem):
    fig = plt.figure(figsize=(7.1, 7.8))
    outer = fig.add_gridspec(1, 2, wspace=0.2, left=0.075, right=0.972, top=0.905, bottom=0.085)
    zooms = []
    for k, (a, atlas_solar, head, by, atlas_label) in enumerate(panels):
        nu = a["wavenumber_cm1"]
        v1, v2 = np.sort(nu[a["mask"]])[[0, -1]]
        z1, z2 = zoom_interval(a, 4.0, by=by)
        zooms.append([z1, z2])
        g = outer[k].subgridspec(2, 1, height_ratios=[1.25, 3.4], hspace=0.42)
        gf = g[0].subgridspec(2, 1, height_ratios=[1.3, 0.7], hspace=0.06)
        gz = g[1].subgridspec(4, 1, height_ratios=[1.3, 1, 1.4, 0.8], hspace=0.06)
        f0 = fig.add_subplot(gf[0])
        f1 = fig.add_subplot(gf[1], sharex=f0)
        zax = [fig.add_subplot(gz[0])]
        zax += [fig.add_subplot(gz[i], sharex=zax[0]) for i in (1, 2, 3)]

        m = a["mask"]
        f0.plot(nu, np.where(m, a["flux"], np.nan), color=C_DATA, lw=0.3)
        f0.plot(nu, a["model_flux"], color=C_MODEL, lw=0.3, alpha=0.9)
        res = np.where(m, a["residual"] / a["continuum"], np.nan) * 100
        f1.plot(nu, res, color=C_DATA, lw=0.3)
        f1.axhline(0, color=C_MODEL, lw=0.5)
        for ax in (f0, f1):
            ax.set_xlim(v1, v2)
            ax.axvspan(z1, z2, color=C_TRANS, alpha=0.15, lw=0, zorder=0)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.ticklabel_format(axis="x", useOffset=False, style="plain")
        lim = float(np.nanpercentile(np.abs(res[m]), 99.5)) * 1.15
        f1.set_ylim(-lim, lim)
        f0.set_ylim(min(-0.02, np.nanmin(a["flux"][m]) - 0.05), 1.08 * np.nanmax(a["flux"][m]))
        plt.setp(f0.get_xticklabels(), visible=False)
        add_top_axes(f0, nu, pixel_offset=17, wl_bins=5)
        f1.set_xlabel(r"wavenumber (cm$^{-1}$)", labelpad=1)
        plot_rows(zax, a, z1, z2, atlas_solar, legend=(k == 0), atlas_label=atlas_label,
                  scale_in_label=False)
        finish_y(zax, a, z1, z2, ylabel=(k == 0))
        add_top_axes(zax[0], nu, pixel_offset=17, wl_bins=4)
        for ax in zax[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        zax[-1].set_xlabel(r"vacuum wavenumber (cm$^{-1}$)")
        if k == 0:
            f0.set_ylabel("flux")
            f1.set_ylabel("res. (%)")
        fig.text(0.075 + k * 0.5, 0.995, head, va="top", ha="left", fontsize=8)
        for ax, lab in zip([f0, f1] + zax, "abcdef"):
            ax.text(0.01, 0.95, f"({lab}{k + 1})", transform=ax.transAxes, va="top", fontsize=6.5)
    save(fig, stem)
    return zooms


def set_atlas_scale(a, solar):
    """One scalar between the authors' solar column and our corrected spectrum.

    The atlases' solar columns carry their own normalisation (an empirical
    ratio of two spectra), so only their shape is comparable; the scale is the
    median ratio over reliable pixels where neither the Sun nor the sky absorbs.
    """
    corr = a["flux"] / a["eff"]
    pz = a["stellar_only_pixels"] / a["continuum"]
    g = a["reliable"] & a["mask"] & np.isfinite(solar) & (pz > 0.99) & (a["eff"] > 0.8)
    if g.sum() < 30:
        g = a["reliable"] & a["mask"] & np.isfinite(solar)
    a["atlas_scale"] = float(np.median(corr[g] / solar[g]))
    a["atlas_scale_pixels"] = int(g.sum())


def describe(a, metrics, all_rms, rich_rms):
    r = a["row"]
    d = dict(metrics)
    d.update({
        "v1_cm1": float(r["v1"]), "v2_cm1": float(r["v2"]),
        "pixel_sigma": a["sigma"], "residual_rms": float(r["residual_rms"]),
        "free_species": r["free_species"].decode(), "at_bound": r["at_bound"].decode(),
        "converged": bool(r["all_stages_converged"]),
        "airmass": float(r["airmass"]), "lsf_sigma_kms": float(r["lsf_sigma_kms"]),
        "min_effective_transmission": float(np.nanmin(a["eff"][a["reliable"] & a["mask"]])),
        "percentile_all_windows": float(100 * np.mean(all_rms <= d["telluric_rms"])),
        "percentile_rich_windows": float(100 * np.mean(rich_rms <= d["telluric_rms"])) if rich_rms.size else None,
    })
    names = [k.removeprefix("log_column_") for k in r.dtype.names if k.startswith("log_column_")]
    d["log_columns_free"] = {n: float(r[f"log_column_{n}"]) for n in names
                             if n in d["free_species"].split("+")}
    rel = a["reliable"] & a["mask"]
    tel = rel & (a["eff"] < TELLURIC_DEPTH)
    clean = rel & (a["eff"] > 0.99)
    res = a["residual"] / a["continuum"]
    d["clean_rms"] = float(np.sqrt(np.mean(res[clean] ** 2))) if clean.sum() > 30 else None
    pz = a["stellar_only_pixels"] / a["continuum"]
    line = tel & (pz < 0.95)
    noline = tel & (pz > 0.99)
    d["telluric_rms_on_solar_lines"] = float(np.sqrt(np.mean(res[line] ** 2))) if line.sum() > 10 else None
    d["telluric_rms_off_solar_lines"] = float(np.sqrt(np.mean(res[noline] ** 2))) if noline.sum() > 10 else None
    d["telluric_pixels_on_solar_lines"] = int(line.sum())
    return d


def main():
    recs = records()
    table = {atlas: [window_metrics(atlas, row) for row in rows] for atlas, rows in recs.items()}
    report = {"rule": __doc__.split("Run:")[0].strip().split("\n\n", 2)[-1]}

    chosen = {}
    for atlas in ("file5", "niratl"):
        t = table[atlas]
        rms = np.array([x["telluric_rms"] for x in t])
        frac = np.array([x["telluric_fraction"] for x in t])
        rich = np.isfinite(rms) & (frac >= RICH_FRACTION)
        med = float(np.median(rms[rich]))
        idx = np.where(rich)[0]
        i = int(idx[np.argmin(np.abs(rms[idx] - med))])
        chosen[atlas] = i
        report[f"{atlas}_windows"] = len(t)
        report[f"{atlas}_rich_windows"] = int(rich.sum())
        report[f"{atlas}_rich_telluric_rms_p10_p50_p90"] = [float(v) for v in np.percentile(rms[rich], [10, 50, 90])]
        report[f"{atlas}_all_telluric_rms_p10_p50_p90"] = [float(v) for v in np.nanpercentile(rms, [10, 50, 90])]

    def rms_arrays(atlas):
        t = table[atlas]
        rms = np.array([x["telluric_rms"] for x in t])
        frac = np.array([x["telluric_fraction"] for x in t])
        return rms[np.isfinite(rms)], rms[np.isfinite(rms) & (frac >= RICH_FRACTION)]

    # (c) typical windows
    for atlas, stem, label in (("file5", "fig_solar_photatl_typical", "photatl solar column"),
                               ("niratl", "fig_solar_niratl_typical", "niratl solar column")):
        i = chosen[atlas]
        a = load_window(atlas, recs[atlas][i])
        solar, check = photatl_solar(a) if atlas == "file5" else niratl_solar(a)
        d = describe(a, table[atlas][i], *rms_arrays(atlas))
        d["atlas_total_vs_observed_median_sigma"] = check
        set_atlas_scale(a, solar)
        d["atlas_solar_scale"] = a["atlas_scale"]
        solar_scaled = solar * a["atlas_scale"]
        rel = a["reliable"] & a["mask"] & np.isfinite(solar)
        corr = a["flux"] / a["eff"]
        d["rms_corrected_minus_atlas_solar"] = float(np.sqrt(np.mean(((corr - solar_scaled) / a["continuum"])[rel] ** 2)))
        d["rms_corrected_minus_payne_zero"] = float(np.sqrt(np.mean(((corr - a["stellar_only_pixels"]) / a["continuum"])[rel] ** 2)))
        d["rms_atlas_solar_minus_payne_zero"] = float(np.sqrt(np.mean(((solar_scaled - a["stellar_only_pixels"]) / a["continuum"])[rel] ** 2)))
        d["atlas_solar_measured_pixels"] = int(rel.sum())
        title = (f"{a['name']}: telluric-pixel residual {100 * d['telluric_rms']:.2f}%, "
                 f"{d['percentile_rich_windows']:.0f}th percentile of telluric-rich windows")
        d["zoom_cm1"] = figure_single(a, solar, stem, title, label)
        report[f"{atlas}_typical"] = d

    # (d) O2 A band and the worst window
    nt = table["niratl"]
    i_o2 = int(np.argmax([x["o2_only_pixels"] for x in nt]))
    a_o2 = load_window("niratl", recs["niratl"][i_o2])
    s_o2, check_o2 = niratl_solar(a_o2)
    d_o2 = describe(a_o2, nt[i_o2], *rms_arrays("niratl"))
    d_o2["atlas_total_vs_observed_median_sigma"] = check_o2
    set_atlas_scale(a_o2, s_o2)
    d_o2["atlas_solar_scale"] = a_o2["atlas_scale"]

    pool = [(atlas, j, x) for atlas in table for j, x in enumerate(table[atlas])
            if x["telluric_pixels"] >= MIN_TELLURIC_PIXELS and np.isfinite(x["telluric_rms"])]
    atlas_w, j_w, _ = max(pool, key=lambda p: p[2]["telluric_rms"])
    a_w = load_window(atlas_w, recs[atlas_w][j_w])
    s_w, check_w = photatl_solar(a_w) if atlas_w == "file5" else niratl_solar(a_w)
    d_w = describe(a_w, table[atlas_w][j_w], *rms_arrays(atlas_w))
    d_w["atlas_total_vs_observed_median_sigma"] = check_w
    set_atlas_scale(a_w, s_w)
    d_w["atlas_solar_scale"] = a_w["atlas_scale"]
    pool_rms = np.array([p[2]["telluric_rms"] for p in pool])
    d_w["pool_size"] = len(pool)
    d_w["pool_p50_p90"] = [float(v) for v in np.percentile(pool_rms, [50, 90])]
    d_o2["percentile_in_pool"] = float(100 * np.mean(pool_rms <= d_o2["telluric_rms"]))
    head_o2 = f"O$_2$ A band: niratl {a_o2['window']} cm$^{{-1}}$, residual {100 * d_o2['telluric_rms']:.2f}%"
    head_w = (f"worst: {'file 5' if atlas_w == 'file5' else 'niratl'} {a_w['window']} cm$^{{-1}}$, "
              f"residual {100 * d_w['telluric_rms']:.2f}%")
    zooms = figure_pair([
        (a_o2, s_o2, head_o2, "telluric", "atlas solar column"),
        (a_w, s_w, head_w, "residual", "atlas solar column"),
    ], "fig_solar_o2_worst")
    d_o2["zoom_cm1"], d_w["zoom_cm1"] = zooms
    report["o2_window"] = d_o2
    report["worst_window"] = d_w

    report["photatl_page_solar_scales"] = PHOTATL_PAGE_SCALES
    (OUT / "fig_solar_pages.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
