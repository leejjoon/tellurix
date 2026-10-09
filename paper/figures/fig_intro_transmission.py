#!/usr/bin/env python
"""Introductory figure: what absorbs where in the near infrared.

The atmosphere is the ERA5 column over Gemini South at 2021-03-17 07 UT
(``data/profiles/gemini_2021_era5.csv``: 12 weighted layers from 734 hPa /
2.72 km up to 8 hPa, 4.33 mm of precipitable water). Ozone is not in that
profile; it is added from the AFGL midlatitude-summer climatology on the same
layer altitudes (``tellurix.site_profile.afgl_dry_vmr``), which misses the
ozone above the 8 hPa top of the profile.

Every species is computed line by line with tellurix's production kernel
(``ExoJAXOpacityBackend``, ``direct_sparse`` with pressure shifts, AER 3.9 line
files) on one constant-velocity grid at R = 45,000 x 4 samples -- the sampling
the IGRINS and FTS drivers use -- from 0.9 to 5.3 um, evaluated in chunks of
the same global grid so the samples keep their phase. Water carries the MT_CKD
4.3 continuum and O2 the LBLRTM O2 collision-induced bands (``o2_cia.py``).
The transmission is along a slant path of air mass 1.5.

The overview is the same calculation smoothed to R = 2,000 so the bands are
legible; the main panels are convolved to R = 45,000 with a Gaussian. The
zoom compares the monochromatic transmission, recomputed on a 16x finer grid,
with the R = 45,000 spectrum an instrument records.

    CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_intro_transmission.py

The per-chunk optical depths are cached under
``paper/figures/cache/intro_transmission/`` (``--recompute`` redoes them;
about 10 min on one GPU). Writes ``fig_intro_transmission_overview.{pdf,png}``
(the 0.9-5.3 um overview), ``fig_intro_transmission.{pdf,png}`` (H and K, each
species, the zoom) and ``fig_intro_transmission.json`` to ``paper/figures/output/``.
O2 and O3 take out under 1% anywhere in 1.4-2.5 um, so they have no row of
their own in the H/K figure; the overview shows where they matter.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

import tellurix  # noqa: F401  (x64 before exojax)
from tellurix import (
    AERLineDatabase, DataPaths, ExoJAXOpacityBackend, MTCKDWaterContinuum, TelluricModel,
    constant_velocity_grid,
)
from tellurix.io import load_atmosphere_csv
from tellurix.o2_cia import O2CollisionInducedContinuum
from tellurix.site_profile import afgl_dry_vmr

from scipy.ndimage import gaussian_filter1d

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper/figures/cache/intro_transmission"
OUTPUT = ROOT / "paper/figures/output"
PROFILE = ROOT / "data/profiles/gemini_2021_era5.csv"

MOLECULE_IDS = {"H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
SPECIES = ("H2O", "CO2", "CH4", "N2O", "CO", "O2", "O3")
OVERVIEW_UM = (0.90, 5.30)
MAIN_UM = (1.40, 2.50)
CHUNK_CM1 = 500.0
RESOLVING_POWER = 45_000.0
OVERVIEW_R = 2_000.0
SAMPLES_PER_RESOLUTION = 4.0
ZOOM_SAMPLES_PER_RESOLUTION = 64.0
LINE_MARGIN_CM1 = 25.0
MAX_PAIRS = 3.0e8
AIRMASS = 1.5
# A 2 nm stretch of K band with CH4, H2O and CO lines side by side.
ZOOM_NM = (2318.0, 2320.0)
C_KMS = 299792.458
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
IGRINS = {"H": (1.49, 1.80), "K": (1.96, 2.48)}
# Photometric windows (MKO-like), for orientation only.
WINDOWS = {"J": (1.17, 1.33), "H": (1.49, 1.78), "K": (2.03, 2.37), "L$'$": (3.42, 4.12),
           "M$'$": (4.57, 4.79)}

WIDTH_IN = 7.1
COLOURS = {"H2O": "#0072B2", "CO2": "#D55E00", "CH4": "#009E73", "N2O": "#CC79A7",
           "CO": "#E69F00", "O2": "#56B4E9", "O3": "#B8A800"}
LABELS = {"H2O": r"H$_2$O", "CO2": r"CO$_2$", "CH4": r"CH$_4$", "N2O": r"N$_2$O",
          "CO": "CO", "O2": r"O$_2$", "O3": r"O$_3$"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})


def profile():
    atmosphere = load_atmosphere_csv(PROFILE)
    dry = afgl_dry_vmr("midlatitude_summer", np.asarray(atmosphere.altitude_km), "2020")
    vmr = dict(atmosphere.vmr)
    vmr["O3"] = dry["O3"] * (1.0 - np.asarray(vmr["H2O"]))
    return dataclasses.replace(atmosphere, vmr=vmr)


def global_grid(samples: float = SAMPLES_PER_RESOLUTION, um=OVERVIEW_UM) -> np.ndarray:
    grid = constant_velocity_grid(um[0] * 1e3, um[1] * 1e3, resolving_power=RESOLVING_POWER,
                                  samples_per_resolution=samples, margin_cm1=0.0)
    return np.asarray(grid)


def species_tau(atmosphere, grid: np.ndarray, v1: float, v2: float, species: str):
    """Vertical line optical depth of one species on ``grid``, or None if no lines.

    ExoJAX's Direct constructor builds the dense (line, grid) offset matrix
    once before SparseCoreDirect discards it, so a stretch dense in lines (the
    4.3 um CO2 band holds ~10^5 AER lines) is split in halves until that
    matrix stays under MAX_PAIRS.
    """

    paths = DataPaths.bootstrapped(ROOT)
    try:
        database = AERLineDatabase(paths.line_file(species, MOLECULE_IDS[species]), species,
                                   (v1, v2), margin_cm1=LINE_MARGIN_CM1)
    except ValueError:
        return None
    n_lines = int(np.asarray(database.nu_lines).size)
    if n_lines * grid.size > MAX_PAIRS and grid.size > 64:
        half = grid.size // 2
        tau, total = np.zeros_like(grid), 0
        for part in (grid[:half], grid[half:]):
            result = species_tau(atmosphere, part, float(part[0]), float(part[-1]), species)
            if result is not None:
                tau[np.isin(grid, part)] = result[0]
                total = max(total, result[1])
        return tau, n_lines
    backend = ExoJAXOpacityBackend.prepare(
        {species: database}, grid, methods="direct_sparse", temperature_range_k=(150.0, 320.0),
        maximum_pressure_bar=1.0, vectorize_layers=True, pressure_shift=True)
    one = dataclasses.replace(atmosphere, vmr={species: atmosphere.vmr[species]})
    model = TelluricModel(one, grid, backend)
    # precompute_opacity jit-compiles the kernel once; an eager call would
    # dispatch it op by op (CLAUDE.md, "What a fit costs"). Frozen is exact at
    # the profile's own abundances, which is all this figure evaluates.
    xs = np.asarray(model.precompute_opacity(self_broadening="frozen").opacity.values[species])
    column = np.asarray(atmosphere.air_column_cm2) * np.asarray(atmosphere.vmr[species])
    return (xs * column[:, None]).sum(axis=0), n_lines


def line_optical_depths(atmosphere, grid: np.ndarray, recompute: bool, tag: str,
                        chunk_cm1: float = CHUNK_CM1) -> tuple[dict, dict]:
    """Per-species vertical optical depth on ``grid``, chunk by chunk, cached."""

    CACHE.mkdir(parents=True, exist_ok=True)
    tau = {s: np.zeros_like(grid) for s in SPECIES}
    counts = {s: 0 for s in SPECIES}
    edges = np.arange(grid[0], grid[-1] + chunk_cm1, chunk_cm1)
    edges[-1] = np.nextafter(grid[-1], np.inf)
    for a, b in zip(edges[:-1], edges[1:]):
        inside = (grid >= a) & (grid < b)
        if not inside.any():
            continue
        path = CACHE / f"{tag}_{a:.3f}.npz"
        sub = grid[inside]
        if path.exists() and not recompute:
            data = np.load(path)
            if np.array_equal(data["grid"], sub):
                for s in SPECIES:
                    tau[s][inside] = data[f"tau_{s}"]
                    counts[s] += int(data[f"n_{s}"])
                continue
        started = time.time()
        pieces, numbers = {}, {}
        for s in SPECIES:
            result = species_tau(atmosphere, sub, float(sub[0]), float(sub[-1]), s)
            pieces[s], numbers[s] = (np.zeros_like(sub), 0) if result is None else result
            tau[s][inside] = pieces[s]
            counts[s] += numbers[s]
        np.savez(path, grid=sub, **{f"tau_{s}": v for s, v in pieces.items()},
                 **{f"n_{s}": v for s, v in numbers.items()})
        print(f"{tag} {a:8.1f}-{b:8.1f} cm-1  {sub.size:6d} points  {time.time() - started:5.1f} s",
              flush=True)
    return tau, counts


def continua(atmosphere, grid: np.ndarray) -> dict:
    """Vertical MT_CKD water and O2 CIA optical depths."""

    vmr = {s: np.asarray(v) for s, v in atmosphere.vmr.items()}
    water = MTCKDWaterContinuum.from_netcdf(DataPaths.bootstrapped(ROOT).mt_ckd, grid)
    oxygen = O2CollisionInducedContinuum(grid)
    out = {}
    for name, backend in (("H2O", water), ("O2", oxygen)):
        bound = backend.bind(atmosphere)
        out[name] = np.asarray(bound.optical_depth(atmosphere, vmr)).sum(axis=0)
    return out


def smooth(transmission: np.ndarray, resolving_power: float, samples: float) -> np.ndarray:
    """Gaussian of FWHM c/R on a constant-velocity grid of ``samples`` per c/R(45k)."""

    step_kms = C_KMS / RESOLVING_POWER / samples
    sigma = C_KMS / resolving_power * FWHM_TO_SIGMA / step_kms
    return gaussian_filter1d(transmission, sigma, mode="nearest")


def um(nu: np.ndarray) -> np.ndarray:
    return 1.0e4 / nu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    started = time.time()

    atmosphere = profile()
    grid = global_grid()
    lines, counts = line_optical_depths(atmosphere, grid, args.recompute, "full")
    cont = continua(atmosphere, grid)
    tau = {s: lines[s] + cont.get(s, 0.0) for s in SPECIES}
    tau_total = sum(tau.values())

    mono = {s: np.exp(-AIRMASS * tau[s]) for s in SPECIES}
    mono_total = np.exp(-AIRMASS * tau_total)
    r45 = {s: smooth(mono[s], RESOLVING_POWER, SAMPLES_PER_RESOLUTION) for s in SPECIES}
    r45_total = smooth(mono_total, RESOLVING_POWER, SAMPLES_PER_RESOLUTION)
    low_total = smooth(mono_total, OVERVIEW_R, SAMPLES_PER_RESOLUTION)

    # The zoom: same physics on a 16x finer grid, so the monochromatic line
    # cores are resolved.
    zoom_grid = global_grid(ZOOM_SAMPLES_PER_RESOLUTION, (ZOOM_NM[0] / 1e3 - 0.004,
                                                          ZOOM_NM[1] / 1e3 + 0.004))
    zoom_lines, _ = line_optical_depths(atmosphere, zoom_grid, args.recompute, "zoom",
                                        chunk_cm1=1.0e4)
    zoom_cont = continua(atmosphere, zoom_grid)
    zoom_tau = {s: zoom_lines[s] + zoom_cont.get(s, 0.0) for s in SPECIES}
    zoom_mono = np.exp(-AIRMASS * sum(zoom_tau.values()))
    zoom_r45 = smooth(zoom_mono, RESOLVING_POWER, ZOOM_SAMPLES_PER_RESOLUTION)
    zoom_species_mono = {s: np.exp(-AIRMASS * zoom_tau[s]) for s in SPECIES}

    lam = um(grid)
    lam_zoom_nm = 1e3 * um(zoom_grid)

    # ---- numbers --------------------------------------------------------
    def band(mask, curve):
        return float(np.min(curve[mask])), float(np.mean(curve[mask]))

    numbers = {
        "profile": str(PROFILE.relative_to(ROOT)),
        "profile_note": "ERA5 Gemini South 2021-03-17T07Z, 12 weighted layers, 4.33 mm PWV, "
                        "top at 8 hPa; O3 from AFGL midlatitude summer on the same layers",
        "airmass": AIRMASS,
        "grid": {"resolving_power": RESOLVING_POWER, "samples_per_resolution": SAMPLES_PER_RESOLUTION,
                 "points": int(grid.size), "range_um": OVERVIEW_UM,
                 "note": "all panels are full line-by-line on this grid (no coarse shortcut); "
                         "the top panel is the same spectrum smoothed to R=2000"},
        "zoom": {"range_nm": ZOOM_NM, "samples_per_resolution": ZOOM_SAMPLES_PER_RESOLUTION,
                 "points": int(zoom_grid.size)},
        "kernel": "ExoJAXOpacityBackend direct_sparse, pressure_shift=True, AER 3.9, "
                  "+ MT_CKD 4.3 (H2O) + LBLRTM O2 CIA",
        "lines_selected_summed_over_chunks": counts,
        "lines_note": "each chunk selects lines within 25 cm-1 of its edges, so margins are counted twice",
        "o3_column_du": float(np.sum(np.asarray(atmosphere.air_column_cm2)
                                     * np.asarray(atmosphere.vmr["O3"])) / 2.687e16),
        "precipitable_water_mm": float(np.sum(np.asarray(atmosphere.air_column_cm2)
                                              * np.asarray(atmosphere.vmr["H2O"]))
                                       * 18.015 / 6.02214076e23 * 10.0),
    }
    bands = {}
    for name, (lo, hi) in IGRINS.items():
        mask = (lam >= lo) & (lam <= hi)
        bands[f"IGRINS_{name}"] = {
            "range_um": [lo, hi],
            "mean_T_R45k": float(np.mean(r45_total[mask])),
            "fraction_of_R45k_samples_below_0.9": float(np.mean(r45_total[mask] < 0.9)),
            "fraction_of_R45k_samples_below_0.5": float(np.mean(r45_total[mask] < 0.5)),
            "per_species_mean_absorption_R45k": {s: float(1.0 - np.mean(r45[s][mask]))
                                                 for s in SPECIES},
            "per_species_deepest_line_R45k": {s: float(1.0 - np.min(r45[s][mask]))
                                              for s in SPECIES},
        }
    named = {
        "H2O_1.4um (1.34-1.48)": ("H2O", 1.34, 1.48),
        "H2O_1.9um (1.80-1.96)": ("H2O", 1.80, 1.96),
        "CO2_1.57um (1.565-1.585)": ("CO2", 1.565, 1.585),
        "CO2_1.60um (1.595-1.615)": ("CO2", 1.595, 1.615),
        "CO2_2.0um (1.995-2.025)": ("CO2", 1.995, 2.025),
        "CO2_2.06um (2.045-2.075)": ("CO2", 2.045, 2.075),
        "CH4_2.3um (2.20-2.40)": ("CH4", 2.20, 2.40),
        "CH4_1.66um (1.63-1.70)": ("CH4", 1.63, 1.70),
        "CO_2.3um (2.29-2.40)": ("CO", 2.29, 2.40),
        "N2O_2.11um (2.10-2.13)": ("N2O", 2.10, 2.13),
        "O2_1.27um (1.25-1.29)": ("O2", 1.25, 1.29),
    }
    for key, (s, lo, hi) in named.items():
        mask = (lam >= lo) & (lam <= hi)
        bands[key] = {"deepest_R45k_line_depth": float(1.0 - np.min(r45[s][mask])),
                      "mean_absorption_R45k": float(1.0 - np.mean(r45[s][mask])),
                      "mean_absorption_R2000": float(1.0 - np.mean(
                          smooth(mono[s], OVERVIEW_R, SAMPLES_PER_RESOLUTION)[mask]))}
    numbers["bands"] = bands
    zoom_in = (lam_zoom_nm >= ZOOM_NM[0]) & (lam_zoom_nm <= ZOOM_NM[1])
    numbers["zoom"]["min_T_monochromatic"] = float(np.min(zoom_mono[zoom_in]))
    numbers["zoom"]["min_T_R45k"] = float(np.min(zoom_r45[zoom_in]))
    numbers["zoom"]["per_species_min_T_monochromatic"] = {
        s: float(np.min(zoom_species_mono[s][zoom_in])) for s in SPECIES}

    # ---- figure 1: the 0.9-5.3 um overview -------------------------------
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(WIDTH_IN, 2.7))
    ax0 = fig.add_axes([0.075, 0.15, 0.91, 0.82])
    for name, (lo, hi) in WINDOWS.items():
        ax0.axvspan(lo, hi, color="#000000", alpha=0.06, lw=0)
        ax0.text(0.5 * (lo + hi), 0.30, name, ha="center", va="center", fontsize=8,
                 color="#555555", fontweight="bold")
    for name, (lo, hi) in IGRINS.items():
        ax0.plot([lo, hi], [-0.075, -0.075], color="#0072B2", lw=3, solid_capstyle="butt")
        ax0.text(0.5 * (lo + hi), -0.115, f"IGRINS {name}", color="#0072B2", ha="center",
                 va="top", fontsize=6.5)
    ax0.plot(lam, low_total, color="black", lw=0.7)
    # Absorbers above the curve, staggered in two rows, each with a leader line
    # down to the band it names.
    overview_labels = [
        (0.94, "H$_2$O", 0), (1.13, "H$_2$O", 1), (1.27, "O$_2$", 0), (1.38, "H$_2$O", 1),
        (1.60, "CO$_2$", 0), (1.87, "H$_2$O", 1), (2.01, "CO$_2$", 0), (2.33, "CH$_4$", 1),
        (2.70, "H$_2$O + CO$_2$", 0), (3.32, "CH$_4$", 1), (4.27, "CO$_2$", 0),
        (4.50, "N$_2$O", 1), (4.75, "CO, O$_3$", 0), (5.10, "H$_2$O", 1),
    ]
    for x, text, row in overview_labels:
        y = 1.13 + 0.15 * row
        ax0.plot([x, x], [1.03, y - 0.03], color="#888888", lw=0.5)
        ax0.text(x, y, text, ha="center", va="bottom", fontsize=6.5, color="#333333")
    ax0.set_xlim(*OVERVIEW_UM)
    ax0.set_ylim(-0.2, 1.43)
    ax0.set_yticks([0, 0.5, 1])
    ax0.spines["left"].set_bounds(0, 1)
    ax0.set_ylabel("Transmission", y=0.42)
    ax0.set_xlabel(r"Vacuum wavelength ($\mu$m)", labelpad=1)
    ax0.text(4.34, 0.45, f"R = {OVERVIEW_R:,.0f}\nair mass {AIRMASS}",
             fontsize=6.5, ha="center", va="center", color="#333333")
    stem = OUTPUT / "fig_intro_transmission_overview"
    fig.savefig(stem.with_suffix(".pdf"), dpi=400)
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)

    # ---- figure 2: H and K at R = 45,000, species, zoom -------------------
    # O2 and O3 take out under 1% anywhere in 1.4-2.5 um (JSON), so they get
    # no row of their own here; the overview shows where they matter.
    shown = ("H2O", "CO2", "CH4", "N2O", "CO")
    magnified = {"N2O": 0.9, "CO": 0.9}
    fig = plt.figure(figsize=(WIDTH_IN, 6.6))
    rows = len(shown)
    outer = GridSpec(1, 2, figure=fig, width_ratios=[3.0, 1.0], wspace=0.08,
                     left=0.115, right=0.925, top=0.905, bottom=0.075)
    left = outer[0, 0].subgridspec(2 + rows, 1, height_ratios=[1.9, 0.18] + [0.62] * rows,
                                   hspace=0.22)
    main = (lam >= MAIN_UM[0]) & (lam <= MAIN_UM[1])

    ax1 = fig.add_subplot(left[0])
    for name, (lo, hi) in IGRINS.items():
        ax1.axvspan(lo, hi, color="#0072B2", alpha=0.07, lw=0)
        ax1.text(hi - 0.004, 1.06, f"IGRINS {name}", ha="right", va="center",
                 color="#0072B2", fontsize=7)
    ax1.plot(lam[main], r45_total[main], color="black", lw=0.25, rasterized=True)
    main_labels = [(1.40, "H$_2$O\n1.4 $\\mu$m"), (1.60, "CO$_2$\n1.6 $\\mu$m"),
                   (1.87, "H$_2$O\n1.9 $\\mu$m"), (2.035, "CO$_2$\n2.0 $\\mu$m"),
                   (2.34, "CH$_4$ + CO\n2.3 $\\mu$m")]
    for x, text in main_labels:
        ax1.text(x, -0.07, text, ha="center", va="top", fontsize=6.5, color="#333333")
    ax1.set_xlim(*MAIN_UM)
    ax1.set_ylim(-0.36, 1.12)
    ax1.set_yticks([0, 0.5, 1])
    ax1.spines["left"].set_bounds(0, 1.0)
    ax1.set_ylabel("Transmission", y=0.62)
    ax1.tick_params(labelbottom=False, bottom=False)
    ax1.spines["bottom"].set_visible(False)
    ax1.set_title(f"(b) all species together, R = {RESOLVING_POWER:,.0f}, air mass {AIRMASS}",
                  loc="left")
    top = ax1.secondary_xaxis("top", functions=(lambda x: 1e4 / np.maximum(x, 1e-3),
                                                lambda x: 1e4 / np.maximum(x, 1e-3)))
    top.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=7, labelpad=2)
    top.set_xticks([4000, 4500, 5000, 5500, 6000, 6500, 7000])
    top.tick_params(labelsize=6.5)
    ax1.axvspan(ZOOM_NM[0] / 1e3, ZOOM_NM[1] / 1e3, ymin=0.36 / 1.48, color="#E69F00", alpha=0.9, lw=0.8)
    ax1.text(ZOOM_NM[1] / 1e3 + 0.004, 0.93, "(d)", color="#B07800", fontsize=6.5, va="top")

    spacer = fig.add_subplot(left[1])
    spacer.axis("off")
    spacer.text(0.0, 0.0, "(c) each species alone (same R and air mass)", transform=spacer.transAxes,
                fontsize=8, va="bottom")

    axes = []
    for i, s in enumerate(shown):
        ax = fig.add_subplot(left[2 + i], sharex=ax1)
        floor = magnified.get(s, 0.0)
        ax.fill_between(lam[main], r45[s][main], 1.0, color=COLOURS[s], alpha=0.35, lw=0, rasterized=True)
        ax.plot(lam[main], r45[s][main], color=COLOURS[s], lw=0.25, rasterized=True)
        ax.set_ylim(floor - 0.03 * (1 - floor), 1.0 + 0.05 * (1 - floor))
        ax.set_yticks([floor, 1.0])
        ax.set_yticklabels([f"{floor:g}", "1"])
        ax.tick_params(labelsize=6)
        label = LABELS[s] + ("\n+ cont." if s == "H2O" else "")
        ax.set_ylabel(label, rotation=0, color=COLOURS[s], fontsize=7.5, ha="right",
                      va="center", labelpad=4)
        if floor:
            ax.text(0.005, 0.35, f"magnified: axis {floor:g}–1", transform=ax.transAxes,
                    fontsize=6, color="#555555", va="center")
        if i < rows - 1:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
    axes[-1].set_xlabel(r"Vacuum wavelength ($\mu$m)")

    axz = fig.add_subplot(outer[0, 1])
    zx = lam_zoom_nm - ZOOM_NM[0]
    axz.plot(zx, zoom_mono, color="#999999", lw=0.6, label="monochromatic")
    axz.plot(zx, zoom_r45, color="black", lw=1.1, label=f"R = {RESOLVING_POWER:,.0f}")
    for k, s in enumerate(SPECIES):
        depth = 1.0 - zoom_species_mono[s]
        if np.max(depth[zoom_in]) > 0.05:
            peaks = np.flatnonzero((depth[1:-1] > depth[:-2]) & (depth[1:-1] >= depth[2:])
                                   & (depth[1:-1] > 0.05) & zoom_in[1:-1]) + 1
            axz.plot(zx[peaks], np.full(peaks.size, 1.06 - 0.012 * k), "v", ms=3.2, color=COLOURS[s],
                     label=LABELS[s] + " line")
    axz.set_xlim(0.0, ZOOM_NM[1] - ZOOM_NM[0])
    axz.set_ylim(-0.16, 1.08)
    axz.spines["right"].set_bounds(0.0, 1.0)
    axz.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axz.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    axz.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
    axz.set_xlabel(f"$\\lambda$ $-$ {ZOOM_NM[0]:.0f} nm")
    axz.yaxis.tick_right()
    axz.yaxis.set_label_position("right")
    axz.tick_params(axis="y", labelright=True, labelleft=False)
    axz.spines["right"].set_visible(True)
    axz.spines["left"].set_visible(False)
    axz.set_ylabel("Transmission", y=0.58)
    axz.legend(loc="lower left", frameon=False, handlelength=1.4, fontsize=6.3)
    axz.set_title(f"(d) zoom: {ZOOM_NM[1] - ZOOM_NM[0]:.0f} nm", fontsize=8, loc="left", pad=28)

    stem = OUTPUT / "fig_intro_transmission"
    fig.savefig(stem.with_suffix(".pdf"), dpi=400)
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    numbers["runtime_s_this_invocation"] = round(time.time() - started, 1)
    stem.with_suffix(".json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers["bands"]["IGRINS_H"], indent=1))
    print(json.dumps(numbers["bands"]["IGRINS_K"], indent=1))


if __name__ == "__main__":
    main()
