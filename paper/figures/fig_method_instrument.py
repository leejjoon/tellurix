#!/usr/bin/env python
"""Method figure: tellurix's instrument model, step by step, on 10 cm-1 of K band.

One real model run: AER 3.9 lines (H2O, CO2, N2O, CO, CH4; ExoJAX direct_sparse
with pressure shifts) plus MT_CKD through the 12-layer weighted DCT column of
``fig_method_layering.py``, at airmass 1.5, over 4290-4300 cm-1 (2.326-2.331 um),
sampled on IGRINS's own pixels: the wavelength solution of K order index 5 of
``data/igrins/20151201_0037/SDCK_20151201_0037.spec.fits`` (3.0 pixels per
R=45,000 element). Every step is ``TelluricModel.predict`` or the instrument
classes it calls; only the truncated-sinc comparison is written here, because
truncation is what tellurix does not do.

The model grid is finer than the pipeline's (16 rather than 4 samples per
resolution element) so that panel (a) shows the monochromatic lines.

    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_method_instrument.py

The line-by-line cross sections are cached in
``paper/figures/cache/instrument_4290.npz`` (``--recompute`` redoes them).
Writes ``fig_method_instrument.{pdf,png,json}``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import tellurix  # noqa: F401  (x64 before exojax)
import jax.numpy as jnp
from tellurix import (
    AERLineDatabase, AtmosphereProfile, DataPaths, ExoJAXOpacityBackend, MTCKDWaterContinuum,
    SpectralOrder, TelluricModel, TelluricParameters, constant_velocity_grid,
)
from tellurix.era5 import build_era5_profile
from tellurix.model import ArrayOpacityBackend, BoxcarFTSInstrumentProfile
from tellurix.site_profile import DEFAULT_EDGES_KM, EPOCH_DRY_VMR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper/figures/cache"
OUTPUT = ROOT / "paper/figures/output"
IGRINS_SPEC = ROOT / "data/igrins/20151201_0037/SDCK_20151201_0037.spec.fits"
ORDER_INDEX = 5

CASE = {"site_altitude_km": 2.360, "station_pressure_hpa": 768.5}
WINDOW_CM1 = (4290.0, 4300.0)
MOLECULE_IDS = {"H2O": 1, "CO2": 2, "N2O": 4, "CO": 5, "CH4": 6}
RESOLVING_POWER = 45_000.0
SAMPLES_PER_RESOLUTION = 16.0
LINE_MARGIN_CM1 = 25.0
ZENITH_DEG = float(np.degrees(np.arccos(1.0 / 1.5)))
C_KMS = 299792.458
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
LSF_SIGMA_KMS = C_KMS / RESOLVING_POWER * FWHM_TO_SIGMA
VELOCITY_KMS = 2.0
STRETCH = 0.0
CONTINUUM = (0.0, 0.06, -0.03)
# LBLRTM SCANFN's SINC: 119.33 half widths = 72 zero crossings from the centre
# (postsub.f90:97-101).
LBLRTM_SINC_ZEROS = 72

WIDTH_IN = 7.1
C_MONO = "#999999"
C_LSF = "black"
C_SIMPSON = "#0072B2"
C_POINT = "#D55E00"
C_CONT = "#009E73"
C_SINC = "#CC79A7"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})


def profile() -> AtmosphereProfile:
    data = np.load(CACHE / "era5_dct_2018.npz")
    column = {k: data[k] for k in ("level_hpa", "temperature_k", "specific_humidity", "height_m")}
    layers = build_era5_profile(column, CASE["station_pressure_hpa"], CASE["site_altitude_km"],
                                EPOCH_DRY_VMR["2020"], DEFAULT_EDGES_KM, layering="weighted")
    edges = np.concatenate([layers["pressure_top_bar"][:1], layers["pressure_bottom_bar"]])
    return AtmosphereProfile(edges, layers["temperature_k"], layers["altitude_km"],
                             {s: layers[s] for s in ("H2O", "CO2", "CH4", "N2O", "CO", "O2")},
                             layers["mean_molecular_weight_g_mol"], layers["gravity_m_s2"],
                             mean_pressure_bar=layers["pressure_bar"])


def pixels() -> tuple[np.ndarray, np.ndarray]:
    from astropy.io import fits
    with fits.open(IGRINS_SPEC) as hdul:
        wavelength_nm = np.asarray(hdul[1].data[ORDER_INDEX], float) * 1000.0
    index = np.arange(wavelength_nm.size)
    order = np.argsort(wavelength_nm)
    wavelength_nm, index = wavelength_nm[order], index[order]
    keep = (wavelength_nm >= 1e7 / WINDOW_CM1[1]) & (wavelength_nm <= 1e7 / WINDOW_CM1[0])
    return wavelength_nm[keep], index[keep]


def cross_sections(atmosphere, grid, recompute: bool) -> dict:
    path = CACHE / "instrument_4290.npz"
    if path.exists() and not recompute:
        data = np.load(path)
        if np.array_equal(data["grid"], grid):
            return {k[3:]: data[k] for k in data.files if k.startswith("xs_")}
    paths = DataPaths.bootstrapped(ROOT)
    v1, v2 = WINDOW_CM1
    databases = {}
    for species, molecule in MOLECULE_IDS.items():
        databases[species] = AERLineDatabase(paths.line_file(species, molecule), species, (v1, v2),
                                             margin_cm1=LINE_MARGIN_CM1)
    backend = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse", temperature_range_k=(150.0, 320.0),
        maximum_pressure_bar=1.0, vectorize_layers=True, pressure_shift=True)
    model = TelluricModel(atmosphere, grid, backend)
    # precompute_opacity jit-compiles the kernel once; an eager call would
    # dispatch it op by op (CLAUDE.md, "What a fit costs").
    frozen = model.precompute_opacity(self_broadening="frozen")
    values = {s: np.asarray(v) for s, v in frozen.opacity.values.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, grid=grid, **{f"xs_{s}": v for s, v in values.items()})
    return values


def truncated_sinc_convolve(spectrum, step_kms, first_zero_kms, zeros, normalise=True):
    """Direct convolution with sinc(pi v / v0) cut at +-``zeros`` first-zero spacings."""

    half = int(np.floor(zeros * first_zero_kms / step_kms))
    v = np.arange(-half, half + 1) * step_kms
    kernel = np.sinc(v / first_zero_kms) * step_kms / first_zero_kms
    if normalise:
        kernel = kernel / kernel.sum()
    padded = np.pad(spectrum, (half, half), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    atmosphere = profile()
    wavelength_nm, pixel_index = pixels()
    grid = constant_velocity_grid(1e7 / WINDOW_CM1[1], 1e7 / WINDOW_CM1[0],
                                  resolving_power=RESOLVING_POWER,
                                  samples_per_resolution=SAMPLES_PER_RESOLUTION,
                                  margin_cm1=LINE_MARGIN_CM1)
    values = cross_sections(atmosphere, grid, args.recompute)
    from tellurix import DataPaths as _DataPaths
    continuum = MTCKDWaterContinuum.from_netcdf(_DataPaths.bootstrapped(ROOT).mt_ckd, grid)
    model = TelluricModel(atmosphere, grid, ArrayOpacityBackend(values), continuum=continuum,
                          accuracy_mode="mt_ckd", max_lsf_sigma_kms=10.0)
    parameters = TelluricParameters(
        log_column_scales={}, velocity_kms=VELOCITY_KMS, wavelength_stretch=STRETCH,
        lsf_sigma_kms=LSF_SIGMA_KMS, continuum_coeffs=jnp.asarray(CONTINUUM), log_jitter=0.0)
    unshifted = parameters._replace(velocity_kms=0.0, continuum_coeffs=jnp.zeros(1))
    order = SpectralOrder(wavelength_nm, np.ones_like(wavelength_nm), np.ones_like(wavelength_nm),
                          zenith_angle_deg=ZENITH_DEG)

    nu = np.asarray(grid)
    mono = np.asarray(model.transmission(parameters, ZENITH_DEG))
    lsf = np.asarray(model._convolve_lsf(jnp.asarray(mono), jnp.asarray(LSF_SIGMA_KMS)))
    point_model = copy.copy(model)
    point_model.pixel_integration = "point"
    simpson = np.asarray(model.predict(order, unshifted))
    point = np.asarray(point_model.predict(order, unshifted))
    final = np.asarray(model.predict(order, parameters))
    x = np.linspace(-1.0, 1.0, wavelength_nm.size)
    continuum_curve = np.exp(np.polynomial.chebyshev.chebval(x, CONTINUUM))

    # The FTS sinc at the same FWHM as the Gaussian, applied the way tellurix
    # does it (boxcar transfer function by FFT) and by truncated kernels.
    step = model.velocity_step_kms
    fwhm_kms = C_KMS / RESOLVING_POWER
    centre_cm1 = 0.5 * sum(WINDOW_CM1)
    mopd_cm = 1.20671 / (2.0 * fwhm_kms / C_KMS * centre_cm1)
    fts = BoxcarFTSInstrumentProfile(mopd_cm, centre_cm1, max_residual_sigma_kms=0.0)
    exact = np.asarray(fts.convolve(jnp.asarray(mono), parameters, step))
    v0 = fts.first_zero_kms
    inside = (nu >= WINDOW_CM1[0]) & (nu <= WINDOW_CM1[1])
    zeros_list = np.unique(np.round(np.geomspace(2, 200, 24))).astype(int)
    truncation = {"normalised": [], "raw": []}
    for zeros in zeros_list:
        for key, norm in (("normalised", True), ("raw", False)):
            approx = truncated_sinc_convolve(mono, step, v0, zeros, norm)
            truncation[key].append(float(np.max(np.abs(approx - exact)[inside])))
    lbl_cut = {key: float(np.max(np.abs(truncated_sinc_convolve(mono, step, v0, LBLRTM_SINC_ZEROS, n)
                                         - exact)[inside]))
               for key, n in (("normalised", True), ("raw", False))}

    simpson_minus_point = simpson - point
    numbers = {
        "window_cm1": WINDOW_CM1, "pixels": [int(pixel_index.min()), int(pixel_index.max())],
        "igrins_file": str(IGRINS_SPEC.relative_to(ROOT)), "order_index": ORDER_INDEX,
        "pixels_per_resolution_element": float(np.median(
            wavelength_nm[1:] / np.diff(wavelength_nm)) / RESOLVING_POWER),
        "model_velocity_step_kms": step, "lsf_sigma_kms": LSF_SIGMA_KMS,
        "airmass": 1.5, "velocity_kms": VELOCITY_KMS, "continuum_coeffs": CONTINUUM,
        "simpson_minus_point": {"max_abs": float(np.max(np.abs(simpson_minus_point))),
                                "rms": float(np.sqrt(np.mean(simpson_minus_point ** 2)))},
        "sinc_first_zero_kms": v0, "sinc_mopd_cm": mopd_cm,
        "truncated_sinc_max_error": {"zeros": zeros_list.tolist(), **truncation},
        "lblrtm_72_zero_truncation_max_error": lbl_cut,
        "min_transmission": float(mono[inside].min()),
    }
    print(json.dumps(numbers, indent=1))

    # ------------------------------------------------------------------ plot
    figure = plt.figure(figsize=(WIDTH_IN, 5.6))
    outer = figure.add_gridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.30)
    left = outer[0].subgridspec(5, 1, height_ratios=[1, 1, 1, 0.55, 1], hspace=0.38)
    right = outer[1].subgridspec(2, 1, hspace=0.45)
    axes = [figure.add_subplot(left[i]) for i in range(5)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    ax_mono, ax_lsf, ax_pix, ax_diff, ax_final = axes
    ax_kernel = figure.add_subplot(right[0])
    ax_trunc = figure.add_subplot(right[1])

    wl_hi = 1e7 / nu
    lo, hi = 1e7 / WINDOW_CM1[1], 1e7 / WINDOW_CM1[0]
    half_pixel = 0.5 * np.gradient(wavelength_nm)

    ax_mono.plot(wl_hi, mono, color=C_MONO, lw=0.6)
    ax_mono.set_title("(a) Monochromatic transmission on the constant-velocity grid "
                      f"({step:.2f} km s$^{{-1}}$ step), airmass 1.5", loc="left", fontsize=6.8, pad=2)

    ax_lsf.plot(wl_hi, mono, color=C_MONO, lw=0.4, alpha=0.6)
    ax_lsf.plot(wl_hi, lsf, color=C_LSF, lw=0.9)
    ax_lsf.set_title(f"(b) Gaussian LSF by direct convolution, $\\sigma$ = {LSF_SIGMA_KMS:.2f} "
                     "km s$^{-1}$ (R = 45,000)", loc="left", fontsize=6.8, pad=2)

    ax_pix.plot(wl_hi, lsf, color=C_LSF, lw=0.6, alpha=0.5)
    ax_pix.errorbar(wavelength_nm, simpson, xerr=half_pixel, fmt="o", ms=2.0, lw=0.6,
                    color=C_SIMPSON, label="Simpson pixel average (default)")
    ax_pix.plot(wavelength_nm, point, "x", ms=3.0, mew=0.7, color=C_POINT,
                label='point sample ("point", FTS)')
    ax_pix.legend(loc="lower left", bbox_to_anchor=(0.05, 0.0), frameon=False, fontsize=6,
                  handlelength=1.2, borderaxespad=0.1)
    ax_pix.set_title("(c) Sampled at the IGRINS pixels (bars: pixel extent)", loc="left",
                     fontsize=6.8, pad=2)

    ax_diff.axhline(0.0, color="#bbbbbb", lw=0.5)
    ax_diff.plot(wavelength_nm, 1e3 * simpson_minus_point, "-", color=C_SIMPSON, lw=0.7,
                 marker=".", ms=2)
    ax_diff.set_ylabel("Simpson $-$\npoint ($10^{-3}$)", labelpad=1, fontsize=6.5)

    ax_final.plot(wavelength_nm, final, "o-", ms=1.8, lw=0.6, color=C_SIMPSON,
                  label=f"predict(), $v$ = {VELOCITY_KMS:+.0f} km s$^{{-1}}$")
    ax_final.plot(wavelength_nm, simpson, "-", lw=0.5, color="#bbbbbb", label="(c), unshifted")
    ax_final.plot(wavelength_nm, continuum_curve, "--", lw=0.9, color=C_CONT,
                  label=r"$\exp(\sum_k c_k T_k(x))$")
    ax_final.legend(loc="lower left", bbox_to_anchor=(0.05, 0.0), frameon=False, fontsize=6,
                    handlelength=1.6, borderaxespad=0.1)
    ax_final.set_title("(d) Pixel model shifted in velocity, times the log-Chebyshev continuum",
                       loc="left", fontsize=6.8, pad=2)
    ax_final.set_xlabel("Vacuum wavelength (nm)")
    ax_final.set_xlim(lo, hi)
    for ax in (ax_mono, ax_lsf, ax_pix, ax_final):
        ax.set_ylim(-0.12, 1.12)
        ax.set_ylabel("Transmission" if ax is not ax_final else "Flux", labelpad=1)
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    # Pixel index on top of the first panel (IGRINS order, detector columns).
    slope, intercept = np.polyfit(wavelength_nm, pixel_index, 1)
    top = ax_mono.secondary_xaxis("top", functions=(lambda w: slope * w + intercept,
                                                    lambda p: (p - intercept) / slope))
    top.set_xlabel(f"IGRINS K pixel index (order {ORDER_INDEX})", labelpad=2)
    top.tick_params(labelsize=6.5)

    # Kernels in velocity space.
    v = np.linspace(-40.0, 40.0, 4001)
    gauss = np.exp(-0.5 * (v / LSF_SIGMA_KMS) ** 2)
    sinc = np.sinc(v / v0)
    ax_kernel.axhline(0.0, color="#bbbbbb", lw=0.5)
    ax_kernel.plot(v, gauss, color=C_LSF, lw=1.0, label="Gaussian (IGRINS)")
    ax_kernel.plot(v, sinc, color=C_SINC, lw=1.0, label="unapodized FTS sinc")
    wing = np.abs(v) > v0
    ax_kernel.plot(np.where(wing, v, np.nan), np.abs(1.0 / (np.pi * v / v0)), color=C_SINC, lw=0.5,
                   ls=(0, (2, 2)), label=r"sinc envelope $1/(\pi v/v_0)$")
    ax_kernel.set_xlim(-40, 40)
    ax_kernel.set_ylim(-0.3, 1.08)
    ax_kernel.set_xlabel(r"Velocity offset (km s$^{-1}$)")
    ax_kernel.set_ylabel("Kernel (peak = 1)", labelpad=1)
    ax_kernel.legend(loc="upper left", frameon=False, fontsize=5.6, handlelength=1.5,
                     borderaxespad=0.0)
    ax_kernel.set_title(f"(e) Kernels at equal FWHM, {fwhm_kms:.2f} km s$^{{-1}}$", loc="left",
                        fontsize=7.5)

    ax_trunc.loglog(zeros_list, truncation["normalised"], "o-", ms=2.5, lw=0.8, color=C_SINC,
                    label="kernel renormalized")
    ax_trunc.loglog(zeros_list, truncation["raw"], "s--", ms=2.3, lw=0.7, color="#555555",
                    mfc="none", label="kernel as cut")
    ax_trunc.axvline(LBLRTM_SINC_ZEROS, color="#999999", lw=0.6, ls=":")
    ax_trunc.text(LBLRTM_SINC_ZEROS * 1.08, 0.5, "LBLRTM SCANFN\nSINC extent\n(72 zeros)",
                  transform=ax_trunc.get_xaxis_transform(), fontsize=5.6, va="center")
    ax_trunc.set_xlabel("Kernel half-width (sinc zeros)")
    ax_trunc.set_ylabel(r"max $|\Delta T|$ vs. FFT", labelpad=1)
    ax_trunc.legend(loc="lower left", frameon=False, fontsize=5.8, handlelength=1.6,
                    borderaxespad=0.1)
    ax_trunc.set_title("(f) Truncated sinc kernel vs. FFT", loc="left", fontsize=7.5)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT / f"fig_method_instrument.{suffix}", dpi=200, bbox_inches="tight")
    (OUTPUT / "fig_method_instrument.json").write_text(json.dumps(numbers, indent=1))
    print("wrote", OUTPUT / "fig_method_instrument.png")


if __name__ == "__main__":
    main()
