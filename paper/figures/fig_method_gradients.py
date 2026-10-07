#!/usr/bin/env python
"""Paper figure: automatic differentiation against finite differences.

The objective is the one every fit minimizes: ``OrderObjective`` (negative
Gaussian log-likelihood with jitter) on the synthetic page of
``benchmarks/benchmark_fit.py`` -- AER 3.9 H2O + CO2, 5005-5025 cm-1, Kitt
Peak 1994 (12 layers), MT_CKD, sparse Direct with pressure shifts, 1001 pixels,
5.5e-3 noise -- evaluated with the *live* line-by-line kernel in float64
(``mixed_precision=False``) at a point away from the optimum. For the H2O
column scale, velocity and LSF width the reverse-mode gradient is compared with
forward and central differences over 13 decades of step: the familiar V of
truncation error (h, h^2) against round-off (eps |f| / h).

    CUDA_VISIBLE_DEVICES=2 UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_method_gradients.py

Cached in ``paper/figures/cache/method_gradients.npz``; ``--recompute`` redoes it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/figures/output"
CACHE = ROOT / "paper/figures/cache/method_gradients.npz"
LINE_ROOT = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
PROFILE = ROOT / "data/profiles/kitt_peak_1994.csv"
MT_CKD = ROOT / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc"
WINDOW_CM1 = (5005.0, 5025.0)
PIXELS = 1001
STEPS = np.logspace(-12, 0, 49)
PROBED = (("H2O", "H$_2$O column, $\\ln s$"), ("velocity_kms", "velocity (km s$^{-1}$)"),
          ("lsf_sigma_kms", "LSF width $\\sigma$ (km s$^{-1}$)"))

BLUE, VERMILLION, GREY = "#0072B2", "#D55E00", "#666666"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42,
})


def compute() -> dict:
    from tellurix import (AERLineDatabase, ExoJAXOpacityBackend, MTCKDWaterContinuum, OrderObjective,
                          SpectralOrder, TelluricModel, TelluricParameters, constant_velocity_grid,
                          load_atmosphere_csv)

    grid = constant_velocity_grid(1.0e7 / WINDOW_CM1[1], 1.0e7 / WINDOW_CM1[0], resolving_power=100_000.0,
                                  samples_per_resolution=4.0, margin_cm1=25.0)
    profile = load_atmosphere_csv(PROFILE)
    databases = {name: AERLineDatabase(LINE_ROOT / f"{stem}/{stem}", name, (float(grid[0]), float(grid[-1])),
                                       margin_cm1=0.0)
                 for stem, name in (("01_H2O", "H2O"), ("02_CO2", "CO2"))}
    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse",
        temperature_range_k=(float(np.min(profile.temperature_k)), float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        vectorize_layers=True, mixed_precision=True, pressure_shift=True)
    continuum = MTCKDWaterContinuum.from_netcdf(MT_CKD, grid)
    model = TelluricModel(profile, grid, opacity, continuum=continuum, accuracy_mode="mt_ckd",
                          max_lsf_sigma_kms=4.0, pixel_integration="point")

    wavelength = 1.0e7 / np.linspace(WINDOW_CM1[0], WINDOW_CM1[1], PIXELS)[::-1]
    uncertainty = np.full(PIXELS, 0.0055)
    truth = TelluricParameters(log_column_scales={"H2O": 0.35, "CO2": -0.12}, velocity_kms=0.3,
                               wavelength_stretch=0.0, lsf_sigma_kms=1.2,
                               continuum_coeffs=np.array([0.02, -0.01, 0.005, 0.0]), log_jitter=-9.0)
    blank = SpectralOrder(wavelength, np.ones(PIXELS), uncertainty, source_flux=np.ones(PIXELS))
    observed = np.asarray(model.predict(blank, truth)) + np.random.default_rng(7).normal(0.0, 0.0055, PIXELS)
    order = SpectralOrder(wavelength, observed, uncertainty, source_flux=np.ones(PIXELS))
    point = TelluricParameters(log_column_scales={"H2O": 0.25, "CO2": -0.05}, velocity_kms=0.1,
                               wavelength_stretch=0.0, lsf_sigma_kms=1.4,
                               continuum_coeffs=np.array([0.01, 0.0, 0.0, 0.0]), log_jitter=-9.0)
    result = {"steps": STEPS}
    for variant, fit_model in (("precomputed", model.precompute_opacity()), ("live", model)):
        objective = OrderObjective(fit_model, order, 4)
        vector = objective.codec.pack(point)
        started = time.perf_counter()
        value, gradient = objective(vector)
        result[f"{variant}_compile_s"] = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(5):
            objective(vector)
        result[f"{variant}_call_ms"] = 1e3 * (time.perf_counter() - started) / 5
        names = objective.codec.names
        result["names"] = np.array(names)
        result[f"{variant}_value"], result[f"{variant}_gradient"] = value, gradient
        for name, _ in PROBED:
            index = names.index(name)
            plus, minus = [], []
            for h in STEPS:
                v = vector.copy()
                v[index] += h
                plus.append(objective(v)[0])
                v[index] -= 2 * h
                minus.append(objective(v)[0])
            result[f"{variant}_plus_{name}"] = np.array(plus)
            result[f"{variant}_minus_{name}"] = np.array(minus)
            print(variant, name, flush=True)
    np.savez(CACHE, **result)
    return dict(np.load(CACHE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data = compute() if args.recompute or not CACHE.exists() else dict(np.load(CACHE))
    names = list(data["names"])
    steps = data["steps"]
    eps = np.finfo(float).eps

    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.9), constrained_layout=True, sharey=True)
    numbers = {f"{v}_{k}": float(data[f"{v}_{k}"]) for v in ("precomputed", "live")
               for k in ("value", "call_ms", "compile_s")}
    for ax, (name, label) in zip(axes, PROBED):
        index = names.index(name)
        for variant, alpha, marker_face in (("live", 0.45, "none"), ("precomputed", 1.0, None)):
            value = float(data[f"{variant}_value"])
            g = float(data[f"{variant}_gradient"][index])
            forward = (data[f"{variant}_plus_{name}"] - value) / steps
            central = (data[f"{variant}_plus_{name}"] - data[f"{variant}_minus_{name}"]) / (2 * steps)
            err_f = np.maximum(np.abs(forward - g) / abs(g), 1e-17)
            err_c = np.maximum(np.abs(central - g) / abs(g), 1e-17)
            suffix = "" if variant == "precomputed" else ", live mixed-precision kernel"
            ax.loglog(steps, err_f, "o-", color=VERMILLION, ms=2.2, lw=1.0, alpha=alpha, mfc=marker_face,
                      label="forward difference" + suffix)
            ax.loglog(steps, err_c, "s-", color=BLUE, ms=2.2, lw=1.0, alpha=alpha, mfc=marker_face,
                      label="central difference" + suffix)
            best = int(np.argmin(err_c))
            numbers[f"{variant}_{name}"] = {
                "autodiff": g, "best_central_relative_error": float(err_c[best]),
                "best_central_step": float(steps[best]),
                "best_forward_relative_error": float(err_f.min()),
                "best_forward_step": float(steps[int(np.argmin(err_f))])}
            if variant == "precomputed":
                # The round-off floor eps |f| / (h |g|) that every difference shares.
                ax.loglog(steps, eps * abs(value) / (steps * abs(g)), ":", color=GREY, lw=0.9,
                          label="round-off $\\epsilon|f|/(h|g|)$")
                ax.set_title(f"$\\partial f/\\partial$ {label.split(' (')[0].split(',')[0]}"
                             f" = {g:.4g}", loc="left", fontsize=7.5)
        if name == "velocity_kms":
            # A velocity step enters as wavelength * (1 + v/c); the step is
            # itself only resolved to eps * c / h relative, far above eps |f|.
            ax.loglog(steps, eps * 299792.458 / steps, "--", color=GREY, lw=0.9)
            ax.text(1e-3, 2e-11, "$\\epsilon c/h$: step lost\nin $\\lambda(1+v/c)$", fontsize=6.3, color=GREY)
        ax.axhline(eps, color="k", lw=1.2, label="reverse-mode autodiff (reference)")
        ax.set_xlabel(f"step $h$ in {label}")
        ax.set_ylim(1e-17, 10)
        ax.set_xlim(steps[0], steps[-1])
        ax.set_xticks([1e-12, 1e-8, 1e-4, 1])
    axes[0].text(2e-12, 6e-2, "live: P$_\\mathrm{self}$ step below\nfloat32 resolution of $a$", fontsize=6.3,
                 color=BLUE, alpha=0.7)
    axes[0].set_ylabel("|FD $-$ autodiff| / |autodiff|")
    axes[0].text(2e-12, 2.5e-16, "float64 $\\epsilon$ (autodiff reference)", fontsize=6.3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, fontsize=6.5)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT / f"fig_method_gradients.{suffix}", dpi=200)
    (OUTPUT / "fig_method_gradients.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
