#!/usr/bin/env python
"""Paper figure: precomputed opacity, first order in the self-broadening pressure.

A column scale s_k enters the optical depth as

    tau(nu) = sum_layers sigma_k(nu; T, P, P_self,k = s_k q_k P) * s_k N_k,

so it is linear except through the self-broadening partial pressure inside
sigma (gamma_L = gamma_air (P - P_self) + gamma_self P_self, scaled in T).
``TelluricModel.precompute_opacity`` evaluates sigma at the reference P_self
and its derivative by a central difference (+-25 % of P_self), and
``LinearizedOpacityBackend`` returns sigma_0 + dsigma/dP_self * (P_self -
P_self,0). ``self_broadening="frozen"`` drops the derivative.

Workload (as ``benchmarks/benchmark_fit.py``): AER 3.9 H2O + CO2, 5005-5025
cm-1, R = 1e5 at 4 samples per resolution element with a 25 cm-1 margin,
Kitt Peak 1994 (12 layers), MT_CKD continuum, sparse Direct with mixed
precision and pressure shifts. Panels:

  (a) max |T_approx - T_exact| over the window against the H2O scale, for the
      linearized and frozen opacity, against 5.5e-3 photon noise;
  (b) the strongest H2O line's cross section in the bottom layer at scales
      0.5, 1 and 2.72: exact, linearized and frozen;
  (c) the column-integrated sensitivity d tau / d ln s_H2O split into the
      linear term (tau itself) and the self-broadening term.

    CUDA_VISIBLE_DEVICES=2 UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_method_precompute.py

Cached in ``paper/figures/cache/method_precompute.npz``; ``--recompute`` redoes it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/figures/output"
CACHE = ROOT / "paper/figures/cache/method_precompute.npz"
LINE_ROOT = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
PROFILE = ROOT / "data/profiles/kitt_peak_1994.csv"
MT_CKD = ROOT / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc"
WINDOW_CM1 = (5005.0, 5025.0)
NOISE = 5.5e-3
LOG_SCALES = np.linspace(np.log(0.5), 1.0, 29)
LINE_SCALES = (0.5, 1.0, np.e)

BLUE, VERMILLION, GREY, ORANGE, GREEN, SKY = ("#0072B2", "#D55E00", "#666666", "#E69F00",
                                              "#009E73", "#56B4E9")
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42,
})


def compute() -> dict:
    import jax
    import jax.numpy as jnp

    from tellurix import (AERLineDatabase, ExoJAXOpacityBackend, MTCKDWaterContinuum, TelluricModel,
                          TelluricParameters, constant_velocity_grid, load_atmosphere_csv)

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
    linear = model.precompute_opacity(self_broadening="linear")
    frozen = model.precompute_opacity(self_broadening="frozen")

    def parameters(log_h2o):
        return TelluricParameters(log_column_scales={"H2O": log_h2o, "CO2": 0.0}, velocity_kms=0.0,
                                  wavelength_stretch=0.0, lsf_sigma_kms=1.0,
                                  continuum_coeffs=np.zeros(4), log_jitter=-9.0)

    curves = {}
    for label, m in (("exact", model), ("linear", linear), ("frozen", frozen)):
        evaluate = jax.jit(lambda s, m=m: m.transmission(parameters(s)))
        curves[label] = np.stack([np.asarray(evaluate(jnp.asarray(s))) for s in LOG_SCALES])
        print(label, flush=True)

    # (b) one strong H2O line, bottom layer
    nu = np.asarray(grid)
    window = (nu >= WINDOW_CM1[0]) & (nu <= WINDOW_CM1[1])
    db = databases["H2O"]
    centres = np.asarray(db.nu_lines)
    inside = (centres > WINDOW_CM1[0] + 1) & (centres < WINDOW_CM1[1] - 1)
    strongest = np.flatnonzero(inside)[np.argmax(np.asarray(db.logsij0)[inside])]
    centre = float(centres[strongest])
    calc = opacity.calculators["H2O"]
    bottom = -1
    t = float(profile.temperature_k[bottom])
    p = float(profile.pressure_layer_bar[bottom])
    q = float(profile.vmr["H2O"][bottom])
    xsv = jax.jit(lambda pself: calc.xsvector(t, p, pself))
    near = np.abs(nu - centre) < 0.3
    line = {"nu": nu[near], "centre": centre, "T": t, "P": p, "q": q}
    lin_backend = linear.opacity
    for s in LINE_SCALES:
        line[f"exact_{s:.3f}"] = np.asarray(xsv(p * q * s))[near]
        line[f"linear_{s:.3f}"] = np.maximum(
            lin_backend.values["H2O"][bottom] + lin_backend.derivative["H2O"][bottom]
            * (p * q * s - lin_backend.reference_partial_pressure_bar["H2O"][bottom]), 0.0)[near]
        line[f"frozen_{s:.3f}"] = lin_backend.values["H2O"][bottom][near]

    # (c) d tau / d ln s, column integrated, at s = 1
    air = np.asarray(profile.air_column_cm2)
    vmr = np.asarray(profile.vmr["H2O"])
    pself = np.asarray(profile.pressure_layer_bar) * vmr
    linear_term = np.sum(lin_backend.values["H2O"] * (air * vmr)[:, None], axis=0)
    self_term = np.sum(lin_backend.derivative["H2O"] * (pself * air * vmr)[:, None], axis=0)

    result = {"log_scales": LOG_SCALES, "window": window, "nu": nu, "linear_term": linear_term,
              "self_term": self_term, "lines_h2o": centres.size, "lines_co2": np.asarray(
                  databases["CO2"].nu_lines).size,
              **{f"T_{k}": v for k, v in curves.items()},
              **{f"line_{k}": np.asarray(v) for k, v in line.items()}}
    np.savez(CACHE, **result)
    return dict(np.load(CACHE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data = compute() if args.recompute or not CACHE.exists() else dict(np.load(CACHE))

    window = data["window"]
    scales = np.exp(data["log_scales"])
    errors = {}
    for label in ("linear", "frozen"):
        diff = np.abs(data[f"T_{label}"] - data["T_exact"])[:, window]
        errors[label] = {"max": diff.max(axis=1), "rms": np.sqrt(np.mean(diff ** 2, axis=1))}
    numbers = {
        "scale_range": [float(scales[0]), float(scales[-1])],
        "linear_max": float(errors["linear"]["max"].max()),
        "linear_rms_max_over_scales": float(errors["linear"]["rms"].max()),
        "linear_rms_over_all": float(np.sqrt(np.mean(np.abs(data["T_linear"] - data["T_exact"])[:, window] ** 2))),
        "frozen_max": float(errors["frozen"]["max"].max()),
        "frozen_over_linear_max": float(errors["frozen"]["max"].max() / errors["linear"]["max"].max()),
        "linear_max_full_grid": float(np.abs(data["T_linear"] - data["T_exact"]).max()),
        "frozen_max_full_grid": float(np.abs(data["T_frozen"] - data["T_exact"]).max()),
        "at_reference_linear_max": float(errors["linear"]["max"][np.argmin(np.abs(data["log_scales"]))]),
        "lines": [int(data["lines_h2o"]), int(data["lines_co2"])],
        "line_centre_cm1": float(data["line_centre"]),
    }
    ratio = data["self_term"][window] / data["linear_term"][window]
    numbers["self_over_linear_sensitivity"] = {"min": float(ratio.min()), "max": float(ratio.max()),
                                               "median_abs": float(np.median(np.abs(ratio)))}

    figure = plt.figure(figsize=(7.1, 4.6), constrained_layout=True)
    left, right = figure.subfigures(1, 2, width_ratios=(1.0, 1.05))
    ax_err, ax_sens = left.subplots(2, 1, height_ratios=(1.3, 1.0))
    ax_line, ax_res = right.subplots(2, 1, sharex=True, height_ratios=(1.6, 1.0))

    # (a)
    ax_err.semilogy(scales, errors["frozen"]["max"], color=VERMILLION, lw=1.4, label="frozen, max")
    ax_err.semilogy(scales, errors["frozen"]["rms"], color=VERMILLION, lw=1.0, ls="--", label="frozen, rms")
    ax_err.semilogy(scales, errors["linear"]["max"], color=BLUE, lw=1.4, label="linearized, max")
    ax_err.semilogy(scales, errors["linear"]["rms"], color=BLUE, lw=1.0, ls="--", label="linearized, rms")
    ax_err.axhline(NOISE, color=GREY, lw=0.9, ls=":")
    ax_err.text(0.52, NOISE * 1.25, "photon noise 5.5e-3", color=GREY, fontsize=6.5)
    ax_err.axvline(1.0, color="k", lw=0.5)
    ax_err.set_xscale("log")
    ax_err.set_xticks([0.5, 1, 2, 2.7])
    ax_err.set_xticklabels(["0.5", "1", "2", "2.7"])
    ax_err.minorticks_off()
    ax_err.set_ylim(1e-10, 1e-1)
    ax_err.set_xlabel("H$_2$O column scale $s$ (reference = 1)")
    ax_err.set_ylabel("$|T_\\mathrm{approx}-T_\\mathrm{exact}|$")
    ax_err.set_title("(a) transmission error, 5005–5025 cm$^{-1}$", loc="left")
    ax_err.legend(loc="lower right", frameon=False, ncol=2, fontsize=6.3)

    # (c) sensitivity split: the self-broadening share of d tau / d ln s
    nu = data["nu"][window]
    share = 100 * data["self_term"][window] / data["linear_term"][window]
    ax_sens.plot(nu, share, color=GREEN, lw=0.7)
    ax_sens.axhline(0, color="k", lw=0.5)
    ax_sens.set_xlim(WINDOW_CM1)
    ax_sens.set_xticks(np.arange(5005, 5026, 5))
    ax_sens.set_xlabel("wavenumber (cm$^{-1}$)")
    ax_sens.set_ylabel("self-broadening share (%)")
    ax_sens.set_title("(c) $\\partial\\tau/\\partial\\ln s$: non-linear part / linear part", loc="left")
    twin = ax_sens.twinx()
    twin.plot(nu, np.exp(-data["linear_term"][window]), color=GREY, lw=0.5, alpha=0.6)
    twin.set_ylim(-1.6, 1.05)
    twin.set_yticks([0, 1])
    twin.set_ylabel("H$_2$O $T$", color=GREY, fontsize=7)
    twin.tick_params(axis="y", colors=GREY, labelsize=6.5)
    twin.spines["right"].set_visible(True)
    twin.grid(False)

    # (b) one line
    centre = float(data["line_centre"])
    dnu = data["line_nu"] - centre
    colors = {0.5: SKY, 1.0: GREY, np.e: VERMILLION}
    for s in LINE_SCALES:
        key = f"{s:.3f}"
        exact = data[f"line_exact_{key}"]
        ax_line.plot(dnu, exact / 1e-22, color=colors[s], lw=1.8, alpha=0.5,
                     label=f"exact, $s={s:.2g}$")
        ax_line.plot(dnu, data[f"line_linear_{key}"] / 1e-22, color=colors[s], lw=0.8, ls="--")
        if s != 1.0:
            peak = exact.max()
            ax_res.plot(dnu, (data[f"line_linear_{key}"] - exact) / peak, color=colors[s], lw=1.0, ls="--")
            ax_res.plot(dnu, (data[f"line_frozen_{key}"] - exact) / peak, color=colors[s], lw=1.0, ls=":")
    from matplotlib.lines import Line2D
    handles, labels = ax_line.get_legend_handles_labels()
    handles += [Line2D([], [], color="k", lw=0.8, ls="--"), Line2D([], [], color="k", lw=0.8, ls=":")]
    labels += ["linearized", "frozen (= $s{=}1$)"]
    ax_line.legend(handles, labels, loc="upper right", frameon=False, fontsize=6.3)
    ax_line.set_ylabel("$\\sigma$ ($10^{-22}$ cm$^2$)")
    ax_line.set_title("(b) one H$_2$O line, bottom layer", loc="left")
    ax_line.text(0.02, 0.97, f"$\\nu_0={centre:.3f}$ cm$^{{-1}}$\n$P={float(data['line_P']):.2f}$ bar, "
                 f"$T={float(data['line_T']):.0f}$ K\n$q_\\mathrm{{H_2O}}={100 * float(data['line_q']):.2f}$%",
                 transform=ax_line.transAxes, va="top", fontsize=6.3)
    ax_res.set_ylabel("(approx $-$ exact) / peak")
    ax_res.set_xlabel("$\\nu-\\nu_0$ (cm$^{-1}$)")
    ax_res.set_xlim(dnu.min(), dnu.max())
    ax_res.axhline(0, color="k", lw=0.5)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT / f"fig_method_precompute.{suffix}", dpi=200)
    (OUTPUT / "fig_method_precompute.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
