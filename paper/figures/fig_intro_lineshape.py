#!/usr/bin/env python
"""Introductory figure: how one absorption line gets its shape.

One real AER 3.9 H2O line, 5004.585 cm-1 (2.0 um, E'' = 649 cm-1 -- the line
``fig_method_lineshape.py`` follows through LBLRTM), is placed in each layer of
the ERA5 column over Gemini South at 2021-03-17 07 UT
(``data/profiles/gemini_2021_era5.csv``), at the layer's own pressure,
temperature and water partial pressure. Everything is plain numpy/scipy, but
the widths and strengths are the formulas tellurix evaluates in
``SparseCoreDirect._line_parameters`` (``packages/tellurix/src/tellurix/direct.py``),
which calls ExoJAX's ``doppler_sigma``, ``gamma_hitran`` and ``gamma_natural``
(``exojax/database/core/broadening.py``) and ``line_strength``
(``exojax/database/core/line_strength.py``), with the TIPS-2017 partition
function that ``AERLineDatabase.qr_interp`` (``aer.py``) reads from HAPI:

* Doppler: sigma_D = 3.0415595e-7 sqrt(T / M) nu0 (cm-1, Gaussian standard
  deviation, M in g/mol), HWHM alpha_D = sqrt(2 ln 2) sigma_D.
* Lorentz: alpha_L = (296 / T)^n_air [gamma_air (P - P_self) + gamma_self P_self] / (1 atm)
  + 2.6544e-12 A (natural, negligible), with P the layer pressure tellurix uses
  for lines (``AtmosphereProfile.pressure_layer_bar``).
* Voigt: scipy.special.voigt_profile(x, sigma_D, alpha_L); its HWHM is found by
  root finding (no approximation).
* Strength: S(T) = S(296) Q(296)/Q(T) exp[-c2 E'' (1/T - 1/296)]
  [1 - exp(-c2 nu0 / T)] / [1 - exp(-c2 nu0 / 296)], c2 = hc/k = 1.43878 cm K.

    JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_intro_lineshape.py

Writes ``fig_intro_lineshape.{pdf,png,json}`` to ``paper/figures/output/``.
"""

from __future__ import annotations

import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import voigt_profile

import tellurix  # noqa: F401  (x64 before exojax)
from tellurix import AERLineDatabase, DataPaths
from tellurix.io import load_atmosphere_csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/figures/output"
PROFILE = ROOT / "data/profiles/gemini_2021_era5.csv"

# The line of fig_method_lineshape.py, and two others of the same band chosen
# for a low and a high lower-state energy.
MAIN_CM1 = 5004.5850
CONTRAST_CM1 = {"low": 5144.7694, "high": 5065.2531}
# Layers shown in (b), by nominal altitude: near the ground, mid troposphere,
# where Doppler and pressure widths are equal, and the top layer.
SHOWN_KM = (2.9, 4.9, 19.7, 27.7)
RESOLVING_POWER = 45_000.0

# ExoJAX constants (exojax/utils/constants.py, broadening.py).
C3 = 3.0415595e-07
P_ATM_BAR = 1.01325
T_REF_K = 296.0
C2_CM_K = 1.4387773538277202
NATURAL = 2.6544188e-12
SQRT2LN2 = np.sqrt(2.0 * np.log(2.0))

WIDTH_IN = 7.1
C_DOPPLER = "#0072B2"
C_LORENTZ = "#D55E00"
C_VOIGT = "black"
C_RES = "#999999"
C_LINES = {"low": "#009E73", "main": "black", "high": "#CC79A7"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})


def lines() -> dict:
    database = AERLineDatabase(DataPaths.bootstrapped(ROOT).line_file("H2O", 1), "H2O",
                               (4990.0, 5160.0), margin_cm1=0.0)
    out = {}
    for key, nu in {"main": MAIN_CM1, **CONTRAST_CM1}.items():
        i = int(np.argmin(np.abs(database.nu_lines - nu)))
        if abs(database.nu_lines[i] - nu) > 1e-3:
            raise RuntimeError(f"line {nu} not found")
        out[key] = {
            "nu_cm1": float(database.nu_lines[i]), "isotope": int(database.isoid[i]),
            "S296_cm_per_molecule": float(database.line_strength_ref_original[i]),
            "A_s-1": float(database.A[i]), "gamma_air_cm1_atm": float(database.gamma_air[i]),
            "gamma_self_cm1_atm": float(database.gamma_self[i]), "n_air": float(database.n_air[i]),
            "elower_cm1": float(database.elower[i]), "delta_air_cm1_atm": float(database.delta_air[i]),
        }
    out["molar_mass_g_mol"] = float(database.molmass)
    return out


def partition(isotope: int):
    with redirect_stdout(open(os.devnull, "w")):
        import hapi
    t = np.asarray(hapi.TIPS_2017_ISOT_HASH[(1, isotope)], float)
    q = np.asarray(hapi.TIPS_2017_ISOQ_HASH[(1, isotope)], float)
    return lambda temperature: np.interp(temperature, t, q)


def doppler_sigma(nu, temperature, mass):
    return C3 * np.sqrt(temperature / mass) * nu


def lorentz_hwhm(line, pressure_bar, temperature, self_bar):
    return ((T_REF_K / temperature) ** line["n_air"]
            * (line["gamma_air_cm1_atm"] * (pressure_bar - self_bar) / P_ATM_BAR
               + line["gamma_self_cm1_atm"] * self_bar / P_ATM_BAR)
            + NATURAL * line["A_s-1"])


def voigt_hwhm(sigma, gamma):
    peak = voigt_profile(0.0, sigma, gamma)
    upper = 2.0 * (SQRT2LN2 * sigma + gamma)
    return brentq(lambda x: voigt_profile(x, sigma, gamma) - 0.5 * peak, 0.0, upper, xtol=1e-12)


def strength_ratio(line, temperature, q):
    nu, e = line["nu_cm1"], line["elower_cm1"]
    stimulated = (1.0 - np.exp(-C2_CM_K * nu / temperature)) / (1.0 - np.exp(-C2_CM_K * nu / T_REF_K))
    return (q(T_REF_K) / q(temperature) * np.exp(-C2_CM_K * e * (1.0 / temperature - 1.0 / T_REF_K))
            * stimulated)


def main() -> None:
    atmosphere = load_atmosphere_csv(PROFILE)
    data = lines()
    line, mass = data["main"], data["molar_mass_g_mol"]
    p_layer = np.asarray(atmosphere.pressure_layer_bar)
    t_layer = np.asarray(atmosphere.temperature_k)
    z_layer = np.asarray(atmosphere.altitude_km)
    w_layer = np.asarray(atmosphere.vmr["H2O"])

    sigma_l = doppler_sigma(line["nu_cm1"], t_layer, mass)
    alpha_d_l = SQRT2LN2 * sigma_l
    alpha_l_l = lorentz_hwhm(line, p_layer, t_layer, p_layer * w_layer)
    alpha_v_l = np.array([voigt_hwhm(s, g) for s, g in zip(sigma_l, alpha_l_l)])

    # A continuous curve through the layers: T, log H2O and altitude
    # interpolated in log pressure between the layer values.
    edges = np.asarray(atmosphere.pressure_edges_bar)
    p_fine = np.geomspace(edges[0], edges[-1], 400)
    t_fine = np.interp(np.log(p_fine), np.log(p_layer), t_layer)
    w_fine = np.exp(np.interp(np.log(p_fine), np.log(p_layer), np.log(w_layer)))
    sigma_f = doppler_sigma(line["nu_cm1"], t_fine, mass)
    alpha_d_f = SQRT2LN2 * sigma_f
    alpha_l_f = lorentz_hwhm(line, p_fine, t_fine, p_fine * w_fine)
    alpha_v_f = np.array([voigt_hwhm(s, g) for s, g in zip(sigma_f, alpha_l_f)])
    log_ratio = np.log(alpha_l_f / alpha_d_f)
    k = int(np.flatnonzero(np.diff(np.sign(log_ratio)) != 0)[0])
    frac = log_ratio[k] / (log_ratio[k] - log_ratio[k + 1])
    p_cross = float(np.exp(np.log(p_fine[k]) + frac * np.log(p_fine[k + 1] / p_fine[k])))
    z_cross = float(np.interp(np.log(p_cross), np.log(p_layer), z_layer))
    width_cross = float(np.exp(np.interp(np.log(p_cross), np.log(p_fine), np.log(alpha_d_f))))
    resolution_hwhm = 0.5 * line["nu_cm1"] / RESOLVING_POWER

    # ---- figure ----------------------------------------------------------
    fig = plt.figure(figsize=(WIDTH_IN, 5.0))
    top_row = GridSpec(1, 2, figure=fig, wspace=0.42, left=0.075, right=0.985, top=0.955,
                       bottom=0.585)
    gs = GridSpec(1, 4, figure=fig, wspace=0.10, left=0.075, right=0.985, top=0.40, bottom=0.085)

    # (a) widths against pressure
    ax = fig.add_subplot(top_row[0, 0])
    ax.axhspan(p_cross * 1e3, edges[-1] * 1e3, color=C_LORENTZ, alpha=0.07, lw=0)
    ax.axhspan(edges[0] * 1e3, p_cross * 1e3, color=C_DOPPLER, alpha=0.07, lw=0)
    ax.plot(alpha_d_f, p_fine * 1e3, color=C_DOPPLER, lw=1.3, label=r"Doppler $\alpha_D$")
    ax.plot(alpha_l_f, p_fine * 1e3, color=C_LORENTZ, lw=1.3, label=r"Lorentz $\alpha_L$")
    ax.plot(alpha_v_f, p_fine * 1e3, color=C_VOIGT, lw=1.0, ls="--", label=r"Voigt $\alpha_V$")
    for values, colour in ((alpha_d_l, C_DOPPLER), (alpha_l_l, C_LORENTZ), (alpha_v_l, C_VOIGT)):
        ax.plot(values, p_layer * 1e3, "o", ms=2.6, color=colour, mec="none")
    ax.axvline(resolution_hwhm, color=C_RES, lw=0.8, ls=":")
    ax.text(resolution_hwhm * 1.08, 380, "R=45,000\nresolution\nelement / 2", fontsize=6,
            color="#666666", va="center")
    ax.axhline(p_cross * 1e3, color="#555555", lw=0.6)
    ax.text(0.0145, p_cross * 1e3 * 1.08,
            f"$\\alpha_L = \\alpha_D$ at\n{p_cross * 1e3:.0f} hPa (~{z_cross:.0f} km)",
            fontsize=6.5, va="top", color="#333333")
    ax.text(0.0012, 30, "Doppler\ndominates", color=C_DOPPLER, fontsize=7, va="center")
    ax.text(0.0012, 400, "pressure\ndominates", color=C_LORENTZ, fontsize=7, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(edges[-1] * 1e3 * 1.03, edges[0] * 1e3)
    ax.set_xlim(1e-3, 0.15)
    ax.set_yticks([10, 20, 50, 100, 200, 500, 700])
    ax.set_yticklabels(["10", "20", "50", "100", "200", "500", "700"])
    ax.set_xlabel(r"Half width at half maximum (cm$^{-1}$)")
    ax.set_ylabel("Pressure (hPa)")
    ax.legend(loc="upper right", frameon=False, handlelength=1.6)
    altitude = ax.secondary_yaxis(
        "right", functions=(lambda p: np.interp(np.log(np.clip(p, 1e-3, None) / 1e3),
                                                np.log(p_layer), z_layer),
                            lambda z: 1e3 * np.exp(np.interp(z, z_layer[::-1],
                                                             np.log(p_layer)[::-1]))))
    altitude.set_ylabel("Altitude (km)", labelpad=1)
    altitude.set_yticks([3, 5, 10, 15, 20, 25])
    altitude.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    altitude.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title(f"(a) widths of the H$_2$O line at {line['nu_cm1']:.3f} cm$^{{-1}}$ in each layer",
                 loc="left")

    # (c) strength against temperature
    axc = fig.add_subplot(top_row[0, 1])
    temperature = np.linspace(200.0, 300.0, 201)
    axc.axvspan(t_layer.min(), t_layer.max(), color="#000000", alpha=0.05, lw=0)
    axc.text(0.5 * (t_layer.min() + t_layer.max()), 0.05, "temperatures in\nthis atmosphere",
             transform=axc.get_xaxis_transform(), ha="center", fontsize=6, color="#666666")
    ratio_numbers = {}
    for key in ("low", "main", "high"):
        entry = data[key]
        q = partition(entry["isotope"])
        ratio = strength_ratio(entry, temperature, q)
        axc.plot(temperature, ratio, color=C_LINES[key], lw=1.3,
                 label=f"{entry['nu_cm1']:.1f} cm$^{{-1}}$, $E''$ = {entry['elower_cm1']:.0f} cm$^{{-1}}$")
        ratio_numbers[key] = {f"{t:.0f}K": float(strength_ratio(entry, t, q))
                              for t in (200.0, 220.0, 250.0, 280.0, 300.0)}
        ratio_numbers[key]["layers"] = [float(v) for v in strength_ratio(entry, t_layer, q)]
    axc.plot(t_layer, strength_ratio(line, t_layer, partition(line["isotope"])), "o", ms=2.6,
             color="black", mec="none")
    axc.axhline(1.0, color="#999999", lw=0.5)
    axc.axvline(T_REF_K, color="#999999", lw=0.5, ls=":")
    axc.set_yscale("log")
    axc.set_xlim(200, 300)
    axc.set_ylim(0.02, 4.0)
    axc.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2])
    axc.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
    axc.set_xlabel("Temperature (K)")
    axc.set_ylabel(r"$S(T)\,/\,S(296\,\mathrm{K})$")
    axc.legend(loc="lower right", frameon=False, handlelength=1.4, bbox_to_anchor=(1.0, 0.12))
    axc.set_title("(c) line strength depends on temperature", loc="left")

    # (b) profiles in four layers
    shown = [int(np.argmin(np.abs(z_layer - z))) for z in SHOWN_KM]
    x = np.geomspace(2e-4, 1.0, 2000)
    layer_numbers = []
    first = None
    for j, i in enumerate(shown):
        axb = fig.add_subplot(gs[0, j], sharey=first)
        first = first or axb
        sigma, gamma = sigma_l[i], alpha_l_l[i]
        gauss = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        lorentz = gamma / np.pi / (x ** 2 + gamma ** 2)
        voigt = voigt_profile(x, sigma, gamma)
        axb.plot(x, gauss, color=C_DOPPLER, lw=1.1, label="Gaussian (Doppler only)")
        axb.plot(x, lorentz, color=C_LORENTZ, lw=1.1, label="Lorentzian (pressure only)")
        axb.plot(x, voigt, color=C_VOIGT, lw=1.0, ls="--", label="Voigt (both)")
        axb.set_xscale("log")
        axb.set_yscale("log")
        axb.set_ylim(1e-4, 3e2)
        axb.set_xlim(2e-4, 1.0)
        axb.set_xticks([1e-3, 1e-2, 1e-1, 1])
        axb.set_xticklabels(["0.001", "0.01", "0.1", "1"])
        for width, colour in ((alpha_d_l[i], C_DOPPLER), (gamma, C_LORENTZ)):
            axb.axvline(width, ymin=0.0, ymax=0.06, color=colour, lw=1.4)
        axb.text(0.04, 0.10,
                 f"{z_layer[i]:.1f} km\n{p_layer[i] * 1e3:.0f} hPa, {t_layer[i]:.0f} K\n"
                 f"$\\alpha_D$ = {alpha_d_l[i] * 1e3:.1f}\n$\\alpha_L$ = {gamma * 1e3:.1f}\n"
                 f"$\\alpha_V$ = {alpha_v_l[i] * 1e3:.1f}",
                 transform=axb.transAxes, fontsize=6, va="bottom")
        if j:
            axb.tick_params(labelleft=False)
        else:
            axb.set_ylabel(r"Profile (cm), unit area")
        far = np.abs(x - 0.3).argmin()
        core = 0
        layer_numbers.append({
            "altitude_km": float(z_layer[i]), "pressure_hpa": float(p_layer[i] * 1e3),
            "temperature_k": float(t_layer[i]), "h2o_vmr": float(w_layer[i]),
            "alpha_doppler_cm1": float(alpha_d_l[i]), "sigma_doppler_cm1": float(sigma),
            "alpha_lorentz_cm1": float(gamma), "alpha_voigt_cm1": float(alpha_v_l[i]),
            "lorentz_over_doppler": float(gamma / alpha_d_l[i]),
            "voigt_over_lorentz_at_0.3cm1": float(voigt[far] / lorentz[far]),
            "gauss_over_voigt_at_0.3cm1": float(gauss[far] / voigt[far]),
            "voigt_over_gauss_at_centre": float(voigt[core] / gauss[core]),
            "voigt_over_lorentz_at_centre": float(voigt[core] / lorentz[core]),
        })
    fig.text(0.53, 0.012, r"|Offset from line centre| (cm$^{-1}$); ticks at the foot mark $\alpha_D$ and $\alpha_L$; widths in $10^{-3}$ cm$^{-1}$", ha="center", fontsize=8)
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", ncol=3, frameon=False,
               bbox_to_anchor=(0.40, 0.432), fontsize=6.5)
    fig.text(0.075, 0.432, "(b) the line profile in four layers", fontsize=8, va="center")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "fig_intro_lineshape"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)

    numbers = {
        "profile": str(PROFILE.relative_to(ROOT)),
        "lines": data,
        "crossover": {"pressure_hpa": p_cross * 1e3, "altitude_km": z_cross,
                      "width_cm1": width_cross,
                      "note": "alpha_L = alpha_D, with T and H2O interpolated in log P "
                              "between the layer values"},
        "resolution_hwhm_cm1_R45000": resolution_hwhm,
        "layers_all": [{"altitude_km": float(z), "pressure_hpa": float(p * 1e3),
                        "temperature_k": float(t), "alpha_doppler_cm1": float(d),
                        "alpha_lorentz_cm1": float(lw), "alpha_voigt_cm1": float(v),
                        "lorentz_over_doppler": float(lw / d)}
                       for z, p, t, d, lw, v in zip(z_layer, p_layer, t_layer, alpha_d_l,
                                                    alpha_l_l, alpha_v_l)],
        "layers_shown": layer_numbers,
        "strength_ratio_S_T_over_S_296": ratio_numbers,
        "formulas": {
            "doppler_sigma": "sigma_D = 3.0415595e-7 * sqrt(T/M) * nu0 [cm-1]; alpha_D = sqrt(2 ln2) sigma_D "
                             "-- exojax/database/core/broadening.py doppler_sigma",
            "lorentz": "alpha_L = (296/T)^n_air * (gamma_air (P-Pself)/1.01325 + gamma_self Pself/1.01325) "
                       "+ 2.6544188e-12 A [P in bar] -- exojax broadening.py gamma_hitran + gamma_natural",
            "voigt": "scipy.special.voigt_profile(x, sigma_D, alpha_L), HWHM by brentq; tellurix evaluates "
                     "the same Voigt with ExoJAX's Faddeeva routines in direct.py SparseCoreDirect",
            "strength": "S(T) = S296 * Q(296)/Q(T) * exp(-c2 E'' (1/T - 1/296)) * (1-exp(-c2 nu0/T))"
                        "/(1-exp(-c2 nu0/296)) -- exojax/database/core/line_strength.py line_strength",
            "partition_function": "TIPS-2017 from HAPI, as AERLineDatabase.qr_interp (tellurix aer.py)",
            "where_tellurix_calls_them": "packages/tellurix/src/tellurix/direct.py "
                                         "SparseCoreDirect._line_parameters",
            "layer_pressure": "AtmosphereProfile.pressure_layer_bar (geometric mean of the edges for "
                              "this CSV, which has no pressure_bar column)",
        },
    }
    stem.with_suffix(".json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps({"crossover": numbers["crossover"], "shown": layer_numbers,
                      "strength": {k: {t: v for t, v in r.items() if t != 'layers'}
                                   for k, r in ratio_numbers.items()}}, indent=1))


if __name__ == "__main__":
    main()
