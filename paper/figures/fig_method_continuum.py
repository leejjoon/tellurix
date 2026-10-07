#!/usr/bin/env python
"""Method figure: the MT_CKD 4.3 water continuum in LBLRTM 12.17 and in tellurix.

The profile is the 12-layer weighted DCT column of ``fig_method_layering.py``
(ERA5 2018-12-21 05 UT), handed to LBLRTM as its own layers (IATM=0,
``LBLRTMRunConfig(user_layers=True)``) so both codes evaluate the continuum at
the same pressure, temperature and columns. LBLRTM is run continuum-only: its
TAPE3 holds O2 lines at 12750-14850 cm-1 only, so nothing but continua enters
4000-6500 cm-1. Self and foreign are separated by LBLRTM's own switches
(lblrtm.f90:713-738): ICNTNM=1 has both, 2 drops self (XSELF=0), 3 drops
foreign (XFRGN=0); every other continuum cancels in the differences.

tellurix is ``MTCKDWaterContinuum`` -- the bound (precomputed) form the model
uses -- on LBLRTM's own output grid. A third curve emulates LBLRTM's two-stage
interpolation (10 cm-1 coefficients -> 1 cm-1 DVABS grid -> output grid,
mt_ckd_h2o_module.f90 ``myxint`` then oprop.f90 ``XINT``) in float64, to show
how much of the residual that second interpolation is.

    JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_method_continuum.py

LBLRTM runs are cached under ``paper/figures/cache/lblrtm_mtckd/`` (18 runs,
a few seconds each). Writes ``fig_method_continuum.{pdf,png,json}``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

import tellurix  # noqa: F401  (x64 first)
from tellurix import AtmosphereProfile, LBLRTMRunConfig, MTCKDWaterContinuum
from tellurix.era5 import build_era5_profile
from tellurix.lblrtm import write_tape5
from tellurix.reference import read_tape12_single_precision
from tellurix.site_profile import DEFAULT_EDGES_KM, EPOCH_DRY_VMR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper/figures/cache"
OUTPUT = ROOT / "paper/figures/output"
LBLRTM = ROOT / "data/lblrtm"
EXECUTABLE = LBLRTM / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
EMPTY_TAPE3 = LBLRTM / "run_lnfl_o2/TAPE3"
COEFFICIENTS = LBLRTM / "LBLRTM/data/absco-ref_wv-mt-ckd.nc"
VALIDATION = ROOT / "packages/tellurix/tests/data/native_mt_ckd_validation.json"

CASE = {"site_altitude_km": 2.360, "station_pressure_hpa": 768.5}
# OPPATH stops above V2 - V1 = 2020 cm-1 (lblrtm.f90:4283), so two runs.
SEGMENTS = ((3990.0, 5260.0), (5240.0, 6510.0))
SPAN = (4000.0, 6500.0)
WATER_SCALES = (0.5, 1.0, 2.0)
FLAGS = {"all": 1, "no_self": 2, "no_foreign": 3}
# IGRINS H and K in vacuum wavenumber.
BANDS = {"K": (1.0e4 / 2.48, 1.0e4 / 1.96), "H": (1.0e4 / 1.80, 1.0e4 / 1.49)}
WINDOW = (4500.0, 4700.0)
# TAPE12 stores transmission in float32, so -ln T carries an absolute error of
# ~6e-8 whatever T is; a vertical continuum of 1e-5 would be resolved to 1%.
# Every column is multiplied by PATH_SCALE (gravity divided by it, so pressure,
# temperature and every mixing ratio -- all MT_CKD depends on besides the
# column -- are untouched) and the optical depths divided by it afterwards.
# The largest, foreign at c = 2, stays near 55: e^-55 is still a normal float32.
PATH_SCALE = 100.0
STRIDE_CM1 = 0.05

WIDTH_IN = 7.1
C_LBL = "black"
C_SELF = "#0072B2"
C_FOREIGN = "#D55E00"
C_EMUL = "#009E73"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})


def profile() -> AtmosphereProfile:
    data = np.load(CACHE / "era5_dct_2018.npz")
    column = {k: data[k] for k in ("level_hpa", "temperature_k", "specific_humidity", "height_m")}
    layers = build_era5_profile(column, CASE["station_pressure_hpa"], CASE["site_altitude_km"],
                                EPOCH_DRY_VMR["2020"], DEFAULT_EDGES_KM, layering="weighted")
    edges = np.concatenate([layers["pressure_top_bar"][:1], layers["pressure_bottom_bar"]])
    species = ("H2O", "CO2", "CH4", "N2O", "CO", "O2")
    return AtmosphereProfile(edges, layers["temperature_k"], layers["altitude_km"],
                             {s: layers[s] for s in species}, layers["mean_molecular_weight_g_mol"],
                             layers["gravity_m_s2"], mean_pressure_bar=layers["pressure_bar"])


def scaled(atmosphere: AtmosphereProfile, scale: float) -> AtmosphereProfile:
    vmr = dict(atmosphere.vmr)
    vmr["H2O"] = np.asarray(vmr["H2O"]) * scale
    return dataclasses.replace(atmosphere, vmr=vmr)


def lblrtm_tau(atmosphere: AtmosphereProfile, scale: float, flag: str, segment) -> tuple:
    directory = CACHE / f"lblrtm_mtckd/s{PATH_SCALE:.0f}_x{scale:.2f}_{flag}_{segment[0]:.0f}"
    tape12 = directory / "TAPE12"
    if not tape12.exists():
        directory.mkdir(parents=True, exist_ok=True)
        config = LBLRTMRunConfig(*segment, continuum_flag=FLAGS[flag], user_layers=True,
                                 description=f"paper MT_CKD continuum x{scale} {flag}")
        path = scaled(atmosphere, scale)
        path = dataclasses.replace(path, gravity_m_s2=np.asarray(path.gravity_m_s2) / PATH_SCALE)
        write_tape5(directory / "TAPE5", path, config)
        for source, name in ((EMPTY_TAPE3, "TAPE3"), (COEFFICIENTS, "absco-ref_wv-mt-ckd.nc")):
            if not (directory / name).exists():
                (directory / name).symlink_to(source)
        import shutil
        import subprocess
        shutil.copy2(EXECUTABLE, directory / "lblrtm")
        result = subprocess.run(["./lblrtm"], cwd=directory, capture_output=True, text=True)
        if result.returncode != 0 or not tape12.exists() or tape12.stat().st_size == 0:
            raise RuntimeError(f"LBLRTM failed in {directory}:\n{result.stdout[-2000:]}")
    spectrum = read_tape12_single_precision(tape12)
    transmission = np.asarray(spectrum.transmission, float)
    if np.any(transmission <= 0.0):
        raise RuntimeError("transmission underflowed; lower PATH_SCALE")
    return np.asarray(spectrum.wavenumber_cm1, float), -np.log(transmission) / PATH_SCALE


def lblrtm_components(atmosphere, scale):
    """Self and foreign vertical optical depth on a common grid, every STRIDE_CM1."""

    nu_out = np.arange(SPAN[0], SPAN[1] + 1e-9, STRIDE_CM1)
    pieces = {}
    for flag in FLAGS:
        nus, taus = [], []
        for segment in SEGMENTS:
            nu, tau = lblrtm_tau(atmosphere, scale, flag, segment)
            nus.append(nu)
            taus.append(tau)
        # Each segment covers the output grid it is used for; the second
        # takes over at the midpoint of the overlap.
        cut = 0.5 * (SEGMENTS[0][1] + SEGMENTS[1][0])
        nu_all = np.concatenate([nus[0][nus[0] < cut], nus[1][nus[1] >= cut]])
        tau_all = np.concatenate([taus[0][nus[0] < cut], taus[1][nus[1] >= cut]])
        pieces[flag] = (nu_all, tau_all)
    nu = pieces["all"][0]
    for flag in FLAGS:
        if not np.array_equal(pieces[flag][0], nu):
            raise RuntimeError("LBLRTM runs differ in output grid")
    index = np.searchsorted(nu, nu_out)
    index = np.clip(index, 0, nu.size - 1)
    return {"nu": nu[index],
            "self": (pieces["all"][1] - pieces["no_self"][1])[index],
            "foreign": (pieces["all"][1] - pieces["no_foreign"][1])[index]}


def tellurix_components(atmosphere, scale, nu):
    continuum = MTCKDWaterContinuum.from_netcdf(COEFFICIENTS, nu)
    bound = continuum.bind(atmosphere)
    water = np.asarray(atmosphere.vmr["H2O"]) * scale
    column = np.asarray(atmosphere.air_column_cm2) * water
    self_tau = (bound.self_cross_section * water[:, None] * column[:, None]).sum(axis=0)
    foreign_tau = (bound.foreign_cross_section * (1.0 - water[:, None]) * column[:, None]).sum(axis=0)
    return {"self": self_tau, "foreign": foreign_tau, "version": continuum.version}


def xint(source_nu, values, target_nu):
    """LBLRTM's four-point XINT (oprop.f90:2201; myxint, mt_ckd_h2o_module.f90)."""

    dva = source_nu[1] - source_nu[0]
    j = np.floor((target_nu - source_nu[0]) / dva + 1.001).astype(int) - 1   # zero-based J
    p = (target_nu - (source_nu[0] + dva * j)) / dva
    c = (3.0 - 2.0 * p) * p * p
    b = 0.5 * p * (1.0 - p)
    b1, b2 = b * (1.0 - p), b * p
    return (-values[j - 1] * b1 + values[j] * (1.0 - c + b2) + values[j + 1] * (c + b1)
            - values[j + 2] * b2)


def emulated_lblrtm(atmosphere, scale, nu, segment_v1, segment_v2):
    """MT_CKD as LBLRTM stages it, in float64: coefficients -> 1 cm-1 -> output grid."""

    from scipy.io import netcdf_file
    with netcdf_file(COEFFICIENTS, "r", mmap=False) as data:
        wn = data.variables["wavenumbers"].data.astype(float)
        self_ref = data.variables["self_absco_ref"].data.astype(float)
        for_ref = data.variables["for_absco_ref"].data.astype(float)
        texp = data.variables["self_texp"].data.astype(float)
        p_ref = float(data.variables["ref_press"].data)
        t_ref = float(data.variables["ref_temp"].data)
    # lblrtm.f90:6217-6222: DVABS = 1, V1ABS = INT(V1) - 3, V2ABS = INT(V2 + 3.5).
    v1abs = float(int(segment_v1)) - 3.0
    v2abs = float(int(segment_v2 + 3.0 + 0.5))
    abs_grid = np.arange(v1abs, v2abs + 0.5, 1.0)
    water = np.asarray(atmosphere.vmr["H2O"]) * scale
    column = np.asarray(atmosphere.air_column_cm2) * water
    out = {"self": np.zeros_like(nu), "foreign": np.zeros_like(nu)}
    for layer in range(len(water)):
        t = float(atmosphere.temperature_k[layer])
        p_hpa = float(atmosphere.continuum_pressure_bar[layer]) * 1000.0
        rho = (p_hpa / p_ref) * (t_ref / t)
        x = wn * 1.4387752 / t
        rad = np.where(x <= 0.01, 0.5 * x * wn, wn * np.tanh(0.5 * x))
        s = self_ref * (t_ref / t) ** texp * water[layer] * rho * rad
        f = for_ref * (1.0 - water[layer]) * rho * rad
        for key, coeff in (("self", s), ("foreign", f)):
            on_abs = xint(wn, coeff, abs_grid)
            out[key] += column[layer] * xint(abs_grid, on_abs, nu)
    return out


def main() -> None:
    atmosphere = profile()
    results = {}
    for scale in WATER_SCALES:
        lbl = lblrtm_components(atmosphere, scale)
        tx = tellurix_components(atmosphere, scale, lbl["nu"])
        results[scale] = {"lbl": lbl, "tx": tx}
    nu = results[1.0]["lbl"]["nu"]
    lbl, tx = results[1.0]["lbl"], results[1.0]["tx"]
    # Emulate per segment, on the side of the overlap each one is used for.
    cut = 0.5 * (SEGMENTS[0][1] + SEGMENTS[1][0])
    emul = {"self": np.empty_like(nu), "foreign": np.empty_like(nu)}
    for (v1, v2), part in zip(SEGMENTS, (nu < cut, nu >= cut)):
        e = emulated_lblrtm(atmosphere, 1.0, nu[part], v1, v2)
        for key in emul:
            emul[key][part] = e[key]

    stats = {}
    for key in ("self", "foreign"):
        rel = tx[key] / lbl[key] - 1.0
        rel_emul = tx[key] / emul[key] - 1.0
        rel_emul_lbl = emul[key] / lbl[key] - 1.0
        stats[key] = {
            "tellurix_vs_lblrtm": {"median_abs": float(np.median(np.abs(rel))),
                                   "p99_abs": float(np.percentile(np.abs(rel), 99)),
                                   "max_abs": float(np.max(np.abs(rel)))},
            "tellurix_vs_emulated_two_stage": {"median_abs": float(np.median(np.abs(rel_emul))),
                                               "p99_abs": float(np.percentile(np.abs(rel_emul), 99))},
            "emulated_vs_lblrtm": {"median_abs": float(np.median(np.abs(rel_emul_lbl))),
                                   "p99_abs": float(np.percentile(np.abs(rel_emul_lbl), 99))},
            "tau_range": [float(lbl[key].min()), float(lbl[key].max())],
        }
    window = (nu >= WINDOW[0]) & (nu <= WINDOW[1])
    scaling = {key: {"lblrtm": [float(results[s]["lbl"][key][window].mean()) for s in WATER_SCALES],
                     "tellurix": [float(results[s]["tx"][key][window].mean()) for s in WATER_SCALES]}
               for key in ("self", "foreign")}
    validation = json.loads(VALIDATION.read_text())
    numbers = {"profile": "ERA5 DCT 2018-12-21 05UT, 12 weighted layers, vertical",
               "mt_ckd": tx["version"], "stats": stats, "window_cm1": WINDOW,
               "water_scales": WATER_SCALES, "window_mean_tau": scaling,
               "committed_validation_median_rel": [c["median_relative_error"]
                                                   for c in validation["cases"]],
               "committed_validation_p99_rel": [c["percentile_99_relative_error"]
                                                for c in validation["cases"]]}
    print(json.dumps(numbers, indent=1))

    figure = plt.figure(figsize=(WIDTH_IN, 4.3))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.42)
    ax_tau = figure.add_subplot(grid[0, :])
    ax_rel = figure.add_subplot(grid[1, :2])
    ax_scale = figure.add_subplot(grid[1, 2])

    for name, (a, b) in BANDS.items():
        for ax in (ax_tau, ax_rel):
            ax.axvspan(max(a, SPAN[0]), min(b, SPAN[1]), color="#f0f0f0", zorder=0, lw=0)
        ax_tau.text(0.5 * (max(a, SPAN[0]) + min(b, SPAN[1])), 0.45, f"IGRINS {name}",
                    ha="center", va="bottom", fontsize=7, color="#666666")
    ax_tau.plot(nu, lbl["self"], color=C_LBL, lw=1.6, label="LBLRTM 12.17")
    ax_tau.plot(nu, lbl["foreign"], color=C_LBL, lw=1.6)
    ax_tau.plot(nu, tx["self"], color=C_SELF, lw=0.8, ls=(0, (4, 2)), label="tellurix self")
    ax_tau.plot(nu, tx["foreign"], color=C_FOREIGN, lw=0.8, ls=(0, (4, 2)), label="tellurix foreign")
    ax_tau.set_yscale("log")
    ax_tau.set_ylim(1e-6, 1.5)
    ax_tau.set_xlim(*SPAN)
    ax_tau.set_ylabel("Vertical optical depth")
    ax_tau.set_xlabel(r"Wavenumber (cm$^{-1}$)", labelpad=1)
    ax_tau.legend(loc="upper right", ncol=1, frameon=False, bbox_to_anchor=(1.0, 0.86))
    ax_tau.text(4300, 3e-5, "self", color=C_SELF, fontsize=7.5)
    ax_tau.text(4300, 2.5e-3, "foreign", color=C_FOREIGN, fontsize=7.5)
    top = ax_tau.secondary_xaxis("top", functions=(lambda v: 1e4 / np.maximum(v, 1.0),
                                                   lambda w: 1e4 / np.maximum(w, 1e-6)))
    top.set_xticks([1.6, 1.7, 1.8, 2.0, 2.2, 2.4])
    top.set_xlabel(r"Wavelength ($\mu$m)", labelpad=1)
    ax_tau.set_title(f"(a) MT_CKD {tx['version'].split('-')[-1].strip()} H$_2$O continuum, "
                     "ERA5 DCT column (12 layers, 2.35 mm)", loc="left", pad=16)

    for key, color in (("self", C_SELF), ("foreign", C_FOREIGN)):
        ax_rel.plot(nu, 100 * (tx[key] / lbl[key] - 1.0), color=color, lw=0.6,
                    label=f"{key}: tellurix / LBLRTM")
    ax_rel.axhline(0.0, color="#999999", lw=0.5)
    ax_rel.set_xlim(*SPAN)
    ax_rel.set_ylim(-0.2, 0.2)
    ax_rel.set_ylabel("Relative difference (%)")
    ax_rel.set_xlabel(r"Wavenumber (cm$^{-1}$)", labelpad=1)
    ax_rel.legend(loc="upper left", ncol=1, frameon=False, fontsize=6, handlelength=1.8)
    s, f = stats["self"]["tellurix_vs_lblrtm"], stats["foreign"]["tellurix_vs_lblrtm"]
    e = stats["self"]["tellurix_vs_emulated_two_stage"]
    ax_rel.text(0.02, 0.03, f"|tellurix / LBLRTM $-$ 1|   median: self {s['median_abs']:.4%}, "
                f"foreign {f['median_abs']:.4%}\n"
                f"99th percentile: self {s['p99_abs']:.3%}, foreign {f['p99_abs']:.3%}",
                transform=ax_rel.transAxes, ha="left", va="bottom", fontsize=5.8)
    ax_rel.set_title("(b) Agreement on LBLRTM's output grid", loc="left")

    scales = np.geomspace(0.3, 3.0, 50)
    x = np.asarray(WATER_SCALES)
    for key, color, marker in (("self", C_SELF, "o"), ("foreign", C_FOREIGN, "s")):
        reference = scaling[key]["tellurix"][1]
        if key == "self":
            curve = reference * scales ** 2
            label = r"self $\propto c^2$"
        else:
            vmr = np.asarray(atmosphere.vmr["H2O"])
            col = np.asarray(atmosphere.air_column_cm2) * vmr
            # Foreign: c * (1 - c x) per layer; evaluated exactly, not as c.
            per_layer = np.asarray([((1.0 - sc * vmr) * col * sc).sum() / ((1.0 - vmr) * col).sum()
                                    for sc in scales])
            curve = reference * per_layer
            label = r"foreign $\propto c(1-c\,x)$"
        ax_scale.plot(scales, curve, color=color, lw=1.0, label=label)
        ax_scale.plot(x, scaling[key]["lblrtm"], marker, ms=4, mfc="none", mec=C_LBL, mew=0.9,
                      label="LBLRTM" if key == "self" else None)
    ax_scale.set_xscale("log")
    ax_scale.set_yscale("log")
    ax_scale.set_xticks([0.5, 1, 2])
    ax_scale.set_xticklabels(["0.5", "1", "2"])
    ax_scale.set_xlabel(r"Water column scale $c$")
    ax_scale.set_ylabel(r"Mean $\tau$", labelpad=1)
    ax_scale.text(0.03, 0.97, fr"{WINDOW[0]:.0f}–{WINDOW[1]:.0f} cm$^{{-1}}$", fontsize=6,
                  transform=ax_scale.transAxes, ha="left", va="top")
    ax_scale.legend(loc="lower right", frameon=False, fontsize=5.6, handlelength=1.4,
                    borderaxespad=0.1)
    ax_scale.set_title("(c) Differentiable in $c$", loc="left")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT / f"fig_method_continuum.{suffix}", dpi=200, bbox_inches="tight")
    (OUTPUT / "fig_method_continuum.json").write_text(json.dumps(numbers, indent=1))
    print("wrote", OUTPUT / "fig_method_continuum.png")


if __name__ == "__main__":
    main()
