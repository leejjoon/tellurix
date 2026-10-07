#!/usr/bin/env python
"""Paper figure: first-order (Rosenkranz) line coupling in LBLRTM 12.17 and tellurix.

Three strong, coupled 12C16O2 lines near 4998 cm-1 (AER 3.9 ``lncpl_lines``, flag
-1, Y and G at 200/250/296/340 K) are put through both codes in one
homogeneous layer, with and without coupling:

* LBLRTM: LNFL builds a TAPE3 from just these records, once as is and once
  with NOCPL; LBLRTM computes the layer optical depth (TAPE10) with rejection
  off (DPTMIN = DPTFAC = 0). Coupled minus uncoupled is LBLRTM's dispersion
  term. In ``oprop.f90`` it is S Y p times the line's own (approximate) Voigt
  shape times its offset in Voigt widths: LNCOR1 sets SP = S (1 + G p^2) and
  SPPSP = S Y p / SP, CNVFNV adds each sub-function again multiplied by
  CLC x ZF = (nu - nu0) / alpha_V, and CONVF4 multiplies the fourth function,
  pedestal included, by 1 + (nu - nu0) SPP / (SP alpha_V).
* tellurix: ``SparseCoreDirect(line_coupling=True)``, S (1 + G p^2) [Re w +
  Y p Im w] with Im w exact (Algorithm 916 / asymptotic, custom JVP) and the
  Im w term cut at 25 cm-1 less the Lorentz-limit pedestal x / pi (gamma^2 + B^2).

Two layers: 800 hPa / 280 K (nearly Lorentzian) and 50 hPa / 220 K (Doppler
comparable to Lorentz). The JSON sidecar holds the numbers quoted in the text.

Runs go to ``data/lblrtm/paper_method_coupling/`` (gitignored), results to
``paper/figures/cache/method_coupling.npz`` (``--recompute``). CPU:

    JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_method_coupling.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper/figures/cache"
OUTPUT = ROOT / "paper/figures/output"
RUNS = ROOT / "data/lblrtm/paper_method_coupling"
LBLRTM = ROOT / "data/lblrtm/LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
LNFL = ROOT / "data/lblrtm/LNFL/lnfl_v3.2_linux_gnu_sgl"
COUPLING_FILE = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/lncpl_lines"

LINES = ("4997.246255", "4998.354232", "4999.432285")
LAYERS = {"800": (800.0, 280.0), "50": (50.0, 220.0)}
CO2_VMR = 4.1e-4
AIR_COLUMN_800 = 2.0e24       # air column scales with pressure: a 2.1 km slab-ish column at 800 hPa
V1, V2 = 4975.0, 5025.0
CUTOFF_CM1 = 25.0

BLACK, ORANGE, SKY, GREEN, BLUE, VERMILLION, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
GREY = "#999999"


def fortran_records(path: Path) -> list[bytes]:
    data, offset, records = path.read_bytes(), 0, []
    while offset + 4 <= len(data):
        size = struct.unpack_from("=i", data, offset)[0]
        records.append(data[offset + 4: offset + 4 + size])
        offset += 8 + size
    return records


def read_optical_depth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    records = fortran_records(path)
    nus, ods, index = [], [], 1
    while index < len(records):
        if len(records[index]) != 24:
            index += 1
            continue
        v1p, _v2p, dvp, nlim = struct.unpack("=ddfi", records[index])
        if nlim <= 0:
            break
        nus.append(v1p + dvp * np.arange(nlim))
        ods.append(np.frombuffer(records[index + 1], dtype="<f4")[:nlim].astype(float))
        index += 2
    return np.concatenate(nus), np.concatenate(ods)


def avrat_table() -> np.ndarray:
    """AVRAT(zeta): LBLRTM's Voigt width / (alpha_L + alpha_D) (oprop.f90, BLOCK DATA VOICON)."""

    import re
    text = (ROOT / "data/lblrtm/LBLRTM/src/oprop.f90").read_text(errors="replace")
    block = text[text.index("BLOCK DATA VOICON"): text.index("end block data VOICON")]
    values: list[float] = []
    for part in ("01", "51"):
        body = re.search(rf"DATA AV{part}\s*/(.*?)/", block, re.S).group(1)
        values += [float(v) for v in re.sub(r"[&\n,]", " ", body).split()]
    return np.asarray(values)


def run_lnfl(directory: Path, records: list[str], coupling: bool) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "TAPE1").write_text("".join(r.rstrip("\n").ljust(100) + "\n" for r in records))
    (directory / "TAPE5").write_text(f"paper coupled CO2\n{4950.0:10.3f}{5050.0:10.3f}\n"
                                     f"{'01'.ljust(47, '0')}    {'' if coupling else 'NOCPL'}\n")
    for name in ("TAPE3", "TAPE6", "TAPE10"):
        (directory / name).unlink(missing_ok=True)
    shutil.copy2(LNFL, directory / "lnfl")
    subprocess.run(["./lnfl"], cwd=directory, capture_output=True, check=False)
    return directory / "TAPE3", (directory / "TAPE6").read_text(errors="replace")


def run_lblrtm(directory: Path, tape3: Path, pressure_hpa: float, temperature_k: float):
    air = AIR_COLUMN_800 * pressure_hpa / 800.0
    co2 = CO2_VMR * air
    directory.mkdir(parents=True, exist_ok=True)
    p, t = pressure_hpa, temperature_k
    (directory / "TAPE5").write_text("\n".join([
        "$paper coupled CO2 layer",
        "".join(f"{v:5d}" for v in (1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        f"{V1:10.3f}{V2:10.3f}{4.0:10.3f}{0.0:10.3f}{0.04:10.3f}{36.0:10.3f}{0.0:10.3f}{0.0:10.3f}"
        f"{0:5d}{0.0:15.3f}{0:5d}",
        f"{t:10.3f}{1.0:10.3f}{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}    s",
        f" {1:1d}{1:3d}{7:5d}{1.0:10.6f}{'':20s}{0.0:8.3f}{'':4s}{1.0:8.3f}{'':5s}{0.0:8.3f}",
        f"{p:15.7E}{t:10.4f}{0.0:10.4f}{'':3s}{0:2d} {0.0:7.2f}{p:8.3f}{t:7.2f}{1.0:7.2f}{p:8.3f}{t:7.2f}",
        "".join(f"{c:15.7E}" for c in (0, co2, 0, 0, 0, 0, 0)) + f"{air - co2:15.7E}",
        "-1.0", "-1.0", "%"]) + "\n")
    link = directory / "TAPE3"
    link.unlink(missing_ok=True)
    link.symlink_to(tape3.resolve())
    for name in ("TAPE10", "TAPE11", "TAPE12", "TAPE6"):
        (directory / name).unlink(missing_ok=True)
    shutil.copy2(LBLRTM, directory / "lblrtm")
    subprocess.run(["./lblrtm"], cwd=directory, capture_output=True, check=False)
    return read_optical_depth(directory / "TAPE10"), co2


def compute() -> dict:
    text = COUPLING_FILE.read_text(errors="replace").splitlines()
    records = []
    for index, line in enumerate(text):
        if line[:3] == " 21" and line[3:15].strip() in LINES:
            records += [line, text[index + 1]]          # the line, then its Y/G record
    assert len(records) == 2 * len(LINES)
    tape3 = {}
    for coupling in (True, False):
        tape3[coupling], log = run_lnfl(RUNS / f"lnfl_{'cpl' if coupling else 'nocpl'}", records, coupling)

    import tellurix  # noqa: F401
    from scipy.special import wofz
    from tellurix import DataPaths
    from tellurix.aer import AERLineDatabase
    from tellurix.direct import SparseCoreDirect

    paths = DataPaths.bootstrapped(ROOT)
    database = AERLineDatabase(paths.line_file("CO2", 2), "CO2", (4997.0, 5000.0), margin_cm1=0.5,
                               line_coupling=paths.line_coupling)
    keep = np.zeros(np.asarray(database.nu_lines).size, bool)
    for position in LINES:
        keep |= (np.abs(np.asarray(database.nu_lines) - float(position)) < 1e-6) & (database.isoid == 1)
    database = database.restrict(keep)

    out: dict = {"y_table": np.asarray(database.coupling_y), "g_table": np.asarray(database.coupling_g)}
    for name, (pressure, temperature) in LAYERS.items():
        results = {}
        for coupling in (True, False):
            (nu, od), co2 = run_lblrtm(RUNS / f"lblrtm_{name}_{'cpl' if coupling else 'nocpl'}", tape3[coupling],
                                       pressure, temperature)
            results[coupling] = od
        p_bar = pressure / 1000.0
        tellurix_od = {}
        for coupling in (True, False):
            calculator = SparseCoreDirect(database, nu, pressure_shift=True, line_coupling=coupling,
                                          maximum_pressure_bar=1.2)
            tellurix_od[coupling] = np.asarray(calculator.xsvector(temperature, p_bar, p_bar * CO2_VMR)) * co2
        calculator = SparseCoreDirect(database, nu, pressure_shift=True, line_coupling=True, maximum_pressure_bar=1.2)
        sigma, gamma, strength = (np.asarray(v) for v in calculator._line_parameters(temperature, p_bar,
                                                                                       p_bar * CO2_VMR))
        mixing, factor = (np.asarray(v) for v in database.coupling(temperature, p_bar))
        centre = np.asarray(database.nu_lines) + np.asarray(calculator._line_shift(temperature, p_bar))
        # The same Im w, cut with LBLRTM's CO2 pedestal (2 - x^2/B^2) L(B) instead
        # of the Lorentz one -- CONVF4 applies the CO2 form to coupled lines too.
        co2_pedestal = np.zeros_like(nu)
        voigt_offset = np.zeros_like(nu)     # LBLRTM's form with the exact Voigt: V(x) x / alpha_V
        for i in range(centre.size):
            x = nu - centre[i]
            scale = 1 / (np.sqrt(2) * sigma[i])
            w = wofz((x + 1j * gamma[i]) * scale)
            imag = np.imag(w) * scale / np.sqrt(np.pi)
            real = np.real(w) * scale / np.sqrt(np.pi)
            lorentz_pedestal = x / (np.pi * (gamma[i] ** 2 + CUTOFF_CM1 ** 2))
            weight = strength[i] * factor[i] * co2 * mixing[i]
            co2_pedestal += np.where(np.abs(x) <= CUTOFF_CM1,
                                     weight * (imag - lorentz_pedestal * (2 - x * x / CUTOFF_CM1 ** 2)), 0.0)
            alfd = sigma[i] * np.sqrt(2 * np.log(2))
            zeta = gamma[i] / (gamma[i] + alfd)
            iz = int(100 * zeta + 1.001)
            avrat = avrat_table()
            alfv = (avrat[iz - 1] + (100 * zeta - iz + 1) * (avrat[iz] - avrat[iz - 1])) * (gamma[i] + alfd)
            voigt_offset += np.where(np.abs(x) <= CUTOFF_CM1, weight * real * x / alfv, 0.0)
        out.update({f"nu_{name}": nu, f"lbl_cpl_{name}": results[True], f"lbl_nocpl_{name}": results[False],
                    f"tlx_cpl_{name}": tellurix_od[True], f"tlx_nocpl_{name}": tellurix_od[False],
                    f"tlx_co2ped_{name}": co2_pedestal, f"voigt_offset_{name}": voigt_offset,
                    f"centre_{name}": centre, f"gamma_{name}": gamma, f"sigma_{name}": sigma,
                    f"strength_{name}": strength * co2, f"mixing_{name}": mixing})
    return out


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42, "savefig.dpi": 200,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })


def plot(data: dict) -> dict:
    import matplotlib.pyplot as plt

    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(left=0.09, right=0.985, top=0.955, bottom=0.085, hspace=0.45, wspace=0.3)

    nu = data["nu_800"]
    lbl_d = data["lbl_cpl_800"] - data["lbl_nocpl_800"]
    tlx_d = data["tlx_cpl_800"] - data["tlx_nocpl_800"]
    centre = data["centre_800"]
    mixing = data["mixing_800"]

    # (a) the three lines with and without coupling
    near = (nu > 4996.4) & (nu < 5000.3)
    ax_a.plot(nu[near], data["lbl_nocpl_800"][near], color=GREY, lw=1.6, label="LBLRTM, uncoupled")
    ax_a.plot(nu[near], data["lbl_cpl_800"][near], color=BLACK, lw=0.8, label="LBLRTM, coupled")
    ax_a.plot(nu[near], data["tlx_cpl_800"][near], color=BLUE, lw=0.8, ls=(0, (3, 2)), label="tellurix, coupled")
    for c, y in zip(centre, mixing):
        ax_a.text(c, 4.0, f"$Yp$={y:+.4f}", ha="center", fontsize=6)
    ax_a.set(yscale="log", ylim=(3e-3, 8), xlim=(4996.4, 5000.3), xlabel=r"wavenumber (cm$^{-1}$)",
             ylabel="optical depth")
    ax_a.legend(loc="lower left", fontsize=6.2)
    ax_a.set_title("(a) Three coupled CO$_2$ lines, 800 hPa, 280 K", loc="left", fontsize=8)

    # (b) the dispersion term near the lines
    ax_b.axhline(0, color=GREY, lw=0.5)
    ax_b.plot(nu[near], lbl_d[near] * 1e3, color=BLACK, lw=1.4, label=r"LBLRTM: $SYp\,V_{\rm LBL}(x)\,x/\alpha_V$")
    ax_b.plot(nu[near], tlx_d[near] * 1e3, color=BLUE, lw=0.8, ls=(0, (3, 2)),
              label=r"tellurix: $SYp\,{\rm Im}\,w(z)$")
    ax_b.plot(nu[near], (tlx_d - lbl_d)[near] * 1e4, color=VERMILLION, lw=0.7, label=r"tellurix $-$ LBLRTM ($\times$10)")
    ax_b.set(xlim=(4996.4, 5000.3), xlabel=r"wavenumber (cm$^{-1}$)",
             ylabel=r"coupled $-$ uncoupled ($10^{-3}$)")
    ax_b.legend(loc="upper right", fontsize=6.0)
    from matplotlib.ticker import MultipleLocator
    for ax in (ax_a, ax_b):
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax_b.set_title("(b) Dispersion term, 800 hPa", loc="left", fontsize=8)

    # (c) the far wing, where the codes' pedestals differ
    far = (nu > 5000.6) & (nu < V2)
    ax_c.axhline(0, color=GREY, lw=0.5)
    ax_c.plot(nu[far], lbl_d[far] * 1e4, color=BLACK, lw=1.4, label="LBLRTM")
    ax_c.plot(nu[far], tlx_d[far] * 1e4, color=BLUE, lw=0.9, ls=(0, (3, 2)),
              label=r"tellurix: Im $w$ less $x/\pi(\gamma^2+B^2)$")
    ax_c.plot(nu[far], data["tlx_co2ped_800"][far] * 1e4, color=GREEN, lw=0.9, ls=(0, (1, 1)),
              label=r"Im $w$ less LBLRTM's CO$_2$ pedestal $(2-x^2/B^2)\,x/\pi(\gamma^2+B^2)$")
    ax_c.set(xlim=(5000.6, V2), xlabel=r"wavenumber (cm$^{-1}$)", ylabel=r"coupled $-$ uncoupled ($10^{-4}$)")
    ax_c.legend(loc="lower right", fontsize=5.9)
    fm = far
    ax_c.text(0.97, 0.42, (f"mean over the panel ($10^{{-4}}$):\nLBLRTM {lbl_d[fm].mean() * 1e4:.2f}\n"
                           f"tellurix {tlx_d[fm].mean() * 1e4:.2f} "
                           f"({tlx_d[fm].mean() / lbl_d[fm].mean() - 1:+.0%})\n"
                           f"with CO$_2$ pedestal {data['tlx_co2ped_800'][fm].mean() * 1e4:.2f} "
                           f"({data['tlx_co2ped_800'][fm].mean() / lbl_d[fm].mean() - 1:+.1%})"),
              transform=ax_c.transAxes, ha="right", va="center", fontsize=6.2)
    ax_c.set_title("(c) Dispersion far wing, 800 hPa (B = 25 cm$^{-1}$)", loc="left", fontsize=8)

    # (d) a Doppler-broadened layer: the shape of the term itself
    nu50 = data["nu_50"]
    lbl50 = data["lbl_cpl_50"] - data["lbl_nocpl_50"]
    tlx50 = data["tlx_cpl_50"] - data["tlx_nocpl_50"]
    c50 = data["centre_50"][1]
    gamma50, sigma50 = data["gamma_50"][1], data["sigma_50"][1]
    alfd50 = sigma50 * np.sqrt(2 * np.log(2))
    zoom = np.abs(nu50 - c50) < 0.06
    ax_d.axhline(0, color=GREY, lw=0.5)
    ax_d.plot(nu50[zoom] - c50, lbl50[zoom] * 1e4, color=BLACK, lw=1.4, label="LBLRTM")
    ax_d.plot(nu50[zoom] - c50, tlx50[zoom] * 1e4, color=BLUE, lw=0.9, ls=(0, (3, 2)), label="tellurix (Im $w$)")
    ax_d.plot(nu50[zoom] - c50, data["voigt_offset_50"][zoom] * 1e4, color=ORANGE, lw=0.8, ls=(0, (1, 1)),
              label=r"exact Voigt $\times\,x/\alpha_V$")
    ax_d.set(xlim=(-0.06, 0.06), xlabel=r"$\nu-\nu_0$ (cm$^{-1}$), line at 4998.35 cm$^{-1}$",
             ylabel=r"coupled $-$ uncoupled ($10^{-4}$)")
    ax_d.legend(loc="upper left", fontsize=6.0)
    ax_d.text(0.98, 0.97, f"50 hPa, 220 K\n$\\gamma_L$={gamma50:.4f}, $\\alpha_D$={alfd50:.4f} cm$^{{-1}}$",
              transform=ax_d.transAxes, ha="right", va="top", fontsize=6.2)
    ax_d.set_title("(d) Dispersion shape when Doppler matters", loc="left", fontsize=8)

    for name in ("pdf", "png"):
        fig.savefig(OUTPUT / f"fig_method_coupling.{name}", dpi=200)

    numbers = {"lines_cm1": list(LINES), "g_all_zero": bool(np.all(data["g_table"] == 0)),
               "y_at_200_250_296_340K": data["y_table"].tolist()}
    for name in LAYERS:
        n = data[f"nu_{name}"]
        dl = data[f"lbl_cpl_{name}"] - data[f"lbl_nocpl_{name}"]
        dt = data[f"tlx_cpl_{name}"] - data[f"tlx_nocpl_{name}"]
        farm = (n > 5000.6) & (n < V2)
        numbers[name + "hPa"] = {
            "mixing_Yp": data[f"mixing_{name}"].tolist(),
            "dispersion_peak_lblrtm": float(np.abs(dl).max()), "dispersion_peak_tellurix": float(np.abs(dt).max()),
            "max_abs_difference_over_lblrtm_peak": float(np.abs(dt - dl).max() / np.abs(dl).max()),
            "far_wing_mean_lblrtm": float(dl[farm].mean()), "far_wing_mean_tellurix": float(dt[farm].mean()),
            "far_wing_mean_tellurix_with_lblrtm_co2_pedestal": float(data[f"tlx_co2ped_{name}"][farm].mean()),
            "integrated_dispersion_lblrtm": float(dl.sum() * np.mean(np.diff(n))),
            "integrated_dispersion_tellurix": float(dt.sum() * np.mean(np.diff(n))),
            "uncoupled_peak_ratio_tellurix_over_lblrtm": float(data[f"tlx_nocpl_{name}"].max()
                                                               / data[f"lbl_nocpl_{name}"].max()),
        }
    return numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "method_coupling.npz"
    if args.recompute or not cache.exists():
        np.savez(cache, **compute())
    data = dict(np.load(cache))
    numbers = plot(data)
    (OUTPUT / "fig_method_coupling.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
