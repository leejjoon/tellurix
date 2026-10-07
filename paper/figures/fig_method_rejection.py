#!/usr/bin/env python
"""Paper figure: LBLRTM's per-layer line rejection against tellurix's line list.

LBLRTM 12.17 drops a line from a layer when its peak-optical-depth proxy falls
below DPTMIN + DPTFAC x (the far-wing optical depth LBLF4 has already put at
that wavenumber): ``oprop.f90`` LNCOR1 for the near-line calculation (HIRAC1,
SPEAK = S N / alpha_V) and CONVF4 for the far-wing one (LBLF4, SPEAK = A3 x the
fourth function's peak). Defaults DPTMIN = 2e-4 and DPTFAC = 1e-3
(``lblrtm.f90``, record 1.3). Coupled lines are exempt. tellurix keeps every
line; ``select_significant_lines`` optionally drops the weakest lines whose
optical-depth *upper bounds* sum to a budget, which then bounds the error.

Setup: the 12-layer ERA5 profile ``data/profiles/gemini_2021_era5.csv`` (H2O
and CO2), 30 degrees zenith, 5000-5020 cm-1, the IATM=0 identical-layer input
of ``docs/lblrtm_tellurix_performance.json``; continua off and an uncoupled
(NOCPL) TAPE3, so only lines and rejection differ. LBLRTM is run with its
default thresholds and with both set to zero, for the bottom layer alone
(optical depth, TAPE10) and for the whole slant path (transmittance, TAPE12).
Which lines are rejected is reproduced in Python from the same rules and
checked against LBLRTM's own TAPE6 counts (written to the JSON sidecar).

Runs go to ``data/lblrtm/paper_method_rejection/`` (gitignored); results are
cached in ``paper/figures/cache/method_rejection.npz`` (``--recompute``). CPU:

    JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_method_rejection.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper/figures/cache"
OUTPUT = ROOT / "paper/figures/output"
RUNS = ROOT / "data/lblrtm/paper_method_rejection"
LBLRTM = ROOT / "data/lblrtm/LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
LNFL = ROOT / "data/lblrtm/LNFL/lnfl_v3.2_linux_gnu_sgl"
LINE_FILE = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_file/aer_v_3.9"
OPROP = ROOT / "data/lblrtm/LBLRTM/src/oprop.f90"
PROFILE = ROOT / "data/profiles/gemini_2021_era5.csv"

V1, V2 = 5000.0, 5020.0
ZENITH_DEG = 30.0
DPTMIN, DPTFAC = 2.0e-4, 1.0e-3          # lblrtm.f90: defaults when record 1.3 gives < 0
CUTOFF_CM1 = 25.0
RADCN2 = 1.4387752
BUDGET = 5.0e-4                           # per species: 1e-3 for H2O + CO2 together
RESOLVING_POWER = 45_000.0

BLACK, ORANGE, SKY, GREEN, BLUE, VERMILLION, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
GREY = "#999999"


# --------------------------------------------------------------------------
# LBLRTM

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


def profile_h2o_co2():
    from tellurix import load_atmosphere_csv
    profile = load_atmosphere_csv(PROFILE)
    return dataclasses.replace(profile, vmr={k: v for k, v in profile.vmr.items() if k in ("H2O", "CO2")})


def run_case(directory: Path, tape3: Path, rejection: bool, bottom_layer_only: bool) -> str:
    from tellurix import LBLRTMRunConfig, write_tape5

    directory.mkdir(parents=True, exist_ok=True)
    config = LBLRTMRunConfig(V1, V2, zenith_angle_deg=ZENITH_DEG, continuum_flag=0, user_layers=True,
                             line_rejection=rejection, description="paper rejection")
    write_tape5(directory / "TAPE5", profile_h2o_co2(), config)
    if bottom_layer_only:
        # Record 2.1 then each layer's two records, from the observer up.
        lines = (directory / "TAPE5").read_text().splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith(" 1 12"))
        lines[first] = " 1  1" + lines[first][5:]
        (directory / "TAPE5").write_text("\n".join(lines[: first + 3] + lines[-3:]) + "\n")
    link = directory / "TAPE3"
    link.unlink(missing_ok=True)
    link.symlink_to(tape3.resolve())
    for name in ("TAPE10", "TAPE11", "TAPE12", "TAPE6"):
        (directory / name).unlink(missing_ok=True)
    shutil.copy2(LBLRTM, directory / "lblrtm")
    subprocess.run(["./lblrtm"], cwd=directory, capture_output=True, check=False)
    return (directory / "TAPE6").read_text(errors="replace")


def tape6_counts(log: str) -> dict:
    hirac = re.findall(r"HIRAC1\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)", log)
    lblf4 = re.findall(r"LBLF4\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)", log)
    linf4 = re.findall(r"LINF4\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)\n", log)
    dv = re.findall(r"HIRAC1 \*  OUTPUT ON FILE\s+\d+\s+DV =\s+([\d.]+)\s+BOUNDF3\(CM-1\) =\s+([\d.]+)", log)
    return dict(hirac=[tuple(map(int, h)) for h in hirac], lblf4=[tuple(map(int, h)) for h in lblf4],
                linf4=[tuple(map(int, h)) for h in linf4], dv=[tuple(map(float, d)) for d in dv])


# --------------------------------------------------------------------------
# The rejection rules, transcribed from oprop.f90

def avrat_table() -> np.ndarray:
    text = OPROP.read_text(errors="replace")
    block = text[text.index("BLOCK DATA VOICON"): text.index("end block data VOICON")]
    values: list[float] = []
    for part in ("01", "51"):
        body = re.search(rf"DATA AV{part}\s*/(.*?)/", block, re.S).group(1)
        values += [float(v) for v in re.sub(r"[&\n,]", " ", body).split()]
    return np.asarray(values)


def voigt_width(alfl, alfd, avrat):
    zeta = alfl / (alfl + alfd)
    iz = (100 * zeta + 1.001).astype(int)
    frac = 100 * zeta - (iz - 1)
    return (avrat[iz - 1] + frac * (avrat[iz] - avrat[iz - 1])) * (alfl + alfd)


def emulate_layer(layer_text: str, dv: float, bound: float) -> dict:
    """Which of this layer's lines HIRAC1 (LNCOR1) and LBLF4 (CONVF4) keep."""

    import tellurix  # noqa: F401
    from tellurix import DataPaths
    from tellurix.aer import AERLineDatabase
    from tellurix.direct import SparseCoreDirect

    fields = layer_text.splitlines()
    pressure_hpa, temperature = float(fields[0][:15]), float(fields[0][15:25])
    amounts = [float(fields[1][15 * k: 15 * (k + 1)]) for k in range(8)]
    secant = 1.0 / np.cos(np.deg2rad(ZENITH_DEG))
    column = {"H2O": amounts[0] * secant, "CO2": amounts[1] * secant}
    total = sum(amounts) * secant
    xkt = temperature / RADCN2
    radfn = lambda v: v * (1 - np.exp(-v / xkt)) / (1 + np.exp(-v / xkt))   # noqa: E731
    avrat = avrat_table()

    paths = DataPaths.bootstrapped(ROOT)
    nu, strength, alfl, alfd, mol = [], [], [], [], []
    for species, molecule in (("H2O", 1), ("CO2", 2)):
        database = AERLineDatabase(paths.line_file(species, molecule), species, (V1, V2), margin_cm1=30.0)
        calculator = SparseCoreDirect(database, np.linspace(V1 - 40, V2 + 40, 16), pressure_shift=True)
        p_bar = pressure_hpa / 1000.0
        sigma, gamma, s = (np.asarray(a) for a in calculator._line_parameters(
            temperature, p_bar, p_bar * column[species] / total))
        nu.append(np.asarray(database.nu_lines) + np.asarray(calculator._line_shift(temperature, p_bar)))
        strength.append(s * column[species])
        alfl.append(gamma)
        alfd.append(sigma * np.sqrt(2 * np.log(2)))
        mol.append(np.full(s.size, molecule))
    order = np.argsort(np.concatenate(nu), kind="stable")
    nu, strength, alfl, alfd, mol = (np.concatenate(a)[order] for a in (nu, strength, alfl, alfd, mol))

    # LINF4 + SHRINK: lines within DVR4/2 merge, CO2 and the rest separately.
    dvr4 = 64 * dv
    v1r4, v2r4 = V1 - 2 * dvr4, V2 + 2 * dvr4
    in_f4 = (nu >= v1r4 - CUTOFF_CM1 - dvr4) & (nu <= v2r4 + CUTOFF_CM1 + 2 * dvr4)
    strength_nr = strength / radfn(nu)          # LINF4 keeps no radiation term
    groups, members = [], {0: [], 2: []}
    index = np.where(in_f4)[0]
    limit = nu[index[0]] + dvr4 / 2
    for k, i in enumerate(index):
        members[2 if mol[i] == 2 else 0].append(i)
        if k + 1 < index.size and nu[index[k + 1]] <= limit:
            continue
        limit = nu[i] + dvr4 / 2
        for key in (0, 2):
            group = members[key]
            if group:
                w = strength_nr[group]
                groups.append((np.dot(nu[group], w) / w.sum(), w.sum(), np.dot(alfl[group], w) / w.sum(),
                               np.dot(alfd[group], w) / w.sum(), key))
            members[key] = []

    # LBLF4 / CONVF4, in wavenumber order against the running R4.
    npt4 = int((v2r4 - v1r4) / dvr4 + 1.001)
    r4 = np.zeros(npt4)
    zsq = 64.0 ** 2
    a3, b3 = (1 + 2 * zsq) / (1 + zsq) ** 2, -1 / (1 + zsq) ** 2
    threshold_nr = DPTMIN / radfn(v2r4)
    kept4 = 0
    for v, s, al, ad, key in groups:
        av = voigt_width(np.array([al]), np.array([ad]), avrat)[0]
        sil, siv = s / np.pi * al, (al / av) * s / np.pi / av
        jj = min(max(int((v - v1r4) / dvr4 + 1.0), 1), npt4)
        if a3 * abs(siv) <= threshold_nr + DPTFAC * r4[jj - 1]:
            continue
        kept4 += 1
        xn = v - v1r4
        j = np.arange(max(int((xn - CUTOFF_CM1) / dvr4 + 2.0), 1), min(int((xn + CUTOFF_CM1) / dvr4 + 1.0), npt4) + 1)
        if j.size == 0:
            continue
        xm = (j - 1) * dvr4 - xn
        pedestal = sil / (al * al + CUTOFF_CM1 ** 2) * ((2 - xm * xm / CUTOFF_CM1 ** 2) if key == 2 else 1.0)
        zv = xm * xm / av ** 2
        r4[j - 1] += np.where(zv <= zsq, siv * (a3 + zv * b3), sil / (al * al + xm * xm)) - pedestal
    grid4 = v1r4 + dvr4 * np.arange(npt4)
    r4 *= radfn(grid4)                             # LBLF4, JRAD = 1

    # HIRAC1 / LNCOR1.
    alfv = np.clip(voigt_width(alfl, alfd, avrat), dv, 16 * dv)
    speak = strength / alfv
    jj = np.clip(((nu - v1r4) / dvr4 + 1.0).astype(int), 1, npt4)
    threshold = DPTMIN + DPTFAC * r4[jj - 1]
    read = (nu >= V1 - bound) & (nu <= V2 + bound)
    reaches = 64 * alfv + nu >= V1 - 32 * dv       # else skipped before the test (HWF3*ALFV+VNU < VFT)
    kept = read & reaches & (speak > threshold)
    return dict(e_nu=nu, e_mol=mol, e_strength=strength, e_speak=speak, e_threshold=threshold, e_read=read,
                e_reaches=reaches, e_kept=kept, e_grid4=grid4, e_r4=r4, e_kept4=kept4, e_groups=len(groups),
                e_in_f4=int(in_f4.sum()))


def compute() -> dict:
    from tellurix import DataPaths, run_lnfl
    from tellurix.aer import AERLineDatabase, select_significant_lines, line_optical_depth_bound
    from tellurix.lblrtm import lblrtm_line_shape_optical_depth

    tape3 = run_lnfl(RUNS / "lnfl", ("H2O", "CO2"), V1 - CUTOFF_CM1 - 6.0, V2 + CUTOFF_CM1 + 6.0,
                     LINE_FILE, LNFL, line_coupling=False)
    logs = {}
    for rejection in (True, False):
        for bottom in (True, False):
            name = f"{'default' if rejection else 'off'}_{'layer1' if bottom else 'path'}"
            logs[name] = run_case(RUNS / name, tape3, rejection, bottom)
    nu1, od_default = read_optical_depth(RUNS / "default_layer1/TAPE10")
    nu1b, od_off = read_optical_depth(RUNS / "off_layer1/TAPE10")
    assert np.allclose(nu1, nu1b)
    from tellurix.reference import read_tape12_single_precision
    path_default = read_tape12_single_precision(RUNS / "default_path/TAPE12")
    path_off = read_tape12_single_precision(RUNS / "off_path/TAPE12")
    assert np.allclose(path_default.wavenumber_cm1, path_off.wavenumber_cm1)

    counts_layer = tape6_counts(logs["default_layer1"])
    counts_path = tape6_counts(logs["default_path"])
    dv, bounf3 = counts_layer["dv"][0]
    tape5 = (RUNS / "default_layer1/TAPE5").read_text().splitlines()
    first = next(i for i, line in enumerate(tape5) if line.startswith(" 1  1"))
    emulated = emulate_layer("\n".join(tape5[first + 1: first + 3]), dv, 2 * bounf3)

    # tellurix's optional selection on the whole slant path, per species.
    profile = profile_h2o_co2()
    paths = DataPaths.bootstrapped(ROOT)
    secant = 1.0 / np.cos(np.deg2rad(ZENITH_DEG))
    nu_path = path_off.wavenumber_cm1
    discarded_tau = np.zeros_like(nu_path)
    selection = {}
    for species, molecule in (("H2O", 1), ("CO2", 2)):
        database = AERLineDatabase(paths.line_file(species, molecule), species, (V1, V2), margin_cm1=CUTOFF_CM1)
        kept = select_significant_lines(database, profile, species, BUDGET, maximum_column_scale=secant)
        keep_mask = np.isin(np.asarray(database.nu_lines), np.asarray(kept.nu_lines))
        bound = line_optical_depth_bound(database, profile, species, secant)
        selection[species] = dict(lines=int(keep_mask.size), discarded=int((~keep_mask).sum()),
                                  bound_sum_discarded=float(bound[~keep_mask].sum()))
        if (~keep_mask).any():
            _truncated, full = lblrtm_line_shape_optical_depth(database.restrict(~keep_mask), profile, nu_path)
            discarded_tau += full * secant
    return dict(layer_pressure_hpa=float(tape5[first + 1][:15]), nu1=nu1, od_default=od_default, od_off=od_off, nu_path=nu_path,
                t_default=path_default.transmission, t_off=path_off.transmission, discarded_tau=discarded_tau,
                dv=dv, bounf3=bounf3, counts_layer=json.dumps(counts_layer), counts_path=json.dumps(counts_path),
                selection=json.dumps(selection), **emulated)


def convolve(nu: np.ndarray, values: np.ndarray) -> np.ndarray:
    """A Gaussian of FWHM nu/R on LBLRTM's evenly spaced output grid."""

    step = float(np.mean(np.diff(nu)))
    sigma = float(np.mean(nu)) / RESOLVING_POWER / (2 * np.sqrt(2 * np.log(2))) / step
    offsets = np.arange(-int(6 * sigma) - 1, int(6 * sigma) + 2)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return np.convolve(values, kernel / kernel.sum(), "same")


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
    counts_layer = json.loads(str(data["counts_layer"]))
    counts_path = json.loads(str(data["counts_path"]))
    selection = json.loads(str(data["selection"]))
    nu, speak, threshold = data["e_nu"], data["e_speak"], data["e_threshold"]
    read, reaches, kept, mol = data["e_read"], data["e_reaches"], data["e_kept"], data["e_mol"]
    tested = read & reaches
    rejected = tested & ~kept

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.6), sharex=False)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.965, bottom=0.07, hspace=0.42)
    ax_a, ax_b, ax_c = axes
    xlim = (V1 - 8.5, V2 + 8.5)

    # (a) the HIRAC1 decision for the bottom layer
    ax_a.axvspan(V1, V2, color="#eeeeee", lw=0, zorder=0)
    from matplotlib.ticker import MultipleLocator
    for ax in (ax_b, ax_c):
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    for molecule, marker in ((1, "o"), (2, "^")):
        m = mol == molecule
        ax_a.plot(nu[tested & ~kept & m], speak[tested & ~kept & m], marker, ms=1.6, mfc=VERMILLION, mec="none",
                  alpha=0.55, ls="none")
        ax_a.plot(nu[kept & m], speak[kept & m], marker, ms=2.4, mfc=BLUE, mec="none", ls="none")
        ax_a.plot(nu[read & ~reaches & m], speak[read & ~reaches & m], marker, ms=1.6, mfc=GREY, mec="none",
                  alpha=0.5, ls="none")
    order = np.argsort(nu)
    grid4, r4 = data["e_grid4"], data["e_r4"]
    ax_a.step(grid4, DPTMIN + DPTFAC * r4, where="post", color=BLACK, lw=0.9)
    ax_a.set(yscale="log", xlim=xlim, ylim=(1e-9, 3e2), ylabel=r"SPEAK $=SN/\alpha_V$")
    ax_a.text(V1 - 1.0, 2.6e-4, r"DPTMIN + DPTFAC$\cdot$R4($\nu$)", fontsize=6.3, va="bottom", ha="right",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="none", mfc=BLUE, mec="none", ms=3, label="kept"),
               Line2D([], [], marker="o", ls="none", mfc=VERMILLION, mec="none", ms=3, label="rejected"),
               Line2D([], [], marker="o", ls="none", mfc=GREY, mec="none", ms=3,
                      label=r"read, but 64$\alpha_V$ short of the panel"),
               Line2D([], [], marker="o", ls="none", mfc="w", mec=BLACK, ms=3, label=r"H$_2$O"),
               Line2D([], [], marker="^", ls="none", mfc="w", mec=BLACK, ms=3, label=r"CO$_2$")]
    ax_a.legend(handles=handles, loc="upper left", ncol=5, fontsize=6.2, columnspacing=1.0, handletextpad=0.3)
    hirac_read, hirac_kept = counts_layer["hirac"][0]
    lblf4_read, lblf4_reported = counts_layer["lblf4"][0]
    ax_a.text(0.995, 0.035, (f"bottom layer ({PRESSURE_TEXT(data)}): HIRAC1 keeps {hirac_kept} of {hirac_read} lines "
                            f"(emulated {int(kept.sum())}); LBLF4 keeps {lblf4_reported // 2} of "
                            f"{lblf4_read} merged lines (emulated {int(data['e_kept4'])} of {int(data['e_groups'])})"),
              transform=ax_a.transAxes, ha="right", va="bottom", fontsize=6.2,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax_a.set_title("(a) LBLRTM line rejection in one layer (LNCOR1 / CONVF4)", loc="left", fontsize=8)

    # (b) the optical depth they carry
    nu1, od_default, od_off = data["nu1"], data["od_default"], data["od_off"]
    lost = od_off - od_default
    ax_b.plot(nu1, od_off, color=GREY, lw=0.5, label="all lines (DPTMIN = DPTFAC = 0)")
    ax_b.plot(nu1, np.where(lost > 0, lost, np.nan), color=VERMILLION, lw=0.7,
              label="lost to rejection (LBLRTM: off $-$ default)")
    ax_b.axhline(DPTMIN, color=BLACK, lw=0.5, ls=":")
    ax_b.text(V1 + 0.05, DPTMIN * 0.55, "DPTMIN", fontsize=6, va="top")
    ax_b.set(yscale="log", xlim=(V1, V2), ylim=(1e-5, 30), ylabel="layer optical depth")
    ax_b.legend(loc="upper left", ncol=2, fontsize=6.3)
    window = (nu1 >= V1) & (nu1 <= V2)
    ax_b.text(0.995, 0.04, f"5000$-$5020 cm$^{{-1}}$: {lost[window].sum() / od_off[window].sum():.2%} of the layer's "
                           f"line optical depth; mean {lost[window].mean():.1e}, max {lost[window].max():.1e}",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6.2)
    ax_b.set_title("(b) Optical depth of the rejected lines, same layer", loc="left", fontsize=8)

    # (c) the slant-path transmission
    nup = data["nu_path"]
    t_off, t_default = data["t_off"], data["t_default"]
    d_lbl = t_default - t_off
    t_select = t_off * np.exp(data["discarded_tau"])
    d_sel = t_select - t_off
    w = (nup >= V1) & (nup <= V2)
    absolute = lambda v: np.where(np.abs(v) > 0, np.abs(v), np.nan)   # noqa: E731
    ax_c.plot(nup[w], absolute(d_lbl[w]), color=VERMILLION, lw=0.35, alpha=0.4)
    ax_c.plot(nup[w], absolute(convolve(nup, d_lbl)[w]), color=VERMILLION, lw=1.1,
              label="LBLRTM default rejection (thin: monochromatic; thick: R=45,000)")
    ax_c.plot(nup[w], absolute(d_sel[w]), color=BLUE, lw=0.35, alpha=0.4)
    ax_c.plot(nup[w], absolute(convolve(nup, d_sel)[w]), color=BLUE, lw=1.1,
              label=f"tellurix select_significant_lines (optional), budget {2 * BUDGET:g}")
    ax_c.axhline(2 * BUDGET, color=BLUE, lw=0.7, ls="--")
    ax_c.text(V2 - 0.05, 2 * BUDGET * 0.8, "guaranteed bound of the selection", fontsize=6.2, ha="right",
              color=BLUE, va="top")
    ax_c.text(V2 - 0.05, 2e-10, "tellurix default keeps every line: $\\Delta T = 0$", fontsize=6.4,
              ha="right", va="bottom")
    ax_c.set(xlim=(V1, V2), yscale="log", ylim=(1e-10, 3e-2), ylabel=r"$|\Delta T|$",
             xlabel=r"wavenumber (cm$^{-1}$)")
    ax_c.legend(loc="lower left", fontsize=6.3, ncol=1)
    ax_c.set_title("(c) Transmission error, 12-layer ERA5 slant path at 30$^\\circ$ zenith", loc="left", fontsize=8)
    for ax in (ax_a, ax_b):
        ax.set_xlabel(r"wavenumber (cm$^{-1}$)")

    for name in ("pdf", "png"):
        fig.savefig(OUTPUT / f"fig_method_rejection.{name}", dpi=200)

    tau_off = -np.log(np.clip(t_off, 1e-30, None))
    tau_default = -np.log(np.clip(t_default, 1e-30, None))
    usable = w & (t_off > 0.05)
    per_layer = [dict(layer=i + 1, dv_cm1=dv[0], hirac_read=h[0], hirac_kept=h[1],
                      hirac_rejected_fraction_of_read=1 - h[1] / h[0], lblf4_merged=l[0],
                      lblf4_after_reject_as_printed=l[1])
                 for i, (h, l, dv) in enumerate(zip(counts_path["hirac"], counts_path["lblf4"], counts_path["dv"]))]
    return {
        "layer1": {"pressure_hpa": PRESSURE_HPA(data), "lblrtm_dv_cm1": float(data["dv"]),
                   "tape6_hirac_read_kept": counts_layer["hirac"][0],
                   "emulated_hirac_read_tested_kept": [int(read.sum()), int(tested.sum()), int(kept.sum())],
                   "hirac_rejected_fraction_of_tested": float(rejected.sum() / tested.sum()),
                   "tape6_linf4_read_after_shrink": counts_layer["linf4"][0],
                   "emulated_linf4_read_after_shrink": [int(data["e_in_f4"]), int(data["e_groups"])],
                   "tape6_lblf4_after_reject_printed": counts_layer["lblf4"][0],
                   "emulated_lblf4_kept_per_call": int(data["e_kept4"]),
                   "optical_depth_lost_fraction": float(lost[window].sum() / od_off[window].sum()),
                   "optical_depth_lost_mean": float(lost[window].mean()),
                   "optical_depth_lost_min_max": [float(lost[window].min()), float(lost[window].max())]},
        "path": {"per_layer_tape6": per_layer,
                 "optical_depth_lost_fraction_T>0.05": float((tau_off - tau_default)[usable].sum() / tau_off[usable].sum()),
                 "dT_monochromatic_max_abs": float(np.abs(d_lbl[w]).max()),
                 "dT_monochromatic_mean": float(d_lbl[w].mean()),
                 "dT_R45000_max_abs": float(np.abs(convolve(nup, d_lbl)[w]).max()),
                 "dT_R45000_mean": float(convolve(nup, d_lbl)[w].mean())},
        "tellurix_select_significant_lines": {
            "budget_per_species": BUDGET, "selection": selection,
            "dT_monochromatic_max_abs": float(np.abs(d_sel[w]).max()),
            "dT_R45000_max_abs": float(np.abs(convolve(nup, d_sel)[w]).max()),
            "discarded_tau_max": float(data["discarded_tau"][w].max())},
    }


def PRESSURE_HPA(data) -> float:
    return float(data["layer_pressure_hpa"])


def PRESSURE_TEXT(data) -> str:
    return f"{PRESSURE_HPA(data):.0f} hPa"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "method_rejection.npz"
    if args.recompute or not cache.exists():
        np.savez(cache, **compute())
    data = dict(np.load(cache))
    numbers = plot(data)
    (OUTPUT / "fig_method_rejection.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
