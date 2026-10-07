#!/usr/bin/env python
"""Paper figure: how LBLRTM 12.17 and tellurix evaluate one line.

One real AER 3.9 H2O line (5004.585 cm-1) in one homogeneous layer (800 hPa,
280 K, 1% H2O) is put through both codes:

* LBLRTM itself. LNFL builds a one-line TAPE3 and LBLRTM computes the layer's
  optical depth (TAPE10, on its own DV grid) with line rejection off, so the
  profile is exactly what HIRAC1 + LBLF4 make of the line.
* A line-for-line Python transcription of the same arithmetic (``oprop.f90``:
  SHAPEL/SHAPEG/VERFN tables, the VOICON zeta tables read from the source, the
  CNVFNV sub-function loops, PANEL's 4-point Lagrange interpolation, CONVF4's
  fourth function with its pedestal and XINT's cubic) to split the LBLRTM
  profile into the parts each sub-function contributes. It is checked against
  the LBLRTM run and the agreement is written to the JSON sidecar.
* tellurix's ``SparseCoreDirect`` on the R=45,000 x 4 constant-velocity grid,
  compared with ``scipy.special.wofz`` (the exact Voigt).

LBLRTM and LNFL are run in ``data/lblrtm/paper_method_lineshape/`` (gitignored)
and the results cached in ``paper/figures/cache/method_lineshape.npz``
(``--recompute`` redoes them). CPU only:

    JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_method_lineshape.py

Writes ``fig_method_lineshape.{pdf,png}`` and ``fig_method_lineshape.json`` to
``paper/figures/output/``.
"""

from __future__ import annotations

import argparse
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
RUNS = ROOT / "data/lblrtm/paper_method_lineshape"
LBLRTM = ROOT / "data/lblrtm/LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
LNFL = ROOT / "data/lblrtm/LNFL/lnfl_v3.2_linux_gnu_sgl"
OPROP = ROOT / "data/lblrtm/LBLRTM/src/oprop.f90"
H2O_FILE = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule/01_H2O/01_H2O"

LINE_POSITION = "5004.584965"
PRESSURE_HPA, TEMPERATURE_K = 800.0, 280.0
H2O_VMR, AIR_COLUMN = 0.01, 2.0e24
V1, V2 = 4975.0, 5035.0           # +-30 cm-1 around the line: past the 25 cm-1 cutoff
P0_HPA, T0_K = 1013.25, 296.0
RADCN2 = 1.4387752
CUTOFF_CM1 = 25.0

# Okabe-Ito
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILLION, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
GREY = "#888888"


# --------------------------------------------------------------------------
# LBLRTM and LNFL

def fortran_records(path: Path) -> list[bytes]:
    data, offset, records = path.read_bytes(), 0, []
    while offset + 4 <= len(data):
        size = struct.unpack_from("=i", data, offset)[0]
        records.append(data[offset + 4: offset + 4 + size])
        offset += 8 + size
    return records


def read_optical_depth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A single-precision layer optical-depth file (TAPE10): header, then panels."""

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


def run_lnfl(directory: Path, records: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "TAPE1").write_text("".join(r.rstrip("\n").ljust(100) + "\n" for r in records))
    (directory / "TAPE5").write_text(f"paper one-line H2O\n{4950.0:10.3f}{5060.0:10.3f}\n"
                                     f"{'1'.ljust(47, '0')}    \n")
    for name in ("TAPE3", "TAPE6", "TAPE10"):
        (directory / name).unlink(missing_ok=True)
    shutil.copy2(LNFL, directory / "lnfl")
    subprocess.run(["./lnfl"], cwd=directory, capture_output=True, check=False)
    if not (directory / "TAPE3").exists():
        raise RuntimeError("LNFL failed")
    return directory / "TAPE3"


def run_lblrtm(directory: Path, tape3: Path, h2o: float, air: float) -> tuple[np.ndarray, np.ndarray, float]:
    """One homogeneous layer, IATM=0, lines only, DPTMIN = DPTFAC = 0 (no rejection)."""

    directory.mkdir(parents=True, exist_ok=True)
    columns = "".join(f"{c:15.7E}" for c in (h2o, 0, 0, 0, 0, 0, 0)) + f"{air - h2o:15.7E}"
    p, t = PRESSURE_HPA, TEMPERATURE_K
    (directory / "TAPE5").write_text("\n".join([
        "$paper one-line H2O layer",
        "".join(f"{v:5d}" for v in (1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        f"{V1:10.3f}{V2:10.3f}{4.0:10.3f}{0.0:10.3f}{0.04:10.3f}{36.0:10.3f}{0.0:10.3f}{0.0:10.3f}"
        f"{0:5d}{0.0:15.3f}{0:5d}",
        f"{t:10.3f}{1.0:10.3f}{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}    s",
        f" {1:1d}{1:3d}{7:5d}{1.0:10.6f}{'':20s}{0.0:8.3f}{'':4s}{1.0:8.3f}{'':5s}{0.0:8.3f}",
        f"{p:15.7E}{t:10.4f}{0.0:10.4f}{'':3s}{0:2d} {0.0:7.2f}{p:8.3f}{t:7.2f}{1.0:7.2f}{p:8.3f}{t:7.2f}",
        columns, "-1.0", "-1.0", "%"]) + "\n")
    link = directory / "TAPE3"
    link.unlink(missing_ok=True)
    link.symlink_to(tape3.resolve())
    for name in ("TAPE10", "TAPE11", "TAPE12", "TAPE6"):
        (directory / name).unlink(missing_ok=True)
    shutil.copy2(LBLRTM, directory / "lblrtm")
    subprocess.run(["./lblrtm"], cwd=directory, capture_output=True, check=False)
    log = (directory / "TAPE6").read_text(errors="replace")
    dv = float(re.search(r"HIRAC1 \*  OUTPUT ON FILE\s+\d+\s+DV =\s+([\d.]+)", log).group(1))
    nu, od = read_optical_depth(directory / "TAPE10")
    return nu, od, dv


# --------------------------------------------------------------------------
# HIRAC1 / LBLF4 for one line, transcribed from oprop.f90

NFPTS = 2001                 # lblparams.f90
NFMX = int(1.3 * NFPTS)
HWF = (4.0, 16.0, 64.0)      # HIRAC1: HWFF1..3, half widths in Voigt widths
DXF = (0.002, 0.008, 0.032)  # HIRAC1: DXFF1..3


def voicon_tables() -> dict[str, np.ndarray]:
    """AVRAT, CGAUSS, CF1-3 and CER as functions of zeta (BLOCK DATA VOICON)."""

    text = OPROP.read_text(errors="replace")
    block = text[text.index("BLOCK DATA VOICON"): text.index("end block data VOICON")]
    tables = {}
    for name in ("AV", "CG", "CFA", "CFB", "CFC", "CER"):
        values: list[float] = []
        for part in ("01", "51"):
            body = re.search(rf"DATA {name}{part}\s*/(.*?)/", block, re.S).group(1)
            values += [float(v) for v in re.sub(r"[&\n,]", " ", body).split()]
        tables[name] = np.asarray(values)
    return tables


def shape_tables():
    """SHAPEL's F1, F2, F3; SHAPEG's FG; VERFN's XVER."""

    recpi = 1.0 / np.pi
    lorentz = lambda x2: 1.0 / (1.0 + x2)                       # noqa: E731
    a = lambda z: (1 + 2 * z * z) / (1 + z * z) ** 2              # noqa: E731
    b = lambda z: -1.0 / (1 + z * z) ** 2                         # noqa: E731
    quad = [lambda x2, z=z: a(z) + b(z) * x2 for z in HWF]
    tables = [np.zeros(NFMX + 2) for _ in range(3)]
    for k in range(3):
        x = np.arange(NFPTS) * DXF[k]
        x2 = x * x
        if k == 0:
            tables[0][:NFPTS] = recpi * (lorentz(x2) - quad[0](x2))
        else:
            inner = x <= HWF[k - 1] + 1e-9
            tables[k][:NFPTS] = recpi * np.where(inner, quad[k - 1](x2) - quad[k](x2), lorentz(x2) - quad[k](x2))
        if k < 2:
            tables[k][NFPTS - 1] = 0.0          # F1(NX1) = F2(NX2) = 0; F3 is not zeroed
    gauss = np.zeros(NFMX + 2)
    gauss[:NFPTS] = np.sqrt(np.log(2) / np.pi) * np.exp(-np.log(2) * (np.arange(NFPTS) * DXF[0]) ** 2)
    gauss[NFPTS - 1] = 0.0
    cexp, ce0, ce2, ce4 = 0.45, 1.0, -0.20737285249, -0.00872684335747
    z2 = (np.arange(NFMX) * DXF[0]) ** 2
    ae2 = 2 * cexp
    xver = np.zeros(NFMX + 2)
    xver[:NFMX] = ((ce0 + ce2 * ae2 * z2 + ce4 * ae2 * ae2 / 3 * z2 * z2) / (ce0 + ce2 + ce4)
                   * np.sqrt(cexp / np.pi) * np.exp(-cexp * z2))
    return tables, gauss, xver


def lagrange4(coarse: np.ndarray) -> np.ndarray:
    """PANEL: a grid 4x coarser onto the finer one, 4-point Lagrange at 1/4, 1/2, 3/4."""

    x00, x01, x02, x03, x10, x11 = -7 / 128, 105 / 128, 35 / 128, -5 / 128, -1 / 16, 9 / 16
    n = coarse.size
    pad = np.concatenate([[0.0], coarse, [0.0, 0.0]])
    rm, r0, r1, r2 = pad[0:n], pad[1:n + 1], pad[2:n + 2], pad[3:n + 3]
    fine = np.zeros(4 * n)
    fine[0::4] = r0
    fine[1::4] = x00 * rm + x01 * r0 + x02 * r1 + x03 * r2
    fine[2::4] = x10 * (rm + r2) + x11 * (r0 + r1)
    fine[3::4] = x03 * rm + x02 * r0 + x01 * r1 + x00 * r2
    return fine


def xint(v1a, dva, values, vft, dvr3, n3):
    """XINT: the R4 array onto R3, four-point cubic with ONEPL = 1.001."""

    vi = vft + dvr3 * np.arange(n3)
    j = np.floor((vi - v1a) / dva + 1.001).astype(int)
    p = (vi - (v1a + dva * (j - 1))) / dva
    c = (3 - 2 * p) * p * p
    b = 0.5 * p * (1 - p)
    b1, b2 = b * (1 - p), b * p
    padded = np.concatenate([[0.0, 0.0], values, [0.0, 0.0, 0.0]])
    get = lambda k: padded[np.clip(k + 1, 0, padded.size - 1)]   # noqa: E731
    out = -get(j - 1) * b1 + get(j) * (1 - c + b2) + get(j + 1) * (c + b1) - get(j + 2) * b2
    return np.where((j - 1 >= 1) & (j + 2 <= values.size), out, 0.0)


def lblrtm_line(nu0, alfl, alfad, dv, v1, v2, radfn):
    """One line of unit strength through CNVFNV + PANEL and LBLF4 (CONVF4 + XINT).

    Returns the R1 grid, the total and its parts. ``radfn`` puts LBLF4's
    radiation term at each R4 point, as LBLF4 does when JRAD=1, against HIRAC1's
    at the line centre (LNCOR1).
    """

    tab = voicon_tables()
    (f1, f2, f3), fg, xver = shape_tables()
    zeta = alfl / (alfl + alfad)
    iz = int(100 * zeta + 1.001)
    frac = 100 * zeta - (iz - 1)
    interp = lambda t: t[iz - 1] + frac * (t[iz] - t[iz - 1])     # noqa: E731
    alfv_raw = interp(tab["AV"]) * (alfl + alfad)
    alfv = min(max(alfv_raw, dv), 16 * dv)       # LNCOR1: DV <= ALFV <= ALFMAX
    cf1, cf2, cf3, cg, cer = (interp(tab[k]) for k in ("CFA", "CFB", "CFC", "CG", "CER"))

    vft = v1 - 32 * dv                           # NSHIFT = 32
    n1 = ((int(round((v2 - vft) / dv)) + 200) // 64 + 2) * 64
    n2, n3 = n1 // 4, n1 // 16
    r1 = np.zeros(n1)
    r2, r3 = np.zeros(n2), np.zeros(n3)
    depth = 1.0 / alfv
    zint = (nu0 - vft) / dv
    zslope = dv / (alfv * DXF[0])
    half = (HWF[0] / DXF[0]) / zslope
    j1 = np.arange(int(zint - half + 1.5), int(zint + half + 1.5) + 1)
    shift = 0.5 if zint >= 0 else -0.5
    j2 = j1 - int(zint * 0.75 + shift)
    j3 = j1 - int(zint * (15 / 16) + shift)
    i1 = (np.abs((j1 - 1 - zint) * zslope) + 1.5).astype(int)
    i2 = (np.abs((j2 - 1 - zint / 4) * zslope) + 1.5).astype(int)
    i3 = (np.abs((j3 - 1 - zint / 16) * zslope) + 1.5).astype(int)
    np.add.at(r1, j1 - 1, depth * (cf1 * f1[i1 - 1] + cg * fg[i1 - 1] + cer * xver[i1 - 1]))
    np.add.at(r2, j2 - 1, depth * cf2 * f2[i2 - 1])
    np.add.at(r3, j3 - 1, depth * cf3 * f3[i3 - 1])

    dvr4 = 64 * dv                               # OPDPTH: DVR4 = 64 ALFAV / SAMPLE
    v1r4 = v1 - 2 * dvr4
    n4 = int((v2 + 2 * dvr4 - v1r4) / dvr4 + 1.001)
    xnui = nu0 - v1r4
    jj = np.arange(max(int((xnui - CUTOFF_CM1) / dvr4 + 2.0), 1), min(int((xnui + CUTOFF_CM1) / dvr4 + 1.0), n4) + 1)
    xm = (jj - 1) * dvr4 - xnui
    sil = alfl / np.pi
    siv = (alfl / alfv_raw) / np.pi / alfv_raw
    zsq = 64.0 ** 2
    a3, b3 = (1 + 2 * zsq) / (1 + zsq) ** 2, -1 / (1 + zsq) ** 2
    pedestal = sil / (alfl ** 2 + CUTOFF_CM1 ** 2)
    zv = xm * xm / alfv_raw ** 2
    r4 = np.zeros(n4)
    r4[jj - 1] = np.where(zv <= zsq, siv * (a3 + zv * b3), sil / (alfl ** 2 + xm * xm)) - pedestal
    grid4 = v1r4 + dvr4 * np.arange(n4)
    r4 *= radfn(grid4) / radfn(nu0)
    r3_from_r4 = xint(v1r4, dvr4, r4, vft, 16 * dv, n3)

    parts = {
        "R1": r1,
        "R2": lagrange4(r2)[:n1],
        "R3": lagrange4(lagrange4(r3)[:n2])[:n1],
        "R4": lagrange4(lagrange4(r3_from_r4)[:n2])[:n1],
    }
    grid = vft + dv * np.arange(n1)
    keep = (grid >= v1 - 1e-9) & (grid <= v2 + 1e-9)
    info = dict(alfv=alfv, zeta=zeta, cf1=cf1, cf2=cf2, cf3=cf3, cgauss=cg, cer=cer, dvr4=dvr4,
                pedestal=pedestal, a3=a3)
    return grid[keep], {k: v[keep] for k, v in parts.items()}, info


# --------------------------------------------------------------------------

def compute() -> dict:
    record = next(line for line in H2O_FILE.open(errors="replace") if line[3:15].strip() == LINE_POSITION)
    tape3 = run_lnfl(RUNS / "lnfl", [record])
    h2o = H2O_VMR * AIR_COLUMN
    nu, od, dv = run_lblrtm(RUNS / "lblrtm", tape3, h2o, AIR_COLUMN)

    # The line as LNCOR1 shapes it. Width and shift agree with tellurix's to
    # 1e-8 (both are the HITRAN expressions); the Doppler width uses LBLRTM's
    # FAD (MOLEC) with the H2-16O mass.
    gamma_air, gamma_self = float(record[35:40]), float(record[40:45])
    n_air, delta_air = float(record[55:59]), float(record[59:67])
    rhorat = (PRESSURE_HPA / P0_HPA) * (T0_K / TEMPERATURE_K)
    tcor = (TEMPERATURE_K / T0_K) ** (1 - n_air)          # LNFL stores 1 - n (lnfl.f:1766)
    alfl = gamma_air * tcor * (rhorat - rhorat * H2O_VMR) + gamma_self * tcor * rhorat * H2O_VMR
    nu0 = float(LINE_POSITION) + rhorat * delta_air
    fad = np.log(2) * 2 * 6.02214076e23 * 1.380649e-16 / 2.99792458e10 ** 2
    alfad = nu0 * np.sqrt(fad * TEMPERATURE_K / 18.010565)
    xkt = TEMPERATURE_K / RADCN2
    radfn = lambda v: v * (1 - np.exp(-v / xkt)) / (1 + np.exp(-v / xkt))   # noqa: E731
    grid, parts, info = lblrtm_line(nu0, alfl, alfad, dv, V1, V2, radfn)
    model = sum(parts.values())
    on_run = np.interp(nu, grid, model)
    strength_lblrtm = float(np.dot(on_run, od) / np.dot(on_run, on_run))   # S(T) N in cm-1

    import tellurix  # noqa: F401  (float64 before ExoJAX)
    from tellurix import DataPaths
    from tellurix.aer import AERLineDatabase
    from tellurix.direct import SparseCoreDirect
    from tellurix.model import constant_velocity_grid

    paths = DataPaths.bootstrapped(ROOT)
    database = AERLineDatabase(paths.line_file("H2O", 1), "H2O", (5004.5, 5004.7), margin_cm1=0.0)
    database = database.restrict(np.abs(np.asarray(database.nu_lines) - float(LINE_POSITION)) < 1e-6)
    tgrid = constant_velocity_grid(1e7 / V2, 1e7 / V1, 45_000.0, 4.0, margin_cm1=0.0)
    calculator = SparseCoreDirect(database, tgrid, pressure_shift=True)
    p_bar = PRESSURE_HPA / 1000.0
    cross = np.asarray(calculator.xsvector(TEMPERATURE_K, p_bar, p_bar * H2O_VMR))
    sigma, gamma, strength = (float(np.asarray(v)[0]) for v in
                              calculator._line_parameters(TEMPERATURE_K, p_bar, p_bar * H2O_VMR))
    tnu0 = float(database.nu_lines[0]) + float(np.asarray(calculator._line_shift(TEMPERATURE_K, p_bar))[0])
    core_half_width = float(np.asarray(calculator._core_half_width)[0])
    return dict(nu=nu, od=od, dv=dv, grid=grid, r1=parts["R1"], r2=parts["R2"], r3=parts["R3"], r4=parts["R4"],
                nu0=nu0, alfl=alfl, alfad=alfad, strength_lblrtm=strength_lblrtm,
                tgrid=tgrid, tprofile=cross / strength, tnu0=tnu0, tsigma=sigma, tgamma=gamma,
                tstrength_column=strength * h2o, core_half_width=core_half_width,
                **{f"info_{k}": v for k, v in info.items()})


def voigt(x, alfd_hwhm, alfl):
    from scipy.special import wofz
    sigma = alfd_hwhm / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j * alfl) / (np.sqrt(2) * sigma))) / (sigma * np.sqrt(2 * np.pi))


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "lines.linewidth": 1.0, "pdf.fonttype": 42, "savefig.dpi": 200, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    })


def plot(data: dict) -> dict:
    import matplotlib.pyplot as plt

    style()
    nu0, alfl, alfad, dv = (float(data[k]) for k in ("nu0", "alfl", "alfad", "dv"))
    alfv = float(data["info_alfv"])
    nu, od = data["nu"], data["od"]
    lbl = od / float(data["strength_lblrtm"])
    grid = data["grid"]
    parts = {k: data[k] for k in ("r1", "r2", "r3", "r4")}
    model = sum(parts.values())
    x_lbl = nu - nu0
    exact_lbl = voigt(x_lbl, alfad, alfl)
    peak = voigt(np.zeros(1), alfad, alfl)[0]

    tgrid, tprof = data["tgrid"], data["tprofile"]
    tnu0, tsig, tgam = (float(data[k]) for k in ("tnu0", "tsigma", "tgamma"))
    x_t = tgrid - tnu0
    from scipy.special import wofz
    exact_t = np.real(wofz((x_t + 1j * tgam) / (np.sqrt(2) * tsig))) / (tsig * np.sqrt(2 * np.pi))
    a_t = tgam / (np.sqrt(2) * tsig)
    alg916 = np.sqrt(max(111.0 - a_t * a_t, 0.0)) * np.sqrt(2) * tsig
    core = float(data["core_half_width"])

    bounds = [4 * alfv, 16 * alfv, 64 * alfv, CUTOFF_CM1]
    xmin, xmax = 2e-3, 40.0
    region_colors = [BLUE, SKY, GREEN, ORANGE]
    region_labels = [f"R1 (F1, FG, XVER), DV={dv:.4f}",
                     f"R2: F2, 4DV={4 * dv:.3f}",
                     f"R3: F3, 16DV={16 * dv:.3f}",
                     f"R4: F4 (LBLF4), 64DV={64 * dv:.3f}"]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.085, hspace=0.42, wspace=0.27)
    pos = x_lbl > 0

    # (a) LBLRTM's sub-functions
    edges = [xmin] + bounds
    for k in range(4):
        ax_a.axvspan(edges[k], edges[k + 1], color=region_colors[k], alpha=0.10, lw=0)
    gpos = grid - nu0 > 0
    xg = grid[gpos] - nu0
    for key, color, label in zip(("r1", "r2", "r3", "r4"), region_colors, region_labels):
        value = parts[key][gpos]
        ax_a.plot(xg, np.where(value > 0, value, np.nan), color=color, lw=1.1, label=label)
        ax_a.plot(xg, np.where(value < 0, -value, np.nan), color=color, lw=0.8, ls=":")
    sel = pos & (np.mod(np.arange(nu.size), 1) == 0)
    xs = x_lbl[sel]
    # Thin the LBLRTM samples to an even spacing in log(offset).
    keep = np.unique(np.searchsorted(xs, np.geomspace(xmin, 30, 140)).clip(0, xs.size - 1))
    ax_a.plot(xs[keep], np.where(lbl[sel][keep] > 0, lbl[sel][keep], np.nan), "o", ms=2.0, mfc="none",
              mec=BLACK, mew=0.5, label="LBLRTM 12.17 output (TAPE10)")
    ax_a.plot(xg, voigt(xg, alfad, alfl), color=GREY, lw=0.8, ls="--", label="exact Voigt")
    ped = float(data["info_pedestal"])
    ax_a.axhline(ped, color=VERMILLION, lw=0.6, ls="-.")
    ax_a.text(5.2, ped * 0.62, "pedestal\n$\\alpha_L/\\pi(\\alpha_L^2+25^2)$", fontsize=5.6, color=VERMILLION, va="top")
    for b, lab in zip(bounds, ["4$\\alpha_V$", "16$\\alpha_V$", "64$\\alpha_V$", "25 cm$^{-1}$"]):
        ax_a.axvline(b, color=GREY, lw=0.4)
        ax_a.text(b, 2e1, lab, fontsize=6, rotation=90, ha="right", va="top", color="#444444")
    ax_a.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(1e-9, 3e1),
             xlabel=r"$\nu-\nu_0$ (cm$^{-1}$)", ylabel=r"$k(\nu)\,/\,S N$ (cm)")
    ax_a.set_title("(a) LBLRTM: HIRAC1 sub-functions + LBLF4", loc="left", fontsize=8)
    ax_a.legend(loc="lower left", fontsize=5.6, handlelength=1.6, borderaxespad=0.2, frameon=True,
                facecolor="white", edgecolor="none", framealpha=0.92)

    # (b) tellurix
    tpos = x_t > 0
    ax_b.axvspan(xmin, alg916, color=PURPLE, alpha=0.15, lw=0)
    ax_b.axvspan(alg916, core, color=PURPLE, alpha=0.06, lw=0)
    ax_b.axvspan(core, xmax, color=YELLOW, alpha=0.15, lw=0)
    xf = np.geomspace(xmin, xmax, 600)
    ax_b.plot(xf, voigt(xf, tsig * np.sqrt(2 * np.log(2)), tgam), color=GREY, lw=0.8, ls="--", label="exact Voigt")
    ax_b.plot(x_t[tpos], tprof[tpos], "s", ms=2.0, mfc="none", mec=BLUE, mew=0.5,
              label=f"tellurix, R=45,000$\\times$4 grid ($\\delta\\nu$={np.diff(tgrid).mean():.4f})")
    ax_b.text(1.2 * xmin, 2e-5, "Algorithm 916\n$x^2+a^2<111$", fontsize=6, color=PURPLE)
    ax_b.text(alg916 * 1.05, 4e-6, "core\nlist:\nhjert", fontsize=5.6, color=PURPLE)
    ax_b.text(core * 1.3, 2e-6, "asymptotic $w(z)$ series only,\nevery grid point, no cutoff\n"
              "(lines within 25 cm$^{-1}$ of the window)", fontsize=6, color="#7a6a00")
    ax_b.axvline(CUTOFF_CM1, color=GREY, lw=0.4)
    ax_b.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(1e-9, 3e1),
             xlabel=r"$\nu-\nu_0$ (cm$^{-1}$)", ylabel=r"$k(\nu)\,/\,S N$ (cm)")
    ax_b.set_title("(b) tellurix: SparseCoreDirect (ExoJAX hjert)", loc="left", fontsize=8)
    ax_b.legend(loc="lower left", fontsize=5.8, borderaxespad=0.2)

    # (c) the core, linear
    zoom = 0.45
    zl = np.abs(x_lbl) < zoom
    ax_c.plot(x_lbl[zl], exact_lbl[zl], color=GREY, lw=0.8, ls="--", label="exact Voigt")
    ax_c.plot(x_lbl[zl], lbl[zl], ".", ms=1.6, color=BLACK, label=f"LBLRTM, DV={dv:.4f} cm$^{{-1}}$")
    zt = np.abs(x_t) < zoom
    ax_c.plot(x_t[zt], tprof[zt], "s", ms=3.0, mfc="none", mec=BLUE, mew=0.7,
              label=f"tellurix, {np.diff(tgrid).mean():.4f} cm$^{{-1}}$")
    ax_c.axvspan(-alg916, alg916, color=PURPLE, alpha=0.10, lw=0)
    for s in (-1, 1):
        ax_c.axvline(s * 4 * alfv, color=BLUE, lw=0.5, ls=":")
    ax_c.text(4 * alfv, 0.97 * peak, " 4$\\alpha_V$", fontsize=6, color=BLUE, va="top")
    ax_c.set(xlim=(-zoom, zoom), ylim=(0, 1.08 * peak), xlabel=r"$\nu-\nu_0$ (cm$^{-1}$)",
             ylabel=r"$k(\nu)\,/\,S N$ (cm)")
    ax_c.set_title("(c) Line core: both codes sample the same Voigt", loc="left", fontsize=8)
    ax_c.legend(loc="upper left", fontsize=6)
    ax_c.text(0.98, 0.55, (f"H$_2$O {LINE_POSITION} cm$^{{-1}}$\n{PRESSURE_HPA / 1000:.1f} bar, "
                           f"{TEMPERATURE_K:.0f} K, 1% H$_2$O\n$\\alpha_L$={alfl:.4f}, $\\alpha_D$={alfad:.4f}\n"
                           f"$\\alpha_V$={alfv:.4f} cm$^{{-1}}$, $\\zeta$={float(data['info_zeta']):.3f}"),
              transform=ax_c.transAxes, ha="right", va="top", fontsize=6)

    # (d) differences from the exact Voigt
    neg = x_lbl < 0
    dl = np.abs(lbl - exact_lbl) / peak
    ax_d.plot(x_lbl[pos], dl[pos], color=BLACK, lw=0.7, label=r"|LBLRTM $-$ Voigt|, $\nu>\nu_0$")
    ax_d.plot(-x_lbl[neg], dl[neg], color=VERMILLION, lw=0.6, alpha=0.8, label=r"|LBLRTM $-$ Voigt|, $\nu<\nu_0$")
    dt = np.abs(tprof - exact_t) / peak
    ax_d.plot(np.abs(x_t), np.where(dt > 0, dt, np.nan), "s", ms=1.6, mfc="none", mec=BLUE, mew=0.4,
              label=r"|tellurix $-$ Voigt|")
    ax_d.axhline(ped / peak, color=VERMILLION, lw=0.6, ls="-.")
    ax_d.text(0.0025, ped / peak * 1.5, "pedestal", fontsize=6, color=VERMILLION)
    for b in bounds:
        ax_d.axvline(b, color=GREY, lw=0.4)
    ax_d.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(1e-16, 1e-2),
             xlabel=r"$|\nu-\nu_0|$ (cm$^{-1}$)", ylabel="absolute difference / Voigt peak")
    ax_d.set_title("(d) Departure from the exact Voigt", loc="left", fontsize=8)
    ax_d.legend(loc="lower left", fontsize=6)

    for name in ("pdf", "png"):
        fig.savefig(OUTPUT / f"fig_method_lineshape.{name}", dpi=200)

    # Numbers for the text.
    region = lambda lo, hi: (np.abs(x_lbl) >= lo) & (np.abs(x_lbl) < hi)   # noqa: E731
    ranges = {"<4aV": (0, bounds[0]), "4-16aV": (bounds[0], bounds[1]), "16-64aV": (bounds[1], bounds[2]),
              "64aV-25": (bounds[2], CUTOFF_CM1), ">25": (CUTOFF_CM1, 40)}
    model_on_run = np.interp(nu, grid, model) * float(data["strength_lblrtm"])
    return {
        "line": f"H2O {LINE_POSITION} cm-1 (AER 3.9)", "layer": f"{PRESSURE_HPA} hPa, {TEMPERATURE_K} K, H2O vmr {H2O_VMR}",
        "lblrtm_dv_cm1": dv, "dvr4_cm1": float(data["info_dvr4"]), "alpha_lorentz_cm1": alfl,
        "alpha_doppler_cm1": alfad, "alpha_voigt_lblrtm_cm1": alfv, "zeta": float(data["info_zeta"]),
        "voicon_coefficients": {k: float(data[f"info_{k}"]) for k in ("cf1", "cf2", "cf3", "cgauss", "cer")},
        "sub_function_limits_cm1": dict(zip(["4aV", "16aV", "64aV", "cutoff"], bounds)),
        "pedestal_over_peak": ped / peak,
        "transcription_vs_lblrtm_max_abs_over_peak": float(np.max(np.abs(model_on_run - od)) / od.max()),
        "lblrtm_SN_fitted_over_tellurix_SN": float(data["strength_lblrtm"]) / float(data["tstrength_column"]),
        "lblrtm_minus_voigt_max_abs_over_peak": {k: float(dl[region(*v)].max()) for k, v in ranges.items()},
        "lblrtm_minus_voigt_max_relative": {k: float((np.abs(lbl - exact_lbl) / exact_lbl)[region(*v)].max())
                                            for k, v in ranges.items() if k != ">25"},
        "lblrtm_min_value_beyond_cutoff_over_peak": float(lbl[np.abs(x_lbl) > CUTOFF_CM1].min() / peak),
        "lblrtm_area_within_window": float(lbl.sum() * dv),
        "tellurix_grid_spacing_cm1": float(np.diff(tgrid).mean()),
        "tellurix_minus_voigt_max_abs_over_peak": float(dt.max()),
        "tellurix_minus_voigt_max_relative": float(np.max(np.abs(tprof - exact_t) / exact_t)),
        "tellurix_algorithm916_half_width_cm1": alg916, "tellurix_core_list_half_width_cm1": core,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "method_lineshape.npz"
    if args.recompute or not cache.exists():
        np.savez(cache, **compute())
    data = dict(np.load(cache))
    numbers = plot(data)
    (OUTPUT / "fig_method_lineshape.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
