#!/usr/bin/env python
"""Method figure: how LBLRTM 12.17 and tellurix turn atmospheric levels into layers.

One ERA5 column -- the DCT night of ``scripts/compare_layering.py``
(2018-12-21 05 UT, anchored at 768.5 hPa / 2.360 km) -- is given to both codes
as the same levels: ERA5's own heights above the telescope, with the pressure
tellurix's hydrostatic integration puts there, ERA5's temperature, and its
water. LBLRTM is then run (LBLATM only matters; an O2-only TAPE3 outside the
window keeps the line calculation empty) with IBMAX=0, so AUTLAY chooses its own
boundaries and ALAYER/FPACK form the Curtis-Godson layers, which are read back
from TAPE6. tellurix builds its 12 default layers both ways
``site_profile.LAYERINGS`` offers.

Panel (d) is the 6-layer ``example_midlatitude.csv`` case of
``docs/lblrtm_corrected_mode.md`` ("What the 5000--5020 cm-1 gap is"): it is
read from the existing ``data/lblrtm/run_corrections/lines_h2o/TAPE6``; nothing
is rerun for it.

    JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_method_layering.py

The ERA5 column is cached in ``paper/figures/cache/era5_dct_2018.npz`` (fetched
from ARCO-ERA5 if missing) and the LBLRTM run in
``paper/figures/cache/lblrtm_layering_dct_2018/``. Writes
``fig_method_layering.{pdf,png}`` and ``fig_method_layering.json`` to
``paper/figures/output/``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import tellurix  # noqa: F401  (x64 before anything else)
from tellurix import AtmosphereProfile
from tellurix.era5 import build_era5_profile, specific_humidity_to_vmr
from tellurix.io import load_atmosphere_csv
from tellurix.site_profile import DEFAULT_EDGES_KM, EPOCH_DRY_VMR, _gravity

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper/figures/cache"
OUTPUT = ROOT / "paper/figures/output"
LBLRTM = ROOT / "data/lblrtm"
EXECUTABLE = LBLRTM / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
# O2 lines only, 12750-14850 cm-1: with no O2 in the column and a window far
# outside it, LBLRTM's line calculation is empty and the run is LBLATM alone.
EMPTY_TAPE3 = LBLRTM / "run_lnfl_o2/TAPE3"
SAG_TAPE6 = LBLRTM / "run_corrections/lines_h2o/TAPE6"
SAG_PROFILE = ROOT / "data/profiles/example_midlatitude.csv"
LAYERING_REPORT = ROOT / "docs/layering_comparison.json"
GAP_REPORT = ROOT / "docs/lblrtm_gap_attribution.json"

CASE = {"era5_time": "2018-12-21T05:00", "latitude": 34.7444, "longitude": -111.4223,
        "site_altitude_km": 2.360, "station_pressure_hpa": 768.5}
TOP_KM = DEFAULT_EDGES_KM[-1]
BOLTZMANN_ERG_K = 1.380649e-16
AVOGADRO = 6.02214076e23
DRY_MOLAR_MASS = 28.9647

WIDTH_IN = 7.1
C_LBL = "black"
C_WEIGHTED = "#0072B2"   # Okabe-Ito blue
C_CENTRE = "#D55E00"     # vermillion
C_LEVEL = "#999999"
C_SAG = "#009E73"        # bluish green

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})


# --------------------------------------------------------------------------
# The atmosphere: ERA5 levels and tellurix's fine hydrostatic column.

def era5_column() -> dict:
    path = CACHE / "era5_dct_2018.npz"
    if not path.exists():
        from tellurix.era5 import fetch_column
        when = datetime.fromisoformat(CASE["era5_time"]).replace(tzinfo=timezone.utc)
        column = fetch_column(CASE["latitude"], CASE["longitude"], when)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: column[k] for k in ("level_hpa", "temperature_k",
                                                  "specific_humidity", "height_m")})
    data = np.load(path)
    return {k: data[k] for k in ("level_hpa", "temperature_k", "specific_humidity", "height_m")}


def fine_atmosphere(column: dict) -> dict:
    """The fine column ``build_era5_profile`` integrates (copied, then checked against it)."""

    site_m = CASE["site_altitude_km"] * 1000.0
    order = np.argsort(column["height_m"])
    height_km = (column["height_m"][order] - site_m) / 1000.0
    temperature = column["temperature_k"][order]
    water = specific_humidity_to_vmr(column["specific_humidity"][order])
    fine = np.linspace(0.0, TOP_KM, 40001)
    fine_t = np.interp(fine, height_km, temperature)
    g = _gravity(CASE["site_altitude_km"] + fine) * 100.0
    integrand = DRY_MOLAR_MASS * g / (AVOGADRO * BOLTZMANN_ERG_K) / fine_t
    log_p = np.log(CASE["station_pressure_hpa"]) - np.concatenate(
        [[0.0], np.cumsum(np.diff(fine) * 1.0e5 * 0.5 * (integrand[1:] + integrand[:-1]))])
    fine_w = np.exp(np.interp(fine, height_km, np.log(np.maximum(water, 1.0e-12))))
    air = np.exp(log_p) * 1000.0 / (BOLTZMANN_ERG_K * fine_t)
    return {"z": fine, "p": np.exp(log_p), "t": fine_t, "w": fine_w, "n_air": air,
            "n_h2o": air * fine_w, "level_z": height_km, "level_t": temperature,
            "level_w": water}


def fine_column(atm: dict, z1: float, z2: float, key: str = "n_h2o") -> float:
    """Trapezoid integral of a fine density between two heights above the site, cm-2."""

    z, values = atm["z"], atm[key]
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(z))])
    return float(np.diff(np.interp([z1, z2], z, cumulative))[0] * 1.0e5)


# --------------------------------------------------------------------------
# LBLRTM: the same levels, its own layers.

def lblrtm_levels(atm: dict) -> dict:
    inside = (atm["level_z"] > 0.0) & (atm["level_z"] < TOP_KM)
    z = np.concatenate([[0.0], atm["level_z"][inside], [TOP_KM]])
    return {"z_km": z + CASE["site_altitude_km"],
            "p_hpa": np.exp(np.interp(z, atm["z"], np.log(atm["p"]))),
            "t_k": np.interp(z, atm["z"], atm["t"]),
            "h2o_vmr": np.exp(np.interp(z, atm["z"], np.log(atm["w"])))}


def write_level_tape5(path: Path, levels: dict) -> None:
    """IATM=1, ITYPE=3, IBMAX=0 (AUTLAY), user levels: the layout of the
    5000-5020 cm-1 template runs, with every continuum off."""

    z = levels["z_km"]
    lines = ["$tellurix paper: LBLRTM auto-layering of one ERA5 column",
             # IEMIT=1 as in the template runs: record 1.4 is read only then.
             "    1    1    0    0    1    0    0    0    0    1    0    0    0    0    0    0",
             "  4990.000  5000.000     4.000     0.000     0.040    36.000    -1.000    -1.000"
             "    0          0.000    0",
             f"{levels['t_k'][0]:10.3f}     1.000     0.000     0.000     0.000     0.000     0.000    s",
             f"    0    3    0    0    0    7    0 0  0     0.000{z[-1]:10.3f}  4995.000"
             "               0.000",
             f"{z[0]:10.3f}     0.000     0.000     0.000     0.000    0     {z[0]:10.3f}",
             # Record 3.3A: AVTRAT, TDIFF1, TDIFF2 zero select 1.5, 5 K, 8 K
             # (lblatm.f90:497, 837-839); ALTD1/ALTD2 span the path.
             f"     0.000     0.000     0.000{z[0]:10.3f}{z[-1]:10.3f}",
             f"{len(z):5d} ERA5 DCT 2018-12-21 05UT"]
    # ``A`` is ppmv of dry air (TAPE6 echoes it as "BASED UPON DRY AIR").
    dry_ppmv = levels["h2o_vmr"] / (1.0 - levels["h2o_vmr"]) * 1.0e6
    for zi, p, t, w in zip(z, levels["p_hpa"], levels["t_k"], dry_ppmv):
        # E10.3 fields read an explicit decimal point as written, so F10.4
        # keeps four more digits of pressure than the template writer did.
        lines.append(f"{zi:10.4f}{p:10.4f}{t:10.4f}     AA L AAAAAAA")
        lines.append(f"{w:15.8E}" + "".join(f"{0.0:15.8E}" for _ in range(6)))
    lines.extend(("-1.0", "-1.0", "%"))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run_lblrtm(levels: dict) -> Path:
    directory = CACHE / "lblrtm_layering_dct_2018"
    tape6 = directory / "TAPE6"
    if tape6.exists():
        return tape6
    directory.mkdir(parents=True, exist_ok=True)
    write_level_tape5(directory / "TAPE5", levels)
    (directory / "TAPE3").symlink_to(EMPTY_TAPE3)
    shutil.copy2(EXECUTABLE, directory / "lblrtm")
    result = subprocess.run(["./lblrtm"], cwd=directory, capture_output=True, text=True)
    if result.returncode != 0 or "FINAL SET OF LAYERS" not in tape6.read_text(errors="replace"):
        raise RuntimeError(f"LBLRTM failed:\n{result.stdout[-2000:]}{result.stderr[-2000:]}")
    return tape6


def read_tape6(path: Path) -> dict:
    """AUTLAY's boundaries and FPACK's final layers (PBAR, TBAR, AIR, H2O)."""

    text = path.read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(text) if "FINAL SET OF LAYERS" in line)
    rows = []
    for line in text[start:]:
        match = re.match(r"^0\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+"
                         r"([\d.E+-]+)\s+([\d.E+-]+)", line)
        if match:
            rows.append([float(v) for v in match.groups()])
        elif rows and "MOLECULAR MIXING RATIOS" in line:
            break
    rows = np.asarray(rows)
    levels_start = next(i for i, line in enumerate(text) if "DENSITY  (MOLS CM-3)" in line)
    level_rows = []
    for line in text[levels_start + 4:]:
        parts = line.split()
        if len(parts) < 7 or not parts[0].isdigit():
            if level_rows:
                break
            continue
        level_rows.append([float(v) for v in parts[1:7]])
    level_rows = np.asarray(level_rows)
    return {"z_bottom": rows[:, 1], "z_top": rows[:, 2], "pbar_hpa": rows[:, 3],
            "tbar_k": rows[:, 4], "air_cm2": rows[:, 5], "h2o_cm2": rows[:, 6],
            "level_z": level_rows[:, 0], "level_p": level_rows[:, 1], "level_t": level_rows[:, 2],
            "level_n_air": level_rows[:, 4], "level_n_h2o": level_rows[:, 5]}


# --------------------------------------------------------------------------
# tellurix: the default 12 layers, both ways.

def tellurix_layers(column: dict) -> dict:
    out = {}
    for layering in ("weighted", "centre"):
        layers = build_era5_profile(column, CASE["station_pressure_hpa"], CASE["site_altitude_km"],
                                    EPOCH_DRY_VMR["2020"], DEFAULT_EDGES_KM, layering=layering)
        edges = np.concatenate([layers["pressure_top_bar"][:1], layers["pressure_bottom_bar"]])
        profile = AtmosphereProfile(edges, layers["temperature_k"], layers["altitude_km"],
                                    {"H2O": layers["H2O"]}, layers["mean_molecular_weight_g_mol"],
                                    layers["gravity_m_s2"],
                                    mean_pressure_bar=layers.get("pressure_bar"))
        # Bottom-up arrays for plotting.
        out[layering] = {
            "p_hpa": profile.pressure_layer_bar[::-1] * 1000.0,
            "t_k": profile.temperature_k[::-1],
            "h2o_vmr": profile.vmr["H2O"][::-1],
            "h2o_cm2": (profile.air_column_cm2 * profile.vmr["H2O"])[::-1],
            "edges_hpa": profile.pressure_edges_bar[::-1] * 1000.0,
        }
    return out


# --------------------------------------------------------------------------

def sag_case() -> dict:
    """The 6-layer example: tellurix's layers, and what LBLRTM made of their 7 levels."""

    profile = load_atmosphere_csv(SAG_PROFILE)
    lbl = read_tape6(SAG_TAPE6)
    tellurix_h2o = profile.air_column_cm2 * profile.vmr["H2O"]
    # tellurix layer altitudes as the TAPE5 writer gave them to LBLRTM: its
    # level altitudes are TAPE6's level table, bottom-up.
    level_z = lbl["level_z"]
    layer_h2o = tellurix_h2o[::-1]
    thickness_cm = np.diff(level_z) * 1.0e5
    # LBLRTM's ALAYER integral between two levels: exponential in altitude,
    # (n_a - n_b) H / ... = mean (n_a - n_b) / ln(n_a / n_b).
    na, nb = lbl["level_n_h2o"][:-1], lbl["level_n_h2o"][1:]
    exponential_mean = (na - nb) / np.log(na / nb)
    return {"level_z": level_z, "level_n_h2o": lbl["level_n_h2o"],
            "tellurix_mean_density": layer_h2o / thickness_cm,
            "linear_mean_density": 0.5 * (na + nb),
            "exponential_mean_density": exponential_mean,
            "tellurix_column": float(tellurix_h2o.sum()),
            "lblrtm_column": float(lbl["h2o_cm2"].sum()),
            "lblrtm_exponential_column": float(np.sum(exponential_mean * thickness_cm)),
            "lblrtm_layers": len(lbl["h2o_cm2"])}


def main() -> None:
    column = era5_column()
    atm = fine_atmosphere(column)
    levels = lblrtm_levels(atm)
    lbl = read_tape6(run_lblrtm(levels))
    tx = tellurix_layers(column)
    site = CASE["site_altitude_km"]

    # tellurix's own weighted builder must agree with the fine column used here.
    tx_edges_z = np.asarray(DEFAULT_EDGES_KM)
    truth_tx = np.array([fine_column(atm, a, b) for a, b in zip(tx_edges_z[:-1], tx_edges_z[1:])])
    truth_lbl = np.array([fine_column(atm, a - site, b - site)
                          for a, b in zip(lbl["z_bottom"], lbl["z_top"])])
    total_truth = fine_column(atm, 0.0, TOP_KM)
    sag = sag_case()
    layering = json.loads(LAYERING_REPORT.read_text())
    gap = json.loads(GAP_REPORT.read_text())

    numbers = {
        "case": CASE,
        "lblrtm_layers": int(len(lbl["h2o_cm2"])),
        "lblrtm_boundaries_km": [float(lbl["z_bottom"][0])] + [float(v) for v in lbl["z_top"]],
        "era5_levels_used": int(len(levels["z_km"])),
        "water_column_cm2": {
            "fine_truth": total_truth,
            "lblrtm_autolayer": float(lbl["h2o_cm2"].sum()),
            "tellurix_weighted": float(tx["weighted"]["h2o_cm2"].sum()),
            "tellurix_centre": float(tx["centre"]["h2o_cm2"].sum()),
        },
        "per_layer_h2o_error": {
            "lblrtm": (lbl["h2o_cm2"] / truth_lbl - 1.0).tolist(),
            "weighted": (tx["weighted"]["h2o_cm2"] / truth_tx - 1.0).tolist(),
            "centre": (tx["centre"]["h2o_cm2"] / truth_tx - 1.0).tolist(),
        },
        "sag_example": {k: v for k, v in sag.items() if not isinstance(v, np.ndarray)},
        "layering_comparison_json": {
            name: {"centre_p99_fitted": [layering["results"][name][w]["centre"]["fitted_p99"]
                                         for w in layering["results"][name] if "-" in w],
                   "weighted_p99_fitted": [layering["results"][name][w]["weighted"]["fitted_p99"]
                                           for w in layering["results"][name] if "-" in w],
                   "water_column_cm2": layering["results"][name]["water_column_cm2"]}
            for name in layering["results"]},
    }
    for name, value in numbers["water_column_cm2"].items():
        print(f"{name:20s} {value:.4e}  {value / total_truth - 1.0:+.4%}")
    print("LBLRTM boundaries", numbers["lblrtm_boundaries_km"])
    print("sag", numbers["sag_example"])

    figure, axes = plt.subplots(1, 4, figsize=(WIDTH_IN, 3.05),
                                gridspec_kw={"width_ratios": [1.05, 1.0, 0.95, 1.1], "wspace": 0.42})
    ax_t, ax_w, ax_c, ax_s = axes
    p_top = levels["p_hpa"][-1]
    pmin, pmax = 0.9 * p_top, 1.05 * CASE["station_pressure_hpa"]

    # (a) temperature against pressure.
    ax_t.plot(atm["t"], atm["p"], color=C_LEVEL, lw=0.8, zorder=1)
    ax_t.plot(levels["t_k"], levels["p_hpa"], "o", ms=2.2, color=C_LEVEL, zorder=2,
              label=f"ERA5 levels ({len(levels['z_km'])} given to LBLRTM)")
    # Boundaries as ladders on either edge: LBLRTM's left, tellurix's right.
    for p in np.concatenate([[lbl["level_p"][0]], np.interp(lbl["z_top"], levels["z_km"],
                                                             levels["p_hpa"])]):
        ax_t.plot([185, 193], [p, p], color=C_LBL, lw=0.8)
    for p in tx["weighted"]["edges_hpa"]:
        ax_t.plot([296, 304], [p, p], color=C_WEIGHTED, lw=0.8)
    def total(key):
        return f"H$_2$O column {numbers['water_column_cm2'][key] / total_truth - 1:+.3%}"

    ax_t.plot(lbl["tbar_k"], lbl["pbar_hpa"], "s", ms=3.2, mfc="none", mec=C_LBL, mew=0.8,
              label=f"LBLRTM AUTLAY, {len(lbl['h2o_cm2'])} layers ({total('lblrtm_autolayer')})")
    ax_t.plot(tx["weighted"]["t_k"], tx["weighted"]["p_hpa"], "D", ms=3.0, color=C_WEIGHTED,
              label=f"tellurix weighted, 12 layers ({total('tellurix_weighted')})")
    ax_t.plot(tx["centre"]["t_k"], tx["centre"]["p_hpa"], "x", ms=3.5, mew=0.9, color=C_CENTRE,
              label=f"tellurix centre, 12 layers ({total('tellurix_centre')})")
    ax_t.set_yscale("log")
    ax_t.set_ylim(pmax, pmin)
    ax_t.set_xlim(178, 305)
    ax_t.set_xlabel("Temperature (K)")
    ax_t.set_ylabel("Pressure (hPa)")
    ax_t.set_yticks([700, 500, 300, 200, 100, 50, 30, 20, 10])
    ax_t.set_yticklabels(["700", "500", "300", "200", "100", "50", "30", "20", "10"])
    ax_t.minorticks_off()
    ax_t.text(186, pmin * 1.12, "LBLRTM\nbounds", fontsize=5.5, ha="left", va="top")
    ax_t.text(304, pmin * 1.12, "tellurix\nbounds", fontsize=5.5, ha="right", va="top",
              color=C_WEIGHTED)
    figure.legend(*ax_t.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
                  bbox_to_anchor=(0.5, 1.09), handlelength=1.4, columnspacing=1.6)
    ax_t.set_title("(a) Temperature", loc="left")

    # (b) water against pressure: the levels, and each layer's column-weighted VMR.
    ax_w.plot(atm["w"] * 1e6, atm["p"], color=C_LEVEL, lw=0.8)
    ax_w.plot(levels["h2o_vmr"] * 1e6, levels["p_hpa"], "o", ms=2.2, color=C_LEVEL)
    lbl_vmr = lbl["h2o_cm2"] / lbl["air_cm2"]
    ax_w.plot(lbl_vmr * 1e6, lbl["pbar_hpa"], "s", ms=3.2, mfc="none", mec=C_LBL, mew=0.8)
    ax_w.plot(tx["weighted"]["h2o_vmr"] * 1e6, tx["weighted"]["p_hpa"], "D", ms=3.0,
              color=C_WEIGHTED)
    ax_w.plot(tx["centre"]["h2o_vmr"] * 1e6, tx["centre"]["p_hpa"], "x", ms=3.5, mew=0.9,
              color=C_CENTRE)
    ax_w.set_xscale("log")
    ax_w.set_yscale("log")
    ax_w.set_ylim(pmax, pmin)
    ax_w.set_yticks([700, 500, 300, 200, 100, 50, 30, 20, 10])
    ax_w.set_yticklabels([])
    ax_w.minorticks_off()
    ax_w.set_xlabel(r"H$_2$O VMR (ppmv)")
    ax_w.set_title(r"(b) Water vapour", loc="left")

    # (c) each layer's water column against the integral of the same atmosphere
    # between its own boundaries.
    tx_mid_p = tx["weighted"]["p_hpa"]
    for key, color, marker, label in (("centre", C_CENTRE, "x", "tellurix centre"),
                                      ("weighted", C_WEIGHTED, "D", "tellurix weighted")):
        error = numbers["per_layer_h2o_error"][key]
        ax_c.plot(np.asarray(error) * 100, tx_mid_p, marker=marker, ms=3.0 if marker == "D" else 3.5,
                  mew=0.9, color=color, lw=0.7, label=label)
    ax_c.plot(np.asarray(numbers["per_layer_h2o_error"]["lblrtm"]) * 100, lbl["pbar_hpa"],
              marker="s", ms=3.2, mfc="none", mec=C_LBL, mew=0.8, color=C_LBL, lw=0.7,
              label="LBLRTM")
    ax_c.axvline(0.0, color="#bbbbbb", lw=0.6, zorder=0)
    ax_c.set_yscale("log")
    ax_c.set_ylim(pmax, pmin)
    ax_c.set_yticks([700, 500, 300, 200, 100, 50, 30, 20, 10])
    ax_c.set_yticklabels([])
    ax_c.minorticks_off()
    ax_c.set_xlabel(r"H$_2$O column error (%)")
    wc = numbers["water_column_cm2"]
    ax_c.set_xlim(-40, 22)
    ax_c.set_title("(c) Integrated amount", loc="left")

    # (d) the 5000-5020 cm-1 template: 6 layers handed over as 7 levels. The
    # two lowest layers hold 93% of its water; linear axes show the sag.
    z = sag["level_z"]
    scale = 1.0e17
    for i in range(2):
        zz = np.linspace(z[i], z[i + 1], 80)
        na, nb = sag["level_n_h2o"][i], sag["level_n_h2o"][i + 1]
        curve = na * (nb / na) ** ((zz - z[i]) / (z[i + 1] - z[i]))
        ax_s.plot([na / scale, nb / scale], [z[i], z[i + 1]], color=C_LEVEL, lw=0.7, ls=(0, (3, 2)),
                  label="linear between levels" if i == 0 else None)
        ax_s.plot(curve / scale, zz, color=C_LBL, lw=1.0,
                  label="LBLRTM (exponential)" if i == 0 else None)
        ax_s.plot([sag["tellurix_mean_density"][i] / scale] * 2, [z[i], z[i + 1]], color=C_WEIGHTED,
                  lw=1.6, label="tellurix layer" if i == 0 else None)
        ax_s.plot([sag["exponential_mean_density"][i] / scale] * 2, [z[i], z[i + 1]], color=C_LBL,
                  lw=0.9, ls=(0, (1.5, 1.2)), label="LBLRTM layer mean" if i == 0 else None)
        change = sag["exponential_mean_density"][i] / sag["tellurix_mean_density"][i] - 1.0
        ax_s.text(sag["tellurix_mean_density"][i] / scale + 0.15, 0.5 * (z[i] + z[i + 1]) + 0.4,
                  f"{change:+.1%}", fontsize=5.6, va="center", color=C_LBL)
    ax_s.plot(sag["level_n_h2o"][:3] / scale, z[:3], "o", ms=2.8, color=C_LEVEL, zorder=3,
              label="levels given")
    ax_s.set_ylim(z[0] - 0.15, z[2] + 1.2)
    ax_s.set_xlim(0.0, 3.0)
    ax_s.set_xlabel(r"H$_2$O density (10$^{17}$ cm$^{-3}$)")
    ax_s.set_ylabel("Altitude (km)", labelpad=1)
    deficit = sag["lblrtm_column"] / sag["tellurix_column"] - 1.0
    ax_s.text(0.97, 0.97,
              f"column, all {len(z) - 1} layers (cm$^{{-2}}$)\ntellurix {sag['tellurix_column']:.3e}\n"
              f"LBLRTM {sag['lblrtm_column']:.3e}\n= {-deficit:.1%} less (tellurix\n"
              f"{sag['tellurix_column'] / sag['lblrtm_column'] - 1:.1%} more)",
              transform=ax_s.transAxes, ha="right", va="top", fontsize=5.6)
    ax_s.legend(loc="center right", bbox_to_anchor=(1.03, 0.36), frameon=False, handlelength=1.6,
                borderaxespad=0.1, fontsize=5.2)
    ax_s.set_title("(d) Interpolation sag", loc="left")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT / f"fig_method_layering.{suffix}", dpi=200, bbox_inches="tight")
    (OUTPUT / "fig_method_layering.json").write_text(json.dumps(numbers, indent=1, default=float))
    print("wrote", OUTPUT / "fig_method_layering.png")


if __name__ == "__main__":
    main()
