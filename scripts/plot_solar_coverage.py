#!/usr/bin/env python
"""Wavelength coverage of the solar spectra: observed, corrected by us, and corrected by others.

One row per data set. Every range is read from the data, never typed in: the
observed spectrum from its own file, ours from the promoted run summaries (a fitted
window, a window with no telluric line above the cut, or one that failed), and the
others' from their corrected columns, with the pixels they interpolated or drew by
hand left out (photatl's ``atmospheric < dg``, niratl's -1.0 fill, Kurucz's straight
stretches, Wallace 2011's blanks below T = 0.8). ACE-FTS is observed from orbit, so it
is telluric-free over its whole range rather than corrected. Coverage is decided on
1 cm-1 bins: a bin counts when more than half of its pixels are valid.

    uv run python scripts/plot_solar_coverage.py   # -> docs/solar_coverage.{png,json}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from tellurix_fts import read_niratl_page, read_photatl_page, read_solar_spectrum
from tellurix_fts.baker import read_baker_solar_atlas
from tellurix_fts.flux_atlas import read_flux_atlas, read_kurucz_irradiance, read_wallace2011_correction

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIBLING = Path("/home/jjlee/work/differentiable_stellar_spectroscopy")
NSO = SIBLING / "data/atlases/nso"
ACE = SIBLING / "references/aux"
SOLAR = ROOT / "data/corrected/solar"
GRID = np.arange(650.0, 35001.0, 1.0)
COLOURS = {"data": "#cfd6da", "ours": "#1f6fb4", "clear": "#a9cbe8", "failed": "#c0392b",
           "author": "#d9822b", "other": "#8e5bb5", "space": "#138d75"}


def binned(nu, valid=None) -> np.ndarray:
    nu = np.asarray(nu)
    valid = np.ones(nu.size, bool) if valid is None else np.asarray(valid, bool)
    total, _ = np.histogram(nu, GRID)
    good, _ = np.histogram(nu[valid], GRID)
    return (total > 0) & (good > 0.5 * total)


def windows(*summaries: Path):
    """Bins of fitted windows, of windows with nothing to correct, of windows that failed; and the fitted count."""
    centre = 0.5 * (GRID[1:] + GRID[:-1])
    fitted, clear, failed = (np.zeros(centre.size, bool) for _ in range(3))
    count = 0
    for summary in summaries:
        for row in json.loads(summary.read_text())["results"]:
            inside = (centre >= row["v1"]) & (centre <= row["v2"])
            if row.get("npz"):
                fitted |= inside
                count += 1
            elif row.get("negligible_telluric"):
                clear |= inside
            else:
                failed |= inside
    return fitted, clear & ~fitted, failed & ~fitted & ~clear, count


def segments(mask: np.ndarray) -> list[tuple[float, float]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return [(GRID[a], GRID[b]) for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0])]


def span_um(mask: np.ndarray):
    if not mask.any():
        return None
    at = np.nonzero(mask)[0]
    return [round(1e4 / GRID[at[-1] + 1], 3), round(1e4 / GRID[at[0]], 3)]


def rows() -> list[dict]:
    out = []
    raw = {name: read_solar_spectrum(NSO / f"telluric_near_ir/ftsspec_{name}.txt")
           for name in ("901218_5", "901218_4", "830626_2", "830626_3")}
    photatl = [read_photatl_page(p) for p in sorted((NSO / "photatl").glob("wn*")) if p.name[2:].isdigit()]
    out.append({"label": "Kitt Peak FTS 1990-12-18 #5", "detail": "disc centre, X 1.99", "group": "centre",
                "data": binned(raw["901218_5"].wavenumber_vacuum_cm1),
                "ours": windows(SOLAR / "ftsspec_901218_5_summary.json"),
                "others": [(binned(np.concatenate([p.wavenumber_vacuum_cm1 for p in photatl]),
                                   np.concatenate([~p.interpolated for p in photatl])),
                            "author", "photatl (Livingston & Wallace 1991)")]})
    out.append({"label": "Kitt Peak FTS 1990-12-18 #4", "detail": "disc centre, X 4.7", "group": "centre",
                "data": binned(raw["901218_4"].wavenumber_vacuum_cm1),
                "ours": windows(SOLAR / "ftsspec_901218_4_summary.json"), "others": []})
    out.append({"label": "Kitt Peak FTS 1983-06-26 #2, #3", "detail": "disc centre, X 5.0 and 3.0", "group": "centre",
                "data": binned(raw["830626_2"].wavenumber_vacuum_cm1) | binned(raw["830626_3"].wavenumber_vacuum_cm1),
                "ours": windows(SOLAR / "ftsspec_830626_2_summary.json", SOLAR / "ftsspec_830626_3_summary.json"),
                "others": []})
    niratl = [read_niratl_page(p) for p in sorted((NSO / "niratl").glob("ph*")) if p.name[2:].isdigit()]
    nu = np.concatenate([p.wavenumber_vacuum_cm1 for p in niratl])
    out.append({"label": "niratl", "detail": "disc centre, 1983 June, X ≈ 1", "group": "centre",
                "data": binned(nu), "ours": windows(SOLAR / "niratl/niratl_summary.json"),
                "others": [(binned(nu, np.concatenate([p.solar != -1.0 for p in niratl])),
                            "author", "solar column (Wallace et al. 1993)")]})
    hase = np.loadtxt(next((ACE / "hase2010_ace_fts").glob("*-mmc1.txt")), delimiter=",", skiprows=1, usecols=0)
    bhusal = np.loadtxt(ACE / "bhusal2026_ace_fts_2026/ACE-FTS_Solar_Spectrum_2026.csv",
                        delimiter=",", skiprows=1, usecols=0)
    nothing = np.zeros(GRID.size - 1, bool)
    out.append({"label": "ACE-FTS solar atlas (SCISAT)", "detail": "near disc centre, from orbit", "group": "centre",
                "data": binned(hase) | binned(bhusal), "ours": (nothing, nothing, nothing, 0),
                "others": [(binned(hase), "space", "Hase et al. 2010"),
                           (binned(bhusal), "space", "Bhusal et al. 2026")]})

    fluxatl = read_flux_atlas("fluxatl", NSO / "fluxatl", (0.0, 1e9), airmass=1.0)
    k_nu, _, straight = read_kurucz_irradiance(SIBLING / "data/atlases/kurucz2005/irradrelwl.dat")
    out.append({"label": "fluxatl (Kurucz et al. 1984)", "detail": "flux, 8 scans 1980-81", "group": "flux",
                "data": binned(fluxatl.wavenumber_vacuum_cm1),
                "ours": windows(*[SOLAR / f"fluxatl/fluxatl_{s}_summary.json" for s in (109, 111, 113, 115)]),
                "others": [(binned(k_nu, ~straight), "author", "Kurucz 2005 re-reduction (irradrelwl)")]})
    data, corrected = nothing.copy(), nothing.copy()
    for region in (1, 2, 3):
        nu, flux, _ = read_wallace2011_correction(NSO / f"wallace2011_flux/sptr.reg{region}")
        data |= binned(nu)
        corrected |= binned(nu, np.isfinite(flux))
    for region in (4, 5, 6):  # wavenumber and flux only: the authors corrected nothing there
        data |= binned(np.loadtxt(NSO / f"wallace2011_flux/sptr.reg{region}", usecols=0))
    out.append({"label": "Wallace et al. 2011 flux atlas", "detail": "flux, regions 1-6, 1989", "group": "flux",
                "data": data,
                "ours": windows(*[SOLAR / f"wallace2011/wallace2011_reg{r}_summary.json" for r in (1, 2, 3)]),
                "others": [(corrected, "author", "authors' corrected flux (T > 0.8 only)")]})
    vis = read_flux_atlas("iag", ROOT / "data/databases/iag/spvis.dat.gz", (0.0, 1e9), airmass=1.0)
    baker = read_baker_solar_atlas(SIBLING / "data/atlases/iag_telfree/iag_telfree_solaratlas.fits")
    out.append({"label": "IAG VIS (Reiners et al. 2016)", "detail": "flux, 9-day co-add, 2014", "group": "flux",
                "data": binned(vis.wavenumber_vacuum_cm1), "ours": windows(SOLAR / "iag/iag_vis_summary.json"),
                "others": [(binned(baker["v"]), "other", "Baker, Blake & Reiners 2020 telluric-free Sun")]})
    nir = read_flux_atlas("iag", ROOT / "data/databases/iag/spnir.dat.gz", (0.0, 1e9), airmass=1.0)
    out.append({"label": "IAG NIR (Reiners et al. 2016)", "detail": "flux, 6-day co-add, 2014", "group": "flux",
                "data": binned(nir.wavenumber_vacuum_cm1), "ours": windows(SOLAR / "iag_nir/iag_nir_summary.json"),
                "others": []})
    return out


def plot(table: list[dict], output: Path) -> None:
    um = lambda nu: 1e4 / nu  # noqa: E731
    bar = lambda ax, y, a, b, h, c: ax.barh(y, um(a) - um(b), left=um(b), height=h, color=c, lw=0)  # noqa: E731
    fig, ax = plt.subplots(figsize=(14, 7.6))
    y, ticks, flux_top = 0.0, [], None
    for row in table:
        if row["group"] == "flux" and flux_top is None:
            ax.axhline(y + 0.5, color="0.6", lw=0.6, ls=":")
            y -= 0.3
            flux_top = y
        fitted, clear, failed, _ = row["ours"]
        for a, b in segments(row["data"]):
            bar(ax, y, a, b, 0.78, COLOURS["data"])
        for mask, colour in ((fitted, "ours"), (clear, "clear"), (failed, "failed")):
            for a, b in segments(mask):
                bar(ax, y + 0.13, a, b, 0.30, COLOURS[colour])
        for j, (mask, kind, name) in enumerate(row["others"]):
            # Two telluric-free releases stack where a corrected spectrum would go.
            yy = y - 0.2 + (0.33 - 0.27 * j if kind == "space" else 0.0)
            for a, b in segments(mask):
                bar(ax, yy, a, b, 0.24, COLOURS[kind])
            at = np.nonzero(mask)[0]
            if kind == "space":
                ax.text(um(GRID[at[-1] + 1]) * 0.985, yy, name, va="center", ha="right", fontsize=7.5,
                        color=COLOURS[kind])
            else:
                ax.text(um(GRID[at[0]]) * 1.012, yy, name, va="center", fontsize=7.5, color=COLOURS[kind])
        ticks.append((y, f"{row['label']}\n{row['detail']}"))
        y -= 1.0
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks], fontsize=8.5)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xscale("log")
    ax.set_xlim(0.29, 15.0)
    ax.xaxis.set_major_locator(FixedLocator([0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 7, 10, 14]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("vacuum wavelength (µm)")
    top = ax.secondary_xaxis("top", functions=(lambda x: 1e4 / np.maximum(x, 1e-9),
                                               lambda n: 1e4 / np.maximum(n, 1e-9)))
    top.xaxis.set_major_locator(FixedLocator([30000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1500, 1000, 700]))
    top.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel("wavenumber (cm⁻¹)")
    ax.text(14.6, 0.55, "disc centre (intensity)", ha="right", va="top", fontsize=8, color="0.4")
    ax.text(14.6, flux_top - 0.25, "disc-integrated (flux)", ha="right", va="top", fontsize=8, color="0.4")
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[Patch(color=COLOURS["data"], label="observed spectrum"),
                       Patch(color=COLOURS["ours"], label="tellurix correction (fitted windows)"),
                       Patch(color=COLOURS["clear"], label="tellurix: no telluric line above the cut"),
                       Patch(color=COLOURS["failed"], label="tellurix: window not fitted"),
                       Patch(color=COLOURS["author"], label="corrected by the atlas authors"),
                       Patch(color=COLOURS["other"], label="corrected by others"),
                       Patch(color=COLOURS["space"], label="telluric-free: observed from orbit")],
              loc="upper center", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0.45, -0.09))
    fig.tight_layout()
    fig.savefig(output, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/solar_coverage.png")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/solar_coverage.json")
    args = parser.parse_args()

    table = rows()
    plot(table, args.output)
    report = {"about": " ".join(__doc__.split("\n\n")[1].split()), "rows": {}}
    for row in table:
        fitted, clear, failed, count = row["ours"]
        report["rows"][row["label"]] = {
            "detail": row["detail"], "observed_um": span_um(row["data"]),
            "tellurix": {"fitted_windows": count, "fitted_um": span_um(fitted),
                         "not_fitted_um": [[round(1e4 / b, 4), round(1e4 / a, 4)] for a, b in segments(failed)]},
            "others": {name: {"kind": kind, "um": span_um(mask)} for mask, kind, name in row["others"]}}
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {args.report}")


if __name__ == "__main__":
    main()
