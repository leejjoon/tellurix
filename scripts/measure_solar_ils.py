#!/usr/bin/env python
"""Measure the FTS instrument line shape of the NSO solar atlases from their data.

The method is ``tellurix.ils``: the transform of a spectrum is its own
interferogram, so the truncation that sets the resolution is measurable and
does not have to be assumed. ``scripts/measure_atlas_ils.py`` does the same for
the Arcturus atlas and shares the implementation.

This matters more here than it did there, because both of these atlases
misreport their own resolution:

* ``photatl`` documents none at all. The only instrumental number in the
  refereed description of the series (Wallace et al. 1996, ApJS 106, 165) is
  ``R ~ 300,000`` quoted once for four atlases, with no MOPD and no apodization.
* the raw ``ftsspec_*`` headers state a ``resolution=`` that their own
  interferogram contradicts by a factor of 2-4.

Both are FTS products and both are zero-filled by a factor their sampling does
not reveal, which is why the sibling project concluded the absolute scale was
not obtainable. Zero-filling does not move the cut; see ``tellurix.ils``.

    UV_CACHE_DIR=.uv-cache uv run python scripts/measure_solar_ils.py --format photatl
    UV_CACHE_DIR=.uv-cache uv run python scripts/measure_solar_ils.py --format ftsspec
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from tellurix import (
    BOXCAR_FWHM_CONSTANT,
    describe_truncation,
    read_fts_spectrum,
    read_niratl_page,
    read_photatl_page,
    window_continuum_snr,
)


NSO_ROOT = Path(
    "/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/nso"
)
# Below this the window is opaque or off the filter, its "interferogram" is
# noise, and the MOPD it reports is meaningless. Calibrated in
# docs/solar_fit_plan.md: live windows reach 188-2702, dead ones 5-11.
MINIMUM_CONTINUUM_SNR = 30.0
SHARP_CUT_RATIO = 3.0


def measure_photatl(root: Path) -> list[dict]:
    """One measurement per page, on the observed (`total`) column."""

    pages = sorted(p for p in root.iterdir() if re.match(r"^wn\d+$", p.name))
    if not pages:
        raise SystemExit(f"no photatl pages found under {root}")
    measurements = []
    for path in pages:
        try:
            page = read_photatl_page(path)
            snr = window_continuum_snr(page.total)
            if snr < MINIMUM_CONTINUUM_SNR:
                raise ValueError(f"opaque page: continuum only {snr:.1f} sigma above zero")
            entry = describe_truncation(page.wavenumber_vacuum_cm1, page.total)
            entry.update({"page": path.name, "continuum_snr": float(snr),
                          "grid_residual_cm1": page.grid_residual_cm1})
        except (ValueError, IndexError) as exc:
            entry = {"page": path.name, "error": str(exc), "mopd_measurable": False}
        measurements.append(entry)
    return measurements


def measure_niratl(root: Path) -> list[dict]:
    """One measurement per page, on the observed column.

    Wallace et al. 1996 say niratl combines five 1983 June spectra. If the
    observed column were itself a co-add of scans with different path
    differences, the cut would not be one sharp edge, which this measurement
    would show as a spread in MOPD or a soft transition.
    """

    pages = sorted(p for p in root.iterdir() if re.match(r"^ph\d+$", p.name))
    if not pages:
        raise SystemExit(f"no niratl pages found under {root}")
    measurements = []
    for path in pages:
        try:
            page = read_niratl_page(path)
            snr = window_continuum_snr(page.observed)
            if snr < MINIMUM_CONTINUUM_SNR:
                raise ValueError(f"opaque page: continuum only {snr:.1f} sigma above zero")
            entry = describe_truncation(page.wavenumber_vacuum_cm1, page.observed)
            entry.update({"page": path.name, "continuum_snr": float(snr),
                          "grid_residual_cm1": page.grid_residual_cm1})
        except (ValueError, IndexError) as exc:
            entry = {"page": path.name, "error": str(exc), "mopd_measurable": False}
        measurements.append(entry)
    return measurements


def measure_ftsspec(root: Path, window_cm1: float) -> list[dict]:
    """One measurement per window of each raw spectrum.

    Per window rather than per file: the instrument response runs from zero to
    unity and back across one of these spectra, and transforming the whole
    thing would measure the envelope as much as the truncation.
    """

    files = sorted(root.glob("ftsspec_*.txt"))
    if not files:
        raise SystemExit(f"no ftsspec files found under {root}")
    measurements = []
    for path in files:
        spectrum = read_fts_spectrum(path)
        edges = np.arange(
            spectrum.wavenumber_vacuum_cm1[0], spectrum.wavenumber_vacuum_cm1[-1], window_cm1
        )
        for low in edges[:-1]:
            common = {
                "file": path.name,
                "window_start_cm1": float(low),
                "stated_resolution_cm1": spectrum.stated_resolution_cm1,
            }
            try:
                window = spectrum.select(float(low), float(low) + window_cm1)
                snr = window_continuum_snr(window.flux)
                if snr < MINIMUM_CONTINUUM_SNR:
                    raise ValueError(f"opaque window: continuum only {snr:.1f} sigma above zero")
                entry = describe_truncation(window.wavenumber_vacuum_cm1, window.flux)
                entry.update(common)
                entry["continuum_snr"] = float(snr)
            except (ValueError, IndexError) as exc:
                entry = dict(common, error=str(exc), mopd_measurable=False)
            measurements.append(entry)
    return measurements


def summarize(measurements: list[dict], group_key: str | None) -> dict:
    """Reduce to the numbers the plan actually leans on.

    ``group_key`` splits the summary by instrument configuration and is None
    when there is only one. It is the *file* for the ftsspec set, never the
    page: photatl's 258 pages are one configuration, and grouping by page would
    report 258 summaries of a single measurement each.
    """

    usable = [
        m for m in measurements
        if m.get("mopd_measurable") and np.isfinite(m.get("mopd_cm", np.nan))
    ]
    summary: dict = {"measured": len(usable), "attempted": len(measurements)}
    # A truncated interferogram falls off within about one path-difference
    # resolution. Where the transition is tens of times wider there is no cut
    # to find, and the steepest-slope search returns anything: on niratl 119 of
    # 180 pages spread from 2 to 28 cm with transitions ~41x the resolution,
    # while the other 61 agree to 1.7%. "sharp_cut" is the number to use.
    sharp = [m for m in usable
             if m["transition_cm"] / m["path_difference_resolution_cm"] < SHARP_CUT_RATIO]
    if sharp and len(sharp) < len(usable):
        summary["sharp_cut"] = summarize(sharp, None)["all"]
        summary["sharp_cut"]["transition_over_resolution_below"] = SHARP_CUT_RATIO
    groups = sorted({m[group_key] for m in usable}) if (usable and group_key) else []
    for name in ["all"] + groups:
        rows = usable if name == "all" else [m for m in usable if m[group_key] == name]
        if not rows:
            continue
        mopd = np.asarray([m["mopd_cm"] for m in rows], dtype=float)
        nu = np.asarray([m["wavenumber_center_cm1"] for m in rows], dtype=float)
        covered = (min(m["wavenumber_min_cm1"] for m in rows),
                   max(m["wavenumber_max_cm1"] for m in rows))
        fwhm = np.asarray([m["ils_fwhm_cm1"] for m in rows], dtype=float)
        transition = np.asarray([m["transition_cm"] for m in rows], dtype=float)
        resolution = np.asarray([m["path_difference_resolution_cm"] for m in rows], dtype=float)
        block = {
            "count": int(mopd.size),
            "median_mopd_cm": float(np.median(mopd)),
            "mad_mopd_cm": float(np.median(np.abs(mopd - np.median(mopd)))),
            "p16_mopd_cm": float(np.percentile(mopd, 16)),
            "p84_mopd_cm": float(np.percentile(mopd, 84)),
            # Arcturus holds the resolving power constant, so its MOPD tracks
            # 1/nu at corr = -0.93 and spreads by a factor 2.87. A constant
            # path difference shows as a spread near 1 and no correlation.
            "p84_over_p16": float(np.percentile(mopd, 84) / np.percentile(mopd, 16)),
            "median_ils_fwhm_cm1": float(np.median(fwhm)),
            "median_transition_cm": float(np.median(transition)),
            "median_transition_over_path_resolution": float(
                np.median(transition / resolution)
            ),
            "wavenumber_range_cm1": [float(nu.min()), float(nu.max())],
        }
        if mopd.size > 2 and np.ptp(nu) > 0.0:
            block["corr_mopd_wavenumber"] = float(np.corrcoef(mopd, nu)[0, 1])
        # R implied by the median FWHM, which is the quantity a fit needs.
        block["resolving_power"] = {
            str(int(point)): float(point / np.median(fwhm))
            for point in (2000, 5000, 9000, 13000)
            if covered[0] <= point <= covered[1]
        }
        summary[name] = block
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=("photatl", "niratl", "ftsspec"), required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--window-cm1", type=float, default=100.0,
                        help="ftsspec only: the span transformed at a time")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.format == "photatl":
        root = args.root or NSO_ROOT / "photatl"
        measurements = measure_photatl(root)
        summary = summarize(measurements, None)
        output = args.output or Path("docs/photatl_ils.json")
    elif args.format == "niratl":
        root = args.root or NSO_ROOT / "niratl"
        measurements = measure_niratl(root)
        summary = summarize(measurements, None)
        output = args.output or Path("docs/niratl_ils.json")
    else:
        root = args.root or NSO_ROOT / "telluric_near_ir"
        measurements = measure_ftsspec(root, args.window_cm1)
        summary = summarize(measurements, "file")
        output = args.output or Path("docs/solar_fts_ils.json")

    report = {
        "root": str(root),
        "format": args.format,
        "method": (
            "Hann-windowed real FFT of the observed column; MOPD located at the "
            "steepest sustained drop of the smoothed log envelope. See tellurix.ils."
        ),
        "boxcar_fwhm_constant": BOXCAR_FWHM_CONSTANT,
        "minimum_continuum_snr": MINIMUM_CONTINUUM_SNR,
        "summary": summary,
        "measurements": measurements,
    }
    if args.format == "ftsspec":
        report["window_cm1"] = args.window_cm1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Every group is printed, not only "all": that aggregate mixes instrument
    # configurations whenever the inputs do, and for the ftsspec set it does --
    # the 1983 and 1990 pairs sit at 14.5 and 34.4 cm, so their combined spread
    # and their correlation with wavenumber are artefacts of pooling them.
    for name, block in summary.items():
        if not isinstance(block, dict) or "median_mopd_cm" not in block:
            continue
        print(
            f"{name}: MOPD = {block['median_mopd_cm']:.3f} +- {block['mad_mopd_cm']:.3f} cm "
            f"over {block['count']} of {summary['attempted']}, spread p84/p16 = "
            f"{block['p84_over_p16']:.3f}, corr(MOPD, nu) = "
            f"{block.get('corr_mopd_wavenumber', float('nan')):+.3f}"
        )
        print(f"    FWHM {block['median_ils_fwhm_cm1']:.5f} cm-1 -> R = " + ", ".join(
            f"{k}:{v:,.0f}" for k, v in block["resolving_power"].items()))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
