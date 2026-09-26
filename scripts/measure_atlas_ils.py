#!/usr/bin/env python
"""Measure the FTS instrument line shape of the Arcturus IR atlas from its data.

An FTS spectrum is the Fourier transform of an interferogram truncated at the
maximum optical path difference, so the transform of a page *is* that
interferogram: its envelope is flat out to the MOPD and then cuts. Both the MOPD
and the apodization are therefore measurable, and neither has to be assumed.

The atlas documents neither. This script recovers them per page and per epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from tellurix.ils import (
    BOXCAR_FWHM_CONSTANT as _BOXCAR_FWHM_CONSTANT,
    interferogram_envelope,
    measure_mopd,
)


ATLAS_ROOT = Path(
    "/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/arcturus/ir"
)
# Columns of the 89-character records; see the atlas table.doc.
_EPOCH_OBSERVED_COLUMN = {"summer": 1, "winter": 4}


def measure_page(path: Path, epoch: str) -> dict:
    values = np.loadtxt(path)
    wavenumber = values[:, 0]
    flux = values[:, _EPOCH_OBSERVED_COLUMN[epoch]]
    path_difference_cm, envelope = interferogram_envelope(wavenumber, flux)
    result = measure_mopd(path_difference_cm, envelope)

    center = float(0.5 * (wavenumber[0] + wavenumber[-1]))
    mopd = result["mopd_cm"]
    result.update(
        {
            "page": path.name,
            "epoch": epoch,
            "wavenumber_min_cm1": float(wavenumber[0]),
            "wavenumber_max_cm1": float(wavenumber[-1]),
            "samples": int(len(wavenumber)),
            "spacing_cm1": float(np.median(np.diff(wavenumber))),
        }
    )
    if np.isfinite(mopd) and mopd > 0.0:
        fwhm = _BOXCAR_FWHM_CONSTANT / (2.0 * mopd)
        result["ils_fwhm_cm1"] = float(fwhm)
        result["resolving_power_at_center"] = float(center / fwhm)
        result["samples_per_fwhm"] = float(fwhm / result["spacing_cm1"])
        # Beyond this the MOPD is longer than the sampling can express and the
        # measurement only recovers the decimation, not the interferogram.
        result["mopd_measurable"] = bool(mopd < 0.95 * result["nyquist_path_difference_cm"])
    else:
        result["mopd_measurable"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=ATLAS_ROOT)
    parser.add_argument("--pages", nargs="*", default=None, help="page file names")
    parser.add_argument("--output", type=Path, default=Path("docs/arcturus_ils.json"))
    args = parser.parse_args()

    if args.pages:
        pages = [args.atlas_root / name for name in args.pages]
    else:
        pattern = re.compile(r"^ab\d+[._]?$")
        pages = sorted(p for p in args.atlas_root.iterdir() if pattern.match(p.name))
    if not pages:
        raise SystemExit(f"no atlas pages found under {args.atlas_root}")

    measurements = []
    for page in pages:
        for epoch in ("summer", "winter"):
            try:
                measurements.append(measure_page(page, epoch))
            except (ValueError, IndexError) as exc:
                measurements.append({"page": page.name, "epoch": epoch, "error": str(exc)})

    usable = [
        m for m in measurements if m.get("mopd_measurable") and np.isfinite(m.get("mopd_cm", np.nan))
    ]
    summary = {}
    for epoch in ("summer", "winter"):
        values = [m["mopd_cm"] for m in usable if m["epoch"] == epoch]
        if values:
            summary[epoch] = {
                "pages": len(values),
                "median_mopd_cm": float(np.median(values)),
                "mad_mopd_cm": float(np.median(np.abs(np.asarray(values) - np.median(values)))),
                "median_transition_cm": float(
                    np.median([m["transition_cm"] for m in usable if m["epoch"] == epoch])
                ),
            }

    report = {
        "atlas_root": str(args.atlas_root),
        "method": (
            "Hann-windowed real FFT of the observed column; MOPD located at the "
            "steepest sustained drop of the smoothed log envelope."
        ),
        "boxcar_fwhm_constant": _BOXCAR_FWHM_CONSTANT,
        "summary": summary,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for epoch, values in summary.items():
        print(
            f"{epoch}: MOPD = {values['median_mopd_cm']:.2f} +- {values['mad_mopd_cm']:.2f} cm "
            f"over {values['pages']} pages, transition {values['median_transition_cm']:.2f} cm"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
