#!/usr/bin/env python
"""Collect ``fit_igrins_science.py`` runs into one compact report.

Per exposure and band: the frame's fitted water and velocity shifts (with and
without clipping), where it sits against the calibrating standards in time --
inside their span or extrapolated -- the median residual, and the H/K water
agreement when both bands were fitted.

    uv run python scripts/summarize_igrins_science.py \\
        --runs data/corrected/igrins/science_dct2018 data/corrected/igrins/science_dct2018_standards \\
        --output docs/igrins_science_dct2018.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The science driver's threshold, 3-4 times the 0.005-0.006 rms by which clean
# H and K water shifts agree; see fit_igrins_science.py.
BAND_DISAGREEMENT = 0.02


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exposures = {}
    for run in args.runs:
        for path in sorted(run.glob("*_science_summary.json")):
            summary = json.loads(path.read_text())
            observation = summary["observation"]
            band = observation["band"]
            key = Path(observation["path"]).name.split(".")[0][5:]
            times = summary["calibration"]["standards_mjd"]
            mjd = observation["mjd"]
            hours = lambda t: round(24.0 * t, 2)  # noqa: E731
            if mjd < times[0]:
                where = f"before the first standard by {hours(times[0] - mjd)} h"
            elif mjd > times[-1]:
                where = f"after the last standard by {hours(mjd - times[-1])} h"
            else:
                where = "between standards"
            rows = summary["results"]
            clipped = [r["clipped_fraction"] for r in rows if r.get("clipped_fraction") is not None]
            entry = exposures.setdefault(key, {
                "run": run.name, "object": observation["object"],
                "object_type": observation["object_type"], "mjd": mjd, "time": where,
                "bands": {},
            })
            entry["bands"][band] = {
                "airmass": round(observation["airmass"], 4),
                "orders": len(rows),
                "order_source": observation["order_source"],
                "water_shift": summary["shifts"]["frame"]["log_column_H2O"],
                "velocity_shift_kms": summary["shifts"]["frame"]["velocity_kms"],
                "water_shift_clipped": (summary["shifts"]["frame_clipped"] or {}).get(
                    "log_column_H2O"),
                "velocity_shift_clipped_kms": (summary["shifts"]["frame_clipped"] or {}).get(
                    "velocity_kms"),
                "median_clipped_fraction": float(np.median(clipped)) if clipped else None,
                "median_residual_over_noise": float(np.nanmedian(
                    [r["residual_rms_over_noise"] for r in rows])),
                "stellar_source": summary["settings"]["stellar"],
            }

    # Recomputed from each band's own summary rather than taken from the run:
    # the bands of one exposure may have been fitted in different runs, and a
    # summary's own comparison is only as current as the other band was then.
    for entry in exposures.values():
        if {"H", "K"} <= set(entry["bands"]):
            water = {b: (entry["bands"][b]["water_shift_clipped"]
                         if entry["bands"][b]["water_shift_clipped"] is not None
                         else entry["bands"][b]["water_shift"]) for b in "HK"}
            difference = water["H"] - water["K"]
            entry["band_agreement"] = {"H": water["H"], "K": water["K"],
                                       "difference": difference,
                                       "flag": bool(abs(difference) > BAND_DISAGREEMENT)}

    def rounded(value):
        if isinstance(value, float):
            return round(value, 5)
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rounded(v) for v in value]
        return value

    differences = {t: [abs(e["band_agreement"]["difference"]) for e in exposures.values()
                       if "band_agreement" in e and e["object_type"] == t] for t in ("STD", "TAR")}
    report = {
        "description": "IGRINS science-frame telluric correction against a night calibration; "
                       "see scripts/fit_igrins_science.py and docs/igrins_science.md",
        "runs": [str(r) for r in args.runs],
        "band_agreement": {
            t: {"exposures": len(v), "median_abs_difference": float(np.median(v)) if v else None,
                "max_abs_difference": float(np.max(v)) if v else None,
                "flagged": sum(e["band_agreement"]["flag"] for e in exposures.values()
                               if "band_agreement" in e and e["object_type"] == t)}
            for t, v in differences.items()},
        "exposures": dict(sorted(exposures.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rounded(report), indent=1) + "\n")
    print(json.dumps(rounded(report["band_agreement"]), indent=1))


if __name__ == "__main__":
    main()
