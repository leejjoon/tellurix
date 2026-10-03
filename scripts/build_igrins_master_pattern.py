#!/usr/bin/env python
"""Build one band's master instrument-response pattern from several nights.

The response belongs to the spectrograph: four nights on three telescopes over
five years correlate at median r = +0.89 to +0.95, and their median captures
74-88% of each night's own pattern (``docs/igrins_a0v.md``). A night with too
few standards to measure its own takes this one instead, in both the standards
fit (``fit_igrins_standard.py --response-pattern``) and its calibration
(``build_igrins_calibration.py --master-pattern``).

    uv run python scripts/build_igrins_master_pattern.py \\
        --calibration data/calibration/igrins_dct2016_h.h5 data/calibration/igrins_mcdonald2017_h.h5 \\
        --output data/calibration/igrins_master_h.h5

To test the pattern on a night, build it without that night.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True,
                        help="NightCalibrations of one band, each with its own pattern")
    parser.add_argument("--minimum-nights", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tellurix_igrins import MasterPattern, NightCalibration

    master = MasterPattern.from_calibrations(
        [NightCalibration.load(path) for path in args.calibration],
        minimum_nights=args.minimum_nights)
    master.save(args.output)
    print(f"{master.band}: {len(master.orders)} orders "
          f"({min(master.orders)}-{max(master.orders)}) from {len(args.calibration)} nights "
          f"-> {args.output}")


if __name__ == "__main__":
    main()
