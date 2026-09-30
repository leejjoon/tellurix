#!/usr/bin/env python
"""Build a night's telluric calibration from its fitted standards.

Takes a ``fit_igrins_standard.py`` run -- one band of one night, fitted with the
fixed-pattern pass -- and writes the per-order calibration a science frame of
that night is fitted against (``fit_igrins_science.py``). See
:mod:`tellurix.night` for what it holds and why.

    uv run python scripts/build_igrins_calibration.py \\
        --run-dir data/corrected/igrins/ladder_a0v --output data/calibration/dct2018_h.h5

``--record`` names the record when it is not in the run directory -- a committed
record beside a cache that lives in another checkout.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run's npz cache, which the response pattern is measured from")
    parser.add_argument("--record", type=Path, default=None,
                        help="default: RUN_DIR/record.h5")
    parser.add_argument("--exclude", default="",
                        help="comma-separated frames to leave out, e.g. to test on one of them")
    parser.add_argument("--master-pattern", type=Path, default=None,
                        help="take the response from a MasterPattern instead of the night's "
                             "own standards -- for a night with too few of them")
    parser.add_argument("--pattern-smooth-pixels", type=int, default=51,
                        help="the guard against the pattern absorbing our own line-list "
                             "error; see leave_one_out_patterns")
    parser.add_argument("--minimum-frames", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tellurix import MasterPattern, NightCalibration

    calibration = NightCalibration.from_run(
        args.record or args.run_dir / "record.h5", args.run_dir,
        exclude=[f for f in args.exclude.split(",") if f],
        master=None if args.master_pattern is None else MasterPattern.load(args.master_pattern),
        smooth_pixels=args.pattern_smooth_pixels, minimum_frames=args.minimum_frames)
    calibration.save(args.output)
    with_pattern = sum(o.pattern is not None for o in calibration.orders.values())
    frames = sorted({f for o in calibration.orders.values() for f in o.frames})
    print(f"{calibration.band}: {len(calibration.orders)} orders "
          f"({min(calibration.orders)}-{max(calibration.orders)}), {with_pattern} with a "
          f"response pattern, from {len(frames)} standards -> {args.output}")


if __name__ == "__main__":
    main()
