#!/usr/bin/env python
"""Measure a night's blaze, per physical order, from its lamp flats.

Reads the night's calibration bundle (RRISA ``CAL_URL``: ``flat_on``,
``flat_off``) and one extracted spectrum of the same night to name the orders,
and writes a :class:`tellurix.FlatBlaze` that the fitting drivers divide out
with ``--blaze``. See :mod:`tellurix.flat` for why.

    uv run python scripts/build_igrins_flat_blaze.py \\
        --cal-dir data/igrins/cals/20181220 --band H \\
        --spec data/igrins/20181220_0100/SDCH_20181220_0100.spec.fits \\
        --output data/calibration/igrins_blaze_dct2018_h.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cal-dir", type=Path, required=True)
    parser.add_argument("--band", choices=("H", "K"), required=True)
    parser.add_argument("--spec", type=Path, required=True,
                        help="any extracted spectrum of the night, to name the traced orders")
    parser.add_argument("--window", type=int, default=31,
                        help="Savitzky-Golay window, pixels: short enough to follow the "
                             "order-end roll-off, long enough not to carry the lamp's own "
                             "small-scale structure into the star")
    parser.add_argument("--interior-window", type=int, default=151,
                        help="the wider window used more than ~200 px from the order ends; "
                             "0 for one window everywhere")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tellurix import FlatBlaze

    on = sorted(args.cal_dir.glob(f"SDC{args.band}_*.flat_on.fits"))
    off = sorted(args.cal_dir.glob(f"SDC{args.band}_*.flat_off.fits"))
    if len(on) != 1 or len(off) != 1:
        raise SystemExit(f"expected one {args.band} flat_on and one flat_off in {args.cal_dir}; "
                         f"found {len(on)} and {len(off)}")
    blaze = FlatBlaze.from_flats(on[0], off[0], args.spec, window=args.window,
                                 interior_window=args.interior_window)
    blaze.save(args.output)
    refused = sorted(n for n in blaze.orders if not blaze.usable(n))
    print(f"{blaze.band}: {len(blaze.orders)} orders ({min(blaze.orders)}-{max(blaze.orders)}) "
          f"from {blaze.source['bands_traced']} traced bands, column mismatch "
          f"{blaze.source['column_mismatch']:.0f}; refused for lamp absorption: "
          f"{[f'{blaze.band}{n}' for n in refused]} -> {args.output}")


if __name__ == "__main__":
    main()
