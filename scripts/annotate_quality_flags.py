#!/usr/bin/env python
"""Add the upper-bound flag to transmission files written before it existed.

Uses the export's own `write_upper_bound_flags`, on the parameters each file
already carries, so an annotated file and a freshly exported one hold the same
flag by construction -- and re-exporting an hour of windows to add one boolean
per row is not needed.

    UV_CACHE_DIR=.uv-cache uv run python scripts/annotate_quality_flags.py \\
        data/corrected/*_transmission.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_transmission_hdf5 import write_upper_bound_flags  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()

    import h5py

    for path in args.files:
        with h5py.File(path, "r+") as handle:
            rows = handle["parameters"][()]
            species = [n[len("log_column_"):] for n in rows.dtype.names
                       if n.startswith("log_column_")]
            flagged = write_upper_bound_flags(handle, rows, species)
            names = [n.decode() for n in handle["species_at_upper_bound"][()] if n]
        print(f"{path.name}: {flagged} of {len(rows)} rows flagged"
              + (f" ({', '.join(names)})" if names else ""))


if __name__ == "__main__":
    main()
