#!/usr/bin/env python
"""Reduce a full atlas summary to the committed record.

The batch driver's own summary carries per-page bookkeeping -- phase timings,
a per-stage optimizer log, line counts per species -- that is worth having
while a run is being judged and worth nothing afterwards. At 598 page-epochs it
reaches 1.6 MB, too much to keep in git and regrow on every re-run.

What survives is the science: the window, the fitted parameters, and the
quality numbers that accuracy claims are made from. Per-page fields that never
vary across the run are lifted into the settings block rather than repeated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEEP = (
    "page", "epoch", "wavenumber_cm1", "pixels", "reliable", "grid_points", "mopd_cm",
    "free_species", "negligible_telluric", "median_transmission",
    "continuum_level", "continuum_level_pixels",
    "parameters", "pixel_sigma", "residual_rms", "residual_rms_over_noise",
    "reduced_chi2", "all_stages_converged",
)
# Recorded once per page by the driver, but constant across a run.
HOIST = ("grid_margin_cm1", "line_margin_cm1", "line_budget", "precomputed_opacity", "layer_chunk")
DIGITS = 6


def round_floats(value):
    if isinstance(value, float):
        return float(f"{value:.{DIGITS}g}")
    if isinstance(value, dict):
        return {k: round_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v) for v in value]
    return value


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, default=root / "data/corrected/atlas/summary.json")
    parser.add_argument("--output", type=Path, default=root / "docs/arcturus_atlas_summary.json")
    parser.add_argument("--record", type=Path,
                        default=root / "data/corrected/atlas/arcturus_atlas.h5",
                        help="the run's authoritative record; named here so the "
                             "readable view points at what it is a view of")
    args = parser.parse_args()

    full = json.loads(args.full.read_text())
    results = full["results"]
    settings = dict(full.get("settings", {}))
    for name in HOIST:
        values = {json.dumps(row.get(name)) for row in results if "error" not in row}
        if len(values) == 1:
            settings[name] = json.loads(values.pop())

    trimmed = []
    for row in results:
        if "error" in row:
            trimmed.append({k: row[k] for k in ("page", "epoch", "error") if k in row})
            continue
        trimmed.append(round_floats({k: row[k] for k in KEEP if k in row}))

    out = {k: full[k] for k in ("generated", "atlas_root", "stellar", "profile", "physics")
           if k in full}
    out["settings"] = settings
    out["source"] = str(args.full.relative_to(root)) if args.full.is_relative_to(root) else str(args.full)
    if args.record.exists():
        # This file is a readable projection. The record is what a rebuild reads,
        # and it keeps full float64 where the rounding below does not.
        from tellurix.record import file_sha256, read_record

        out["record"] = {
            "path": str(args.record.relative_to(root)) if args.record.is_relative_to(root)
                    else str(args.record),
            "sha256": file_sha256(args.record),
            "format_version": read_record(args.record).format_version,
        }
    out["trimmed"] = (
        "Per-page timings, the per-stage optimizer log and per-species line counts are in "
        "the full summary named above, which is not kept in git. Floats here are rounded "
        f"to {DIGITS} significant digits, so rebuild from the record rather than from this."
    )
    # One page-epoch per line: a table of 598 rows reads better this way than
    # as 60,000 lines of nesting, and is a third of the size.
    head = json.dumps(out, indent=1)[:-2].rstrip()
    rows = ",\n  ".join(json.dumps(row, separators=(",", ":")) for row in trimmed)
    args.output.write_text(head + ',\n "results": [\n  ' + rows + "\n ]\n}\n", encoding="utf-8")
    print(f"{len(results)} page-epochs: {args.full.stat().st_size/1e6:.2f} MB "
          f"-> {args.output.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
