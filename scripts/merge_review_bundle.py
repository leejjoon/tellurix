#!/usr/bin/env python
"""Join sharded review bundles into one the page can load.

Each shard numbers its chunks from zero, so merging is a renumber plus a
concatenation of the manifests -- the chunk *contents* and the byte offsets
inside them are untouched, which is what keeps this safe: nothing is re-encoded,
so nothing can be re-encoded wrongly.

    uv run python scripts/merge_review_bundle.py data/review_0 data/review_1 \\
        --output data/review
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def compact(entry: dict, global_states: set[str]) -> dict:
    """Shrink one window's record without losing anything the page shows.

    The manifest is fetched before the page can draw anything, and the scan
    detail dominates it: 46 molecules a window, with optical depths carried to
    17 digits that mean nothing past the fourth. Rows whose state is the same in
    every window (a molecule with no line file at all) are lifted out to a
    single global list instead of repeating 445 times.
    """

    def round_to(value, digits=4):
        if not isinstance(value, float) or value == 0.0 or not math.isfinite(value):
            return value
        return round(value, -int(math.floor(math.log10(abs(value)))) + (digits - 1))

    scan = []
    for row in entry["scan"]:
        if row["state"] in global_states:
            continue
        scan.append({k: round_to(v) for k, v in row.items() if v is not None})
    quality = {k: round_to(v, 6) if isinstance(v, float) else v
               for k, v in entry["quality"].items()}
    species = {name: {"log_scale": round_to(info["log_scale"], 6),
                      "at_bound": info["at_bound"]}
               for name, info in entry["species"].items()}
    trimmed = {k: v for k, v in entry.items()
               if k not in ("scan", "quality", "species", "product_drift")}
    trimmed["scan"] = scan
    trimmed["quality"] = quality
    trimmed["species"] = species
    return trimmed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("chunk_*.bin"):
        stale.unlink()

    windows, failures, species_all, offset, note = [], [], [], 0, None
    for shard in args.shards:
        blob = json.loads((shard / "manifest.json").read_text())
        note = note or blob.get("note")
        species_all = species_all or blob.get("species_all", [])
        chunks = sorted(shard.glob("chunk_*.bin"))
        for index, path in enumerate(chunks):
            shutil.copyfile(path, args.output / f"chunk_{offset + index:03d}.bin")
        for entry in blob["windows"]:
            windows.append({**entry, "chunk": entry["chunk"] + offset})
        failures.extend(blob.get("failures", []))
        offset += len(chunks)

    # The page steps through windows in this order, so sort once here rather
    # than in the browser: file 5 and file 4 of one window sit together.
    windows.sort(key=lambda w: (w["v1"], w["file"]))
    # A state that is identical in every window belongs to the run, not to a
    # window; the page shows it once rather than 445 times.
    global_states = {"no line file", "not in the profile"}
    by_state: dict[str, set[str]] = {}
    for entry in windows:
        for row in entry["scan"]:
            if row["state"] in global_states:
                by_state.setdefault(row["state"], set()).add(row["species"])
    windows = [compact(entry, global_states) for entry in windows]

    (args.output / "manifest.json").write_text(json.dumps({
        "windows": windows, "failures": failures,
        "species_all": species_all, "note": note,
        "global_states": {k: sorted(v) for k, v in by_state.items()},
    }, separators=(",", ":")) + "\n", encoding="utf-8")

    size = sum(f.stat().st_size for f in args.output.glob("chunk_*.bin"))
    print(f"{len(windows)} windows, {len(failures)} failed, {offset} chunks, "
          f"{size / 1e6:.1f} MB -> {args.output}")


if __name__ == "__main__":
    main()
