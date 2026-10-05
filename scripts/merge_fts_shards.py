#!/usr/bin/env python
"""Join the per-shard summaries of a sharded `fit_fts_batch.py` run into one.

Each shard (`--shard I/N`) must write its own `--summary`, or two processes
rewrite one file; this concatenates them into the run's default summary name.
Then rerun the batch driver with the same arguments and no `--shard`: `--resume`
skips every window already in the summary, and the driver writes one record
from the saved fits -- the record is assembled from the `.npz` files, not from
memory, so nothing is refitted.

    uv run python scripts/merge_fts_shards.py data/corrected/solar ftsspec_901218_5 \\
        --shards ftsspec_901218_5_shard0.json ftsspec_901218_5_shard1.json

Refuses a window present in two shards, which would mean the shards overlapped.
"""
import argparse, json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("stem", help="the spectrum's stem, which names the merged <stem>_summary.json")
    parser.add_argument("--shards", nargs="+", required=True, help="shard summaries, relative to output_dir")
    args = parser.parse_args()
    blobs = [json.loads((args.output_dir / name).read_text()) for name in args.shards]
    rows = sorted((row for blob in blobs for row in blob["results"]), key=lambda r: r["v1"])
    if len({round(r["v1"], 3) for r in rows}) != len(rows):
        raise SystemExit("a window appears in more than one shard")
    if len({json.dumps(blob["physics"], sort_keys=True) for blob in blobs}) != 1:
        raise SystemExit("the shards were not run with the same physics")
    merged = {**blobs[0], "results": rows}
    out = args.output_dir / f"{args.stem}_summary.json"
    out.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"{out}: {len(rows)} windows, {sum(1 for r in rows if r.get('npz'))} fitted, "
          f"{sum(1 for r in rows if r.get('error'))} errors")


if __name__ == "__main__":
    main()
