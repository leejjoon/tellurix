#!/usr/bin/env python
"""Merge the per-shard records of one sharded run into a single record.

`fit_arcturus_batch.py` writes a record holding the pages *that process* fitted,
so a run split over four GPUs leaves four partial records. This joins them.

    uv run python scripts/merge_atlas_records.py \\
        --output data/corrected/atlas/arcturus_atlas.h5 \\
        data/corrected/atlas_full/record_shard*.h5

It works on the HDF5 datasets directly rather than rebuilding rows through
`write_record`. That is deliberate: the row dtype packs the column scales into
`log_column_*` fields and the continuum into a vector, and unpacking them back
into the mapping form `write_record` wants is exactly where a merge would
silently put a value in the wrong column. Concatenating the tables cannot.

Shards must agree on everything that is not per-row -- the parameter names, the
species, the key fields, and the whole of /config, /physics and /inputs -- and
the merge refuses if they do not, because two shards that disagree there did not
come from one run. Duplicate keys are refused for the same reason.

`--check` re-reads the output and confirms every input row survived with its
values intact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

BLOCKS = ("config", "physics", "inputs")
PER_ROW = ("pages", "sigma", "correlation", "ils_profile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", type=Path, nargs="+", help="the per-shard records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true",
                        help="re-read the merged file and verify every input row is in it")
    args = parser.parse_args()

    import h5py
    import numpy as np

    if len(args.records) < 2:
        raise SystemExit("merging needs at least two records")
    missing = [p for p in args.records if not p.exists()]
    if missing:
        raise SystemExit(f"these shards do not exist: {', '.join(str(p) for p in missing)}")

    handles = [h5py.File(path, "r") for path in args.records]
    try:
        first = handles[0]

        def fields_of(handle) -> list[str]:
            # A record written before key_fields was configurable has none, and
            # read_record takes that to mean the Arcturus pair. Match it.
            if "key_fields" not in handle.attrs:
                return ["page", "epoch"]
            return [k.decode() if isinstance(k, bytes) else str(k)
                    for k in handle.attrs["key_fields"]]

        key_fields = fields_of(first)

        # Everything that is not per-row has to be identical, or these shards
        # are not one run and the merged provenance would be a fiction.
        for path, handle in zip(args.records[1:], handles[1:]):
            for name in ("parameter_names", "species"):
                if not np.array_equal(first[name][:], handle[name][:]):
                    raise SystemExit(f"{path} disagrees with {args.records[0]} on {name}")
            if fields_of(handle) != key_fields:
                raise SystemExit(f"{path} disagrees on key_fields")
            if handle["pages"].dtype != first["pages"].dtype:
                raise SystemExit(f"{path} has a different row dtype")
            for block in BLOCKS:
                for key in set(first[block].attrs) | set(handle[block].attrs):
                    a, b = first[block].attrs.get(key), handle[block].attrs.get(key)
                    if not np.array_equal(a, b):
                        raise SystemExit(
                            f"{path} disagrees on /{block} {key}: {b!r} against {a!r}")
            if not np.allclose(handle["ils_velocity_kms"][:], first["ils_velocity_kms"][:]):
                raise SystemExit(f"{path} disagrees on the ILS velocity grid")

        joined = {name: np.concatenate([h[name][:] for h in handles]) for name in PER_ROW}
        counts = [h["pages"].shape[0] for h in handles]
        print(f"{len(handles)} shards, {' + '.join(str(c) for c in counts)} = {sum(counts)} rows")

        keys = np.array([tuple(row[f].decode() if isinstance(row[f], bytes) else str(row[f])
                               for f in key_fields) for row in joined["pages"]])
        flat = np.array([" ".join(k) for k in keys])
        unique, counts_of = np.unique(flat, return_counts=True)
        if unique.size != flat.size:
            repeated = unique[counts_of > 1][:5]
            raise SystemExit(
                f"{flat.size - unique.size} keys appear in more than one shard, starting with "
                f"{', '.join(repeated)}. Shards must not overlap.")

        order = np.argsort(flat, kind="stable")
        joined = {name: values[order] for name, values in joined.items()}

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, "w") as out:
            for key, value in first.attrs.items():
                out.attrs[key] = value
            for block in BLOCKS:
                group = out.create_group(block)
                for key, value in first[block].attrs.items():
                    group.attrs[key] = value
            for name in ("parameter_names", "species", "ils_velocity_kms"):
                out.create_dataset(name, data=first[name][:])
            for name in PER_ROW:
                dataset = out.create_dataset(name, data=joined[name], compression="gzip")
                for key, value in first[name].attrs.items():
                    dataset.attrs[key] = value
    finally:
        for handle in handles:
            handle.close()

    size = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({size:.2f} MB)")

    if not args.check:
        return
    from jax_telluric import read_record
    merged = read_record(args.output)
    seen = 0
    for path in args.records:
        shard = read_record(path)
        for index in range(len(shard.pages)):
            key = shard.key(index)
            row = merged.row(*key)
            for field in shard.pages.dtype.names:
                a, b = row[field], shard.pages[index][field]
                if not np.array_equal(a, b):
                    raise SystemExit(f"{' '.join(key)} differs on {field}: {a!r} against {b!r}")
            position = int(np.flatnonzero(
                np.all([merged.pages[f] == shard.pages[index][f] for f in merged.key_fields],
                       axis=0))[0])
            if not np.array_equal(merged.sigma[position], shard.sigma[index]):
                raise SystemExit(f"{' '.join(key)} differs on sigma")
            if not np.array_equal(merged.correlation[position], shard.correlation[index]):
                raise SystemExit(f"{' '.join(key)} differs on correlation")
            if not np.array_equal(merged.ils_profile[position], shard.ils_profile[index]):
                raise SystemExit(f"{' '.join(key)} differs on ils_profile")
            seen += 1
    print(f"checked {seen} rows against their shards: every field identical")


if __name__ == "__main__":
    main()
