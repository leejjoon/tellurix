#!/usr/bin/env python
"""Compare fit_igrins_standard.py runs of the same frames, order by order.

For each run, over the orders every run has: the median residual after the
fixed-pattern pass and before it, the residual in the reddest 150 reliable
samples of each order (where the blaze rolls off), the rms of the response
pattern the run still needed, the block-averaged residual at several scales
(scaled so white noise reads 1), and for H the order-to-order spread of the
fitted stellar velocity over Brackett orders.

    uv run python scripts/compare_igrins_runs.py --band H \\
        --run "degree 9=data/corrected/igrins/ladder_a0v" \\
        --run "blaze + degree 9=data/corrected/igrins/blazeB9_dct2018_h" \\
        --output docs/igrins_blaze_dct2018.json

``label=record_dir[:cache_dir]``: the record, and the npz/summary cache when it
lives elsewhere. ``--orders-from-blaze`` restricts to orders a FlatBlaze can
serve, so a blaze run and a run without one are compared like for like.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

BLOCKS = (1, 4, 16, 64, 256)
# Orders whose Brackett line sits well inside the usable pixels (docs/igrins_a0v.md).
BRACKETT = {98, 103, 106, 109, 111, 113, 114, 115, 116, 117, 119, 120}


def scaled_block_rms(z, index, block):
    m = index.size // block
    if m < 2:
        return np.nan
    means = z[index[:m * block]].reshape(m, block).mean(axis=1)
    return float(np.sqrt(np.mean(means ** 2)) * np.sqrt(block))


def measure(record_dir: Path, cache: Path, band: str, orders: set[int]) -> dict:
    import h5py

    pages = h5py.File(record_dir / "record.h5", "r")["pages"][:]
    pages = pages[[int(n) in orders for n in pages["order_number"]]]
    first = []
    for path in glob.glob(str(cache / f"SDC{band}_*_summary.json")):
        for row in json.loads(Path(path).read_text())["results"]:
            if int(row["order"]) in orders and "residual_rms_over_noise_uncorrected" in row:
                first.append(row["residual_rms_over_noise_uncorrected"])
    edge, pattern, interior, red = [], [], {b: [] for b in BLOCKS}, {b: [] for b in BLOCKS}
    for page in pages:
        path = cache / f"{page['frame'].decode()}_{page['order'].decode()}.npz"
        if not path.exists():
            continue
        with np.load(path) as d:
            ok = d["reliable"].astype(bool)
            z = np.where(ok, d["residual"] / d["uncertainty"], np.nan)
            index = np.flatnonzero(ok)
            if index.size > 600:
                # ascending wavenumber: the red end comes first
                edge.append(float(np.sqrt(np.nanmean(z[index[:150]] ** 2))))
                for b in BLOCKS:
                    interior[b].append(scaled_block_rms(z, index[300:-300], b))
                    red[b].append(scaled_block_rms(z, index[:300], b))
            if "response_pattern" in d.files:
                p = d["response_pattern"][d["mask"].astype(bool)]
                pattern.append(float(np.sqrt(np.mean(p ** 2))))
    out = {
        "order_frames": int(len(pages)),
        "median_residual_with_pattern": float(np.median(pages["residual_rms_over_noise"])),
        "median_residual_without_pattern": float(np.median(first)) if first else None,
        "median_red_edge_rms": float(np.median(edge)) if edge else None,
        "median_pattern_rms": float(np.median(pattern)) if pattern else None,
        "interior_scaled_rms": {str(b): float(np.nanmedian(v)) for b, v in interior.items() if v},
        "red_end_scaled_rms": {str(b): float(np.nanmedian(v)) for b, v in red.items() if v},
    }
    if band == "H":
        brackett = np.isin(pages["order_number"], list(BRACKETT))
        spreads = []
        for frame in set(pages["frame"]):
            v = pages["stellar_velocity_kms"][(pages["frame"] == frame) & brackett]
            v = v[np.abs(np.abs(v) - 60.0) > 1e-3]
            if v.size > 3:
                spreads.append(float(np.std(v)))
        out["brackett_stellar_velocity_spread_kms"] = float(np.median(spreads)) if spreads else None
        out["brackett_stellar_velocity_railed"] = float(np.mean(
            np.abs(np.abs(pages["stellar_velocity_kms"][brackett]) - 60.0) < 1e-3))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--band", choices=("H", "K"), required=True)
    parser.add_argument("--run", action="append", required=True,
                        help="label=record_dir[:cache_dir]")
    parser.add_argument("--orders-from-blaze", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import h5py

    runs = {}
    for spec in args.run:
        label, paths = spec.split("=", 1)
        record_dir, _, cache = paths.partition(":")
        runs[label] = (Path(record_dir), Path(cache or record_dir))
    orders = None
    for record_dir, _ in runs.values():
        present = {int(n) for n in h5py.File(record_dir / "record.h5", "r")["pages"]["order_number"]}
        orders = present if orders is None else orders & present
    if args.orders_from_blaze is not None:
        from tellurix import FlatBlaze

        blaze = FlatBlaze.load(args.orders_from_blaze)
        orders = {n for n in orders if blaze.usable(n)}
    report = {"band": args.band, "orders": sorted(orders),
              "runs": {label: {"record": str(r), "cache": str(c), **measure(r, c, args.band, orders)}
                       for label, (r, c) in runs.items()}}
    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    existing[args.band] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(existing, indent=1) + "\n")
    for label, r in report["runs"].items():
        print(f"{args.band} {label:24s} with pattern {r['median_residual_with_pattern']:.3f}, without "
              f"{r['median_residual_without_pattern'] or float('nan'):.3f}, red edge "
              f"{r['median_red_edge_rms'] or float('nan'):.2f}, pattern rms {r['median_pattern_rms'] or float('nan'):.4f}")


if __name__ == "__main__":
    main()
