#!/usr/bin/env python
"""Compare two `fit_fts_batch.py` runs of the same spectrum, window by window.

The check before promoting a rerun (docs/solar_fit_plan.md, "Done 2026-10-05"):
the same windows fitted, every species' column change, the residual change, and
any column newly at a bound. A column that moved is judged against its formal
sigma from the window's `.npz` -- several percent in an ill-conditioned window
with an identical residual is a flat objective, not a change of physics.

    uv run python scripts/compare_fts_runs.py OLD_summary.json NEW_summary.json [--sigma]
"""
import argparse, json
from pathlib import Path
import numpy as np


def rows(path):
    return {round(r["v1"], 3): r for r in json.loads(Path(path).read_text())["results"] if r.get("parameters")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--threshold", type=float, default=0.01, help="list column changes above this (ln scale)")
    parser.add_argument("--sigma", action="store_true", help="read each listed window's formal sigma from its npz")
    args = parser.parse_args()
    old, new = rows(args.old), rows(args.new)
    common = sorted(set(old) & set(new))
    print(f"windows fitted: old {len(old)}, new {len(new)}, common {len(common)}; "
          f"only old {sorted(set(old) - set(new))[:5]}, only new {sorted(set(new) - set(old))[:5]}")
    for species in sorted({s for r in new.values() for s in r["species"]}):
        d = np.array([new[k]["parameters"][species] - old[k]["parameters"][species] for k in common
                      if species in new[k]["parameters"] and species in old[k]["parameters"]])
        if d.size:
            print(f"  {species:5s} n={d.size:3d} column change: median {np.median(abs(d)) * 100:.3f}%, "
                  f"max {abs(d).max() * 100:.3f}%")
    ratio = np.array([new[k]["residual_rms"] / old[k]["residual_rms"] - 1 for k in common])
    print(f"  residual rms change: median {np.median(ratio) * 100:+.3f}%, "
          f"range {ratio.min() * 100:+.2f}% .. {ratio.max() * 100:+.2f}%")
    bound = [(k, new[k]["at_bound"]) for k in common
             if set(new[k]["at_bound"]) - set(old[k]["at_bound"]) - {"lsf_sigma_kms", "velocity_kms"}]
    print("  newly at a column bound:", bound[:10])
    print(f"columns that moved more than {args.threshold:g} in ln scale:")
    for k in common:
        for species in new[k]["species"]:
            if species not in old[k]["parameters"]:
                continue
            change = new[k]["parameters"][species] - old[k]["parameters"][species]
            if abs(change) <= args.threshold:
                continue
            sigma = ""
            if args.sigma:
                with np.load(args.new.parent / new[k]["npz"]) as z:
                    names = list(z["parameter_names"])
                    sigma = f" (sigma {float(z['sigma'][names.index(species)]):.3f})"
            print(f"  {k:8.1f} {species:5s} {change:+.3f}{sigma}  rms {old[k]['residual_rms']:.5f} -> "
                  f"{new[k]['residual_rms']:.5f}  bound {new[k]['at_bound']}")


if __name__ == "__main__":
    main()
