"""Score a line-coupled FTS run against the uncoupled run it replaced.

A coupled `fit_fts_batch.py --line-coupling` run fits every window with coupled
lines both ways and keeps the better (`tellurix_fts.window.fit_window`); its
summary carries both residuals per window. This reads that summary and the
uncoupled run it replaced and writes, per file, which windows chose coupling,
by how much, and how each window's residual moved against the old run, to
``docs/solar_line_coupling.json``.

    uv run python scripts/score_fts_coupling.py \\
        --pair data/corrected/superseded_pre_coupling/solar/ftsspec_901218_5_summary.json \\
               data/corrected/solar/ftsspec_901218_5_summary.json \\
        --pair ...4_summary.json ...4_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fitted(path: Path) -> dict:
    return {round(r["v1"], 3): r for r in json.loads(path.read_text())["results"] if r.get("parameters")}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pair", nargs=2, type=Path, action="append", required=True,
                        metavar=("OLD", "NEW"), help="the uncoupled summary and the coupled one")
    parser.add_argument("--output", type=Path, default=root / "docs/solar_line_coupling.json")
    args = parser.parse_args()

    files = {}
    for old_path, new_path in args.pair:
        old, new = fitted(old_path), fitted(new_path)
        if set(old) != set(new):
            raise SystemExit(f"{new_path.name}: the two runs did not fit the same windows")
        windows = []
        for v1 in sorted(new):
            row, before = new[v1], old[v1]
            trial = row.get("line_coupling_trial")
            windows.append({
                "v1": row["v1"], "v2": row["v2"],
                "choice": trial["chosen"] if trial else "no coupled lines",
                "rms_coupled": trial and trial["rms_coupled"],
                "rms_uncoupled": trial and trial["rms_uncoupled"],
                "refused": row.get("line_coupling_refused"),
                "rms_over_noise_old": round(before["residual_rms_over_noise"], 4),
                "rms_over_noise_new": round(row["residual_rms_over_noise"], 4),
                "change": round(row["residual_rms_over_noise"] / before["residual_rms_over_noise"] - 1, 5),
            })
        change = np.array([w["change"] for w in windows])
        lost = [w for w in windows if w["choice"] == "uncoupled" and w["rms_coupled"]]
        loss = np.array([w["rms_coupled"] / w["rms_uncoupled"] - 1 for w in lost])
        files[new_path.name.replace("_summary.json", "")] = {
            "old": str(old_path), "new": str(new_path),
            "windows": len(windows),
            "chose_coupled": sum(w["choice"] == "coupled" for w in windows),
            "chose_uncoupled": sum(w["choice"] == "uncoupled" for w in windows),
            "no_coupled_lines": sum(w["choice"] == "no coupled lines" for w in windows),
            "refused": sum(bool(w["refused"]) for w in windows),
            "better_by_2pct": int(np.sum(change < -0.02)),
            "better_by_10pct": int(np.sum(change < -0.10)),
            "worse_by_0p1pct": int(np.sum(change > 0.001)),
            "coupling_loss_where_rejected": {
                "median": round(float(np.median(loss)), 5) if loss.size else None,
                "max": round(float(np.max(loss)), 4) if loss.size else None,
            },
            "per_window": windows,
        }
    report = {
        "about": "Line coupling chosen window by window on the NSO Kitt Peak solar runs, against "
                 "the uncoupled runs they replaced. Written by scripts/score_fts_coupling.py.",
        "files": files,
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, f in files.items():
        print(name, {k: v for k, v in f.items() if k not in ("per_window", "old", "new")})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
