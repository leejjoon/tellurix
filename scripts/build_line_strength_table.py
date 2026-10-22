#!/usr/bin/env python
"""Build a per-line strength table from the measured feature ratios.

A feature's factor is the mean of its observed/model optical-depth ratios over
the solar fits named in ``--fits`` (docs/line_depth_ratios.json) and, with
``--baker``, Baker et al. 2020's ratio where ``vet_baker_line_corrections.py``
vetted it (docs/baker_line_corrections.csv). It needs ``--min-measurements`` of
them, each within ``OUTLIER`` of 1; elsewhere the factor is 1 and the line keeps
AER's strength. The factor goes to every member line of the feature's cluster
(``tellurix_fts.line_depth.cluster_features``): these spectra cannot split a
blend. The table is keyed by molecule, isotopologue, wavenumber and lower-state
energy, as AER prints them (``tellurix.LineStrengthTable``).

``--max-scatter`` keeps a factor only where its measurements agree: their sample
standard deviation at most that. Two fits that disagree otherwise make a factor, and
in the held-out Wallace test (docs/line_intensity_corrections.md, step 2) those were
the windows that got worse. 0.025 is one solar fit's own scatter, the rms by which the
other fits predict it (docs/baker_telluric_ratios.json, ``held_out``); for a pair it
asks for agreement within 0.035.

Baker's verdicts were made against the mean of *every* solar fit, so a held-out
test -- build from three fits, refit the fourth -- must leave ``--baker`` off.

    UV_CACHE_DIR=.uv-cache uv run python scripts/build_line_strength_table.py \\
        --fits j83_2,j83_3,niratl --output data/line_strength/held_out_wallace.npz
    UV_CACHE_DIR=.uv-cache uv run python scripts/build_line_strength_table.py \\
        --fits j83_2,j83_3,niratl --max-scatter 0.025 --output data/line_strength/held_out_wallace_agree.npz
    UV_CACHE_DIR=.uv-cache uv run python scripts/build_line_strength_table.py \\
        --baker --output data/line_strength/all_fits_baker.npz
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AER = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
OUTLIER = 0.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratios", type=Path, default=ROOT / "docs/line_depth_ratios.json")
    parser.add_argument("--fits", default="wallace,j83_2,j83_3,niratl", help="solar fits whose ratios to average")
    parser.add_argument("--baker", action="store_true", help="add Baker et al.'s vetted ratio")
    parser.add_argument("--baker-table", type=Path, default=ROOT / "docs/baker_line_corrections.csv")
    parser.add_argument("--min-measurements", type=int, default=2)
    parser.add_argument("--max-scatter", type=float, default=None,
                        help="keep a factor only if its measurements' standard deviation is at most this")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tellurix import LineStrengthTable
    from tellurix_fts.line_depth import cluster_features, read_feature_lines

    report = json.loads(args.ratios.read_text())
    fits = [name.strip() for name in args.fits.split(",") if name.strip()]
    unknown = set(fits) - set(report["ratio_and_peak_tau"])
    if unknown:
        raise SystemExit(f"no such solar fit in {args.ratios.name}: {sorted(unknown)}")
    lines = read_feature_lines(AER)
    features = cluster_features(lines)
    # The ratios are keyed by the catalogue's feature index; a rule that drifted
    # would hand each factor to the wrong lines.
    if [f["nu"] for f in features] != [f["nu"] for f in report["features"]]:
        raise SystemExit(f"the clusters no longer match {args.ratios.name}'s features")

    measured = {i: [] for i in range(len(features))}
    for name in fits:
        for index, (ratio, _) in report["ratio_and_peak_tau"][name].items():
            measured[int(index)].append(ratio)
    baker_used = 0
    if args.baker:
        with args.baker_table.open() as handle:
            for row in csv.DictReader(handle):
                if row["verdict"] == "vetted":
                    measured[int(row["feature"])].append(float(row["baker_ratio"]))
                    baker_used += 1

    factor_of, rejected = {}, 0
    for index, values in measured.items():
        values = [v for v in values if abs(v - 1) < OUTLIER]
        if len(values) < args.min_measurements:
            continue
        if args.max_scatter is not None and np.std(values, ddof=1) > args.max_scatter:
            rejected += 1
            continue
        factor_of[index] = float(np.mean(values))

    rows = []
    for index, factor in factor_of.items():
        for member in features[index]["members"]:
            rows.append((lines["nu"][member], lines["molecule"][member], lines["isotope"][member],
                         lines["elower"][member], factor))
    rows.sort()
    nu, molecule, isotope, elower, factor = (np.array(column) for column in zip(*rows))
    built_from = dict(fits=fits, baker=args.baker, min_measurements=args.min_measurements, outlier=OUTLIER,
                      max_scatter=args.max_scatter)
    provenance = dict(
        built_from=built_from, built=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        ratios=args.ratios.name, ratios_sha256=sha256(args.ratios),
        baker_table=args.baker_table.name if args.baker else None,
        baker_table_sha256=sha256(args.baker_table) if args.baker else None,
        features=len(factor_of), lines=int(nu.size), baker_ratios_used=baker_used,
        rejected_for_scatter=rejected,
        factor_rms=float(np.sqrt(np.mean((np.array(list(factor_of.values())) - 1) ** 2))))
    table = LineStrengthTable(molecule_id=molecule.astype(int), isotope=isotope.astype(int),
                              wavenumber_cm1=nu.astype(float), elower_cm1=elower.astype(float),
                              factor=factor.astype(float), provenance=provenance)
    table.save(args.output)
    print(f"{args.output}: {len(factor_of)} of {len(features)} features ({rejected} disagreeing), {nu.size} lines, "
          f"factor rms {provenance['factor_rms']:.3f}; built from {built_from}")


if __name__ == "__main__":
    main()
