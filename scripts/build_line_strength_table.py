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

``--detrend`` first divides each fit's ratios by its own linear trend,
1 + a + b E''/1000 cm-1 + c (log10 S + 24), fitted over that fit's features. The
held-out factors exceeded Wallace 2011's own ratios by +2.8% per 1000 cm-1 of E'':
each date's temperature profile, carried by the factors to a date it does not fit,
and shared by the building dates, so an agreement cut cannot catch it. The strength
term is each fit's width scale (a wider line reads a saturated core shallower). What
is left is what every date has in common: the line list's. The trend coefficients go
in the provenance.

``--min-dates`` averages within an observing date first and asks for that many
dates (``DATE``): `j83_2` and `j83_3` are one morning at two air masses, and a factor
they agree on can be that morning's error -- one-date factors scattered 2.6% rms about
Wallace 2011's own ratios, two-date ones 1.9%. The agreement cut then applies to the
date means. ``--baker-spectra`` (``analyse_baker_telluric.py --per-spectrum``) takes
Baker et al.'s ratios spectrum by spectrum, detrended one spectrum at a time with
``--detrend``, and their mean over spectra as the measurement of one more date.

``--previous-table`` iterates: the ratios were measured on fits that used that table,
so each line's factor is the previous one times the new feature's; a line whose
feature fails the cuts this time keeps the previous factor.

Baker's verdicts were made against the mean of *every* solar fit, so a held-out
test -- build from three fits, refit the fourth -- must leave ``--baker`` off.

    UV_CACHE_DIR=.uv-cache uv run python scripts/build_line_strength_table.py \\
        --fits j83_2,j83_3,niratl --output data/line_strength/held_out_wallace.npz
    UV_CACHE_DIR=.uv-cache uv run python scripts/build_line_strength_table.py \\
        --fits j83_2,j83_3,niratl --max-scatter 0.025 --output data/line_strength/held_out_wallace_agree.npz
    UV_CACHE_DIR=.uv-cache uv run python scripts/build_line_strength_table.py \\
        --fits j83_2,j83_3,niratl --detrend --max-scatter 0.025 \\
        --output data/line_strength/held_out_wallace_detrended.npz
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
# Which observing date each measurement belongs to. Wallace 2011 regions 1-2 are two
# scans of one morning (1989/10/13 #8, #7), measured as one fit.
DATE = {"wallace": "1989-10-13", "j83_2": "1983-06-26", "j83_3": "1983-06-26", "niratl": "niratl",
        "baker": "baker2020"}
MIN_BAKER_SPECTRA = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratios", type=Path, default=ROOT / "docs/line_depth_ratios.json")
    parser.add_argument("--fits", default="wallace,j83_2,j83_3,niratl", help="solar fits whose ratios to average")
    parser.add_argument("--baker", action="store_true", help="add Baker et al.'s vetted ratio")
    parser.add_argument("--baker-table", type=Path, default=ROOT / "docs/baker_line_corrections.csv")
    parser.add_argument("--min-measurements", type=int, default=2)
    parser.add_argument("--detrend", action="store_true",
                        help="divide each fit's ratios by its own trend in E'' and log S first")
    parser.add_argument("--min-dates", type=int, default=None,
                        help="average within an observing date first, and require this many dates")
    parser.add_argument("--baker-spectra", type=Path, default=None,
                        help="per-spectrum Baker ratios (analyse_baker_telluric.py --per-spectrum); with --baker")
    parser.add_argument("--previous-table", type=Path, default=None,
                        help="the table the measured fits used: factors multiply onto it")
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

    def detrended(name, ratios):
        """``ratios`` over this fit's own trend in E'' and strength, and the trend."""
        index = np.array(sorted(ratios))
        value = np.array([ratios[i] for i in index])
        x = np.column_stack([np.ones(index.size),
                             np.array([features[i]["e_lower_strongest"] for i in index]) / 1000.0,
                             np.log10([features[i]["s_max"] for i in index]) + 24.0])
        keep = np.abs(value - 1) < OUTLIER
        coef, *_ = np.linalg.lstsq(x[keep], value[keep] - 1, rcond=None)
        trends[name] = dict(zip(("offset", "per_1000_cm1_e_lower", "per_decade_s"), map(float, coef)))
        return dict(zip(index.tolist(), (value / (1 + x @ coef)).tolist()))

    trends = {}
    measured = {i: [] for i in range(len(features))}   # (date, ratio)
    for name in fits:
        ratios = {int(i): r for i, (r, _) in report["ratio_and_peak_tau"][name].items()}
        if args.detrend:
            ratios = detrended(name, ratios)
        for index, ratio in ratios.items():
            measured[index].append((DATE[name], ratio))
    baker_used = 0
    if args.baker:
        with args.baker_table.open() as handle:
            vetted = {int(row["feature"]): float(row["baker_ratio"])
                      for row in csv.DictReader(handle) if row["verdict"] == "vetted"}
        if args.baker_spectra is not None:
            data = np.load(args.baker_spectra)
            per = {}
            for spectrum in np.unique(data["spectrum"]):
                take = data["spectrum"] == spectrum
                ratios = dict(zip(data["feature"][take].tolist(), data["ratio"][take].tolist()))
                if args.detrend:
                    ratios = detrended(f"baker {spectrum}", ratios)
                for index, ratio in ratios.items():
                    if index in vetted and abs(ratio - 1) < OUTLIER:
                        per.setdefault(index, []).append(ratio)
            baker = {i: float(np.mean(v)) for i, v in per.items() if len(v) >= MIN_BAKER_SPECTRA}
        else:
            baker = detrended("baker", vetted) if args.detrend else vetted
        for index, ratio in baker.items():
            measured[index].append((DATE["baker"], ratio))
        baker_used = len(baker)

    factor_of, rejected = {}, 0
    for index, values in measured.items():
        values = [(date, v) for date, v in values if abs(v - 1) < OUTLIER]
        if args.min_dates is not None:
            dates = {}
            for date, v in values:
                dates.setdefault(date, []).append(v)
            estimates = [float(np.mean(v)) for v in dates.values()]
            if len(estimates) < args.min_dates:
                continue
        else:
            estimates = [v for _, v in values]
            if len(estimates) < args.min_measurements:
                continue
        if (args.max_scatter is not None and len(estimates) > 1
                and np.std(estimates, ddof=1) > args.max_scatter):
            rejected += 1
            continue
        factor_of[index] = float(np.mean(estimates))

    previous = None if args.previous_table is None else LineStrengthTable.load(args.previous_table)

    def earlier(members):
        if previous is None:
            return np.ones(len(members))
        factor = np.ones(len(members))
        for molecule in np.unique(lines["molecule"][members]):
            take = lines["molecule"][members] == molecule
            factor[take] = previous.factors(int(molecule), lines["isotope"][members][take],
                                            lines["nu"][members][take], lines["elower"][members][take])[0]
        return factor

    rows = []
    covered = set()
    for index in range(len(features)):
        members = features[index]["members"]
        before = earlier(members)
        if index not in factor_of and not np.any(before != 1.0):
            continue
        new = factor_of.get(index, 1.0)
        for member, old in zip(members, before):
            rows.append((lines["nu"][member], lines["molecule"][member], lines["isotope"][member],
                         lines["elower"][member], old * new))
        covered.add(index)
    rows.sort()
    nu, molecule, isotope, elower, factor = (np.array(column) for column in zip(*rows))
    built_from = dict(fits=fits, baker=args.baker, min_measurements=args.min_measurements, outlier=OUTLIER,
                      max_scatter=args.max_scatter, detrend=args.detrend, min_dates=args.min_dates,
                      baker_spectra=args.baker_spectra is not None,
                      previous_table=None if previous is None else args.previous_table.name)
    provenance = dict(
        built_from=built_from, built=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        ratios=args.ratios.name, ratios_sha256=sha256(args.ratios),
        baker_table=args.baker_table.name if args.baker else None,
        baker_table_sha256=sha256(args.baker_table) if args.baker else None,
        baker_spectra_sha256=sha256(args.baker_spectra) if args.baker and args.baker_spectra else None,
        previous_table_sha256=None if previous is None else sha256(args.previous_table),
        previous_built_from=None if previous is None else previous.provenance.get("built_from"),
        features=len(factor_of), features_in_table=len(covered), lines=int(nu.size), baker_ratios_used=baker_used,
        rejected_for_scatter=rejected, trends=trends,
        # This round's change, and the table's total.
        factor_rms=float(np.sqrt(np.mean((np.array(list(factor_of.values())) - 1) ** 2))),
        table_factor_rms=float(np.sqrt(np.mean((factor - 1) ** 2))))
    table = LineStrengthTable(molecule_id=molecule.astype(int), isotope=isotope.astype(int),
                              wavenumber_cm1=nu.astype(float), elower_cm1=elower.astype(float),
                              factor=factor.astype(float), provenance=provenance)
    table.save(args.output)
    print(f"{args.output}: {len(factor_of)} of {len(features)} features ({rejected} disagreeing), {nu.size} lines, "
          f"factor rms {provenance['factor_rms']:.3f}; built from {built_from}")


if __name__ == "__main__":
    main()
