#!/usr/bin/env python
"""Per telluric feature, observed over model optical depth, in every solar fit that covers it.

A feature is a cluster of AER H2O and O2 lines linked within 0.08 cm-1 whose strongest
line has S >= 3e-25. In each fit its ratio is sum(tau_obs) / sum(tau_model) over the
pixels within 0.05 cm-1 of the cluster's S-weighted centre that are in the fit mask,
solar-clean (stellar model above 0.97 of continuum), not saturated (model above 0.02)
and absorbing (model tau above 0.1), tau taken against the stellar model. A feature
under a solar line or saturated in every fit is not measured.

If one ratio repeats from fit to fit, the error is the line list's. Two checks of
that are reported: the correlation between fits, and, for a same-day pair at two air
masses (``--pair``), the total-least-squares slope of one's (ratio - 1) on the
other's -- 1 for a line-list error, X_low/X_high for a solar-model error, whose
optical depth does not grow with air mass. And a held-out test: each fit's ratios
predicted by the mean of the others'. See docs/line_intensity_corrections.md.

    uv run python scripts/measure_line_depth_ratios.py \\
        wallace='data/corrected/solar/wallace2011/*.npz' \\
        j83_2='data/corrected/solar/ftsspec_830626_2_*.npz' \\
        j83_3='data/corrected/solar/ftsspec_830626_3_*.npz' \\
        niratl='data/corrected/solar/niratl/*.npz' \\
        --pair j83_2=5.37 j83_3=3.085 --output docs/line_depth_ratios.json
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AER = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
MOLECULES = ("01_H2O", "07_O2")
RANGE_CM1 = (8800.0, 17600.0)
LINK_CM1 = 0.08
MIN_FEATURE_S = 3e-25
MIN_LINE_S = 3e-27
CORE_CM1 = 0.05
# A ratio this far from 1 is a missing line, a solar line the clean cut let through,
# or a misfit window; it would dominate every rms.
OUTLIER = 0.5


def read_lines():
    lines = []
    for molecule in MOLECULES:
        for record in open(AER / molecule / molecule):
            try:
                nu = float(record[3:15]); strength = float(record[15:25].replace("D", "E"))
                lower = float(record[45:55])
            except ValueError:
                continue
            if RANGE_CM1[0] < nu < RANGE_CM1[1] and strength > MIN_LINE_S:
                lines.append((nu, strength, lower, int(record[:2])))
    return np.array(sorted(lines)).T


def features(nu, strength, lower, molecule):
    out = []
    for index in np.split(np.arange(nu.size), np.where(np.diff(nu) > LINK_CM1)[0] + 1):
        weight = strength[index]
        if weight.max() < MIN_FEATURE_S:
            continue
        strongest = index[np.argmax(weight)]
        out.append(dict(nu=round(float(np.sum(nu[index] * weight) / weight.sum()), 4), lines=int(index.size),
                        s_max=float(weight.max()), nu_strongest=float(nu[strongest]),
                        e_lower_strongest=float(lower[strongest]), molecule=int(molecule[strongest])))
    return out


def measure(files, centres):
    ratios = {}
    for path in files:
        with np.load(path) as data:
            nu = np.asarray(data["wavenumber_cm1"]); order = np.argsort(nu); nu = nu[order]
            take = lambda key: np.asarray(data[key])[order]
            continuum = take("continuum")
            observed = take("flux") / continuum; model = take("model_flux") / continuum
            solar = take("stellar_only_pixels") / continuum; mask = take("mask").astype(bool)
        tau_model = -np.log(np.clip(model / solar, 1e-6, None))
        tau_observed = -np.log(np.clip(observed / solar, 1e-6, None))
        good = mask & (solar > 0.97) & (model > 0.02) & (tau_model > 0.1)
        for i in np.where((centres > nu[10]) & (centres < nu[-10]))[0]:
            core = (np.abs(nu - centres[i]) < CORE_CM1) & good
            if core.sum() >= 3:
                ratios[int(i)] = [round(float(tau_observed[core].sum() / tau_model[core].sum()), 4),
                                  round(float(tau_model[core].max()), 3)]
    return ratios


def paired(table, a, b):
    common = [i for i in table[a] if i in table[b]]
    x = np.array([table[a][i][0] for i in common]) - 1; y = np.array([table[b][i][0] for i in common]) - 1
    keep = (np.abs(x) < OUTLIER) & (np.abs(y) < OUTLIER)
    return x[keep], y[keep]


def held_out(table, target, others):
    observed, predicted = [], []
    for i, (ratio, _) in table[target].items():
        rest = [table[k][i][0] for k in others if i in table[k]]
        if rest:
            observed.append(ratio); predicted.append(np.mean(rest))
    observed, predicted = np.array(observed), np.array(predicted)
    keep = (np.abs(observed - 1) < OUTLIER) & (np.abs(predicted - 1) < OUTLIER)
    observed, predicted = observed[keep], predicted[keep]
    far = np.abs(observed - 1) > 0.08
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    return dict(features=int(keep.sum()), rms_before=rms(observed - 1), rms_after=rms(observed / predicted - 1),
                features_over_8pct=int(far.sum()), rms_over_8pct_before=rms(observed[far] - 1),
                rms_over_8pct_after=rms(observed[far] / predicted[far] - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fits", nargs="+", metavar="NAME=GLOB", help="a fit_fts_batch.py run's window npz files")
    parser.add_argument("--pair", nargs=2, metavar="NAME=AIRMASS", help="a same-day pair, for the air-mass test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fits = dict(item.split("=", 1) for item in args.fits)

    nu, strength, lower, molecule = read_lines()
    clusters = features(nu, strength, lower, molecule)
    centres = np.array([c["nu"] for c in clusters])
    table = {}
    for name, pattern in fits.items():
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"{name}: no files match {pattern}")
        table[name] = measure(files, centres)
        print(f"{name}: {len(files)} windows, {len(table[name])} features measured")
    names = list(table)

    pairs = {}
    for a, b in itertools.combinations(names, 2):
        x, y = paired(table, a, b)
        pairs[f"{a}/{b}"] = dict(features=int(x.size), correlation=float(np.corrcoef(x, y)[0, 1]),
                                 rms=[float(x.std()), float(y.std())])
        print(f"  {a:8s} {b:8s} n={x.size:5d} r={pairs[f'{a}/{b}']['correlation']:.2f}")
    airmass_test = None
    if args.pair:
        (high, x_high), (low, x_low) = (p.split("=") for p in args.pair)
        x, y = paired(table, low, high)
        _, vectors = np.linalg.eigh(np.cov(np.vstack([x, y])))
        slope = float(vectors[1, -1] / vectors[0, -1])
        airmass_test = dict(high=high, low=low, tls_slope_high_on_low=slope, line_list_expects=1.0,
                            solar_model_expects=float(x_low) / float(x_high))
        print(f"air-mass test {high} on {low}: slope {slope:.2f} (line list 1, solar {float(x_low) / float(x_high):.2f})")
    prediction = {}
    for target in names:
        prediction[target] = held_out(table, target, [k for k in names if k != target])
        p = prediction[target]
        print(f"held out {target}: rms {p['rms_before']:.3f} -> {p['rms_after']:.3f}; "
              f"{p['features_over_8pct']} over 8%: {p['rms_over_8pct_before']:.3f} -> {p['rms_over_8pct_after']:.3f}")
    coverage = {}
    consistent = {f"{t:.2f}": 0 for t in (0.03, 0.05, 0.10)}
    for i in range(len(clusters)):
        ratios = np.array([table[k][i][0] for k in names if i in table[k]])
        coverage[str(ratios.size)] = coverage.get(str(ratios.size), 0) + 1
        if ratios.size >= 3:
            mean = ratios.mean(); error = max(ratios.std(ddof=1) / np.sqrt(ratios.size), 0.01)
            for t in consistent:
                consistent[t] += int(abs(mean - 1) > float(t) and abs(mean - 1) / error > 3)
    print("features by number of fits measuring them:", coverage)
    print("measured in >= 3 fits and off at > 3 sigma by more than:", consistent)

    args.output.write_text(json.dumps(dict(
        method=__doc__.split("\n\n    uv run")[0],
        fits=fits, pairs=pairs, airmass_test=airmass_test, held_out=prediction,
        coverage=coverage, consistent_in_three_or_more=consistent,
        features=clusters, ratio_and_peak_tau={k: {str(i): v for i, v in d.items()} for k, d in table.items()},
    ), indent=None, separators=(",", ":")) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
