#!/usr/bin/env python
"""How far corrected IAG 2014 spectra are from Baker, Blake & Reiners 2020's telluric-free IAG atlas.

The flatness score (`score_telluric_correction.py`) judges a correction against a
running mean of nearby clean pixels, which assumes the Sun is flat under a telluric
line. Baker et al.'s atlas is the Sun itself, from the same instrument and the same
quantity (disc-integrated flux) but other spectra (2015-16, through an iodine cell),
with the atmosphere removed by a joint fit over air mass: the corrected 2014 atlas
divided by it should be 1 wherever the correction is right, under solar lines too.

Per window: ours is sampled on their grid (ours is four times finer, theirs coarser
than the resolution, so interpolating theirs onto ours would add its own error in
every line), shifted by the velocity that best aligns the two where the atmosphere
leaves the pixel alone (transmission above 0.995), and the ratio is divided by a
straight line fitted there, since the two normalisations differ. The transmission
is taken relative to the window's 99th percentile, because the water and O2
continua keep it under 0.995 between lines over much of this range. Scored is the
rms of ratio - 1 in percent, at least 50 of their samples from the window edges,
with transmission at least 0.15 (the products blank deeper cores): ``reference``
(T > 0.995 and flagged clean, ``flags == 0``: what the two atlases disagree by with
no atmosphere -- epochs, noise, their solar model; inside a band, where fewer than
50 samples clear 0.995, the first of 0.98 and 0.95 that 50 clear, recorded as
``reference_above``), ``telluric`` (T < 0.97),
``deep`` (T < 0.7), and ``telluric`` split by the solar model's depth into ``clear``
(> 0.995 of continuum) and ``solar_line``. Their flags mark essentially every
telluric pixel, so they cannot select those; their own ``err`` there is reported
beside the score instead, as the size of their correction's uncertainty.

    uv run python scripts/score_iag_against_baker.py \\
        aer=data/corrected/solar/iag_pilot_aer table=data/corrected/solar/iag_pilot_table \\
        --output docs/iag_pilot_baker_score.json

Each RUN is NAME=DIRECTORY of `fit_fts_batch.py --iag` window files; the first run's
transmission chooses the scored pixels for all of them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tellurix_fts.baker import read_baker_solar_atlas

TELFREE = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/iag_telfree/"
               "iag_telfree_solaratlas.fits")
C_KMS = 299792.458
EDGE_SAMPLES = 50
MIN_TRANSMISSION = 0.15
VELOCITIES_KMS = np.arange(-3.0, 3.0001, 0.05)
REFERENCE_CEILINGS = (0.995, 0.98, 0.95)


def corrected(path: Path):
    """Wavenumber (ascending), corrected/continuum, solar model/continuum, T, mask."""

    with np.load(path) as data:
        nu = np.asarray(data["wavenumber_cm1"])
        order = np.argsort(nu)
        take = lambda key: np.asarray(data[key])[order]
        flat = take("flux") / (take("model_flux") / take("stellar_only_pixels")) / take("continuum")
        return (nu[order], flat, take("stellar_only_pixels") / take("continuum"),
                take("transmission_pixels"), take("mask").astype(bool))


def detrended(v, ratio, where):
    coeffs = np.polyfit(v[where] - v.mean(), ratio[where], 1)
    return ratio / np.polyval(coeffs, v - v.mean())


def align(v, nu, flat, s, where):
    """The velocity of ours relative to theirs, by least scatter of the ratio where nothing absorbs."""

    def cost(velocity):
        ours = np.interp(v * (1.0 + velocity / C_KMS), nu, flat)
        return float(np.std(detrended(v, ours / s, where)[where]))

    costs = np.array([cost(u) for u in VELOCITIES_KMS])
    best = int(np.argmin(costs))
    if 0 < best < len(costs) - 1:
        a, b, c = costs[best - 1: best + 2]
        step = VELOCITIES_KMS[1] - VELOCITIES_KMS[0]
        denominator = a - 2.0 * b + c
        if denominator > 0:
            return float(VELOCITIES_KMS[best] + 0.5 * step * (a - c) / denominator)
    return float(VELOCITIES_KMS[best])


def rms(values) -> float:
    return float(100.0 * np.sqrt(np.mean((values - 1.0) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", metavar="NAME=DIRECTORY")
    parser.add_argument("--stem", default="iag_vis")
    parser.add_argument("--atlas", type=Path, default=TELFREE)
    parser.add_argument("--minimum-pixels", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs = dict(item.split("=", 1) for item in args.runs)
    first = next(iter(runs))
    atlas = read_baker_solar_atlas(args.atlas)
    windows = {}
    for path in sorted(Path(runs[first]).glob(f"{args.stem}_*.npz")):
        found = {name: Path(d) / path.name for name, d in runs.items()}
        if not all(p.exists() for p in found.values()):
            continue
        nu, flat, solar, transmission, mask = corrected(path)
        inside = (atlas["v"] > nu[0] + 0.05) & (atlas["v"] < nu[-1] - 0.05)
        if inside.sum() < 4 * EDGE_SAMPLES:
            continue
        v, s, flags = atlas["v"][inside], atlas["s"][inside], atlas["flags"][inside]
        err = atlas["err"][inside]
        t = np.interp(v, nu, transmission)
        sol = np.interp(v, nu, solar)
        # A sample is usable only if every fine pixel it is drawn from is.
        usable = np.interp(v, nu, mask.astype(float)) > 0.999
        if not usable.any():
            continue
        t = t / np.percentile(t[usable], 99)
        interior = np.zeros(v.size, bool)
        interior[EDGE_SAMPLES:-EDGE_SAMPLES] = True
        good = usable & interior & (t >= MIN_TRANSMISSION)
        # Inside a band nothing reaches 0.995; the reference only aligns and
        # normalises, so it may step down, and the window records how far.
        for ceiling in REFERENCE_CEILINGS:
            reference = good & (t > ceiling) & (flags == 0)
            if reference.sum() >= 50:
                break
        else:
            continue
        bins = {"reference": reference, "telluric": good & (t < 0.97), "deep": good & (t < 0.7),
                "clear": good & (t < 0.97) & (sol > 0.995), "solar_line": good & (t < 0.97) & (sol <= 0.995)}
        row = {"v1": float(nu[0]), "reference_above": ceiling,
               "pixels": {k: int(m.sum()) for k, m in bins.items()},
               "baker_err_percent": {k: (float(100.0 * np.median(err[m])) if m.any() else None)
                                     for k, m in bins.items()},
               "runs": {}}
        for name, file in found.items():
            r_nu, r_flat = corrected(file)[:2]
            velocity = align(v, r_nu, r_flat, s, reference)
            ratio = detrended(v, np.interp(v * (1.0 + velocity / C_KMS), r_nu, r_flat) / s, reference)
            row["runs"][name] = {"velocity_kms": velocity,
                                 **{k: (rms(ratio[m]) if m.sum() >= args.minimum_pixels else None)
                                    for k, m in bins.items()}}
        windows[path.stem.rsplit("_", 1)[1]] = row
        print(f"{path.stem}: " + "  ".join(
            f"{name} ref {r['reference']:.2f} tell {r['telluric'] if r['telluric'] is None else round(r['telluric'], 2)}"
            f" deep {r['deep'] if r['deep'] is None else round(r['deep'], 2)} v {r['velocity_kms']:+.2f}"
            for name, r in row["runs"].items()))

    summary = {}
    for label in ("reference", "telluric", "deep", "clear", "solar_line"):
        keys = [k for k, w in windows.items() if all(w["runs"][n][label] is not None for n in runs)]
        if not keys:
            continue
        table = np.array([[windows[k]["runs"][n][label] for n in runs] for k in keys])
        summary[label] = {"windows": len(keys),
                          "median_percent": {n: float(x) for n, x in zip(runs, np.median(table, axis=0))},
                          f"windows_better_than_{first}": {n: int(np.sum(table[:, i] < table[:, 0]))
                                                           for i, n in enumerate(runs) if i}}
        summary[label]["baker_err_median_percent"] = float(np.median(
            [windows[k]["baker_err_percent"][label] for k in keys]))
        print(f"{label}: {len(keys)} windows, median %: "
              + "  ".join(f"{n} {x:.3f}" for n, x in summary[label]["median_percent"].items())
              + f"  (their err {summary[label]['baker_err_median_percent']:.3f})")
    if args.output:
        args.output.write_text(json.dumps({
            "method": __doc__.split("\n\n")[1].replace("\n", " "),
            "atlas": str(args.atlas), "runs": runs, "summary": summary, "windows": windows},
            indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
