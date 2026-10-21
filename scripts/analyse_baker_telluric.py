#!/usr/bin/env python
"""Per-line strength and position errors in Baker et al. 2020's spectra, against the solar fits'.

Reads the windows ``fit_baker_telluric.py`` wrote and, for every feature of
docs/line_depth_ratios.json, measures in each spectrum the observed/model optical
depth ratio and the centroid offset (``tellurix_fts.line_depth``; positions with
each window's median removed, because the velocity is fitted per window). Their
solar model is divided out already, so the solar-clean cut is theirs: ``stel_mod``
above 0.97 everywhere within 0.1 cm-1, the most the Sun moves against the
atmosphere in a year (2-3 km/s here). Their iodine template is divided out too,
about 1% of its depth too deep (``diagnose_baker_residuals.py``), which leaves the
same residue in every spectrum as a line-list error would: pixels within 0.03 cm-1
of the template below 0.98 are cut as well. The cell is at rest with the
atmosphere, so the reach is only the lines' width. Above 15500 cm-1 that is 7-48%
of the pixels.

What it reports, each a test of "the error is the line list's":

- **repeatability**: spectra split into two halves, the correlation of the
  halves' per-feature mean (ratio - 1), and of their mean offsets;
- **air mass**: per feature, ratio - 1 = a + b (1/X - mean). A line-list error is
  all a. Error their solar model left under a line does not grow with air mass,
  so it is b, relative to the optical depth;
- **the solar fits**: the correlation with their mean ratio, and the held-out
  rms of each solar fit's (ratio - 1) predicted by the other solar fits, by Baker
  alone, and by both -- whether these spectra could build a correction table.

    uv run python scripts/analyse_baker_telluric.py --output docs/baker_telluric_ratios.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATLAS = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/iag_telfree")
SOLAR_CLEAN = 0.97
SOLAR_SHIFT_CM1 = 0.1
IODINE_CLEAN = 0.98
IODINE_REACH_CM1 = 0.03
MIN_SPECTRA = 3
# As in measure_line_depth_ratios.py: further off is a missing line or a misfit window.
OUTLIER = 0.5


def clean_pixels(atlas: Path, group: int, cache: dict, iodine: bool) -> tuple[np.ndarray, np.ndarray]:
    """The group's grid and the pixels clear of their solar model's lines and, with ``iodine``, the cell's."""
    if group not in cache:
        from scipy.ndimage import minimum_filter1d
        from tellurix_fts.baker import read_baker_table

        nu, columns = read_baker_table(atlas, group, ["stel_mod", "iodine"])
        step = np.median(np.diff(nu))
        reach = int(np.ceil(SOLAR_SHIFT_CM1 / step))
        clean = minimum_filter1d(columns["stel_mod"], 2 * reach + 1) > SOLAR_CLEAN
        if iodine:
            reach = int(np.ceil(IODINE_REACH_CM1 / step))
            clean &= minimum_filter1d(columns["iodine"], 2 * reach + 1) > IODINE_CLEAN
        cache[group] = nu, clean
    return cache[group]


def json_safe(value):
    """JSON has no NaN: a value that is not measured is null."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def rms(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits", type=Path, default=ROOT / "data/corrected/baker2020")
    parser.add_argument("--features-from", type=Path, default=ROOT / "docs/line_depth_ratios.json")
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-iodine", action="store_true",
                        help="do not cut pixels under the iodine template's lines (the analysis before 2026-10-21)")
    args = parser.parse_args()

    from tellurix_fts.line_depth import feature_measurements

    report = json.loads(args.features_from.read_text())
    features = report["features"]
    centres = np.array([f["nu"] for f in features])
    solar_fits = report["ratio_and_peak_tau"]

    rows = {}   # feature -> list of (spectrum, airmass, ratio, offset)
    spectra, cache = [], {}
    files = sorted(glob.glob(str(args.fits / "g*_*_*.npz")))
    if not files:
        raise SystemExit(f"no windows in {args.fits}")
    for path in files:
        meta = json.loads(Path(path[:-4] + ".json").read_text())
        spectrum = f"{meta['group']:02d}_{meta['number']}"
        if spectrum not in spectra:
            spectra.append(spectrum)
        with np.load(path) as data:
            nu = np.asarray(data["wavenumber_cm1"]); order = np.argsort(nu); nu = nu[order]
            take = lambda key: np.asarray(data[key])[order]
            continuum = take("continuum")
            observed, model = take("flux") / continuum, take("model_flux") / continuum
            mask = take("mask").astype(bool)
        grid, clean = clean_pixels(args.atlas, meta["group"], cache, iodine=not args.keep_iodine)
        clean = np.interp(nu, grid, clean.astype(float)) > 0.999
        found = feature_measurements(nu, observed, model, np.ones_like(nu), mask, centres, clean=clean)
        if not found:
            continue
        median = np.median([offset for _, _, offset in found.values()]) if len(found) >= 3 else np.nan
        for i, (ratio, _, offset) in found.items():
            rows.setdefault(i, []).append((spectrum, meta["airmass"], ratio, offset - median))

    half = {s: k % 2 for k, s in enumerate(spectra)}
    table = []
    for i, values in sorted(rows.items()):
        names = [v[0] for v in values]
        airmass, ratio, offset = (np.array([v[k] for v in values], dtype=float) for k in (1, 2, 3))
        keep = np.abs(ratio - 1) < OUTLIER
        if keep.sum() < MIN_SPECTRA:
            continue
        airmass, ratio, offset = airmass[keep], ratio[keep], offset[keep]
        halves = np.array([half[n] for n, k in zip(names, keep) if k])
        inverse = 1.0 / airmass - np.mean(1.0 / airmass)
        spread = np.ptp(airmass) > 0.5 and keep.sum() > 3
        b = float(np.sum(inverse * (ratio - ratio.mean())) / np.sum(inverse ** 2)) if spread else float("nan")
        # Its error from the scatter about the line, so b / error is ~N(0, 1)
        # when nothing in the ratio follows 1/X.
        residual = ratio - ratio.mean() - b * inverse
        b_error = (float(np.sqrt(np.sum(residual ** 2) / (keep.sum() - 2) / np.sum(inverse ** 2)))
                   if spread else float("nan"))
        finite = np.isfinite(offset)
        mean_half = lambda v, h: float(np.mean(v[halves == h])) if np.any(halves == h) else float("nan")
        table.append(dict(
            feature=i, nu=features[i]["nu"], spectra=int(keep.sum()),
            ratio=round(float(ratio.mean()), 4), ratio_std=round(float(ratio.std(ddof=1)), 4),
            inverse_airmass_slope=round(b, 4), inverse_airmass_slope_error=round(b_error, 4),
            offset_ms=round(float(np.mean(offset[finite])), 1) if finite.any() else None,
            offset_std_ms=round(float(np.std(offset[finite], ddof=1)), 1) if finite.sum() > 1 else None,
            halves=[mean_half(ratio, 0), mean_half(ratio, 1)],
            offset_halves=[mean_half(np.where(finite, offset, np.nan), 0) if finite.all() else float("nan"),
                           mean_half(np.where(finite, offset, np.nan), 1) if finite.all() else float("nan")],
            solar={name: fit[str(i)][0] for name, fit in solar_fits.items() if str(i) in fit}))

    def correlation(x, y):
        x, y = np.asarray(x), np.asarray(y)
        good = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[good], y[good])[0, 1]), int(good.sum())

    split, n_split = correlation([r["halves"][0] - 1 for r in table], [r["halves"][1] - 1 for r in table])
    split_offset, n_offset = correlation([r["offset_halves"][0] for r in table],
                                         [r["offset_halves"][1] for r in table])
    slopes = np.array([r["inverse_airmass_slope"] for r in table], dtype=float)
    pulls = slopes / np.array([r["inverse_airmass_slope_error"] for r in table], dtype=float)
    pulls = pulls[np.isfinite(pulls)]
    intercepts = np.array([r["ratio"] - 1 for r in table])
    print(f"{len(spectra)} spectra, {len(files)} windows; {len(table)} features in >= {MIN_SPECTRA} spectra")
    print(f"  split halves: ratio r = {split:.2f} ({n_split}), offset r = {split_offset:.2f} ({n_offset})")
    print(f"  per-feature 1/X slope: median {np.nanmedian(slopes):+.3f}, rms {rms(slopes[np.isfinite(slopes)]):.3f}"
          f" against rms(ratio - 1) {rms(intercepts):.3f}; slope/error rms {rms(pulls):.2f}"
          f" (1 if nothing follows 1/X), |slope/error| > 3 for {np.mean(np.abs(pulls) > 3):.1%}")

    both = [r for r in table if len(r["solar"]) >= 2
            and all(abs(v - 1) < OUTLIER for v in r["solar"].values())]
    solar_mean = np.array([np.mean(list(r["solar"].values())) for r in both])
    baker = np.array([r["ratio"] for r in both])
    r_solar, n_solar = correlation(baker - 1, solar_mean - 1)
    comparison = dict(features=n_solar, correlation=r_solar,
                      baker_rms_before=rms(baker - 1), baker_rms_after_solar_mean=rms(baker / solar_mean - 1))
    far = np.abs(solar_mean - 1) > 0.08
    comparison.update(features_over_8pct=int(far.sum()),
                      same_sign_over_8pct=int(np.sum(np.sign(baker[far] - 1) == np.sign(solar_mean[far] - 1))))
    print(f"  against the solar fits ({n_solar}): r = {r_solar:.2f}; Baker rms {comparison['baker_rms_before']:.3f}"
          f" -> {comparison['baker_rms_after_solar_mean']:.3f}; over 8%: {far.sum()}, same sign "
          f"{comparison['same_sign_over_8pct']}")
    held_out = {}
    for name in solar_fits:
        observed, others, alone, joint = [], [], [], []
        for r in table:
            if name not in r["solar"] or abs(r["solar"][name] - 1) >= OUTLIER:
                continue
            rest = [v for k, v in r["solar"].items() if k != name and abs(v - 1) < OUTLIER]
            if not rest:
                continue
            observed.append(r["solar"][name]); others.append(np.mean(rest)); alone.append(r["ratio"])
            joint.append(np.mean([*rest, r["ratio"]]))
        observed = np.array(observed)
        held_out[name] = dict(features=len(observed), rms_before=rms(observed - 1),
                              by_other_solar_fits=rms(observed / np.array(others) - 1),
                              by_baker=rms(observed / np.array(alone) - 1),
                              by_both=rms(observed / np.array(joint) - 1))
        h = held_out[name]
        print(f"  held out {name:7s} ({h['features']}): {h['rms_before']:.3f} -> other solar fits "
              f"{h['by_other_solar_fits']:.3f}, Baker {h['by_baker']:.3f}, both {h['by_both']:.3f}")

    offsets = np.array([r["offset_ms"] for r in table if r["offset_ms"] is not None])
    scatter = np.array([r["offset_std_ms"] for r in table if r["offset_std_ms"] is not None])
    print(f"  offsets: median |offset| {np.median(np.abs(offsets)):.1f} m/s, median between spectra "
          f"{np.median(scatter):.1f} m/s")

    args.output.write_text(json.dumps(json_safe(dict(
        method=__doc__.split("\n\n    uv run")[0],
        settings=dict(fits=str(args.fits.relative_to(ROOT)) if args.fits.is_relative_to(ROOT) else str(args.fits),
                      features_from=str(args.features_from.relative_to(ROOT)), solar_clean=SOLAR_CLEAN,
                      solar_shift_cm1=SOLAR_SHIFT_CM1, min_spectra=MIN_SPECTRA,
                      iodine_clean=None if args.keep_iodine else IODINE_CLEAN,
                      iodine_reach_cm1=None if args.keep_iodine else IODINE_REACH_CM1),
        spectra=spectra, windows=len(files),
        repeatability=dict(ratio_split_half_r=split, features=n_split,
                           offset_split_half_r=split_offset, offset_features=n_offset),
        airmass=dict(inverse_airmass_slope_median=float(np.nanmedian(slopes)),
                     inverse_airmass_slope_rms=rms(slopes[np.isfinite(slopes)]), ratio_rms=rms(intercepts),
                     slope_over_error_rms=rms(pulls), fraction_over_3_sigma=float(np.mean(np.abs(pulls) > 3))),
        solar_fits=comparison, held_out=held_out,
        positions=dict(median_abs_offset_ms=float(np.median(np.abs(offsets))),
                       median_between_spectra_ms=float(np.median(scatter))),
        features=[{k: v for k, v in r.items() if k not in ("halves", "offset_halves")} for r in table],
    )), separators=(",", ":"), allow_nan=False) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
