#!/usr/bin/env python
"""What else than the line list could leave a residual common to every Baker et al. spectrum.

A residual in every spectrum of ``fit_baker_telluric.py``'s fits is not by that alone
the line list's. This reads those fits and ``analyse_baker_telluric.py``'s ratios,
numpy only, and tests the other candidates:

- **isotopologues**: each feature tagged with the AER lines within 0.1 cm-1, by
  their share of S. An HDO depletion would show in HDO-dominated features;
- **strength and E''**: per fit, over features at least 80% H2-16O, the joint
  least-squares (ratio - 1) = a + b log10 S + c E''/1000 cm-1. A line-list error is
  the same in every fit; a temperature profile error is the fit's own (E''); a width
  error reads as strength in saturated cores, so it follows the fit's width scale;
- **line shape**: the residual of isolated H2O lines (>= 95% H2-16O, no other line
  over 3% of S within 0.5 cm-1, clear of 1% solar lines and 2% iodine lines within
  0.4 cm-1), over the model's depth, median over lines and spectra, set beside the
  residual a Voigt fit leaves on a line that is a wider Voigt or speed-dependent.
  Those templates are one layer at 14000 cm-1 and 280 K: an order of magnitude;
- **iodine**: off the telluric lines (model above 0.995) and the solar ones (the
  analysis's cut), the median residual over a window's spectra, where at least half
  of them are clean, against their iodine template's depth. A
  template divided out k too deep leaves +k (1 - iodine) in every spectrum.

    uv run python scripts/diagnose_baker_residuals.py --output docs/baker_residual_causes.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATLAS = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/iag_telfree")
AER = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
ISOTOPOLOGUES = {11: "H2-16O", 12: "H2-18O", 13: "H2-17O", 14: "HD-16O", 15: "HD-18O", 16: "HD-17O",
                 71: "O2-16-16", 72: "O2-16-18", 73: "O2-16-17"}
# The pressure-width scale each fit was run at (docs/line_intensity_corrections.md).
WIDTH_SCALE = {"baker": 1.00, "wallace": 1.00, "j83_2": 1.06, "j83_3": 1.06, "niratl": 1.11}
OUTLIER = 0.5
OFFSETS_CM1 = np.round(np.arange(-0.35, 0.3501, 0.01), 3)
SHOW_CM1 = (0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3)
# One line at 14000 cm-1: H2O's Doppler HWHM at 280 K and a column-weighted pressure HWHM.
DOPPLER_HWHM_CM1, LORENTZ_HWHM_CM1 = 0.0198, 0.065


def read_lines(path: Path, v1: float, v2: float, s_min: float) -> np.ndarray:
    """(nu, S, E'', molecule * 10 + isotope) of the AER records in (v1, v2) over ``s_min``."""
    rows = []
    with open(path) as handle:
        for record in handle:
            try:
                nu, strength = float(record[3:15]), float(record[15:25].replace("D", "E"))
                e_lower, key = float(record[45:55]), int(record[:2]) * 10 + int(record[2:3])
            except ValueError:
                continue
            if v1 < nu < v2 and strength > s_min:
                rows.append((nu, strength, e_lower, key))
    return np.array(rows)


def regression(ratio, log_s, e_lower) -> dict:
    """(ratio - 1) on centred log10 S and E''/1000, with errors from the scatter."""
    y = ratio - 1
    design = np.column_stack([np.ones_like(y), log_s - log_s.mean(), (e_lower - e_lower.mean()) / 1000])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    cov = np.linalg.inv(design.T @ design) * np.sum(residual ** 2) / (y.size - 3)
    error = np.sqrt(np.diag(cov))
    return dict(features=int(y.size), per_decade_of_s=float(coef[1]), per_decade_of_s_error=float(error[1]),
                per_1000_cm1_of_e_lower=float(coef[2]), per_1000_cm1_of_e_lower_error=float(error[2]))


def voigt(x, lorentz):
    from scipy.special import wofz

    sigma = DOPPLER_HWHM_CM1 / np.sqrt(2 * np.log(2))
    return wofz((x + 1j * lorentz) / (sigma * np.sqrt(2))).real / (sigma * np.sqrt(2 * np.pi))


def speed_dependent(x, lorentz, ratio):
    """Lorentz width Gamma0 (1 + ratio ((v / v_p)^2 - 3/2)), averaged over Maxwell speeds and directions."""
    speeds = np.linspace(0.02, 3.5, 140)
    weights = speeds ** 2 * np.exp(-speeds ** 2); weights /= weights.sum()
    cosines = np.linspace(-1, 1, 81)
    shift_cm1 = DOPPLER_HWHM_CM1 / np.sqrt(np.log(2))
    out = np.zeros_like(x)
    for speed, weight in zip(speeds, weights):
        width = lorentz * (1 + ratio * (speed ** 2 - 1.5))
        shifts = shift_cm1 * speed * cosines[:, None]
        out += weight * np.mean(width / np.pi / ((x - shifts) ** 2 + width ** 2), axis=0)
    return out


def voigt_misfit(truth, depth):
    """What a Voigt at the database width, its strength refitted, leaves on ``truth``, over the depth."""
    from scipy.optimize import minimize_scalar

    x = np.arange(-0.35, 0.3501, 0.001)
    shape = voigt(x, LORENTZ_HWHM_CM1)
    tau0 = -np.log(1 - depth) / shape[np.argmin(np.abs(x))]
    observed = np.exp(-tau0 * truth(x))
    core = np.abs(x) < 0.3
    scale = minimize_scalar(lambda k: np.sum((observed - np.exp(-k * tau0 * shape))[core] ** 2),
                            bounds=(0.5, 2), method="bounded").x
    model = np.exp(-scale * tau0 * shape)
    return np.interp(OFFSETS_CM1, x, (observed - model) / (1 - model[np.argmin(np.abs(x))]))


def at_offsets(profile) -> dict:
    return {f"{x:.2f}": round(float(np.interp(x, OFFSETS_CM1, profile)), 5) for x in SHOW_CM1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits", type=Path, default=ROOT / "data/corrected/baker2020")
    parser.add_argument("--ratios", type=Path, default=ROOT / "docs/baker_telluric_ratios.json")
    parser.add_argument("--solar-ratios", type=Path, default=ROOT / "docs/line_depth_ratios.json")
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from scipy.ndimage import minimum_filter1d
    from tellurix_fts.baker import read_baker_table

    lines = np.concatenate([read_lines(AER / name / name, 8800, 17600, 3e-27) for name in ("01_H2O", "07_O2")])
    lines = lines[np.argsort(lines[:, 0])]
    nu_lines, s_lines, key_lines = lines[:, 0], lines[:, 1], lines[:, 3].astype(int)
    solar_report = json.loads(args.solar_ratios.read_text())
    features = solar_report["features"]
    baker = {r["feature"]: r["ratio"] for r in json.loads(args.ratios.read_text())["features"]}
    ratios = {"baker": baker, **{name: {int(i): v[0] for i, v in fit.items()}
                                 for name, fit in solar_report["ratio_and_peak_tau"].items()}}

    tags = {}
    for i, feature in enumerate(features):
        near = np.abs(nu_lines - feature["nu"]) < 0.1
        if not near.any():
            continue
        shares = {k: s_lines[near][key_lines[near] == k].sum() / s_lines[near].sum() for k in np.unique(key_lines[near])}
        dominant = max(shares, key=shares.get)
        tags[i] = (int(dominant), float(shares[dominant]))

    isotopologues = {}
    for name, table in ratios.items():
        rows = {}
        for i, ratio in table.items():
            if i in tags and tags[i][1] >= 0.8 and abs(ratio - 1) < OUTLIER:
                rows.setdefault(ISOTOPOLOGUES.get(tags[i][0], str(tags[i][0])), []).append(ratio - 1)
        isotopologues[name] = {k: dict(features=len(v), median=round(float(np.median(v)), 4))
                               for k, v in sorted(rows.items())}
    trends = {}
    for name, table in ratios.items():
        pick = [i for i, r in table.items() if i in tags and tags[i][0] == 11 and tags[i][1] >= 0.8
                and abs(r - 1) < OUTLIER]
        trends[name] = dict(width_scale=WIDTH_SCALE[name], **regression(
            np.array([table[i] for i in pick]), np.log10([features[i]["s_max"] for i in pick]),
            np.array([features[i]["e_lower_strongest"] for i in pick])))
        t = trends[name]
        print(f"{name:7s} x{t['width_scale']:.2f} ({t['features']}): {100 * t['per_decade_of_s']:+.2f} +- "
              f"{100 * t['per_decade_of_s_error']:.2f}% per decade of S, {100 * t['per_1000_cm1_of_e_lower']:+.2f} +- "
              f"{100 * t['per_1000_cm1_of_e_lower_error']:.2f}% per 1000 cm-1 of E''")
    for name, rows in isotopologues.items():
        print(f"  {name:7s}", ", ".join(f"{k} {v['median']:+.3f} ({v['features']})" for k, v in rows.items()))

    isolated = []
    for i, feature in enumerate(features):
        if tags.get(i, (0, 0))[0] != 11 or tags[i][1] < 0.95 or feature["s_max"] < 1e-24:
            continue
        others = (np.abs(nu_lines - feature["nu"]) < 0.5) & (np.abs(nu_lines - feature["nu"]) > 0.02)
        if not others.any() or s_lines[others].max() <= 0.03 * feature["s_max"]:
            isolated.append(feature["nu"])
    isolated = np.array(isolated)

    clear, iodine, grid = {}, None, None
    for path in sorted(glob.glob(str(args.atlas / "telluric_spectra_*.fits"))):
        group = int(Path(path).stem.rsplit("_", 1)[1])
        grid, columns = read_baker_table(args.atlas, group, ["stel_mod", "iodine"])
        step = grid[1] - grid[0]
        wide, narrow = int(0.4 / step), int(np.ceil(0.1 / step))
        clear[group] = (minimum_filter1d(columns["stel_mod"], 2 * wide + 1) > 0.99,
                        minimum_filter1d(columns["stel_mod"], 2 * narrow + 1) > 0.97)
        if iodine is None:
            iodine = columns["iodine"]
    iodine_clear = minimum_filter1d(iodine, 2 * int(0.4 / (grid[1] - grid[0])) + 1) > 0.98

    stacks = {"weak": [], "mid": [], "deep": []}
    background = []   # per window: iodine depth and median residual off the lines
    by_window = {}
    for path in sorted(glob.glob(str(args.fits / "g*_*_*.npz"))):
        by_window.setdefault(float(Path(path).stem.rsplit("_", 1)[1]), []).append(Path(path))
    for v1, paths in sorted(by_window.items()):
        here = isolated[(isolated > v1 + 0.5) & (isolated < v1 + 29.5)]
        residuals = []
        for path in paths:
            group = int(path.stem[1:3])
            with np.load(path) as data:
                nu = np.asarray(data["wavenumber_cm1"]); order = np.argsort(nu); nu = nu[order]
                continuum = np.asarray(data["continuum"])[order]
                observed = np.asarray(data["flux"])[order] / continuum
                model = np.asarray(data["model_flux"])[order] / continuum
                mask = np.asarray(data["mask"])[order].astype(bool)
            start = int(np.argmin(np.abs(grid - nu[0])))
            span = slice(start, start + nu.size)
            wide, narrow = clear[group][0][span], clear[group][1][span]
            for centre in here:
                near = np.abs(nu - centre) < 0.4
                if not (mask[near] & wide[near] & iodine_clear[span][near]).all():
                    continue
                depth = 1 - np.interp(centre, nu, model)
                if 0.05 <= depth <= 0.9:
                    key = "weak" if depth < 0.2 else "mid" if depth < 0.5 else "deep"
                    stacks[key].append(np.interp(centre + OFFSETS_CM1, nu, observed - model) / depth)
            residuals.append(np.where(mask & narrow & (model > 0.995), observed - model, np.nan))
        residuals = np.array(residuals)
        # Each group moves the Sun differently: a pixel counts where half the spectra see it clean.
        enough = np.isfinite(residuals).sum(axis=0) >= len(paths) / 2
        median = np.where(enough, np.nanmedian(np.where(enough, residuals, 0.0), axis=0), np.nan)
        depth = 1 - iodine[start:start + median.size]
        take = np.isfinite(median)
        if take.sum() > 50:
            background.append((v1, depth[take], median[take]))

    shapes = {}
    for key, rows in stacks.items():
        if rows:
            stack = np.median(rows, axis=0)
            even = 0.5 * (stack + stack[::-1])
            shapes[key] = dict(profiles=len(rows), even=at_offsets(even), odd=at_offsets(stack - even))
            print(f"shape {key:4s} ({len(rows)}): " + "  ".join(f"{k} {100 * v:+.2f}%" for k, v in shapes[key]["even"].items()))
    templates = {}
    for depth in (0.35, 0.7):
        for name, truth in (("widths x0.94", lambda x: voigt(x, 0.94 * LORENTZ_HWHM_CM1)),
                            ("widths x1.06", lambda x: voigt(x, 1.06 * LORENTZ_HWHM_CM1)),
                            ("speed-dependent 0.10", lambda x: speed_dependent(x, LORENTZ_HWHM_CM1, 0.10)),
                            ("speed-dependent 0.15", lambda x: speed_dependent(x, LORENTZ_HWHM_CM1, 0.15))):
            templates[f"{name} at depth {depth}"] = at_offsets(voigt_misfit(truth, depth))
            print(f"template {name:21s} depth {depth}: " + "  ".join(
                f"{k} {100 * v:+.2f}%" for k, v in templates[f"{name} at depth {depth}"].items()))

    def fit(rows):
        depth = np.concatenate([r[1] for r in rows]); median = np.concatenate([r[2] for r in rows])
        slope = float(np.sum(depth * (median - median.mean())) / np.sum((depth - depth.mean()) ** 2))
        bins = {}
        for lo, hi in ((0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 1.5)):
            inside = (depth >= lo) & (depth < hi)
            if inside.any():
                bins[f"{lo}-{hi}"] = dict(pixels=int(inside.sum()), median_residual=float(np.median(median[inside])),
                                          median_depth=float(np.median(depth[inside])))
        return dict(pixels=int(depth.size), correlation=float(np.corrcoef(depth, median)[0, 1]), slope=slope,
                    by_iodine_depth=bins)

    blue = [r for r in background if r[0] >= 15500]
    windows = [dict(v1=v1, max_depth=round(float(d.max()), 3), correlation=round(float(np.corrcoef(d, m)[0, 1]), 3))
               for v1, d, m in blue if d.max() > 0.1]
    red = fit([r for r in background if r[0] < 15500])
    iodine_report = dict(above_15500=fit(blue), below_15500=red, windows=windows)
    b = iodine_report["above_15500"]
    print(f"iodine above 15500: r {b['correlation']:+.3f}, slope {b['slope']:+.4f} over {b['pixels']} pixels; "
          f"below: r {red['correlation']:+.3f}; per window r " + ", ".join(f"{w['v1']:.0f} {w['correlation']:+.2f}" for w in windows))

    args.output.write_text(json.dumps(dict(
        method=__doc__.split("\n\n    uv run")[0],
        inputs=dict(fits=str(args.fits.relative_to(ROOT)), ratios=str(args.ratios.relative_to(ROOT)),
                    solar_ratios=str(args.solar_ratios.relative_to(ROOT))),
        isotopologues=isotopologues, strength_and_e_lower=trends,
        line_shape=dict(isolated_lines=int(isolated.size), offsets_cm1=list(SHOW_CM1), observed=shapes,
                        templates=templates,
                        template_line=dict(doppler_hwhm_cm1=DOPPLER_HWHM_CM1, lorentz_hwhm_cm1=LORENTZ_HWHM_CM1)),
        iodine=iodine_report,
    ), indent=1) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
