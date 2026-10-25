#!/usr/bin/env python
"""How much of the IAG co-add fit's residual against Baker et al.'s Sun is per-line strength error.

`score_iag_against_baker.py` finds the corrected IAG atlas 1.1% from Baker, Blake &
Reiners 2020's telluric-free IAG Sun at telluric pixels, against their quoted 0.35%.
Their atmosphere was removed by an empirical fit per line, so a line our list has
wrong is right in theirs. This asks how much of the difference that is.

Per fitted window, Baker's Sun replaces ours: it is aligned by the velocity the score
found and put on the corrected spectrum's scale by a line fitted where nothing
absorbs, so observed / it is the transmission the atlas saw (up to their error), and
model / stellar model is ours. Then, per telluric feature of the strength table's
catalogue (`tellurix_fts.line_depth`: AER H2O and O2 clusters, 8800-17600 cm-1), the
same measurement the table was built from -- the optical depth ratio over the core
pixels, and a shift-and-scale fit for its position -- now free of our solar model.

Four questions:

1. **Corrected or not.** Features the table scales (``data/line_strength/step3_iter1.npz``)
   against those it leaves at AER, split by why: measured elsewhere but refused
   (sources disagree, or one date only), or never measured.
2. **Is it the line list?** For features the table leaves alone, does the IAG ratio
   repeat what Baker's single 2015-16 spectra and the Kitt Peak fits measured
   (``docs/baker_line_corrections.csv``)? A line-list error repeats across
   instruments, epochs and solar models; a fit error does not.
3. **Pixel budget.** The rms of ln(observed / Baker / our T) at telluric pixels,
   before and after removing each feature's error from its core pixels: (a) with the
   other datasets' ratio (only what is known without this atlas), (b) with the
   feature's own IAG ratio, and (c) its own ratio and shift -- (b) and (c) are upper
   bounds on what any per-line strength and position table could do.
4. **Does it repeat?** The median residual of our fits of Baker's 53 single spectra
   (same model and table, flat Sun), scaled to IAG's optical depth and aligned to its
   frame, fitted to the IAG residual with one scale a window; and, as a control that
   shares nothing with Baker's Sun, our Wallace 2011 fits at solar-clean pixels.

Baker's spectrum is sampled at about one point per resolution element and is
interpolated onto our pixels with a cubic spline, which adds its own error in narrow
lines; the baseline rms is measured the same way, so the fractions compare like with
like, but are not the score's numbers.

    uv run python scripts/attribute_iag_residuals.py --output docs/iag_residual_attribution.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from tellurix_fts.baker import read_baker_solar_atlas
from tellurix_fts.line_depth import (CORE_CM1, SHIFT_CM1, cluster_features, read_feature_lines,
                                     shift_and_scale)

ROOT = Path(__file__).resolve().parents[1]
AER = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
TELFREE = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/iag_telfree/"
               "iag_telfree_solaratlas.fits")
C_KMS = 299792.458
MIN_TRANSMISSION = 0.15
EDGE_PIXELS = 200  # the score's 50 of Baker's samples, on pixels ~4x finer


def table_factor_per_feature(features, lines, table_path):
    """S-weighted mean table factor over each feature's member lines (1 where a line has none)."""

    with np.load(table_path, allow_pickle=False) as t:
        key = {(int(m), round(float(v), 4)): float(f) for m, v, f in
               zip(t["molecule_id"], t["wavenumber_cm1"], t["factor"])}
    out = []
    for feature in features:
        members = feature["members"]
        strength = lines["strength"][members]
        factors = np.array([key.get((int(lines["molecule"][i]), round(float(lines["nu"][i]), 4)), 1.0)
                            for i in members])
        corrected = bool(np.any(factors != 1.0))
        out.append((float(np.sum(strength * factors) / strength.sum()), corrected))
    return out


def independent_ratios(path):
    """Per feature index: Baker's single-spectrum ratio, the Kitt Peak mean, and the verdict."""

    out = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            number = lambda k: float(row[k]) if row[k] not in ("", "nan") else np.nan
            out[int(row["feature"])] = {"baker": number("baker_ratio"), "kitt_peak": number("kitt_peak_mean"),
                                        "verdict": row["verdict"]}
    return out


class FitResiduals:
    """Our fits of other spectra, as a residual template on any pixels.

    ``files`` maps a spectrum to ``{window v1: npz}``. The residual is ln(observed /
    model); with ``solar_clean`` only pixels whose stellar model is above that
    fraction of its window's continuum are kept, so a solar fit's template is its
    telluric error and not its solar model's.
    """

    def __init__(self, files: dict, solar_clean: float | None = None, edge: int = 10):
        self.files, self.solar_clean, self.edge = files, solar_clean, edge
        self.cache = {}

    @classmethod
    def baker(cls, directory: Path):
        files = {}
        for path in directory.glob("g*_*_*.npz"):
            group, spectrum, v1 = path.stem.split("_")
            # A spectrum in several groups was divided by each group's solar model; one copy.
            files.setdefault(spectrum, {}).setdefault(int(v1), path)
        return cls(files)

    @classmethod
    def solar_run(cls, directory: Path, stem: str, solar_clean: float = 0.97):
        files = {}
        for path in directory.glob(f"{stem}_*.npz"):
            spectrum, v1 = path.stem.rsplit("_", 1)
            if v1.isdigit():
                files.setdefault(spectrum, {})[int(v1)] = path
        return cls(files, solar_clean)

    def _window(self, path):
        if path not in self.cache:
            with np.load(path) as d:
                order = np.argsort(d["wavenumber_cm1"])
                take = lambda k: np.asarray(d[k], float)[order]
                nu = take("wavenumber_cm1")
                with np.errstate(invalid="ignore", divide="ignore"):
                    residual = np.log(take("flux") / take("model_flux"))
                    tau = -np.log(np.clip(take("transmission_pixels"), 1e-6, None))
                good = np.asarray(d["mask"])[order].astype(bool) & np.isfinite(residual)
                if self.solar_clean is not None:
                    solar = take("stellar_only_pixels") / take("continuum")
                    good &= solar > self.solar_clean * np.percentile(solar, 99)
                good[: self.edge] = good[-self.edge:] = False
            self.cache[path] = (nu, residual, tau, good)
        return self.cache[path]

    def _covering(self, windows, nu):
        return [(v1, path) for v1, path in windows.items() if not (v1 + 30 < nu[0] or v1 > nu[-1])]

    def on(self, nu, tau_target, minimum_spectra: int = 10):
        """Median residual over spectra on ``nu``, each scaled to ``tau_target``; and how many spectra."""

        stack = []
        for windows in self.files.values():
            residual = np.full(nu.size, np.nan)
            tau = np.full(nu.size, np.nan)
            for _, path in self._covering(windows, nu):
                b_nu, b_res, b_tau, good = self._window(path)
                if not good.any():
                    continue
                inside = (nu > b_nu[good][0]) & (nu < b_nu[good][-1])
                ok = np.interp(nu[inside], b_nu, good.astype(float)) > 0.999
                take = np.where(inside)[0][ok]
                residual[take] = np.interp(nu[take], b_nu, np.where(good, b_res, 0.0))
                tau[take] = np.interp(nu[take], b_nu, b_tau)
            both = np.isfinite(residual) & (tau_target > 0.03) & (tau > 0.0)
            if both.sum() < 100:
                continue
            # The residual of an optical-depth error grows with the column: put each
            # spectrum on the atlas's by the scale that best maps its tau onto the atlas'.
            scale = np.sum(tau_target[both] * tau[both]) / np.sum(tau[both] ** 2)
            stack.append(residual * scale)
        if len(stack) < minimum_spectra:
            return None, len(stack)
        stack = np.array(stack)
        count = np.sum(np.isfinite(stack), axis=0)
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(np.where(count[None, :] >= minimum_spectra, stack, np.nan), axis=0)
        return median, len(stack)

    def aligned(self, nu, tau_target, minimum_spectra: int = 10):
        """``on``, moved into the target's frame by aligning one spectrum's model optical depth with it."""

        template, spectra = self.on(nu, tau_target, minimum_spectra)
        if template is None:
            return None, spectra, None
        reference = np.full(nu.size, np.nan)
        for windows in self.files.values():
            for _, path in self._covering(windows, nu):
                b_nu, _, b_tau, _ = self._window(path)
                inside = (nu > b_nu[0]) & (nu < b_nu[-1])
                reference[inside] = np.interp(nu[inside], b_nu, b_tau)
            if np.isfinite(reference).sum() > 0.8 * nu.size:
                break
        fill = np.isfinite(reference)
        shift = frame_shift(nu[fill], tau_target[fill], reference[fill]) if fill.sum() > 500 else 0.0
        return np.interp(nu * (1.0 + shift / C_KMS), nu, template), spectra, shift


class RepeatTest:
    """How much of the IAG residual one template explains, per pixel class and per window (one scale a window)."""

    def __init__(self, classes):
        self.classes = classes
        self.totals = {c: {"pixels": 0, "base": 0.0, "after": 0.0, "cross": 0.0, "template": 0.0} for c in classes}
        self.windows = []

    def add(self, window, residual, template, use, kind, spectra, shift):
        use = use & np.isfinite(template)
        if use.sum() < 100:
            return None
        a = float(np.sum(residual[use] * template[use]) / np.sum(template[use] ** 2))
        after = residual - a * template
        self.windows.append({"window": window, "spectra": spectra, "frame_shift_kms": round(shift, 2),
                             "scale": round(a, 3), "pixels": int(use.sum()),
                             "r": round(float(np.corrcoef(residual[use], template[use])[0, 1]), 3),
                             "rms_percent": round(float(100 * np.sqrt(np.mean(residual[use] ** 2))), 3),
                             "rms_after_percent": round(float(100 * np.sqrt(np.mean(after[use] ** 2))), 3)})
        for k, c in enumerate(self.classes):
            m = use & (kind == k)
            t = self.totals[c]
            t["pixels"] += int(m.sum())
            t["base"] += float(np.sum(residual[m] ** 2))
            t["after"] += float(np.sum(after[m] ** 2))
            t["cross"] += float(np.sum(residual[m] * template[m]))
            t["template"] += float(np.sum(template[m] ** 2))
        return a

    def report(self):
        median = lambda key: (round(float(np.median([w[key] for w in self.windows])), 3) if self.windows else None)
        return {"by_pixel_class": {c: {
                    "pixels": v["pixels"],
                    "rms_percent": round(100.0 * np.sqrt(v["base"] / v["pixels"]), 3),
                    "rms_after_percent": round(100.0 * np.sqrt(v["after"] / v["pixels"]), 3),
                    "variance_explained_percent": round(100.0 * (1.0 - v["after"] / v["base"]), 1),
                    "correlation": (round(v["cross"] / np.sqrt(v["base"] * v["template"]), 3)
                                    if v["template"] else None)}
                    for c, v in self.totals.items() if v["pixels"]},
                "windows": len(self.windows), "median_window_r": median("r"),
                "median_window_rms_percent": median("rms_percent"),
                "median_window_rms_after_percent": median("rms_after_percent"),
                "median_scale": median("scale"), "per_window": self.windows}


def frame_shift(nu, tau_target, template_tau, velocities=np.arange(-1.5, 1.5001, 0.02)):
    """Velocity (km/s) moving a curve on ``nu`` onto the target, by best correlation of optical depths."""

    best, score = 0.0, -np.inf
    for v in velocities:
        moved = np.interp(nu * (1.0 + v / C_KMS), nu, template_tau)
        c = np.corrcoef(moved[50:-50], tau_target[50:-50])[0, 1]
        if c > score:
            best, score = float(v), c
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=ROOT / "data/corrected/solar/iag")
    parser.add_argument("--stem", default="iag_vis")
    parser.add_argument("--score", type=Path, default=ROOT / "docs/iag_baker_score.json")
    parser.add_argument("--table", type=Path, default=ROOT / "data/line_strength/step3_iter1.npz")
    parser.add_argument("--vetting", type=Path, default=ROOT / "docs/baker_line_corrections.csv")
    parser.add_argument("--atlas", type=Path, default=TELFREE)
    parser.add_argument("--baker-fits", type=Path, default=ROOT / "data/corrected/baker2020_step3_iter1")
    parser.add_argument("--wallace-fits", type=Path, default=ROOT / "data/corrected/solar/wallace2011")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None,
                        help="also draw the evidence: an example window at the median, the per-feature "
                             "ratios, the budget by pixel class and the per-window rms")
    args = parser.parse_args()

    lines = read_feature_lines(AER)
    features = cluster_features(lines)
    centres = np.array([f["nu"] for f in features])
    table = table_factor_per_feature(features, lines, args.table)
    vetting = independent_ratios(args.vetting)
    atlas = read_baker_solar_atlas(args.atlas)
    score = json.loads(args.score.read_text())["windows"]
    velocity = {int(round(w["v1"])): next(iter(w["runs"].values()))["velocity_kms"] for w in score.values()}

    measured = {}       # feature -> (ratio, peak tau, offset m/s, window)
    budget = {k: [] for k in ("base", "known", "own_strength", "own_strength_shift")}
    pixels_in_cores = 0
    # Where the residual sits: sum of squares and pixels per class, before and after (c).
    classes = ("core", "wing", "weak line", "little absorption")
    share = {c: {"pixels": 0, "base": 0.0, "own_strength_shift": 0.0} for c in classes}
    templates = {"baker_single_spectra": (FitResiduals.baker(args.baker_fits), 10),
                 "wallace2011_solar_clean": (FitResiduals.solar_run(args.wallace_fits, "wallace2011_reg[12]"), 1)}
    repeats = {name: RepeatTest(classes) for name in templates}
    # Where both exist: each alone and both together, on the same pixels.
    examples = {}
    joint = {"pixels": 0, "base": 0.0, "baker": 0.0, "wallace": 0.0, "both": 0.0, "cross_templates": [],
             "windows": 0}
    pixels_total = 0
    for path in sorted(args.run.glob(f"{args.stem}_*.npz")):
        v1 = int(path.stem.rsplit("_", 1)[1])
        if v1 not in velocity or abs(velocity[v1]) >= 0.99 or v1 > 17600:
            continue
        with np.load(path) as d:
            order = np.argsort(d["wavenumber_cm1"])
            take = lambda k: np.asarray(d[k], dtype=float)[order]
            nu, observed, model, stellar, continuum = (take(k) for k in (
                "wavenumber_cm1", "flux", "model_flux", "stellar_only_pixels", "continuum"))
            mask = np.asarray(d["mask"])[order].astype(bool)
        transmission = model / stellar
        corrected = observed / transmission
        inside = (atlas["v"] > nu[0] - 1.0) & (atlas["v"] < nu[-1] + 1.0)
        at = nu / (1.0 + velocity[v1] / C_KMS)
        baker = CubicSpline(atlas["v"][inside], atlas["s"][inside])(at)
        clean_flag = np.interp(at, atlas["v"][inside], (atlas["flags"][inside] == 0).astype(float)) > 0.999
        relative = transmission / np.percentile(transmission[mask], 99)
        for ceiling in (0.995, 0.98, 0.95):
            reference = mask & clean_flag & (relative > ceiling)
            if reference.sum() >= 50:
                break
        else:
            continue
        x = nu - nu.mean()
        sun = baker * np.polyval(np.polyfit(x[reference], (corrected / baker)[reference], 1), x)
        # Over the continuum, as feature_measurements wants; the Sun is Baker's.
        obs_c, sun_c = observed / continuum, sun / continuum
        t_obs = obs_c / sun_c
        tau_model = -np.log(np.clip(transmission, 1e-6, None))
        interior = np.zeros(nu.size, bool)
        interior[EDGE_PIXELS:-EDGE_PIXELS] = True
        # The score's telluric bin: T < 0.97 and above the blanking floor.
        telluric = mask & interior & (relative < 0.97) & (transmission >= MIN_TRANSMISSION) & (t_obs > 0)
        if telluric.sum() < 30:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            residual = np.log(t_obs) - np.log(transmission)

        # Each telluric pixel belongs to the nearest feature centre within CORE_CM1.
        here = np.where((centres > nu[0]) & (centres < nu[-1]))[0]
        owner = np.full(nu.size, -1)
        if here.size:
            distance = np.abs(nu[:, None] - centres[here][None, :])
            nearest = np.argmin(distance, axis=1)
            owner = np.where(distance[np.arange(nu.size), nearest] < CORE_CM1, here[nearest], -1)
        spline = CubicSpline(nu, transmission)
        known = residual.copy()
        own = residual.copy()
        own_shift = residual.copy()
        usable = mask & (sun_c > 0.97 * np.percentile(sun_c[mask], 99)) & (transmission > 0.02)
        for i in here:
            core = (owner == i) & usable & (tau_model > 0.1)
            if core.sum() < 3:
                continue
            ratio = float(-np.log(t_obs[core]).sum() / tau_model[core].sum())
            near = (np.abs(nu - centres[i]) < SHIFT_CM1) & usable
            shift, scale = shift_and_scale(spline, nu[near], t_obs[near])
            measured[int(i)] = (ratio, float(tau_model[core].max()),
                                float(shift / centres[i] * C_KMS * 1e3) if np.isfinite(shift) else np.nan, v1)
            pixels = (owner == i) & telluric
            # -tau_obs = -r tau_model, so the residual a pure strength error leaves is -(r - 1) tau.
            own[pixels] = residual[pixels] + (ratio - 1.0) * tau_model[pixels]
            if np.isfinite(shift):
                moved = np.clip(spline(nu[pixels] - shift), 1e-6, 1.0) ** (1.0 + scale)
                own_shift[pixels] = np.log(t_obs[pixels]) - np.log(moved)
            other = vetting.get(int(i), {})
            ratios = [r for r in (other.get("baker"), other.get("kitt_peak")) if r is not None and np.isfinite(r)]
            if ratios:
                # Relative to the model the fit used: AER times the table's factor.
                expected = float(np.exp(np.mean(np.log(ratios)))) / table[i][0]
                known[pixels] = residual[pixels] + (expected - 1.0) * tau_model[pixels]
        near_feature = np.zeros(nu.size, bool)
        if here.size:
            near_feature = np.min(np.abs(nu[:, None] - centres[here][None, :]), axis=1) < 3 * CORE_CM1
        kind = np.where(owner >= 0, 0, np.where(near_feature, 1, np.where(tau_model > 0.03, 2, 3)))
        for k, c in enumerate(classes):
            m = telluric & (kind == k)
            share[c]["pixels"] += int(m.sum())
            share[c]["base"] += float(np.sum(residual[m] ** 2))
            share[c]["own_strength_shift"] += float(np.sum(own_shift[m] ** 2))
        pixels_total += int(telluric.sum())

        # 4. Does the residual repeat in other spectra fitted with the same model? Each
        # template's tau is matched to ours and its frame aligned on the models.
        found, scales = {}, {}
        for name, (fits, minimum) in templates.items():
            template, spectra, shift = fits.aligned(nu, tau_model, minimum)
            a = (repeats[name].add(v1, residual, template, telluric, kind, spectra, shift)
                 if template is not None else None)
            if a is not None:
                found[name], scales[name] = template, a
        if args.figure and len(found) == 2:
            examples[v1] = {"nu": nu, "residual": residual, "telluric": telluric, "transmission": transmission,
                            "templates": {k: scales[k] * t for k, t in found.items()}}
        if len(found) == 2:
            tb, tw = found["baker_single_spectra"], found["wallace2011_solar_clean"]
            m = telluric & np.isfinite(tb) & np.isfinite(tw)
            if m.sum() >= 100:
                y = residual[m]
                single = lambda t: float(np.sum((y - np.sum(y * t) / np.sum(t * t) * t) ** 2))
                design = np.stack([tb[m], tw[m]], axis=1)
                both = float(np.sum((y - design @ np.linalg.lstsq(design, y, rcond=None)[0]) ** 2))
                joint["pixels"] += int(m.sum()); joint["windows"] += 1
                joint["base"] += float(np.sum(y ** 2))
                joint["baker"] += single(tb[m]); joint["wallace"] += single(tw[m]); joint["both"] += both
                joint["cross_templates"].append(float(np.corrcoef(tb[m], tw[m])[0, 1]))
        pixels_in_cores += int((telluric & (owner >= 0)).sum())
        for name, values in (("base", residual), ("known", known), ("own_strength", own),
                             ("own_strength_shift", own_shift)):
            budget[name].append(float(100.0 * np.sqrt(np.mean(values[telluric] ** 2))))

    # 1 and 2: per feature, by what the table did.
    groups = {"in table": [], "refused: sources disagree": [], "refused: one source": [],
              "not measured elsewhere": []}
    pairs = {"baker": [], "kitt_peak": []}
    pairs_table = {"baker": [], "kitt_peak": []}
    rows = []
    for i, (ratio, peak, offset, v1) in measured.items():
        factor, in_table = table[i]
        other = vetting.get(i, {})
        if in_table:
            group = "in table"
        elif other.get("verdict") == "disagrees with Kitt Peak":
            group = "refused: sources disagree"
        elif other:
            group = "refused: one source"
        else:
            group = "not measured elsewhere"
        groups[group].append(ratio)
        for source in ("baker", "kitt_peak"):
            value = other.get(source)
            if value is not None and np.isfinite(value):
                # Both relative to AER: the IAG model carried the factor, so undo it.
                (pairs_table if in_table else pairs)[source].append((np.log(ratio * factor), np.log(value)))
        rows.append({"feature": i, "nu": features[i]["nu"], "window": v1, "group": group,
                     "iag_ratio": round(ratio, 4), "peak_tau": round(peak, 3),
                     "offset_ms": None if not np.isfinite(offset) else round(offset, 1),
                     "table_factor": round(factor, 4),
                     "baker_ratio": other.get("baker"), "kitt_peak_ratio": other.get("kitt_peak")})

    def spread(values):
        values = np.log(np.asarray(values))
        if values.size == 0:
            return None
        return {"features": int(values.size), "median_abs_percent": round(float(100 * np.median(np.abs(values))), 2),
                "rms_percent": round(float(100 * np.sqrt(np.mean(values ** 2))), 2),
                "beyond_5_percent": int(np.sum(np.abs(values) > np.log(1.05)))}

    def agreement(points):
        if len(points) < 5:
            return None
        a, b = np.array(points).T
        slope = float(np.sum(a * b) / np.sum(b * b))
        return {"features": len(points), "r": round(float(np.corrcoef(a, b)[0, 1]), 3),
                "slope_through_zero": round(slope, 3),
                "rms_iag_percent": round(float(100 * np.sqrt(np.mean(a ** 2))), 2),
                "rms_iag_minus_other_percent": round(float(100 * np.sqrt(np.mean((a - b) ** 2))), 2)}

    report = {
        "method": __doc__.split("\n\n")[1].replace("\n", " ") + " " + __doc__.split("\n\n")[2].replace("\n", " "),
        "windows": len(budget["base"]),
        "telluric_pixels": pixels_total,
        "telluric_pixels_in_feature_cores_percent": round(100.0 * pixels_in_cores / max(pixels_total, 1), 1),
        "feature_ratio_spread": {k: spread(v) for k, v in groups.items()},
        "uncorrected_vs_other_datasets": {k: agreement(v) for k, v in pairs.items()},
        "corrected_vs_other_datasets_relative_to_aer": {k: agreement(v) for k, v in pairs_table.items()},
        "residual_by_pixel_class": {c: {
            "pixels_percent": round(100.0 * v["pixels"] / max(pixels_total, 1), 1),
            "sum_of_squares_percent": round(100.0 * v["base"] / sum(w["base"] for w in share.values()), 1),
            "rms_percent": round(100.0 * np.sqrt(v["base"] / max(v["pixels"], 1)), 3),
            "rms_after_own_strength_shift_percent": round(100.0 * np.sqrt(v["own_strength_shift"] / max(v["pixels"], 1)), 3)}
            for c, v in share.items()},
        "repeats_in_other_fits": {name: test.report() for name, test in repeats.items()},
        "baker_and_wallace_on_shared_pixels": ({
            "windows": joint["windows"], "pixels": joint["pixels"],
            "rms_percent": round(100.0 * np.sqrt(joint["base"] / joint["pixels"]), 3),
            "variance_explained_percent": {k: round(100.0 * (1.0 - joint[k] / joint["base"]), 1)
                                           for k in ("baker", "wallace", "both")},
            "median_correlation_between_templates": round(float(np.median(joint["cross_templates"])), 3)}
            if joint["pixels"] else None),
        "pixel_class_definition": ("core: within 0.05 cm-1 of a catalogue feature (S >= 3e-25); wing: 0.05-0.15 "
                                   "from one; weak line: elsewhere with model tau > 0.03; little absorption: the rest"),
        "pixel_rms_percent_median": {k: round(float(np.median(v)), 3) for k, v in budget.items()},
        "pixel_rms_percent_by_window": {k: [round(x, 3) for x in v] for k, v in budget.items()},
        "features": sorted(rows, key=lambda r: r["nu"]),
    }
    print(json.dumps({k: v for k, v in report.items() if k not in ("features", "pixel_rms_percent_by_window", "method")},
                     indent=1))
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    if args.figure:
        draw(report, examples, args.figure)


def draw(report, examples, path):
    """Four panels: an example window, the per-feature ratios, the budget by class, the windows."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baker = report["repeats_in_other_fits"]["baker_single_spectra"]
    wallace = report["repeats_in_other_fits"]["wallace2011_solar_clean"]
    windows = {w["window"]: w for w in baker["per_window"]}
    # The example is the window nearest the median in both correlation and rms, not the best.
    candidates = [v for v in examples if v in windows]
    pick = min(candidates, key=lambda v: abs(windows[v]["r"] - baker["median_window_r"]) / 0.1
               + abs(windows[v]["rms_percent"] - baker["median_window_rms_percent"]) / 0.3)
    e, w = examples[pick], windows[pick]
    nu, telluric = e["nu"], e["telluric"]
    show = lambda x: np.where(telluric, 100.0 * x, np.nan)
    colours = {"iag": "#10181c", "baker": "#d1493f", "wallace": "#2f7ad6", "table": "#8a9aa1"}

    fig = plt.figure(figsize=(13, 11.5))
    grid = fig.add_gridspec(4, 3, height_ratios=[0.55, 1.0, 1.0, 1.25], hspace=0.75, wspace=0.32)

    # (a) The window, and the 1.5 cm-1 stretch with the most telluric pixels.
    width = 1.5
    starts = np.arange(nu[0] + 1.0, nu[-1] - width - 1.0, 0.25)
    start = max(starts, key=lambda a: int(np.sum(telluric & (nu >= a) & (nu < a + width))))
    pixel_axis = (lambda x: np.interp(x, nu, np.arange(nu.size)),
                  lambda i: np.interp(i, np.arange(nu.size), nu))
    ax_t = fig.add_subplot(grid[0, :])
    ax_t.plot(nu, e["transmission"], color=colours["iag"], lw=0.6)
    ax_t.axvspan(start, start + width, color="#f2d16b", alpha=0.35, lw=0)
    ax_t.set_ylabel("our T")
    ax_t.set_xlim(nu[0], nu[-1])
    ax_t.set_title(f"(a) IAG window {pick} cm$^{{-1}}$ ({1e7 / nu[-1]:.0f}-{1e7 / nu[0]:.0f} nm), chosen at the "
                   f"median of all windows (r {w['r']:.2f}, rms {w['rms_percent']:.2f}%), not the best",
                   loc="left", fontsize=10, pad=22)
    ax_t.secondary_xaxis("top", functions=pixel_axis).set_xlabel("pixel in window", fontsize=8)
    ax = fig.add_subplot(grid[1, :2])
    zoom = fig.add_subplot(grid[1, 2])
    for axis, (lo, hi) in ((ax, (nu[0], nu[-1])), (zoom, (start, start + width))):
        axis.axhline(0, color="#c5cdd0", lw=0.6)
        axis.plot(nu, show(e["residual"]), color=colours["iag"], lw=0.7,
                  label="IAG: ln(observed / Baker Sun / our T)")
        axis.plot(nu, show(e["templates"]["baker_single_spectra"]), color=colours["baker"], lw=0.8, alpha=0.85,
                  label=f"our fits of Baker's {w['spectra']} single spectra, median residual (x{w['scale']:.2f})")
        axis.plot(nu, show(e["templates"]["wallace2011_solar_clean"]), color=colours["wallace"], lw=0.7, alpha=0.7,
                  label="our Wallace 2011 fit, solar-clean pixels (control)")
        axis.set_xlim(lo, hi)
        axis.set_xlabel("wavenumber (cm$^{-1}$)")
        axis.secondary_xaxis("top", functions=pixel_axis).tick_params(labelsize=7)
    zoom.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    zoom.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    zoom.set_ylim(ax.get_ylim())
    ax.set_ylabel("residual (%)")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.13), ncol=3, frameon=False)
    zoom.text(0.02, 0.96, "zoom: the yellow stretch above", transform=zoom.transAxes, fontsize=8, va="top")
    ax.text(0.99, 0.04, f"rms {w['rms_percent']:.2f}% -> {w['rms_after_percent']:.2f}% after removing the red curve",
            transform=ax.transAxes, ha="right", fontsize=8)

    # (b) Per feature, the IAG depth error against Baker's single spectra, both relative to AER.
    feature_ax = fig.add_subplot(grid[2:, 0])
    rows = [r for r in report["features"] if r["baker_ratio"] is not None and np.isfinite(r["baker_ratio"])]
    for in_table, colour, label in ((True, colours["table"], "corrected by the table"),
                                    (False, colours["baker"], "left at AER")):
        pick_rows = [r for r in rows if (r["group"] == "in table") == in_table]
        x = 100 * np.log([r["baker_ratio"] for r in pick_rows])
        y = 100 * np.log([r["iag_ratio"] * r["table_factor"] for r in pick_rows])
        stats = report["corrected_vs_other_datasets_relative_to_aer" if in_table
                       else "uncorrected_vs_other_datasets"]["baker"]
        feature_ax.scatter(x, y, s=6 if in_table else 9, color=colour, alpha=0.45 if in_table else 0.8, lw=0,
                           label=f"{label}: {stats['features']}, r {stats['r']:.2f}, slope {stats['slope_through_zero']:.2f}")
    lim = 30
    feature_ax.plot([-lim, lim], [-lim, lim], color="#10181c", lw=0.6, ls="--")
    feature_ax.set_xlim(-lim, lim); feature_ax.set_ylim(-lim, lim)
    feature_ax.set_xlabel("Baker's single spectra: depth error vs AER (%)")
    feature_ax.set_ylabel("IAG atlas, Baker's Sun: depth error vs AER (%)")
    feature_ax.set_title("(b) per feature: the errors repeat", loc="left", fontsize=10)
    feature_ax.legend(fontsize=7, loc="upper left", frameon=False)

    # (c) rms by pixel class: as fitted, with a perfect per-feature table, with the shared part removed.
    budget_ax = fig.add_subplot(grid[2:, 1])
    classes = [c for c in ("core", "wing", "weak line") if c in baker["by_pixel_class"]]
    share = report["residual_by_pixel_class"]
    bars = [("as fitted", [share[c]["rms_percent"] for c in classes], "#10181c"),
            ("perfect strength + position table\n(each feature's own IAG error)",
             [share[c]["rms_after_own_strength_shift_percent"] for c in classes], colours["table"]),
            ("minus Wallace 2011 residual", [wallace["by_pixel_class"][c]["rms_after_percent"] *
                                              share[c]["rms_percent"] / wallace["by_pixel_class"][c]["rms_percent"]
                                              for c in classes], colours["wallace"]),
            ("minus Baker single-spectra residual", [baker["by_pixel_class"][c]["rms_after_percent"] *
                                                     share[c]["rms_percent"] / baker["by_pixel_class"][c]["rms_percent"]
                                                     for c in classes], colours["baker"])]
    x = np.arange(len(classes))
    for k, (label, values, colour) in enumerate(bars):
        budget_ax.bar(x + (k - 1.5) * 0.2, values, 0.19, color=colour, label=label)
    budget_ax.set_xticks(x, [f"{c}\n{share[c]['pixels_percent']:.0f}% of px,\n{share[c]['sum_of_squares_percent']:.0f}% of rms$^2$"
                             for c in classes], fontsize=8)
    budget_ax.set_ylabel("residual rms (%)")
    budget_ax.set_title("(c) where it is, and what removes it", loc="left", fontsize=10)
    budget_ax.legend(fontsize=6.5, loc="upper right", frameon=False)

    # (d) Every window: rms before and after removing the shared part.
    window_ax = fig.add_subplot(grid[2:, 2])
    before = np.array([w["rms_percent"] for w in baker["per_window"]])
    after = np.array([w["rms_after_percent"] for w in baker["per_window"]])
    centre = np.array([w["window"] for w in baker["per_window"]])
    points = window_ax.scatter(before, after, c=centre, cmap="viridis", s=14, lw=0)
    top_value = float(np.percentile(before, 99)) * 1.1
    window_ax.plot([0, top_value], [0, top_value], color="#10181c", lw=0.6, ls="--")
    window_ax.axhline(0.35, color=colours["baker"], lw=0.6, ls=":")
    window_ax.text(top_value * 0.98, 0.37, "Baker's quoted error", ha="right", fontsize=7, color=colours["baker"])
    window_ax.set_xlim(0, top_value); window_ax.set_ylim(0, top_value)
    window_ax.set_xlabel("IAG residual rms (%)")
    window_ax.set_ylabel("after removing the Baker single-spectra residual (%)")
    window_ax.set_title(f"(d) {len(before)} windows: median {np.median(before):.2f}% -> {np.median(after):.2f}%",
                        loc="left", fontsize=10)
    fig.colorbar(points, ax=window_ax, label="window (cm$^{-1}$)", shrink=0.8)

    fig.suptitle("Half of the IAG co-add fit's residual against Baker's telluric-free Sun is model error "
                 "shared by every spectrum fitted with this model", fontsize=12, x=0.5, y=0.995)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
