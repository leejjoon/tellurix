#!/usr/bin/env python
"""Which per-line corrections Baker et al. 2020's spectra support: three tests, one verdict.

A feature's optical-depth ratio (``analyse_baker_telluric.py``) is a line-list error
only if it is a *strength* error: tau multiplied by one factor, in every spectrum.
An error their reduction left in the spectrum -- a solar-model residue, an iodine
residue, a misplaced zero -- is *fixed*: the same additive amount eps of the
continuum whatever the water column. The two have different signatures, and which
one a test can see depends on the line's depth:

- **air mass** (weak lines): a fixed eps is -eps / (T tau) of the ratio, which falls
  as 1/X while a strength error does not. Read from the analysis's slope of the
  ratio against 1/X. A saturated core's T tau barely moves with X (x1.2-1.4 between
  the low and high third of the spectra), so the test is blind there;
- **profile** (deep lines): jointly over every spectrum, in optical depth,
  tau_obs - tau_mod = (rho - 1) tau + eps (-1/T) + w (-x dtau/dx) + s (-dtau/dnu) + c_k.
  Across a deep line tau runs from ~3 to 0.03 and 1/T is far from linear in it, so
  rho and eps separate. In a weak line T ~ 1 - tau and -eps/T is a constant plus
  eps tau, the strength shape: the profile is blind there (corr(rho, eps) -> 1). The
  width term is free because a saturated line's strength and width meet in its
  wings; a fit without it reads a shape error as strength. With both eps and the
  width free, rho itself is poor -- against the Kitt Peak ratios it correlates 0.07-0.35
  by depth, where Baker's core ratio does 0.69-0.86 -- so the profile is used only to
  *detect* eps, never as the correction;
- **another instrument**: the solar fits' ratios (Kitt Peak, their own solar
  removal). A Baker artefact is not in them; a line-list error is.

What the profile flags on real data is not Baker's: the features it flags agree
with Kitt Peak as well as those it passes (``summary.profile_flags_against_kitt_peak``)
and their eps takes either sign. It reads a profile that one strength and one width
do not describe -- a shape error both instruments see -- so it is reported as a
shape misfit, beside the strength verdict, not as an artefact.

Verdicts: **fixed part** (the air-mass test: a reduction artefact, not a line);
**disagrees with Kitt Peak**; **vetted** (Kitt Peak agrees, the air-mass test
passes); **Baker only** (no solar fit measures it). ``shape_misfit`` is a flag on
any of them. The correction is the mean of Baker's ratio and the solar fits',
which the analysis's held-out test found predicts a further fit best.

Pixels are those the analysis uses (fit mask, Baker's solar model above 0.97 within
0.1 cm-1, iodine template above 0.98 within 0.03 cm-1, model above 0.02), out from the
feature's model minimum to the transmission maxima on either side, at most
``REACH_CM1``. The pixel noise is ``NOISE_SCALE`` x the quoted uncertainty and
correlated over two pixels (measured on clear pixels at 14300 cm-1). The fit
weights pixels as independent, so its errors are scaled by ``CORRELATION_INFLATION``,
what that correlation does to a sum (1.59; without it the injected pulls were
1.3-1.7), and by sqrt(reduced chi2) when it exceeds one.

The profile fit is calibrated by injection: in every ``--inject-every``-th feature,
the spectra's own model with a known rho (eps = 0) and, separately, a known eps
(rho = 1), with noise of that size and correlation, refitted the same way.

Two outputs: ``--output``, a JSON of the method, settings, summary and injection
calibration -- the numbers the prose quotes -- and ``--table`` beside it, one CSV row
per feature with every test's result, so a regeneration diffs feature by feature and
the strength-table builder reads it with numpy alone.

    UV_CACHE_DIR=.uv-cache uv run python scripts/vet_baker_line_corrections.py \\
        --output docs/baker_line_corrections.json --table docs/baker_line_corrections.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATLAS = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/iag_telfree")
SOLAR_CLEAN, SOLAR_SHIFT_CM1 = 0.97, 0.1
IODINE_CLEAN, IODINE_REACH_CM1 = 0.98, 0.03
REACH_CM1 = 0.25
TAU_RANGE = (0.03, 3.0)
NOISE_SCALE = 2.1
# The kernel [a, 1, a] gives neighbouring pixels the measured correlation of 0.63.
NOISE_KERNEL_A = 0.433
CORRELATION_INFLATION = (1 + 2 * NOISE_KERNEL_A) / np.sqrt(1 + 2 * NOISE_KERNEL_A ** 2)
MIN_SPECTRA, MIN_PIXELS = 10, 60
# Beyond this, rho and eps are one parameter and the profile says nothing.
MAX_PROFILE_CORRELATION = 0.98
# Held-out rms of one solar fit's (ratio - 1) predicted by the others
# (docs/baker_telluric_ratios.json, "held_out"): its scatter as a measurement.
SOLAR_FIT_SCATTER = 0.025
SIGNIFICANT = 3.0
SPEED_OF_LIGHT_MS = 299792458.0


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def clean_pixels(atlas: Path, group: int, cache: dict):
    """The group's grid and its pixels clear of their solar model's and the iodine cell's lines."""
    if group not in cache:
        from scipy.ndimage import maximum_filter1d, minimum_filter1d
        from tellurix_fts.baker import read_baker_table

        nu, columns = read_baker_table(atlas, group, ["stel_mod", "iodine"])
        step = np.median(np.diff(nu))
        solar = minimum_filter1d(columns["stel_mod"], 2 * int(np.ceil(SOLAR_SHIFT_CM1 / step)) + 1) > SOLAR_CLEAN
        iodine = ~maximum_filter1d((columns["iodine"] < IODINE_CLEAN).astype(np.uint8),
                                   2 * int(np.ceil(IODINE_REACH_CM1 / step)) + 1).astype(bool)
        cache[group] = (nu, solar & iodine)
    return cache[group]


def feature_span(model: np.ndarray, nu: np.ndarray, centre: float) -> slice:
    """From the model minimum nearest ``centre`` out to the transmission maxima either side."""
    step = nu[1] - nu[0]
    near = np.where(np.abs(nu - centre) < 0.03)[0]
    if near.size == 0:
        return slice(0, 0)
    i = near[np.argmin(model[near])]
    reach = int(round(REACH_CM1 / step))
    lo = i
    while lo > max(0, i - reach) and model[lo - 1] >= model[lo]:
        lo -= 1
    hi = i
    while hi < min(nu.size - 1, i + reach) and model[hi + 1] >= model[hi]:
        hi += 1
    return slice(lo, hi + 1)


def design(spectra, span, centre):
    """Stacked per-spectrum rows: the four shapes, the spectrum, weights, and what is fitted."""
    blocks = []
    for k, s in enumerate(spectra):
        nu, t, obs = s["nu"][span], s["model"][span], s["observed"][span]
        tau = -np.log(np.clip(t, 1e-6, None))
        tau_f = tau - tau.min()
        use = s["usable"][span] & (obs > 0.02) & (tau_f > TAU_RANGE[0]) & (tau_f < TAU_RANGE[1])
        if use.sum() < 3:
            continue
        slope = np.gradient(tau_f, nu)
        shapes = np.column_stack([tau_f, -1.0 / t, -(nu - centre) * slope, -slope])
        blocks.append(dict(k=k, shapes=shapes[use], t=t[use], tau=tau[use],
                           y=(-np.log(np.clip(obs, 1e-6, None)) - tau)[use], sigma=s["sigma"] * NOISE_SCALE))
    return blocks


def solve(blocks, y_of=lambda b: b["y"]):
    """rho - 1, eps, width, shift (cm-1), their errors and corr(rho, eps); one offset per spectrum."""
    shapes = np.vstack([b["shapes"] for b in blocks])
    weight = np.concatenate([b["t"] / b["sigma"] for b in blocks])
    y = np.concatenate([y_of(b) for b in blocks])
    owner = np.concatenate([np.full(b["y"].size, j) for j, b in enumerate(blocks)])
    offsets = (owner[:, None] == np.arange(len(blocks))[None, :]).astype(float)
    x = np.hstack([shapes, offsets]) * weight[:, None]
    coef, *_ = np.linalg.lstsq(x, y * weight, rcond=None)
    chi2 = np.sum((y * weight - x @ coef) ** 2) / max(1, y.size - coef.size)
    cov = np.linalg.pinv(x.T @ x)[:4, :4] * max(1.0, chi2) * CORRELATION_INFLATION ** 2
    err = np.sqrt(np.diag(cov))
    return coef[:4], err, float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])), float(chi2)


def correlated_noise(rng, n, sigma):
    white = rng.normal(size=n + 2)
    a = NOISE_KERNEL_A
    return sigma * (a * white[:-2] + white[1:-1] + a * white[2:]) / np.sqrt(1 + 2 * a * a)


def inject(blocks, rng, rho, eps):
    """Synthetic tau_obs - tau_mod for a model whose truth is (rho, eps), with the real noise."""
    out = []
    for b in blocks:
        tau_f = b["shapes"][:, 0]
        t_true = b["t"] * np.exp(-(rho - 1.0) * tau_f) + eps
        t_obs = t_true + correlated_noise(rng, t_true.size, b["sigma"])
        out.append(-np.log(np.clip(t_obs, 1e-6, None)) - b["tau"])
    return out


def shown(path: Path) -> str:
    """A path as the report records it: from the repository root when it is inside it."""
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def flatten(row: dict) -> dict:
    """One feature as flat columns: the nested tests become prefixed fields."""
    flat = {k: v for k, v in row.items() if k not in ("airmass", "kitt_peak", "profile")}
    for test in ("airmass", "kitt_peak"):
        flat.update({f"{test}_{k}": v for k, v in row[test].items()})
    profile = row["profile"] or {}
    flat.update({f"profile_{k}": profile.get(k) for k in PROFILE_COLUMNS})
    return flat


PROFILE_COLUMNS = ("rho", "rho_error", "eps", "eps_error", "width", "width_error", "shift_ms", "shift_error_ms",
                   "correlation", "chi2", "peak_tau", "spectra", "pixels", "used", "passes")


def write_table(path: Path, rows: list) -> None:
    def cell(value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return ""
        if isinstance(value, (bool, np.bool_)):
            return "1" if value else "0"
        if isinstance(value, (float, np.floating)):
            # Six significant digits suit ratios and errors; a wavenumber needs its
            # 1e-4 cm-1, which is how the feature catalogue gives it.
            return f"{value:.4f}" if abs(value) >= 1000 else f"{value:.6g}"
        return value
    flat = [flatten(r) for r in rows]
    columns = list(dict.fromkeys(k for f in flat for k in f))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows([[cell(f.get(c)) for c in columns] for f in flat])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits", type=Path, default=ROOT / "data/corrected/baker2020")
    parser.add_argument("--analysis", type=Path, default=ROOT / "docs/baker_telluric_ratios.json")
    parser.add_argument("--features-from", type=Path, default=ROOT / "docs/line_depth_ratios.json")
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--inject-every", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()

    catalogue = json.loads(args.features_from.read_text())["features"]
    analysed = {f["feature"]: f for f in json.loads(args.analysis.read_text())["features"]}
    rng = np.random.default_rng(20261022)

    by_window = {}
    for path in sorted(glob.glob(str(args.fits / "g*_*_*.npz"))):
        by_window.setdefault(float(Path(path).stem.rsplit("_", 1)[1]), []).append(Path(path))

    cache, profile, injections = {}, {}, []
    for v1, paths in sorted(by_window.items()):
        spectra = []
        for path in paths:
            meta = json.loads(path.with_suffix(".json").read_text())
            with np.load(path) as data:
                nu = np.asarray(data["wavenumber_cm1"]); order = np.argsort(nu); nu = nu[order]
                take = lambda key: np.asarray(data[key])[order]
                continuum = take("continuum")
                spectrum = dict(nu=nu, observed=take("flux") / continuum, model=take("model_flux") / continuum,
                                sigma=float(np.median(take("uncertainty") / continuum)), airmass=meta["airmass"])
                mask = take("mask").astype(bool)
            grid, clean = clean_pixels(args.atlas, meta["group"], cache)
            clean = np.interp(nu, grid, clean.astype(float)) > 0.999
            spectrum["usable"] = mask & clean & (spectrum["model"] > 0.02)
            spectra.append(spectrum)
        nu = spectra[0]["nu"]
        median_model = np.median([s["model"] for s in spectra], axis=0)
        for index in analysed:
            centre = catalogue[index]["nu"]
            if not nu[10] < centre < nu[-10]:
                continue
            span = feature_span(median_model, nu, centre)
            blocks = design(spectra, span, centre)
            if len(blocks) < MIN_SPECTRA or sum(b["y"].size for b in blocks) < MIN_PIXELS:
                continue
            coef, err, corr, chi2 = solve(blocks)
            peak = float(np.median([b["tau"].max() for b in blocks]))
            profile[index] = dict(rho=1 + coef[0], rho_error=err[0], eps=coef[1], eps_error=err[1],
                                  width=coef[2], width_error=err[2],
                                  shift_ms=coef[3] / centre * SPEED_OF_LIGHT_MS,
                                  shift_error_ms=err[3] / centre * SPEED_OF_LIGHT_MS,
                                  correlation=corr, chi2=chi2, peak_tau=peak, spectra=len(blocks),
                                  pixels=int(sum(b["y"].size for b in blocks)))
            if args.inject_every and index % args.inject_every == 0:
                for truth in ((1.05, 0.0), (1.0, 0.005)):
                    fake = inject(blocks, rng, *truth)
                    lookup = {id(b): f for b, f in zip(blocks, fake)}
                    got, got_err, got_corr, _ = solve(blocks, y_of=lambda b: lookup[id(b)])
                    injections.append(dict(feature=index, peak_tau=peak, rho_true=truth[0], eps_true=truth[1],
                                           rho=1 + got[0], rho_error=got_err[0], eps=got[1], eps_error=got_err[1],
                                           correlation=got_corr))
        print(f"{v1:.0f}: {len(spectra)} spectra, {sum(1 for i in profile if v1 <= catalogue[i]['nu'] < v1 + 30)} features",
              flush=True)

    rows = []
    for index, a in sorted(analysed.items(), key=lambda kv: kv[1]["nu"]):
        p = profile.get(index)
        n = a["spectra"]
        baker_error = a["ratio_std"] / np.sqrt(n)
        b, b_err = a.get("inverse_airmass_slope"), a.get("inverse_airmass_slope_error")
        airmass_ok = None if b is None or b_err in (None, 0) or not np.isfinite(b_err) else bool(abs(b) < SIGNIFICANT * b_err)
        solar = list(a.get("solar", {}).values())
        if solar:
            solar_mean = float(np.mean(solar))
            agree_err = np.hypot(baker_error, SOLAR_FIT_SCATTER / np.sqrt(len(solar)))
            solar_ok = bool(abs(solar_mean - a["ratio"]) < SIGNIFICANT * agree_err)
        else:
            solar_mean, solar_ok = None, None
        profile_used = bool(p is not None and abs(p["correlation"]) < MAX_PROFILE_CORRELATION)
        profile_ok = bool(abs(p["eps"]) < SIGNIFICANT * p["eps_error"]) if profile_used else None
        if airmass_ok is False:
            verdict = "fixed part"
        elif solar_ok is False:
            verdict = "disagrees with Kitt Peak"
        elif solar_ok:
            verdict = "vetted"
        else:
            verdict = "Baker only"
        estimates = [a["ratio"]] + solar
        rows.append(dict(
            feature=index, nu=a["nu"], lines=catalogue[index].get("lines"), s_max=catalogue[index].get("s_max"),
            e_lower=catalogue[index].get("e_lower_strongest"), molecule=catalogue[index].get("molecule"),
            spectra=n, baker_ratio=a["ratio"], baker_error=baker_error,
            airmass=dict(slope=b, error=b_err, passes=airmass_ok),
            kitt_peak=dict(fits=len(solar), mean=solar_mean, passes=solar_ok),
            profile=(dict(p, used=profile_used, passes=profile_ok) if p is not None else None),
            verdict=verdict, shape_misfit=profile_ok is False,
            # The analysis's held-out test: Baker and the solar fits together predict a
            # further fit best. A saturated core's ratio is not its strength: see profile.rho.
            correction_ratio=float(np.mean(estimates)),
            saturated=bool(p is not None and p["peak_tau"] > 1.5)))

    verdicts = [r["verdict"] for r in rows]
    used = [r for r in rows if r["profile"] and r["profile"]["used"]]
    deep = [r["profile"] for r in used if r["profile"]["peak_tau"] > 0.7]

    def calibration(rho_true, eps_true, lo, hi):
        sel = [j for j in injections if j["rho_true"] == rho_true and j["eps_true"] == eps_true
               and lo <= j["peak_tau"] < hi and abs(j["correlation"]) < MAX_PROFILE_CORRELATION]
        if not sel:
            return None
        rho = np.array([j["rho"] for j in sel]); eps = np.array([j["eps"] for j in sel])
        return dict(features=len(sel), rho_bias=float(np.median(rho) - rho_true),
                    rho_pull_rms=float(np.std((rho - rho_true) / np.array([j["rho_error"] for j in sel]))),
                    eps_bias=float(np.median(eps) - eps_true),
                    eps_pull_rms=float(np.std((eps - eps_true) / np.array([j["eps_error"] for j in sel]))),
                    false_fixed=float(np.mean(np.abs(eps) > SIGNIFICANT * np.array([j["eps_error"] for j in sel])))
                    if eps_true == 0 else None,
                    eps_detected=float(np.mean(np.abs(eps) > SIGNIFICANT * np.array([j["eps_error"] for j in sel])))
                    if eps_true != 0 else None)

    bins = ((0.1, 0.7), (0.7, 1.5), (1.5, 10.0))

    def flagged(test):
        sel = [r for r in rows if test(r) and r["kitt_peak"]["fits"]]
        diff = np.array([r["baker_ratio"] - r["kitt_peak"]["mean"] for r in sel])
        return dict(features=len(sel), baker_minus_kitt_peak_rms=float(np.sqrt(np.mean(diff ** 2))),
                    disagree=float(np.mean([r["kitt_peak"]["passes"] is False for r in sel])))

    def agreement(lo, hi):
        sel = [r for r in rows if r["profile"] and r["kitt_peak"]["fits"] and lo <= r["profile"]["peak_tau"] < hi]
        if len(sel) < 3:
            return None
        kp = [r["kitt_peak"]["mean"] for r in sel]
        return dict(features=len(sel), baker_core=float(np.corrcoef([r["baker_ratio"] for r in sel], kp)[0, 1]),
                    profile_rho=float(np.corrcoef([r["profile"]["rho"] for r in sel], kp)[0, 1]))
    summary = dict(
        features=len(rows),
        verdicts={v: verdicts.count(v) for v in sorted(set(verdicts))},
        profile_fitted=sum(1 for r in rows if r["profile"]),
        profile_used=len(used),
        profile_used_by_peak_tau={f"{lo}-{hi}": sum(1 for r in used if lo <= r["profile"]["peak_tau"] < hi) for lo, hi in bins},
        fixed_by_airmass=sum(1 for r in rows if r["airmass"]["passes"] is False),
        shape_misfit=sum(1 for r in rows if r["shape_misfit"]),
        vetted_without_shape_misfit=sum(1 for r in rows if r["verdict"] == "vetted" and not r["shape_misfit"]),
        profile_flags_against_kitt_peak={name: flagged(test) for name, test in
                                         (("flagged", lambda r: r["shape_misfit"]),
                                          ("passed", lambda r: r["profile"] and r["profile"]["passes"]))},
        deep_line_width=dict(features=len(deep), median=float(np.median([d["width"] for d in deep])) if deep else None,
                             rms=float(np.std([d["width"] for d in deep])) if deep else None),
        # Against the independent instrument, by depth: the core ratio, and the
        # profile's rho with eps and width free -- why the profile only detects.
        against_kitt_peak={f"{lo}-{hi}": agreement(lo, hi) for lo, hi in bins},
        vetted_correction_rms=float(np.sqrt(np.mean([(r["correction_ratio"] - 1) ** 2 for r in rows if r["verdict"] == "vetted"]))),
        injection={f"rho{rt}_eps{et}": {f"{lo}-{hi}": calibration(rt, et, lo, hi) for lo, hi in bins}
                   for rt, et in ((1.05, 0.0), (1.0, 0.005))})
    report = dict(method=__doc__, settings=dict(
        fits=shown(args.fits), analysis=shown(args.analysis), reach_cm1=REACH_CM1, tau_range=TAU_RANGE,
        noise_scale=NOISE_SCALE, max_profile_correlation=MAX_PROFILE_CORRELATION,
        solar_fit_scatter=SOLAR_FIT_SCATTER, significant=SIGNIFICANT, inject_every=args.inject_every),
        table=shown(args.table),
        summary=summary)
    args.output.write_text(json.dumps(json_safe(report), indent=1) + "\n")
    write_table(args.table, rows)
    print(json.dumps(json_safe(summary), indent=1))


if __name__ == "__main__":
    main()
