#!/usr/bin/env python
"""What is left in a night's standards residual after the blaze and the pattern?

Reads one fit_igrins_standard.py run (blaze and response pattern applied), the
night's lamp flat, and optionally other nights' runs, and measures four things:

* **fringe** -- the lamp's and the stars' periodic structure in wavenumber:
  period, amplitude, harmonic, phase per order, its stability over the night,
  and how much of it the committed blaze carries into the star. Both signals
  are taken against a *wide* smooth (Savitzky-Golay 401, cubic), because the
  blaze's own filters pass part of a ~2.8 cm-1 fringe.
* **scales** -- the residual band-passed into a short (0.7-4 cm-1 boxcars) and
  a long (4-15 cm-1) band, correlated between frames: same star, different stars
  under an hour apart and over three hours apart; clear and absorbing orders;
  in the stellar rest frame; against airmass, telluric depth and the lamp; and
  any sub-pixel shift between frames (flexure).
* **nights** -- whether the static short-band structure repeats on other nights,
  aligned on detector pixels, with a 40-pixel shifted control; and what a master
  from the other nights would remove.
* **pixel scale** -- the residual under 15 pixels on clear pixels, which only
  a high signal-to-noise night resolves: how much each frame shares with the
  others, what is left after a leave-one-out template, and how concentrated
  that remainder is and where it sits on the detector (order x 128-column
  blocks); across nights, on detector columns, with a lag search, how much of
  each night's template is isolated spike pixels that recur, and what masking
  another night's spike pixels buys against subtracting its whole template.
* **fringe per frame** -- each frame's own fringe amplitude and phase per
  order, and removing it per frame against removing the other frames' mean.
* **fixes** -- candidate corrections estimated by subtracting them from the
  residual, without a refit: a leave-one-out fringe (one sinusoid at the band's
  period per order, fitted to the other frames' mean), a leave-one-out
  short-band term on line-free pixels only, and a time-weighted (1 h)
  leave-one-out long-band term.

    uv run python scripts/analyze_igrins_residual_structure.py --band H \\
        --run data/corrected/igrins/blazeB9_dct2018_h \\
        --blaze data/calibration/igrins_blaze_dct2018_h.h5 \\
        --cal-dir data/igrins/cals/20181220 \\
        --spec data/igrins/20181220_0100/SDCH_20181220_0100.spec.fits \\
        --night DCT18=data/corrected/igrins/ladder_a0v --own DCT18 \\
        --night DCT16=data/corrected/igrins/dct2016_h ... \\
        --output docs/igrins_residual_structure_h.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

TRIM = 100                   # samples dropped at each end of the reliable range
MINIMUM_FRAMES = 5           # the fit driver's own minimum for a leave-one-out pattern
PIXEL_SCALE = 15             # pixels: the high-pass behind the "pixel-scale" measures
SPIKE_SIGMA = 5.0            # robust sigmas: a pixel of a night's template counted as a spike
BLOCK = 128                  # detector columns per block in the per-frame excess map
LAMP_TRIM = 150
PERIODS = np.linspace(2.5, 3.1, 241)
# H orders whose Brackett line sits inside the usable pixels, where the fitted
# stellar velocity is not pulled by a wing (docs/igrins_a0v.md).
CENTRAL = (98, 103, 109, 113, 120)


def boxcar(y, width, ok):
    k = np.ones(max(int(width), 1))
    num = np.convolve(np.where(ok, y, 0.0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    return np.where(den > 0.6 * k.size, num / np.maximum(den, 1e-9), np.nan)


def band_pass(y, ok, per_cm1, lo, hi):
    return boxcar(y, per_cm1 * lo, ok) - boxcar(y, per_cm1 * hi, ok)


def corr(a, b, minimum=300):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < minimum:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def sinusoid(nu, y, period, trend=3):
    """Least-squares amplitude and phase at one period, over a polynomial trend."""
    x = 2 * np.pi * nu / period
    t = (nu - nu.mean()) / np.ptp(nu)
    design = np.column_stack([np.cos(x), np.sin(x)] + [t ** k for k in range(trend + 1)])
    c, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(np.hypot(c[0], c[1])), float(np.arctan2(c[1], c[0])), design[:, :2] @ c[:2]


def best_period(nu, y, periods=PERIODS):
    amplitudes = [sinusoid(nu, y, p)[0] for p in periods]
    k = int(np.argmax(amplitudes))
    return float(periods[k]), float(amplitudes[k])


def wide(y, window=401):
    from scipy.signal import savgol_filter

    out = np.full(y.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(y))
    out[idx] = savgol_filter(y[idx], window, 3)
    return out


def circular_scatter_deg(phases):
    return float(np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * np.asarray(phases))))))))


def wrap_deg(x):
    return float(np.degrees(np.angle(np.exp(1j * x))))


def clipped(idx, y, k=5.0):
    med = np.median(y[idx])
    mad = 1.4826 * np.median(np.abs(y[idx] - med))
    return idx[np.abs(y[idx] - med) < k * mad]


def shift(y, dv_kms, dlnnu):
    s = dv_kms / 299792.458 / dlnnu
    x = np.arange(y.size)
    ok = np.isfinite(y)
    moved = np.interp(x - s, x[ok], y[ok])
    return np.where(np.interp(x - s, x, ok.astype(float)) > 0.99, moved, np.nan)


def lag(a, b, maxlag=6):
    ok = np.isfinite(a) & np.isfinite(b)
    a0, b0 = np.where(ok, a, 0), np.where(ok, b, 0)
    lags = np.arange(-maxlag, maxlag + 1)
    cc = np.array([np.sum(a0 * np.roll(b0, -L)) for L in lags])
    k = int(np.argmax(cc))
    if 0 < k < lags.size - 1:
        y0, y1, y2 = cc[k - 1:k + 2]
        return float(lags[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))
    return float(lags[k])


def median(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def summaries(run: Path):
    """Each frame's summary, merged over order shards (``.s0``, ``.s1``, ...)."""
    frames = {}
    for path in sorted(glob.glob(str(run / "*_summary.json"))):
        stem = re.sub(r"(\.s\d+)?_summary\.json$", "", path)
        summary = json.loads(Path(path).read_text())
        entry = frames.setdefault(stem, {"observation": summary["observation"], "results": []})
        entry["results"] += summary["results"]
    return frames


def load_run(run: Path, band: str, repo: Path, pixels_from=None):
    """Per frame and order: the arrays the analysis needs, on the order's detector pixels."""
    from tellurix import read_igrins_observation

    frames, meta = {}, {}
    pixel_cache = {}
    for stem, summary in summaries(run).items():
        obs = summary["observation"]
        frame = Path(stem).name.split("_")[-1]
        rows = {r["order"]: r for r in summary["results"]}
        meta[frame] = dict(star=re.split(r"\s+V\s+", obs["object"])[0].strip(), mjd=obs["mjd"],
                           airmass=float(np.median([r["airmass"] for r in rows.values()])),
                           vstar=median([rows[o]["stellar_velocity_kms"] for o in CENTRAL if o in rows]))
        for n, row in rows.items():
            npz = Path(f"{stem}_{row['name']}.npz")
            if not npz.exists():
                continue
            if n not in pixel_cache:
                path = Path(obs["path"])
                order = read_igrins_observation(path if path.is_absolute() else repo / path).order(n)
                pixel_cache[n] = (1e7 / np.asarray(order.wavelength_vacuum_nm), np.asarray(order.pixel))
            d = dict(np.load(npz))
            # The cache runs in ascending wavenumber, the order in ascending
            # wavelength: match samples to detector columns by wavenumber, never
            # by position -- reading one against the other reverses the order.
            nu, pix = pixel_cache[n]
            rank = np.argsort(nu)
            column = np.interp(d["wavenumber_cm1"], nu[rank], pix[rank].astype(float))
            if d["observed"].size != pix.size or np.max(np.abs(column - np.rint(column))) > 0.01:
                raise SystemExit(f"{npz}: samples do not match the order's pixels")
            d["pixel"] = np.rint(column).astype(int)
            frames.setdefault(n, {})[frame] = d
    return frames, meta


def interior(d):
    ok = d["reliable"].astype(bool).copy()
    idx = np.flatnonzero(ok)
    if idx.size < 600:
        return None
    ok[:idx[0] + TRIM] = False
    ok[idx[-1] - TRIM + 1:] = False
    return ok


def per_cm1(d):
    return 1 / np.median(np.abs(np.diff(d["wavenumber_cm1"])))


def fringe(frames, meta, band, blaze, lamp):
    """The lamp's and the stars' periodic structure, per order."""
    lamp_rows, star_rows = {}, {}
    data = {}
    for n in sorted(frames):
        if n not in lamp or not blaze.usable(n):
            continue
        any_frame = next(iter(frames[n].values()))
        pix = any_frame["pixel"]
        nu = any_frame["wavenumber_cm1"]
        reference = wide(np.where(lamp[n] > 0.1 * np.nanmax(lamp[n]), lamp[n], np.nan))
        ell = (lamp[n] / reference - 1)[pix]
        leak = (blaze.orders[n] / reference - 1)[pix]
        idx = np.flatnonzero(np.isfinite(ell) & np.isfinite(leak))[LAMP_TRIM:-LAMP_TRIM]
        if idx.size < 800:
            continue
        # the driver saved the blaze it divided by: the pixel mapping must reproduce it
        applied = any_frame.get("blaze")
        if applied is not None:
            expected = blaze.orders[n][pix] / np.nanmax(blaze.orders[n])
            if np.nanmax(np.abs(applied[idx] - expected[idx])) > 1e-5:
                raise SystemExit(f"{band}{n}: the pixel mapping does not reproduce the applied blaze")
        data[n] = (nu, ell, leak, clipped(idx, ell), idx)
    # the band's period: the one the lamp's orders agree on best
    trial = np.linspace(2.6, 3.0, 161)
    score = [np.median([sinusoid(nu[k], ell[k], p)[0] for nu, ell, _, k, _ in data.values()]) for p in trial]
    period = float(trial[int(np.argmax(score))])
    centres, lamp_phases, star_phases = [], [], []
    for n, (nu, ell, leak, keep, idx) in data.items():
        p_l, a_l = best_period(nu[keep], ell[keep])
        amp_l, ph_l, _ = sinusoid(nu[keep], ell[keep], period)
        harmonic = sinusoid(nu[keep], ell[keep], p_l / 2)[0]
        amp_leak, ph_leak, _ = sinusoid(nu[idx], leak[idx], period)
        lamp_rows[f"{band}{n}"] = dict(
            best_period_cm1=p_l, amplitude=amp_l, second_harmonic=harmonic,
            pixels_per_cycle=float(period / np.median(np.abs(np.diff(nu[idx])))),
            blaze_carries=amp_leak / amp_l if amp_l > 0 else np.nan,
            blaze_phase_offset_deg=wrap_deg(ph_leak - ph_l))
        # the stars: the fit's residual with the pattern and the blaze's copy put back
        stack, phases, amps, times = [], [], [], []
        for f, d in frames[n].items():
            ok = d["reliable"].astype(bool)
            r = np.where(ok, d["observed"] / d["model_flux"], np.nan)
            full = r * (1 + d["response_pattern"]) * (1 + leak) - 1
            after = r - 1
            k = np.flatnonzero(np.isfinite(full))[LAMP_TRIM:-LAMP_TRIM]
            if k.size < 600:
                continue
            k = clipped(k, full)
            a, ph, _ = sinusoid(nu[k], full[k], period)
            a_after = sinusoid(nu[k], after[k], period)[0]
            smooth = np.convolve(after[k], np.ones(15) / 15, "valid")
            phases.append(ph)
            amps.append((a, a_after, (a_after ** 2 / 2) / np.var(smooth)))
            times.append(meta[f]["mjd"])
            stack.append(full)
        if len(stack) < 5:
            continue
        mean = np.nanmean(stack, axis=0)
        k = clipped(np.flatnonzero(np.isfinite(mean))[LAMP_TRIM:-LAMP_TRIM], mean)
        p_s, _ = best_period(nu[k], mean[k])
        order_time = np.argsort(times)
        drift = np.polyfit((np.array(times)[order_time] - min(times)) * 24,
                           np.degrees(np.unwrap(np.array(phases)[order_time])), 1)[0]
        mean_phase = float(np.angle(np.mean(np.exp(1j * np.array(phases)))))
        star_rows[f"{band}{n}"] = dict(
            best_period_cm1=p_s, amplitude=median([a[0] for a in amps]),
            amplitude_after_pattern=median([a[1] for a in amps]),
            share_of_15px_smoothed_variance=median([a[2] for a in amps]),
            frame_phase_scatter_deg=circular_scatter_deg(phases),
            phase_drift_deg_per_hour=float(drift),
            phase_minus_lamp_deg=wrap_deg(mean_phase - ph_l))
        centres.append(float(np.median(nu[idx])))
        lamp_phases.append(ph_l)
        star_phases.append(mean_phase)

    def coherence(phases):
        """One etalon at one angle: phase linear in the order's centre wavenumber."""
        order = np.argsort(centres)
        x, u = np.array(centres)[order], np.unwrap(np.array(phases)[order])
        c = np.polyfit(x, u, 1)
        resid = np.angle(np.exp(1j * (u - np.polyval(c, x))))
        return float(np.degrees(np.sqrt(np.mean(resid ** 2))))

    strong = [r for r in star_rows.values() if r["amplitude"] > 0.0015]
    return dict(
        period_cm1=period,
        optical_thickness_nd_mm=10.0 / (2 * period),
        lamp=dict(orders=lamp_rows,
                  median_amplitude=median([r["amplitude"] for r in lamp_rows.values()]),
                  median_second_harmonic=median([r["second_harmonic"] for r in lamp_rows.values()]),
                  median_blaze_carries=median([r["blaze_carries"] for r in lamp_rows.values()]),
                  phase_rms_about_one_etalon_deg=coherence(lamp_phases)),
        star=dict(orders=star_rows,
                  median_amplitude=median([r["amplitude"] for r in star_rows.values()]),
                  median_amplitude_after_pattern=median([r["amplitude_after_pattern"] for r in star_rows.values()]),
                  median_frame_phase_scatter_deg_where_strong=median([r["frame_phase_scatter_deg"] for r in strong]),
                  median_abs_drift_deg_per_hour_where_strong=median([abs(r["phase_drift_deg_per_hour"]) for r in strong]),
                  phase_rms_about_one_etalon_deg=coherence(star_phases)))


def scales(frames, meta, band, lamp, blaze, period):
    pairs = {}
    rest = {"detector frame": [], "stellar rest frame": []}
    airmass_slope = {"clear": [], "absorbing": []}
    depth_corr = {"clear": [], "absorbing": []}
    lamp_corr, fringe_share, lags = [], [], {}

    def add(key, value):
        pairs.setdefault(key, []).append(value)

    for n, per_frame in sorted(frames.items()):
        bands, transmission = {}, []
        for f, d in per_frame.items():
            ok = interior(d)
            if ok is None:
                continue
            c = per_cm1(d)
            y = d["residual"] / d["model_flux"]
            short = band_pass(y, ok, c, 0.7, 4.0)
            m = np.isfinite(short)
            fringe_share.append(1 - np.var(short[m] - sinusoid(d["wavenumber_cm1"][m], short[m], period,
                                                               trend=2)[2]) / np.var(short[m]))
            both = band_pass(y, ok, c, 1.0, 15.0)
            bands[f] = dict(short=short, long=band_pass(y, ok, c, 4.0, 15.0), both=both,
                            rest=shift(both, -meta[f]["vstar"], np.median(np.abs(np.diff(np.log(d["wavenumber_cm1"]))))),
                            depth=band_pass(1 - d["transmission"], ok, c, 0.7, 4.0), ok=ok)
            transmission.append(np.median(d["transmission"][ok]))
        if len(bands) < MINIMUM_FRAMES:
            continue
        kind = "clear" if np.median(transmission) > 0.97 else "absorbing"
        fs = sorted(bands)
        for i, a in enumerate(fs):
            for b in fs[i + 1:]:
                hours = abs(meta[a]["mjd"] - meta[b]["mjd"]) * 24
                if meta[a]["star"] == meta[b]["star"]:
                    when = "same star"
                elif hours < 1:
                    when = "different stars, <1 h apart"
                elif hours > 3:
                    when = "different stars, >3 h apart"
                else:
                    when = None
                if when:
                    for scale in ("short", "long"):
                        add((scale, kind, when), corr(bands[a][scale], bands[b][scale]))
                        add((scale, "all", when), corr(bands[a][scale], bands[b][scale]))
                if meta[a]["star"] != meta[b]["star"]:
                    rest["detector frame"].append(corr(bands[a]["both"], bands[b]["both"]))
                    rest["stellar rest frame"].append(corr(bands[a]["rest"], bands[b]["rest"]))
        # the static part: its size frame by frame against airmass -- an atmospheric
        # error would grow ~in proportion; and where it sits against the lines
        template = np.nanmedian([bands[f]["short"] for f in fs], axis=0)
        m = np.isfinite(template)
        coefficient = []
        for f in fs:
            k = m & np.isfinite(bands[f]["short"])
            coefficient.append(np.sum(bands[f]["short"][k] * template[k]) / np.sum(template[k] ** 2))
        airmass = np.array([meta[f]["airmass"] for f in fs])
        airmass_slope[kind].append(float(np.polyfit(airmass, coefficient, 1)[0] / np.mean(coefficient)))
        depth_corr[kind].append(corr(template, np.nanmedian([bands[f]["depth"] for f in fs], axis=0)))
        if n in lamp and blaze.usable(n):
            d = next(iter(per_frame.values()))
            relative = (lamp[n] / blaze.orders[n] - 1)[d["pixel"]]
            ok = np.all([bands[f]["ok"] for f in fs], axis=0) & np.isfinite(relative)
            lamp_corr.append(corr(template, band_pass(relative, ok, per_cm1(d), 0.7, 4.0)))
        for f in fs:
            others = np.nanmedian([bands[g]["short"] for g in fs if g != f], axis=0)
            lags.setdefault(f, []).append(lag(others, bands[f]["short"]))
    table = {}
    for (scale, kind, when), values in pairs.items():
        table.setdefault(scale, {}).setdefault(kind, {})[when] = dict(r=median(values), pairs=len(values))
    return dict(
        bands_cm1={"short": [0.7, 4.0], "long": [4.0, 15.0]},
        frame_pair_correlation=table,
        different_stars_by_frame={k: median(v) for k, v in rest.items()},
        static_short_scale={
            kind: dict(orders=len(airmass_slope[kind]),
                       relative_growth_per_unit_airmass=median(airmass_slope[kind]),
                       expected_if_atmospheric=float(1 / np.mean([m["airmass"] for m in meta.values()])),
                       correlation_with_telluric_depth=median(depth_corr[kind]))
            for kind in airmass_slope},
        static_short_scale_vs_lamp=median(lamp_corr),
        fringe_share_of_short_scale_variance=median(fringe_share),
        shift_against_the_others_pixels={f: median(v) for f, v in sorted(lags.items())})


def high_pass(d, ok):
    """The residual's pixel-scale part: what a PIXEL_SCALE boxcar does not follow."""
    y = d["residual"] / d["model_flux"]
    return np.where(ok, y - boxcar(y, PIXEL_SCALE, ok), np.nan)


def clear_interior(d):
    """Interior pixels whose 1 cm-1-smoothed model transmission is above 0.99."""
    ok = interior(d)
    if ok is None:
        return None
    return ok & (boxcar(d["transmission"], per_cm1(d), ok) > 0.99)


def templates(frames, minimum=5, scale="short"):
    out, clear = {}, {}
    for n, per_frame in frames.items():
        rows, transmission = [], []
        for d in per_frame.values():
            ok = interior(d) if scale == "short" else clear_interior(d)
            if ok is None:
                continue
            row = np.full(2048, np.nan)
            row[d["pixel"]] = (band_pass(d["residual"] / d["model_flux"], ok, per_cm1(d), 0.7, 4.0)
                               if scale == "short" else high_pass(d, ok))
            rows.append(row)
            transmission.append(np.median(d["transmission"][ok]))
        if len(rows) >= minimum:
            out[n] = np.nanmedian(rows, axis=0)
            clear[n] = np.median(transmission) > 0.97
    return out, clear


def nights(runs, own, band, repo):
    loaded = {name: load_run(path, band, repo)[0] for name, path in runs.items()}
    t = {name: templates(frames) for name, frames in loaded.items()}
    names = list(t)
    # the pixel scale, on clear pixels: does it repeat, and does another night's remove it?
    tp = {name: templates(frames, scale="pixel")[0] for name, frames in loaded.items()}
    pixel_table = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(tp[a]) & set(tp[b]))
            lag0, lags = [], []
            for n in common:
                rs = {L: corr(tp[a][n], np.roll(tp[b][n], L), 200) for L in range(-4, 5)}
                if np.isfinite(rs[0]):
                    lag0.append(rs[0])
                    lags.append(max(rs, key=lambda k: np.nan_to_num(rs[k], nan=-9.0)))
            pixel_table[f"{a} x {b}"] = dict(
                r=median(lag0), orders=len(lag0), best_lags=sorted(set(lags)),
                shifted_20px_control=median([corr(tp[a][n], np.roll(tp[b][n], 20), 200) for n in common]))
    # Most of that template is isolated pixels that recur between nights: how much,
    # and what masking another night's spike pixels buys against subtracting its template.
    spikes = {name: {n: np.abs(robust_z(tmp)) > SPIKE_SIGMA for n, tmp in tp[name].items()} for name in names}
    spike_runs = {}
    for name in names:
        values = np.concatenate([tmp[np.isfinite(tmp)] for tmp in tp[name].values()]) if tp[name] else np.array([])
        if values.size:
            sq = np.sort(values ** 2)[::-1]
            zs = np.concatenate([robust_z(tmp)[np.isfinite(tmp)] for tmp in tp[name].values()])
            spike_runs[name] = dict(top_1pct_share_of_template_variance=float(sq[:sq.size // 100].sum() / sq.sum()),
                                    spike_pixel_fraction=float(np.mean(np.abs(zs) > SPIKE_SIGMA)))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = pixel_table[f"{a} x {b}"]
            cut, again = [], []
            for n in set(tp[a]) & set(tp[b]):
                keep = ~spikes[a][n] & ~spikes[b][n]
                cut.append(corr(np.where(keep, tp[a][n], np.nan), np.where(keep, tp[b][n], np.nan), 200))
                sb = spikes[b][n] | np.roll(spikes[b][n], 1) | np.roll(spikes[b][n], -1)
                if spikes[a][n].sum():
                    again.append((spikes[a][n] & sb).sum() / spikes[a][n].sum())
            row["r_without_spike_pixels"] = median(cut)
            row["spike_pixels_recurring_within_1px"] = median(again)
    pixel_gain = {}
    for source in names:
        if source == own:
            continue
        acc = [0.0, 0.0, 0.0, 0]
        masked = [0.0, 0, 0]
        for n, per_frame in loaded[own].items():
            if n not in tp[source]:
                continue
            rows = {}
            for f, d in per_frame.items():
                ok = clear_interior(d)
                if ok is not None:
                    rows[f] = (high_pass(d, ok), d)
            for f, (h, d) in rows.items():
                other = tp[source][n][d["pixel"]]
                loo = np.nanmedian([rows[g][0] for g in rows if g != f], axis=0)
                k = np.isfinite(h) & np.isfinite(other) & np.isfinite(loo)
                scale = (d["model_flux"] / d["uncertainty"])[k]
                acc[0] += np.sum((h[k] * scale) ** 2)
                acc[1] += np.sum(((h - other)[k] * scale) ** 2)
                acc[2] += np.sum(((h - loo)[k] * scale) ** 2)
                acc[3] += int(k.sum())
                keep = k & ~spikes[source][n][d["pixel"]]
                masked[0] += np.sum((h[keep] * (d["model_flux"] / d["uncertainty"])[keep]) ** 2)
                masked[1] += int(keep.sum())
                masked[2] += int(k.sum())
        if acc[3]:
            pixel_gain[source] = dict(pixels=acc[3], now=float(np.sqrt(acc[0] / acc[3])),
                                      other_night_template=float(np.sqrt(acc[1] / acc[3])),
                                      other_night_spike_mask=float(np.sqrt(masked[0] / masked[1])),
                                      mask_removes_fraction=float(1 - masked[1] / masked[2]),
                                      own_night_leave_one_out=float(np.sqrt(acc[2] / acc[3])))
    table = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(t[a][0]) & set(t[b][0]))
            row = {}
            for kind, want in (("clear", True), ("absorbing", False)):
                values = [corr(t[a][0][n], t[b][0][n]) for n in common if t[a][1].get(n) == want]
                row[kind] = dict(r=median(values), orders=len(values))
            row["shifted_40px_control"] = median([corr(t[a][0][n], np.roll(t[b][0][n], 40)) for n in common])
            table[f"{a} x {b}"] = row
    # what a master from the other nights would take out of this one
    master = {}
    for name in names:
        if name == own:
            continue
        for n, template in t[name][0].items():
            master.setdefault(n, []).append(template)
    master = {n: np.nanmean(v, axis=0) for n, v in master.items() if len(v) >= 2}
    acc = {"clear": [[], [], [], []], "absorbing": [[], [], [], []]}
    for n, per_frame in loaded[own].items():
        if n not in master:
            continue
        for d in per_frame.values():
            ok = interior(d)
            if ok is None:
                continue
            m = master[n][d["pixel"]]
            use = ok & np.isfinite(m)
            y = d["residual"] / d["model_flux"]
            short = band_pass(y, ok, per_cm1(d), 0.7, 4.0)
            k = use & np.isfinite(short)
            a = acc["clear" if np.median(d["transmission"][ok]) > 0.97 else "absorbing"]
            a[0].append(np.mean((d["residual"][use] / d["uncertainty"][use]) ** 2))
            a[1].append(np.mean(((y - m)[use] * d["model_flux"][use] / d["uncertainty"][use]) ** 2))
            a[2].append(np.std(short[k]))
            a[3].append(np.std((short - m)[k]))
    gain = {kind: dict(order_frames=len(a[0]),
                       z_rms=[float(np.sqrt(np.mean(a[0]))), float(np.sqrt(np.mean(a[1])))],
                       short_band_rms=[median(a[2]), median(a[3])])
            for kind, a in acc.items() if a[0]}
    return dict(runs={k: str(v) for k, v in runs.items()}, own=own,
                template_correlation=table, master_from_other_nights=gain,
                pixel_scale_template_correlation=pixel_table,
                pixel_scale_spikes=spike_runs,
                pixel_scale_z_rms_on_own_night=pixel_gain)


def robust_z(values):
    """Values over their robust (MAD) scale."""
    return values / (1.4826 * np.nanmedian(np.abs(values)))


def pixel_scale(frames, meta):
    """The residual under PIXEL_SCALE pixels, on clear pixels: shared between frames, or each frame's own?

    Per frame: z rms now and after subtracting the median of the other frames
    (leave-one-out); how concentrated the remainder is (the share of chi-squared
    in its top 1% of pixels, and a robust width), its correlation with that
    leave-one-out template, and its lag-1 autocorrelation.
    """
    per, pair_r = {}, []
    for n, per_frame in frames.items():
        rows = {}
        for f, d in per_frame.items():
            ok = clear_interior(d)
            if ok is not None:
                rows[f] = (high_pass(d, ok), d)
        if len(rows) < MINIMUM_FRAMES:
            continue
        fs = sorted(rows)
        for i, a in enumerate(fs):
            for b in fs[i + 1:]:
                if meta[a]["star"] != meta[b]["star"]:
                    pair_r.append(corr(rows[a][0], rows[b][0], 200))
        for f, (h, d) in rows.items():
            loo = np.nanmedian([rows[g][0] for g in rows if g != f], axis=0)
            k = np.isfinite(h) & np.isfinite(loo)
            if k.sum() < 200:
                continue
            scale = d["model_flux"] / d["uncertainty"]
            r = h - loo
            a, b = r[:-1], r[1:]
            m = np.isfinite(a) & np.isfinite(b)
            e = per.setdefault(f, dict(z0=[], z1=[], snr=[], r=[], lag1=[], blocks={}))
            zz = r * scale
            e["blocks"][int(n)] = [
                (lambda m: round(float(np.sqrt(np.mean(zz[m] ** 2))), 2) if m.sum() > 20 else None)(
                    (d["pixel"] >= c0) & (d["pixel"] < c0 + BLOCK) & np.isfinite(zz))
                for c0 in range(0, 2048, BLOCK)]
            e["z0"].append((h * scale)[k])
            e["z1"].append((r * scale)[k])
            e["snr"].append(np.median(scale[k]))
            e["r"].append(corr(h, loo, 200))
            e["lag1"].append(np.corrcoef(a[m], b[m])[0, 1])
    out = {}
    for f, e in sorted(per.items()):
        z0, z1 = np.concatenate(e["z0"]), np.concatenate(e["z1"])
        top = np.sort(z1 ** 2)[::-1]
        out[f] = dict(star=meta[f]["star"], pixel_snr=median(e["snr"]),
                      z_rms=float(np.sqrt(np.mean(z0 ** 2))),
                      z_rms_after_leave_one_out=float(np.sqrt(np.mean(z1 ** 2))),
                      robust_width_after=float(1.4826 * np.median(np.abs(z1))),
                      top_1pct_share_of_chi2=float(top[:max(1, top.size // 100)].sum() / top.sum()),
                      r_with_leave_one_out=median(e["r"]), lag1_autocorrelation=median(e["lag1"]),
                      z_rms_by_order_and_column_block=dict(sorted(e["blocks"].items())))
    return dict(scale_pixels=PIXEL_SCALE, column_block=BLOCK,
                different_star_frame_correlation=median(pair_r), frames=out)


def varying_fringe(nu, k, y, period):
    """A sinusoid at a fixed period, amplitude quadratic along the order, over a cubic trend; fitted on k."""
    x = 2 * np.pi * nu / period
    t = (nu - nu[k].mean()) / np.ptp(nu[k])
    wave = np.column_stack([np.cos(x), np.sin(x), t * np.cos(x), t * np.sin(x),
                            t * t * np.cos(x), t * t * np.sin(x)])
    trend = np.column_stack([t ** j for j in range(4)])
    c, *_ = np.linalg.lstsq(np.hstack([wave[k], trend[k]]), y[k], rcond=None)
    return wave @ c[:6]


def fringe_frames(frames, meta, band, period):
    """Each frame's own fringe, and what removing it buys against a template from the other frames.

    On clear interior pixels: the frame's own fringe (varying_fringe) against
    the same model fitted to the other frames' mean. Per frame, the per-pixel z
    rms and the rms of PIXEL_SCALE-pixel means scaled so white noise reads 1.
    The phase table fits each frame alone, with the pattern put back.
    """
    phases, gains = {}, {}
    for n, per_frame in sorted(frames.items()):
        rows = {}
        for f, d in per_frame.items():
            ok = clear_interior(d)
            if ok is not None and ok.sum() > 400:
                rows[f] = (ok, d, np.where(ok, d["residual"] / d["model_flux"], np.nan))
        if len(rows) < MINIMUM_FRAMES:
            continue
        for f, (ok, d, y) in rows.items():
            nu = d["wavenumber_cm1"]
            full = np.where(ok, d["observed"] / d["model_flux"] * (1 + d["response_pattern"]) - 1, np.nan)
            k = np.flatnonzero(np.isfinite(full))
            if k.size > 400:
                k = clipped(k, full)
                a, ph, _ = sinusoid(nu[k], full[k], period)
                phases.setdefault(f"{band}{n}", {})[f] = dict(amplitude=a, phase_deg=float(np.degrees(ph)))
            mean = np.nanmean([rows[g][2] for g in rows if g != f], axis=0)
            scale = d["model_flux"] / d["uncertainty"]
            models = {"now": 0.0}
            for name, source in (("other frames' fringe", mean), ("own fringe", y)):
                k = np.flatnonzero(np.isfinite(source))
                if k.size >= 400:
                    models[name] = varying_fringe(nu, clipped(k, source), source, period)
            for name, m in models.items():
                r = y - m
                sm = boxcar(np.nan_to_num(r), PIXEL_SCALE, ok)
                e = gains.setdefault(f, {}).setdefault(name, [0.0, 0.0, 0])
                e[0] += float(np.sum((r * scale)[ok] ** 2))
                e[1] += float(np.nansum((sm * scale * np.sqrt(PIXEL_SCALE))[ok] ** 2))
                e[2] += int(ok.sum())
    return dict(
        period_cm1=period,
        own_fringe_by_order=phases,
        clear_pixel_z={f: dict(star=meta[f]["star"],
                               **{name: dict(per_pixel=float(np.sqrt(v[0] / v[2])),
                                             mean_of_pixels=float(np.sqrt(v[1] / v[2])))
                                  for name, v in g.items()})
                       for f, g in sorted(gains.items())})


def fringe_model(nu_fit, y_fit, period, nu):
    """The fitted sinusoid at one period, evaluated on another grid."""
    x = 2 * np.pi * nu_fit / period
    t = (nu_fit - nu_fit.mean()) / np.ptp(nu_fit)
    design = np.column_stack([np.cos(x), np.sin(x)] + [t ** k for k in range(4)])
    c, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
    return c[0] * np.cos(2 * np.pi * nu / period) + c[1] * np.sin(2 * np.pi * nu / period)


def fixes(frames, meta, period):
    keys = ("now", "fringe", "fine", "drift", "both")
    acc = {k: {key: [] for key in keys} for k in ("line-free", "line")}
    snr = []
    for n, per_frame in frames.items():
        per = {}
        for f, d in per_frame.items():
            ok = interior(d)
            if ok is None:
                continue
            c = per_cm1(d)
            y = d["residual"] / d["model_flux"]
            snr.append(float(np.median((d["model_flux"] / d["uncertainty"])[ok])))
            per[f] = dict(ok=ok, y=y, scale=d["model_flux"] / d["uncertainty"], nu=d["wavenumber_cm1"],
                          short=band_pass(y, ok, c, 0.7, 4.0), long=band_pass(y, ok, c, 4.0, 15.0),
                          clear=boxcar(d["transmission"], c, ok) > 0.99)
        fs = sorted(per)
        if len(fs) < MINIMUM_FRAMES:
            continue
        for f in fs:
            p = per[f]
            others = [g for g in fs if g != f]
            # the fringe: one sinusoid at the band's period, fitted to the other frames' mean
            mean = np.nanmean([np.where(per[g]["ok"], per[g]["y"], np.nan) for g in others], axis=0)
            k = np.flatnonzero(np.isfinite(mean))
            k = clipped(k, mean) if k.size > 600 else k
            fringe = fringe_model(p["nu"][k], mean[k], period, p["nu"]) if k.size > 600 else 0.0
            fine = np.where(p["clear"], np.nanmedian([per[g]["short"] for g in others], axis=0), 0.0)
            w = np.array([np.exp(-0.5 * ((meta[g]["mjd"] - meta[f]["mjd"]) * 24) ** 2) for g in others])
            stack = np.array([per[g]["long"] for g in others])
            good = np.isfinite(stack)
            drift = np.nansum(stack * w[:, None], 0) / np.maximum((good * w[:, None]).sum(0), 1e-9)
            fine, drift = np.nan_to_num(fine), np.nan_to_num(np.where(good.any(0), drift, 0.0))
            for kind, sel in (("line-free", p["ok"] & p["clear"]), ("line", p["ok"] & ~p["clear"])):
                if sel.sum() < 50:
                    continue
                for key, y in (("now", p["y"]), ("fringe", p["y"] - fringe), ("fine", p["y"] - fine),
                               ("drift", p["y"] - drift), ("both", p["y"] - fine - drift)):
                    acc[kind][key].append((np.sum((y * p["scale"])[sel] ** 2), sel.sum()))
    out = {kind: dict(pixels=int(sum(c for _, c in v["now"])),
                      **{key: float(np.sqrt(sum(s for s, _ in rows) / sum(c for _, c in rows)))
                         for key, rows in v.items()})
           for kind, v in acc.items() if v["now"]}
    out["median_pixel_snr"] = median(snr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--band", choices=("H", "K"), required=True)
    parser.add_argument("--run", type=Path, required=True,
                        help="a fit_igrins_standard.py run with --blaze and the response pattern")
    parser.add_argument("--blaze", type=Path, required=True)
    parser.add_argument("--cal-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True, help="one extracted frame of the night")
    parser.add_argument("--velocity-run", type=Path,
                        help="an H run of the same frames to take stellar velocities from (for K)")
    parser.add_argument("--night", action="append", default=[], metavar="NAME=RUN_DIR",
                        help="runs of other nights, and this night's own, for the cross-night test")
    parser.add_argument("--own", help="which --night is this night")
    parser.add_argument("--repo", type=Path, default=Path("."),
                        help="where the summaries' relative frame paths resolve")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tellurix import FlatBlaze
    from tellurix.flat import lamp_spectra

    on = sorted(args.cal_dir.glob(f"SDC{args.band}_*.flat_on.fits"))
    off = sorted(args.cal_dir.glob(f"SDC{args.band}_*.flat_off.fits"))
    if len(on) != 1 or len(off) != 1:
        raise SystemExit(f"expected one {args.band} flat_on and one flat_off in {args.cal_dir}")
    lamp, _, _ = lamp_spectra(on[0], off[0], args.spec)
    blaze = FlatBlaze.load(args.blaze)
    frames, meta = load_run(args.run, args.band, args.repo)
    if args.velocity_run:
        _, other = load_run(args.velocity_run, "H", args.repo)
        for f in meta:
            meta[f]["vstar"] = other[f]["vstar"]
    report = dict(
        band=args.band, run=str(args.run), blaze=str(args.blaze), flat_on=str(on[0]),
        frames={f: dict(m, vstar=round(m["vstar"], 2), airmass=round(m["airmass"], 3))
                for f, m in sorted(meta.items())},
        trim_samples=TRIM)
    report["fringe"] = fringe(frames, meta, args.band, blaze, lamp)
    report["scales"] = scales(frames, meta, args.band, lamp, blaze, report["fringe"]["period_cm1"])
    if args.night:
        runs = dict(item.split("=", 1) for item in args.night)
        report["nights"] = nights({k: Path(v) for k, v in runs.items()}, args.own, args.band, args.repo)
    report["fixes_without_refit"] = fixes(frames, meta, report["fringe"]["period_cm1"])
    report["pixel_scale"] = pixel_scale(frames, meta)
    report["fringe_per_frame"] = fringe_frames(frames, meta, args.band, report["fringe"]["period_cm1"])

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return None if not np.isfinite(x) else round(float(x), 6 if abs(x) > 1e4 else 5)
        if isinstance(x, np.integer):
            return int(x)
        return x

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean(report), indent=1) + "\n")
    f, s = report["fringe"], report["scales"]
    print(f"{args.band}: fringe {f['period_cm1']:.3f} cm-1, lamp {100 * f['lamp']['median_amplitude']:.2f}%, "
          f"stars {100 * f['star']['median_amplitude']:.2f}%; short-band r "
          f"{s['frame_pair_correlation']['short']['all']}; fixes {report['fixes_without_refit']} -> {args.output}")


if __name__ == "__main__":
    main()
