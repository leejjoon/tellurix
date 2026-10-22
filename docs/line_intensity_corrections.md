# Per-line intensity corrections from the solar fits -- future work

Status: **proposed, not started** (2026-10-20); tested without our solar model the same day (Baker et al. 2020, below). Measured on CPU from existing runs; no
code in the forward model yet. Report: `docs/line_depth_ratios.json`, written by
`scripts/measure_line_depth_ratios.py` (command in its docstring, ~5 min on CPU).

## Where this came from

Reviewing Wallace 2011 region 1, window 10510 (`wallace2011.html`), the corrected spectrum
spikes either side of the H2O feature at 10511.5 cm-1. The model is too deep there: core
0.04 of continuum against 0.09 observed, flanks 0.05-0.07 too deep, the observed optical
depth 0.82 of the model's. The minima coincide (10511.514), so it is not a shift. The same
feature is too deep in the June 1983 pair too: 0.80 in `_2` (X 5.37), 0.81 in `_3` (X 3.09).
Those are other dates, seasons, air masses and profiles, so the cause is the line list. The
feature is a blend:

| cm-1 | band | S | E'' | gamma_air |
|---|---|---|---|---|
| 10511.503 | 300 | 2.5e-24 | 222 | 0.096 |
| 10511.541 | 300 | 1.16e-23 | 300 | 0.085 |
| 10511.545 | 121 | 3.1e-24 | 886 | 0.053 |

AER v3.9 and HITRAN agree on all three intensities. Which line is wrong cannot be told from
these spectra.

It is not systematic. Over 664 isolated H2O lines in region 1, the median observed/model
optical depth is 1.00 at every depth from tau 0.2 to 4, and 1.00 in every E'' bin. There is
no curve-of-growth trend, so the x1.00 width scale and the temperature profile are not the
cause.

## What the four fits show

The fits measured are:

- `wallace`: Wallace 2011 regions 1-2, the x1.00 runs, promoted 2026-10-20 to `data/corrected/solar/wallace2011/`;
- `j83_2` and `j83_3`: the June 1983 pair, x1.06;
- `niratl`: x1.11.

The feature list has 3149 entries: AER H2O and O2 lines clustered within 0.08 cm-1, with
strongest-line S at least 3e-25.

**The ratios repeat.** The correlation of (ratio - 1) between fits is 0.83-0.88 between
Wallace (Oct 1989) and the 1983 pair, 0.92 within the pair, and 0.82-0.85 between niratl and
the rest. The rms per fit is 4-6%.

**The same-day air-mass test says line list, not Sun.** For `_2` against `_3`, the
total-least-squares slope is 0.96. A line-list error gives 1. A solar-model error under a
telluric line gives 3.09/5.37 = 0.57, because its optical depth does not grow with air mass.

**The held-out test is predicting each fit's ratios from the mean of the other three:**

| held out | rms (ratio - 1) | features > 8% off, their rms |
|---|---|---|
| wallace | 5.3% -> 2.7% | 147: 14.3% -> 4.1% |
| j83_2 | 5.8% -> 2.6% | 212: 13.1% -> 4.9% |
| j83_3 | 5.9% -> 2.2% | 238: 13.5% -> 3.9% |
| niratl | 5.3% -> 2.7% | 119: 13.4% -> 4.8% |

From the 1983 pair alone, Wallace goes from 5.5% to 2.9%.

**Consistent outliers.** Of the 1573 features measured in three or more fits, 640 are off by
more than 3% at more than 3 sigma, 375 by more than 5%, and 93 by more than 10%. The largest
are about 30%: for example, 13941.44 is 1.31 +- 0.004, 12321.65 is 0.73 +- 0.01, and
10628.26 is 1.29.

**The temperature profile barely matters.** Wallace minus `_3` is +0.1% at E'' < 300,
-0.6% at 300-700, and -1.4% at 700-1500. One table can serve all the dates.

**Coverage.** 703 features are measured in no fit, and 459 in only one. A feature is not
measured when it sits under a solar line or is saturated in every fit. 10511.5 is one of
these: the solar model is 0.91 under it, so the solar-clean cut drops it everywhere.

## A shortcut that does not answer the question

I also tried a pixel-space trial (scratch only, not committed). It rescaled each Wallace
window's model optical depth near every feature by the held-out factor, shrunk x0.86, and
rescored with `score_telluric_correction.py --relative-transmission`. The medians improved:

- authors' pixels: region 1 0.61% -> 0.57%, region 2 0.60% -> 0.55%;
- region 2 deep pixels: 2.02% -> 1.48%.

But the tail got worse: region 1 window 12010 went from 0.49% to 1.54%, and 11980 from
1.87% to 3.43%. No measured feature in either window disagrees with its factor. So the
damage is the shortcut's own. It applies a core-measured factor to the wings. It also
ignores the instrument convolution. And the four fits have different width scales
(x1.00, x1.06, x1.11). Treat these numbers as no evidence either way. Only a refit with
scaled intensities tells whether flatness improves.

## Line positions: O2 too, and neither list fixes them

Found 2026-10-20 reviewing Wallace 2011 region 2 window 15910 (the O2 gamma band, 628 nm),
where the corrected spectrum swings positive-negative across each O2 line. The window's
median offset is -4 m/s, so the wavenumber scale and the fitted velocity are right. What
is left is per-line. The method is the S-weighted centroid of 1 - observed/solar minus
that of 1 - model/solar, over +-6 pixels around each model minimum.

| line (cm-1) | Wallace (X 1.4) | June 1983 `_2` (X 5.37) | June 1983 `_3` (X 3.09) |
|---|---|---|---|
| 15914.2 | +17 | +20 | +38 |
| 15916.2 | +11 | +15 | +38 |
| 15918.2 | +10 | +21 | +41 |
| 15925.7 | -32 | -28 | -47 |
| 15927.8 | -40 | -35 | -32 |

The offsets are observed minus AER v3.9, in m/s. `_3` runs about 20 m/s high throughout,
because its own velocity differs. The pattern repeats over two dates and three air masses,
so it belongs to the line data, and it is +-20 m/s (+-0.001 cm-1).

Two candidates were tested in scratch, refitting windows 15910 (gamma), 14500 (B), 13060
and 13120 (A) with the x1.00 settings:

- **HITRAN2024's O2 positions** (HITRANonline, isotopologue ids 36-38, 12900-16000 cm-1,
  fetched 2026-10-20, 195 matched lines substituted). They differ from AER's by 0.5 mK rms
  in the A band, 1.0 in the B band and 2.9 in the gamma band. In the gamma band they are
  13-93 m/s bluer, increasingly so toward the head. Pressure shifts and intensities are
  identical to AER's.
  - HITRAN flattens the 15914-15919 offsets (to +1 to +10 m/s), but band-head lines stay
    30 m/s off, and the scatter about the mean only drops from 20 to 17 m/s.
  - Flatness, AER -> HITRAN, at T < 0.97 (authors' pixels) and deep pixels (T < 0.7):

    | window | T < 0.97 | deep |
    |---|---|---|
    | 15910 | 1.74 -> 1.67% | 1.05 -> 1.28% |
    | 14500 | 0.60 -> 0.61% | 2.36 -> 2.58% |
    | 13060 | 0.95 -> 0.95% | 3.34 -> 3.33% |
    | 13120 | 1.93 -> 1.96% | 1.78 -> 1.82% |

  - No gain. It is not the H2O pressure-shift story (§4q), where HITRAN helped everywhere.
- **AER's first-order O2 line coupling** (`--line-coupling`; `lncpl_lines` carries 182 O2
  lines in 12900-16000 cm-1). Coupling would make a line's shape asymmetric, which leaves
  this kind of S-shape. It was applied at 15910 and 14500 and changed nothing to four
  decimals, per-line offsets included. The A band was the same.

So positions are a per-line error neither list resolves, like the intensities above. They
can be measured the same way: centroid offsets per feature across the fits, with each
fit's mean removed. They would be carried as a position column beside the strength
multiplier (step 1 below), with the same held-out test. A position error is invisible
to the optical-depth ratio, which uses core sums, so it needs its own measurement.

## Without our solar model: Baker et al. 2020's telluric spectra

Every fit above divides by Payne Zero, so a shared solar-model error could pass for a
line-list one. Baker, Blake & Reiners 2020 (ApJS 247, 24) published what their joint
fits of 141 Goettingen spectra left once the Sun and the iodine cell were divided out
(`tellurix_fts.baker`). That is the atmosphere alone, at another site (0.2 km), with
another instrument (the IAG FTS, R ~ 1e6), in another decade (2015-16).

- **The run** (`scripts/fit_baker_telluric.py`, 2026-10-20):
  - one spectrum per ERA5 hour, 53 of the 106 distinct spectra, each with its own profile
    (`data/profiles/baker2020/`);
  - the 146 windows of 9020-17600 cm-1 holding at least three features the solar fits
    measured;
  - a flat Sun, H2O and O2, otherwise the solar fits' physics at x1.00 widths;
  - 7733 windows fitted. The other 5 are spectrum 0139 at 9020-9140, too opaque to fit.
- **The measurement** (`scripts/analyse_baker_telluric.py` -> `docs/baker_telluric_ratios.json`):
  - the same per-feature ratio as the solar fits (`tellurix_fts.line_depth`);
  - their solar model's lines masked (within 0.1 cm-1 of `stel_mod` < 0.97), and since
    2026-10-21 their iodine template's (within 0.03 cm-1 of `iodine` < 0.98; below);
  - 1908 features measured in at least three spectra, a median 52. The iodine cut took
    54 of them, all above 15500 cm-1, where it removes 7-48% of the pixels; it moved the
    rest by 0.9% rms. `--keep-iodine` gives the analysis without it.

**The ratios are the line list's.**

- **Repeatable:** split into two halves of the spectra, the per-feature mean ratio
  correlates at r = 0.98. One spectrum scatters 1.8% about the mean, so a feature's
  mean is good to 0.3%.
- **Grows with air mass:** fitting ratio - 1 = a + b (1/X - mean) per feature over
  X 1.1-3.5, b / its error has rms 1.17 (1 if nothing in the ratio follows 1/X), and
  1.2% of features exceed 3 sigma. Error their solar model left under a line would
  not grow with air mass, so it would show up in b. Almost none does.
- **The same errors as the solar fits:** over the 1574 features both measure, r = 0.79.
  Of the 159 features more than 8% off in the solar fits' mean, 150 are off the same
  way here. The window-13690 lines:

  | cm-1 | Wallace | `_2` | `_3` | Baker (53 spectra) |
  |---|---|---|---|---|
  | 13701.89 | 1.157 | 1.217 | 1.216 | 1.132 +- 0.014 |
  | 13704.54 | 0.858 | 0.844 | 0.834 | 0.832 +- 0.016 |

- **The features the solar fits could not reach:** 79 features no solar fit measured are
  measured here, 29 of them more than 5% off. One is **10511.5, where this started**:
  0.789 +- 0.014 over 34 spectra, the 0.80-0.82 read off by eye in Wallace and the 1983 pair.

**As a source for the table, Baker alone does a little worse than the other solar fits.**
This is the held-out test above, rms of (ratio - 1) predicted three ways:

| held out | before | by the other solar fits | by Baker | by both |
|---|---|---|---|---|
| wallace | 4.9% | 2.6% | 3.0% | 2.3% |
| j83_2 | 5.6% | 2.3% | 3.5% | 2.2% |
| j83_3 | 5.7% | 2.0% | 3.9% | 2.4% |
| niratl | 5.0% | 2.6% | 3.6% | 2.6% |

The 1-1.5% Baker leaves beyond the solar fits is not noise (0.3%). It is something the
Kitt Peak fits share and Goettingen does not. Two candidates:
- the width scales: these fits use AER's widths, the solar fits x1.06-1.11, and a width
  error reads as a strength error in a saturated core;
- the temperature profile (E'' terms, -1.4% at 700-1500 between two Kitt Peak dates).

Partly settled (below): the temperature profile shows as an E'' slope that differs from
fit to fit. The width scale is not it: Baker refitted at x1.06 predicts the solar fits
worse (3.9-4.1%), not better. Baker helps most where the other fits are fewest: Wallace, predicted by
the 1983 pair and niratl only, goes from 2.6% to 2.3%.

**Positions repeat too, but are noisier than the strengths.**
- One spectrum's offset scatters 35 m/s per feature, against 0.2-0.3% for its ratio.
- The halves correlate at r = 0.95 over the 320 features measured in every spectrum
  (0.89 before the iodine cut).
- The median |offset| is 37 m/s.
- The offset is a joint fit of a shift and an optical-depth scale. A centroid over a
  window fixed on the model's centre read offsets 3-5x too small.
- The measurement uses H2O features only. The O2 gamma-band lines of the section above are
  below the feature list's S >= 3e-25, and need their own list.

### What else a residual common to every spectrum could be

A residual the same at every air mass is not by that alone the line list's. The
candidates were tested on 2026-10-21 (`scripts/diagnose_baker_residuals.py` ->
`docs/baker_residual_causes.json`, numpy only, half a minute):

- **Their iodine template, divided out slightly too deep. Real, small, and now cut.**
  Off the telluric and solar lines, above 15500 cm-1, the median residual over a
  window's spectra follows the template's depth: r = 0.15-0.47 in 11 of the 12 windows of
  16670-17030, and +0.07%, +0.30% of the continuum at a median iodine depth of 15% and
  34%. So the template is 0.5-1% of its depth too deep. It is the same in every spectrum,
  so under a telluric line it would pass for a line-list error. The analysis now cuts
  those pixels (above). Below 15500 there is no iodine to speak of (r = 0.02). The
  template also holds a spike at 15796 cm-1, to -0.12, which is not iodine. Near the
  632.8 nm HeNe line, it is presumably the metrology laser.
- **HDO: not testable here.** No feature in 9020-17600 cm-1 is HDO-dominated. The
  16 H2-18O features read -0.6% (Wallace -1.8%, the 1983 pair -1.7% and -0.9%), too few to
  be an abundance.
- **The temperature profile, through E''. Each fit's own.** Over features at least 80%
  H2-16O, the joint fit (ratio - 1) = a + b log10 S + c E''/1000 cm-1:

  | fit | widths | per decade of S | per 1000 cm-1 of E'' |
  |---|---|---|---|
  | Baker | x1.00 | -2.42 +- 0.21% | +1.03 +- 0.31% |
  | Wallace | x1.00 | -1.33 +- 0.26% | -0.55 +- 0.37% |
  | `j83_2` | x1.06 | +0.32 +- 0.36% | +3.52 +- 0.36% |
  | `j83_3` | x1.06 | +1.12 +- 0.30% | +2.46 +- 0.34% |
  | niratl | x1.11 | +0.64 +- 0.27% | +3.91 +- 0.37% |

  The E'' slope differs from fit to fit, so it is not the line list. It is each
  date's temperature profile: Baker's +1% is about 0.5 K.
- **The strength slope seemed to follow the width scale.** At AER's widths the strong
  lines read weak, -2.4% and -1.3% a decade. At x1.06-1.11 the slope is +0.3% to +1.1%.
  But refitting Baker at x1.06 moves its slope only to -2.0% (below), so across fits it
  is the dates, not the widths.
- **The line shape: our Voigt, not AER's numbers.** The residual of 971 profiles of 482
  isolated H2-16O lines, each over its model depth, with the median taken over lines and
  spectra and the even part kept. The Voigt rows are what a Voigt at the database width,
  with its strength refitted, leaves on a line of the other shape. Single layer, 14000 cm-1,
  280 K, so they are an order of magnitude:

  | at offset (cm-1) | 0 | 0.03 | 0.06 | 0.10 | 0.20 | 0.30 |
  |---|---|---|---|---|---|---|
  | observed, depth 0.5-0.9 (460) | -0.70% | -0.07 | +0.65 | +0.65 | +0.06 | +0.03 |
  | observed, depth 0.2-0.5 (453) | -1.38 | -0.68 | +0.24 | +0.24 | -0.11 | -0.06 |
  | speed-dependent, Gamma2/Gamma0 = 0.10, depth 0.7 | -1.12 | -0.38 | +0.43 | +0.35 | +0.02 | 0.00 |
  | speed-dependent, Gamma2/Gamma0 = 0.10, depth 0.35 | -1.43 | -0.25 | +0.70 | +0.46 | +0.07 | +0.02 |
  | true widths x0.94, depth 0.7 | -2.43 | -1.70 | -0.08 | +1.15 | +0.89 | +0.49 |

  - The observed shape has a deeper core and shallower shoulders, and nothing beyond
    0.2 cm-1. That is speed-dependent narrowing at Gamma2/Gamma0 ~ 0.1, a usual value for
    H2O in air.
  - A width error leaves a tail in the far wings that is not there, so the shape is not a
    width error. A width error in the other direction has the opposite sign throughout.
  - The pattern is 0.06-0.1 cm-1 wide, against the instrument's 0.0145 cm-1 sinc, so it is
    not the instrument profile.
  - It is about 1% of a line's depth in every spectrum. A line-strength table cannot fix
    it, because it is a shape. In a core ratio it reads as a strength error of the same
    sign for every line.
- **Tested against an SDV with real line parameters (2026-10-21): Gamma2/Gamma0 = 0.09-0.13
  accounts for all of it.**
  - The method: for each of 698 isolated-line profiles, one Voigt is fitted to the
    model's own optical depth on the fit grid (`speed_dependence_terms`). The change an
    SDV at the same Gamma0 would make is put through the instrument kernel and regressed
    against the residual, with the line's strength, and then also its width, free.
    One amplitude a over all lines; a = 1 means it accounts for all of the residual.
  - At Gamma2/Gamma0 = 0.13, HITRAN 2024's median for H2O-air (below):
    a = 1.03 +- 0.05 with strength and width free, and 0.72 +- 0.03 with strength alone.
    So the data want 0.09-0.13.
  - With the strength refitted, the stacked shape follows the prediction, as % of depth:

    | depth | | 0 | 0.03 | 0.06 | 0.10 | 0.15 | 0.20 | 0.30 cm-1 |
    |---|---|---|---|---|---|---|---|---|
    | 0.2-0.5 (383) | observed | -1.17 | -0.31 | +0.62 | +0.48 | +0.02 | -0.08 | 0.00 |
    | | SDV 0.13 | -1.26 | -0.48 | +0.60 | +0.66 | +0.22 | -0.01 | -0.17 |
    | 0.5-0.9 (260) | observed | -1.06 | -0.52 | +0.44 | +0.57 | +0.12 | -0.03 | -0.06 |
    | | SDV 0.13 | -1.02 | -0.50 | +0.31 | +0.48 | +0.21 | +0.03 | -0.10 |

  - On the x1.06 fits the amplitude falls to 0.54 with the width free and rises to 1.34
    without it. A width 6% off is beyond the first-order width term, which then trades
    against the SDV term. The estimate holds only where the widths are right, which is
    x1.00 here.
  - **HITRAN 2024 gives no Gamma2 here.** Fetched from HITRANonline on 2026-10-21, it gives
    H2O Gamma0 and Gamma2 (`gamma_HT_0_air_296`, `gamma_HT_2_air_296`, with no Dicke term)
    for 2361 H2-16O lines, all between 1850 and 6575 cm-1. There Gamma2/Gamma0 has a median
    of 0.13 and a 10-90% range of 0.105-0.21. There are none from 6575 to 17600 cm-1,
    so the per-line ratio cannot be checked in these spectra. A model here would assume one
    ratio or fit it.
  - **IGRINS cannot test it.** HITRAN's lines in 4000-6700 cm-1 strong enough to see
    (41 at S >= 3e-24) include no isolated ones, and none fall in the 1.8-2 um band.
    At a fixed 0.13, 142 isolated A0V-fit profiles (McDonald 2017, DCT 2018) give
    a = +1.85 +- 0.46 with strength free and -1.73 +- 0.94 with width too, and the
    nights disagree. At R ~ 45,000 the instrument smooths the pattern into what the
    strength and width terms absorb. (A scratch analysis: the IGRINS npz keep the
    unconvolved `transmission`, so the one-layer fit used that.)
- **A hypothesis joining the last two, refuted at Goettingen.** A database width fitted to a lab line
  that is really speed-dependent comes out below the true Gamma0. A Voigt at that width
  has wings that are too weak. They want x1.06-1.11, the strength slope, and niratl's
  width excess. Its core is too shallow at the same time. If so, a uniform width scale
  trades one error for the other, and only a speed-dependent profile fixes both.
- **Tested, and refuted for Goettingen: the x1.06 refit (2026-10-21).** All 53 spectra,
  the same 146 windows and settings, with H2O and O2 widths x1.06
  (`fit_baker_telluric.py --pressure-width-scale 1.06`, `data/corrected/baker2020_w106/`;
  `docs/baker_telluric_ratios_w106.json`, `docs/baker_residual_causes_w106.json`):

  | | x1.00 | x1.06 |
  |---|---|---|
  | fit rms, median over windows | 0.664% | 0.722% (worse in 89% of the 7733 pairs) |
  | extra Gaussian LSF at its floor | 29% | 79% |
  | strength slope, per decade of S | -2.42 +- 0.21% | -2.00 +- 0.24% |
  | E'' slope, per 1000 cm-1 | +1.03 +- 0.31% | -0.13 +- 0.34% |
  | Baker against the solar fits, r | 0.79 | 0.76 |
  | Wallace held out, predicted by Baker | 3.0% | 3.9% |
  | isolated lines, depth 0.5-0.9: core / 0.1 / 0.3 cm-1 | -0.70 / +0.65 / +0.03% | -2.51 / -0.03 / +0.28% |

  Wider lines fit Baker's spectra worse almost everywhere. The fitter pushes the extra LSF
  to its floor to get narrower lines back. The isolated-line residual takes on the
  x1.06 template's far-wing tail, with model wings too strong at 0.2-0.3 cm-1, which
  x1.00 does not have. The strength slope hardly moves. So at Goettingen AER's widths
  are right in the wings, and the strength slope is not a width error there.
  **The x1.06-1.11 the Kitt Peak fits want is Kitt Peak's**, not the database's. That also makes the
  hypothesis above wrong for the wings: what stays at x1.00 is the core-and-shoulder
  pattern of speed dependence alone. The strength slope of -2.4% per decade at x1.00
  is still open: a line-list trend, or the core pattern read as strength, since it
  grows with depth.
- **The same stack on the Kitt Peak fits (2026-10-21): it cannot tell the instrument from the
  lines.** Each run is one spectrum, so the solar cut is looser (no solar line deeper than 2%
  within 0.25 cm-1; `--stack`, same report). Isolated lines of depth 0.2-0.9, the even part
  over model depth, against the fitted instrument FWHM (sinc and free Gaussian together):

  | fit | widths | lines | instrument FWHM, cm-1 | 0 | 0.06 | 0.10 | 0.20 | 0.30 cm-1 | +- |
  |---|---|---|---|---|---|---|---|---|---|
  | Baker | x1.00 | 913 | 0.018 | -0.95% | +0.53 | +0.49 | 0.00 | -0.01 | 0.11 |
  | Baker | x1.06 | 917 | 0.012 | -3.10 | -1.01 | 0.00 | +0.30 | +0.27 | 0.13 |
  | Wallace | x1.00 | 75 | 0.028 | -0.73 | +0.14 | +0.14 | -0.19 | -0.01 | 0.31 |
  | 1983 pair | x1.00 | 153 | 0.051 | -1.11 | +0.61 | +0.55 | -0.27 | -0.37 | 0.28 |
  | 1983 pair | x1.06 | 164 | 0.035 | -1.44 | -0.31 | +0.13 | -0.11 | -0.19 | 0.19 |
  | niratl | x1.00 | 63 | 0.046 | -0.60 | +1.75 | +0.82 | -0.34 | -0.10 | 0.33 |
  | niratl | x1.11 | 68 | 0.015 | -0.68 | -0.32 | -0.05 | +0.18 | +0.13 | 0.27 |

  - At x1.00 the Kitt Peak lines leave Baker's core-and-shoulder pattern, with the
    shoulders larger in niratl, and perhaps a far-wing deficit of 0.3%, at one sigma.
  - What separates the sites is the free Gaussian. At AER's widths the Kitt Peak fits
    broaden their instrument to 0.046-0.051 cm-1, 1.5-3 times its sinc, and Baker's
    stays near its own. At x1.06-1.11 the Kitt Peak instrument falls back to the sinc.
    The excess is real broadening, but the shapes fit both a wider instrument and wider
    lines within 0.3%. These stacks cannot say which.
  - The evidence that it is the lines is older. A Lorentzian instrument kernel lost to the
    width scale in every test, partly because it broadens the solar lines too. The
    interferogram ratio scales with each species' own width
    (`docs/solar_fit_plan.md` §4r). Together with Goettingen at x1.00, that points to
    something in the Kitt Peak fits' atmosphere, and the profiles' pressure or
    temperature are where to look. That is not tested.

**What it changes in the plan:** the table can be built from Baker and the solar fits
together. Baker adds coverage under solar lines and many air masses, and is held out for
the solar fits. Before mixing them, find the 1-1.5% difference above. The width scale is
ruled out, so the Kitt Peak line profile comes next.

### Vetting the corrections: strength, artefact, or shape (2026-10-22)

A ratio is a line-list correction only if it is a *strength* error: tau times one
factor, in every spectrum. Baker et al.'s reduction can leave a *fixed* error instead
-- a solar-model or iodine residue, the same fraction of the continuum at any water
column. `scripts/vet_baker_line_corrections.py` (49 s on CPU) puts every analysed feature
through three tests and gives one verdict: the summary in `docs/baker_line_corrections.json`,
a row per feature in `docs/baker_line_corrections.csv`.

**One window first, 14300-14330 cm-1.** Over the 53 spectra the residual variance is 30%
common to all of them, ~50% pixel noise and ~20% that varies between spectra. The
quoted uncertainty is low by 2.1x: the pixel-to-pixel scatter of clear pixels is
1.27x it, and neighbouring pixels correlate at 0.63. Most of the between-spectrum part
sits on Baker's solar lines and tracks each spectrum's fitted `lsf_sigma_kms` (r 0.95),
which absorbs what their solar division left. AER v3.9 and HITRAN2024 agree on every
line there (S and widths identical, positions within 17 m/s), so neither list is the
fix. Per-line strength, shift and width corrections learned on half the spectra remove
62-71% of the common residual in the other half -- but that test cannot tell a
strength error from a fixed one, since both repeat.

**Which test sees what depends on depth:**

- *Air mass* (weak lines): a fixed eps is -eps/(T tau) of the ratio and falls as 1/X; a
  strength error does not. The test is blind in a saturated core, whose T tau moves only
  x1.2-1.4 between the low and high third of the spectra.
- *Profile* (deep lines): jointly over the spectra, in optical depth, rho (strength),
  eps (as -eps/T), a width and a shift. Across a deep line tau runs from ~3 to 0.03 and
  1/T is far from linear in it; in a weak line -eps/T is a constant plus the strength
  shape, and the profile is blind. Injection at Baker's real noise (2.1x quoted,
  correlated over two pixels) gives unbiased rho and eps, pulls 0.8-1.1 once the errors
  carry the correlation (x1.59), 0-3% false detections, and finds an eps of 0.5% of the
  continuum in 76% of lines with peak tau > 1.5, 28% at 0.7-1.5 and 2% below.
- *Another instrument*: the Kitt Peak solar fits. A Baker artefact is not in them.

At 14304.29, the window's worst line, the profile separates them: rho 0.960 +- 0.002
with eps 0.14 +- 0.09%, a strength error -- as Baker's ratio (0.961) and Kitt Peak's
(0.968-0.987) say. 14301.45 and 14310.79, which a regression in transmission had
called fixed, are strength errors by all three tests.

**Over all 1908 features:**

| verdict | features |
|---|---|
| vetted (Kitt Peak agrees, no air-mass trend) | 1576 |
| disagrees with Kitt Peak | 232 |
| Baker only (no solar fit measures it) | 78 |
| fixed part (air-mass trend) | 22 |

Fixed artefacts are rare: 22 features. The vetted corrections have an rms of 4.9%; 354
are more than 5% from 1 and 80 more than 10%.

The profile flags 350 features where one strength and one width do not describe the
profile. They are *not* Baker's: they agree with Kitt Peak as well as the rest do
(Baker - Kitt Peak rms 0.031 for both, 14% against 11% disagreeing) and their eps takes
either sign. They are shape errors both instruments see, flagged `shape_misfit`; a
strength factor corrects their core ratio but not their wings. 1302 vetted features
carry no such flag. Deep lines want to be 1.1% narrower at the median (rms 13% from
line to line).

The profile is a detector, not the correction. With eps and the width free, its rho
correlates with the Kitt Peak ratios only 0.07-0.35 by depth, where Baker's core ratio
does 0.69-0.86: the extra freedom takes real signal. The correction stays the mean of
Baker's and the solar fits' ratios.

## Plan

1. **A strength-multiplier table in the line database. Done (2026-10-22).**
   `tellurix.LineStrengthTable` (`tellurix/line_strength.py`), applied by
   `AERLineDatabase(line_strength_table=...)`, which multiplies `line_strength_ref_original`
   and `logsij0` at read time for the lines it lists, keyed by molecule, isotopologue,
   wavenumber and lower-state energy as AER prints them; the AER index is untouched.
   - Built by `scripts/build_line_strength_table.py` into `data/line_strength/`
     (gitignored): the mean ratio over the solar fits named in `--fits`, with `--baker`
     also Baker's vetted ratio; at least two measurements within 0.5 of 1, else AER's
     strength. A feature's factor goes to every member line of its cluster; the
     clustering now lives in `tellurix_fts.line_depth` (`cluster_features`) and the
     builder refuses to run if it no longer reproduces the catalogue. From three fits
     (`j83_2,j83_3,niratl`): 1806 of 3149 features, 3535 lines, factor rms 5.7%. From all
     four with Baker: 2212 features, 4287 lines, 5.6%.
   - Off by default. `WindowSettings.line_strength_table`, `--line-strength-table` on
     `fit_baker_telluric.py`, `fit_fts_window.py` and `fit_fts_batch.py`. A record states
     the table's name, SHA-256 and what it was built from (`line_strength_provenance`, no key
     at all without one); `rebuild_fts_window.py`, `export_transmission_hdf5.py` and
     `export_solar_review.py` find it by name in `data/line_strength/` and refuse one whose
     hash differs (`recorded_line_strength_table`).
   - Covers 8800-17600 cm-1 only, where the features were measured, so the IGRINS and
     Arcturus pipelines do not take it. The LBLRTM comparisons and the synthetic checks
     stay without it.
   - First use, Baker window 14300 with factors from Kitt Peak alone: every one of the 53
     spectra fits better, rms -10% at the median, the residual common to all of them
     0.31% to 0.23%.
2. **The honest test: build from three fits, refit the fourth.** Build the table from
   `j83_2`, `j83_3` and `niratl`. Refit Wallace regions 1-2 with it, with the same settings
   as the x1.00 run (the command is in the `fit_fts_batch.py` docstring;
   `--scan-profile data/profiles/kitt_peak_19830626_file3.csv --scan-fwhm-cm1 0.04167`).
   Then score with the authors' pixels and with all pixels. That is one GPU run on two
   GPUs, the same cost as the x1.00 run. Re-check the width scale in the same pass (x1.00
   against x1.05 with the table): a per-line strength and a global width both move
   strong-line depths, and the width preferred today may have been compensating.
   - **Success:** the median flatness drops and the tail does not grow. Compare per window
     against the x1.00 run, and against the authors' 0.71%.
   - **Failure modes to look for:** the factors carry solar-model error (check features
     whose factor differs most between Wallace, a flux atlas, and the 1983 intensity
     spectra); or the gain shows only in the measured features while unmeasured blends
     get worse.
3. **Iterate.** Re-measure from the refitted runs, rebuild the table from all four, and
   repeat once or twice. The ratios should converge to 1 with the scatter at the noise.
   Then refit every product with the final table. Settle that table's provenance before it
   reaches the sibling project, because the factors are derived from the same spectra they
   correct. The held-out result in step 2 is the evidence that this does not overfit.
4. **What this cannot reach.** Features under solar lines or saturated in every fit,
   10511.5 among them, have no measured factor. They need the factors as free parameters
   in the joint solar fit ("Fit the solar spectra jointly" in `solar_fit_plan.md`, Also
   open). That fit would also separate the Payne Zero errors that the clean-pixel cut only
   avoids.

## Limits and risks

- Coverage is 8800-17600 cm-1, where these fits are. IGRINS (H, K) and Arcturus get nothing
  from this table.
- The ratio is measured in line cores. A wrong width looks like a wrong strength in a
  saturated core, which is why step 2 re-checks the width scale.
- Payne Zero errors shared by every fit would pass for line-list error. The air-mass test
  bounds this for the 1983 pair. Wallace sees the Sun as a disc-integrated flux rather than
  an intensity, which gives a partial check.
- The 1983 pair's B-band O2 is about 5% deep for reasons inside those files (`solar_fit_plan.md`,
  "The O2 excess"). Exclude O2 B-band features of `j83_*` from the table, or the excess
  becomes a line-list correction.
