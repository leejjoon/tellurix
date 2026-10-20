# Per-line intensity corrections from the solar fits -- future work

Status: **proposed, not started** (2026-10-20). Measured on CPU from existing runs; no
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

## Plan

1. **A strength-multiplier table in the line database.** Model it on the H2O pressure-shift
   table (`tellurix.hitran`, `AERLineDatabase(pressure_shift_table=...)`,
   `DataPaths.pressure_shift_table`):
   - Key lines by molecule, isotopologue and wavenumber to 1e-6 cm-1. Multiply
     `logsij0` at read time. The AER index stays a superset, so it is untouched.
   - Off by default. Every fit setting and driver takes it (`--line-strength-table`), and
     every record states it with the table's SHA-256. A reader that rebuilds a record takes
     `.get(..., None)`.
   - The LBLRTM comparisons and the synthetic checks stay without it.
   - A blend's factor goes to every line in its cluster. Splitting a blend is not possible
     from these data.
   - Build the table from `docs/line_depth_ratios.json`: the mean ratio over the fits that
     measure a feature. Require at least two fits and |ratio - 1| < 0.5, and leave 1 elsewhere.
   - About a day of code and tests. No GPU.
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
