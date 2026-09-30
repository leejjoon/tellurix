# Carrying a night's standards to a frame without one

Numbers here come from `docs/igrins_transfer_*.json`, written by
`scripts/validate_igrins_transfer.py`; the `inject_` reports are the injection
test below. The per-order-frame rows behind them are
gitignored under `data/corrected/igrins/transfer/`.

## The question

A science target has no telluric standard of its own. Its telluric model has to
come from the night's A0V standards, carried to the target's airmass and time,
with as little as possible fitted on the target itself -- the target's own lines
are what a refit would absorb. This measures how well that transfer works on
the standards, where a full fit gives the right answer to compare against.

Each standard in turn is held out. The others form the calibration, per order:

- **dry columns and the LSF** -- the median of the other frames. The model's
  columns are vertical and the zenith angle supplies the slant path, so a
  well-mixed gas needs no airmass interpolation at all;
- **water** -- linear in time between the bracketing standards, held flat
  outside them. Per order, because orders disagree about the water column by
  10-20% within a frame but each order's offset from the frame repeats to 1-2%;
- **velocity** -- the median;
- **the instrument response** -- the pattern the run already gave this frame,
  built by `leave_one_out_patterns` from the other frames. A science frame gets
  the same pattern from all of them.

The held-out frame is then fitted with progressively more freedom:

| level | refitted on the held-out frame |
|---|---|
| L0 | the continuum only |
| L1 | + velocity, per order |
| L2 | + water, per order |
| L3 | + LSF, per order |
| S1 | one velocity shift for the frame -- the median over orders of L2's |
| S2 | S1 + one water scale for the frame, the same way |

The S levels are what a science frame can afford: two numbers constrained by
every order, instead of per-order freedom that a target's own lines can pull
on. The stellar velocity is free throughout; it is the star's, and a science
target's equivalent is its own model.

Two measures, both against the frame's own full fit and on its reliable pixels:
the residual as a ratio to the full fit's, and the **correction operator's**
difference from the full fit's -- `model_flux / stellar_only`, in continuum
units, over the pixel noise in the same units. The second is what a science
user inherits.

## Results

Five night-bands, 1,282 held-out order-frames. Median / 90th percentile of
the residual ratio, then the median correction difference in units of the
noise. All orders:

| night | standards | full fit | L0 | L1 | L2 | S1 | S2 |
|---|---|---|---|---|---|---|---|
| DCT 2018 H | 10 | 1.33 | 1.018 / 1.24 / 0.39 | 1.010 / 1.15 / 0.31 | 1.002 / 1.03 / 0.16 | 1.011 / 1.15 / 0.33 | 1.005 / 1.04 / 0.18 |
| DCT 2018 K | 10 | 1.31 | 1.066 / 1.35 / 0.60 | 1.025 / 1.15 / 0.40 | 1.007 / 1.04 / 0.21 | 1.029 / 1.15 / 0.43 | 1.010 / 1.05 / 0.27 |
| McDonald 2017 H | 8 | 1.78 | 1.047 / 1.41 / 0.78 | 1.026 / 1.38 / 0.66 | 1.003 / 1.06 / 0.24 | 1.030 / 1.39 / 0.68 | 1.010 / 1.06 / 0.36 |
| McDonald 2017 K | 8 | 2.38 | 1.085 / 1.38 / 1.34 | 1.041 / 1.28 / 0.96 | 1.009 / 1.05 / 0.43 | 1.043 / 1.28 / 0.98 | 1.016 / 1.06 / 0.56 |
| Gemini S. 2021 H | 14 | 1.76 | 1.018 / 1.25 / 0.47 | 1.005 / 1.13 / 0.25 | 1.001 / 1.00 / 0.09 | 1.006 / 1.13 / 0.26 | 1.003 / 1.01 / 0.16 |

Orders with median transmission below 0.9, where the correction matters:

| night | order-frames | full fit | L0 | L2 | S2 |
|---|---|---|---|---|---|
| DCT 2018 H | 52 | 2.04 | 1.053 / 1.52 / 1.18 | 1.002 / 1.04 / 0.30 | 1.011 / 1.05 / 0.41 |
| DCT 2018 K | 94 | 1.53 | 1.127 / 1.47 / 1.33 | 1.005 / 1.05 / 0.24 | 1.012 / 1.07 / 0.45 |
| McDonald 2017 H | 62 | 2.62 | 1.050 / 1.89 / 1.74 | 1.000 / 1.02 / 0.26 | 1.007 / 1.03 / 0.49 |
| McDonald 2017 K | 95 | 2.90 | 1.096 / 1.42 / 1.82 | 1.008 / 1.09 / 0.56 | 1.020 / 1.10 / 0.76 |
| Gemini S. 2021 H | 98 | 2.32 | 1.051 / 1.38 / 1.47 | 1.000 / 1.01 / 0.13 | 1.004 / 1.02 / 0.33 |

### Interpolating water is not enough

Pure transfer (L0) is close in the median and bad in the tail: in absorbing
orders the 90th percentile is 40-90% worse than the full fit and the correction
is off by 1.2-1.8 times the noise. The cause is water. Interpolated in time
between standards, the per-order water column is off from the full fit by a
robust rms of 0.04-0.08 in log, which is what a leave-one-out interpolation of
the frame medians showed before any of this ran. Water moves on timescales the
standards do not sample: on DCT 2018 frame 0045 sits 13% above the
interpolation between standards 42 and 78 minutes either side of it, in both
bands; on McDonald the last frame, past the last standard and so held flat,
needed 19% less in both bands; on Gemini South the first, before the first
standard, needed 10% less.

Velocity alone (L1, S1) recovers half or more of L0's excess; water is the
rest.

### One water scale per frame is enough

S2 recovers nearly all of what per-order water freedom (L2) does: within 1% of
the full fit's residual in the median (1.6% on the wet McDonald K), 1-10% at
the 90th percentile, and a correction within 0.3-0.5 of the noise in absorbing
orders. It works because the water error is a property of the frame, not the
order: after subtracting the frame's shift, what remains of the per-order
refit is a robust rms of **0.009-0.014** in log column, against 0.04-0.08
before.

The one place the per-order remainder shows is McDonald K -- the wettest night
in the band with the most water -- where S2 leaves 0.76 of the noise in
absorbing orders against L2's 0.56. That is still under the noise, and less
than half of what interpolation alone leaves.

Two cross-checks that the frame shift is the atmosphere and not a fitting
artefact:

- **H and K agree.** The two bands are different detectors and different
  lines, fitted independently, and their per-frame water shifts correlate at
  r = +0.951 on DCT 2018 and +0.969 on McDonald 2017, differing by 0.018 and
  0.021 rms against a spread of about 0.06. A science frame's water can
  therefore be measured in whichever band its star contaminates less, and
  applied to both.
- **Velocity is also per-frame**, less cleanly: H and K correlate at r = +0.82
  and +0.96 and differ by 0.13 and 0.11 km/s, which is the two detectors' own
  zero points. S1 is as good as L1 on every night, so one shift per band per
  frame is enough.

## A target with lines of its own

An A0V has no lines outside the hydrogen series, so nothing above competes with
the telluric lines for the water scale and the shift. A science target does.
`--inject` multiplies every held-out frame by a second star's normalized
spectrum -- here Payne Zero's Arcturus, K1.5 III, at +37 km/s, smoothed to the
order's calibrated LSF -- and leaves it out of the model, so its lines are pure
contamination. In three H orders sampled, 20-42% of pixels are then more than
5% deep, 6-40% in three K orders, and the 2.3 um CO bandheads reach 57%.

Correction difference from the full fit, over the noise, absorbing orders
(DCT 2018, 10 frames):

| | clean S2 | injected: L0 | S1 | L2 | S2 | S2c |
|---|---|---|---|---|---|---|
| H | 0.41 | 1.18 | 1.08 | 0.71 | 0.58 | **0.45** |
| K | 0.45 | 1.32 | 0.80 | 0.74 | 0.64 | **0.61** |

and the per-frame shifts against the clean run's:

| | water, S2 | water, S2c | velocity, S2 | velocity, S2c |
|---|---|---|---|---|
| H | -0.015 (rms 0.017) | **-0.006 (0.008)** | +0.010 (0.017) km/s | +0.027 (0.032) |
| K | +0.018 (0.019) | +0.019 (0.021) | -0.040 (0.050) | **+0.003 (0.015)** |

Three conclusions.

- **Per-order water is unsafe.** L2 goes from the best level to worse than S2:
  each order's water is pulled by whatever stellar lines that order has. A
  science frame must not free water per order.
- **One water scale per frame survives, with a floor of about 2%.** The stellar
  lines bias it by -1.5% in H and +1.8% in K -- same size, opposite sign, so a
  line-rich target makes the two bands disagree by about 3%, and that
  disagreement is a warning sign worth computing. Against the full fit's own
  water it is still 1.7-1.9% rms, where time interpolation alone is 4-8%. The
  bias does not come from weak-water orders: restricting the median to orders
  with median transmission below 0.9 or 0.8 leaves it at 1.3-1.9%.
- **Clipping helps H and not K.** S2c drops pixels more than 3 sigma below S2's
  model, widened by 2 pixels either side (5% of pixels in H, 4% in K), and
  measures the frame again. In H that takes the water bias from -1.5% to -0.6%
  and the correction to 0.45, back at the clean 0.41. In K it removes the
  velocity bias but leaves water at +1.9%. H's contaminating lines are
  discrete; K's include the CO bandhead blends and a weak-line forest that never
  crosses 3 sigma, and clipping cannot see either. That is the likely reason,
  not a tested one.

What would remove the K floor is a model of the target -- its synthetic
spectrum in the source slot, which turns a science frame into the A0V case --
or, for a target observed at several barycentric velocities, a stellar template
learned from the frames themselves. Neither is tried here.

### Still not tested

- **A night with few standards.** Every night here has 8-14; the calibration
  had 7-13 after holding one out. Fewer standards mean wider gaps in time,
  which is exactly where interpolation failed, so S2 should matter more, not
  less -- and the response pattern then has to be the master pattern.
- **Other target types and velocities.** One star at one velocity. A cooler or
  line-richer target, or one whose lines sit on the water lines, will do worse.

### Something this exposed in the full fit

The per-order stellar velocity is mostly not measuring the star. In the full
A0V fits it rails at its +-60 km/s bound in 42% of order-frames (113 of 270 on
DCT 2018 H, 157 of 378 on Gemini South), and the pattern belongs to the order,
not the star: on DCT 2018 H, orders 2 and 10 rail on the same side for all ten
frames of five stars, and one frame of k Tau reads +7.8 km/s from Br10 (H05),
+42.7 from Br12 (H11) and -60 from Br14 (H15), each at 1.5 sigma. H11 is the
exception that does track the star -- chi Cap +58.8 and +57.6, k Tau +42.7 and
+43.8, HD 53205 +10.0 and +7.3 -- and is where the per-star check in
`igrins_a0v.md` comes from. The Brackett lines are hundreds of km/s wide, the
usable pixels start 450-550 pixels into each order so a line near the blue end
is seen on one side only, and a degree-9 continuum absorbs what a shift would
otherwise explain. The right parameterization is one stellar velocity per frame
-- the same argument as for water. The transfer frees it per order too, so the
comparisons above are like for like.

## Running it

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_igrins_transfer.py \
    --run-dir data/corrected/igrins/ladder_a0v --output docs/igrins_transfer_dct2018_h.json
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_igrins_transfer.py \
    --run-dir data/corrected/igrins/ladder_a0v --inject data/stellar/arcturus_payne_zero_full.npz \
    --inject-velocity-kms 37 --clip-sigma 3 --output docs/igrins_transfer_inject_dct2018_h.json
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_igrins_transfer.py --resummarize \
    --run-dir data/corrected/igrins/ladder_a0v --output docs/igrins_transfer_dct2018_h.json
```

It needs a finished `fit_igrins_standard.py` run with the fixed-pattern pass,
since it reads the record for the calibration and the npz cache for each
frame's response pattern. 20-45 minutes a night per band on one GPU, most of
it the per-order levels; `--resummarize` rewrites the summaries from the rows
and fits nothing.
