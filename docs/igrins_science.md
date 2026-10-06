# Correcting IGRINS science frames with a night's calibration

Numbers here come from `docs/igrins_science_dct2018.json`, written by
`scripts/summarize_igrins_science.py` from `fit_igrins_science.py` runs. Why the
method is what it is -- which parameters carry over and which must be refitted
-- is measured in `docs/igrins_transfer.md`.

## The method

A science target has no telluric standard of its own. Its telluric model comes
from the night's A0V standards, per physical echelle order
(`tellurix.NightCalibration`, built by `scripts/build_igrins_calibration.py` from
a `fit_igrins_standard.py` run):

- dry columns, LSF and velocity zero point: the median over the standards;
- water: linear in time between standards, flat outside them;
- the instrument response pattern: the median fractional residual over all the
  standards, smoothed by 51 pixels.

On the science frame, `scripts/fit_igrins_science.py` then fits, per order, the
continuum and a trial velocity and water shift, and takes the **median over
orders** as the frame's one velocity shift and one water scale; each order is
refitted with those, freeing only its continuum. Per-order water is never
kept: with a target's own lines in the data it chases them
(`igrins_transfer.md`, injection test). `--clip-sigma 3` measures the shifts a
second time without pixels far below the model. The products divide the data by
the *convolved* effective transmission, `model_flux / stellar_only`.

When both bands of an exposure are fitted, their water shifts are compared.
They measure the same atmosphere with different detectors and lines, so a
disagreement beyond 0.02 in log column -- 3-4 times the 0.005-0.006 by which
the bands agree on held-out standards (`igrins_transfer.md`), and under the
0.033 an injected K giant produced -- says one band's water scale is absorbing
something that is not water. The report
recomputes the comparison from each band's own summary, so bands fitted in
separate runs are still compared.

## DCT 2018-12-20

The calibration is the night's ten A0V standards (frames 33-108). It was tested
on frames it never saw.

**Four more standards** of the same night, dropped from the calibration run
only for their S/N (130-200), fitted exactly as a science frame would be but
with the A0V model as their source:

| frame | star | airmass | time | residual H / K | water shift H / K | H - K |
|---|---|---|---|---|---|---|
| 55 | HD 31069 | 1.35 | between standards | 1.04 / 1.11 | -0.024 / -0.022 | -0.002 |
| 72 | HD 31069 | 1.10 | between standards | 1.02 / 1.01 | +0.033 / +0.025 | +0.009 |
| 114 | k Tau | 1.32 | 0.5 h after the last | 1.02 / 1.02 | +0.017 / +0.018 | -0.000 |
| 122 | k Tau | 1.41 | 0.8 h after the last | 0.89 / 0.89 | -0.026 / -0.029 | +0.003 |

Residuals of 0.9-1.1 sigma with two parameters fitted per frame, and the two
bands agreeing on the frame's water to 0.009 or better.

**Six science targets**, T Tauri stars and M dwarfs, flat source, clipped:

| frame | target | airmass | time | residual H / K | water shift H / K | H - K |
|---|---|---|---|---|---|---|
| 37 | Hubble 4 | 1.83 | between standards | 4.71 / 4.36 | -0.016 / -0.014 | -0.001 |
| 53 | V1075 Tau | 1.51 | between standards | 2.81 / 2.18 | +0.007 / +0.012 | -0.005 |
| 59 | LkCa 15 | 1.32 | between standards | 3.51 / 2.31 | -0.002 / +0.008 | -0.010 |
| 74 | UY Aur | 1.09 | between standards | 3.04 / 1.90 | +0.041 / +0.044 | -0.004 |
| 92 | GJ 281 | 1.43 | between standards | 8.06 / 6.30 | -0.002 / +0.020 | -0.022 |
| 142 | YY Gem | 1.16 | 2.5 h after the last | 2.07 / 1.82 | +0.153 / +0.138 | +0.015 |

The residuals are the targets' own lines, which a flat source leaves in; they
are not a measure of the correction. What measures it is the band agreement:
median 0.007, and one target flagged -- GJ 281, an M dwarf, at 0.022, just past
the threshold and at the size the injection test gives for a line-rich star.
Its K band wants 2% more water than its H band. That is where an M dwarf's own
water would put it, but it is not shown to be the cause; what it says for use
is that GJ 281's water, and so its deepest corrected lines, are good to about
2%, not 0.5%. Clipping removes 1-4% of pixels and
moves the water shift by 0.018 at most and the velocity by 0.063 km/s.

**YY Gem is the case for the frame's own water scale.** Observed 2.5 hours
after the last standard, its water was held at that standard's value; the frame
wanted 15-17% more, and both bands say so independently (+0.153 and +0.138).

## The flag caught a header defect first

In the first run GJ 281 *was* flagged, the bands disagreeing by 0.093. The cause
was not its lines: its K file's end-of-exposure cards describe another frame
(`DATE-END` earlier than `DATE-OBS`), so the averaged zenith distance put the K
slant path 5.8% short of H's for the same exposure, and the K fit absorbed that
into water. The reader now refuses end cards the sidereal rate could not have
produced (`zenith_angle_deg`); 9 of 111 archive files, all K, are affected,
including calibrating standard 94 of this night, whose columns moved by +0.038
-- exactly ln(1.365/1.314) -- with its residual unchanged at 1.33. With the fix
GJ 281's bands differ by 0.022, a quarter as much.

A residual cannot see this: the fit puts a slant-path error into the columns
and fits as well as before. A second band measuring the same water can. The
same fix is why the threshold is 0.02 and not the 0.04 first chosen: the clean
agreement it was set from (0.018-0.021) was itself mostly this defect, and is
0.005-0.006 without it.

## A night with three standards

46% of archive nights have fewer than five standards, too few to measure the
night's own response pattern. The response belongs to the spectrograph, so the
median of other nights' patterns stands in (`tellurix.MasterPattern`,
`scripts/build_igrins_master_pattern.py`). Built from DCT 2016, McDonald 2017 and
Gemini South 2021 only, it matches DCT 2018's own pattern at median r = +0.940
in H and +0.914 in K and captures 88% and 84% of its variance.

The test thins DCT 2018 to three standards spread over the night -- 39 (HR 1558,
early), 84 (HD 53205) and 108 (k Tau, late) -- fits them with that master in
place of the leave-one-out pattern (`fit_igrins_standard.py --response-pattern`),
builds the calibration from the three (`build_igrins_calibration.py
--master-pattern --minimum-frames 3`) and corrects the other eleven standards as
science frames with the A0V model as their source. Against each frame's
full-night reference -- its own full fit for the seven calibrating standards,
the full-night science fit for the four extras:

| | median residual ratio | worst | H - K water, median / worst |
|---|---:|---:|---:|
| H | 1.049 | 1.099 | 0.005 / 0.012 |
| K | 1.066 | 1.124 | |

The three standards themselves fit at 1.26-1.39 sigma in H and 1.25-1.41 in K
with the borrowed pattern, against 1.19-1.35 and 1.21-1.32 with the night's own.
So three standards and a master pattern cost about 5-7% of residual against a
full night, while the bands still agree on every frame's water to 0.005 --
the water interpolation, not the pattern, is what the frame's own water scale
has to rescue, and three standards spread over the night bracket it well
enough. Which part of the 5-7% is the pattern and which the fewer standards is
not separated here. Numbers: `docs/igrins_science_dct2018_sparse.json`, whose
residuals divide by `docs/igrins_science_dct2018.json` and the full-fit records.

`data/calibration/igrins_master_{h,k}.h5` are built from all four nights, for
use; the `_without_dct2018` masters exist only for this test.

## A model of the target

The injection test said a target's own model in the source slot should remove
the line-rich floor. Two real targets were refitted with Payne Zero spectra at
approximate literature labels (`scripts/generate_payne_zero_star.py`; both
atmospheres unconverged, the metadata beside each npz says how the labels were
chosen): GJ 281 at 4000 K, log g 4.7 -- Payne Zero's floor, the star itself
being slightly cooler -- and LkCa 15 at 4370 K, log g 3.9, vsini 13 km/s.

| | residual H / K, flat | with model | H - K water, flat | with model |
|---|---|---|---:|---:|
| GJ 281 | 8.06 / 6.30 | 5.48 / 3.14 | -0.022 | +0.019 |
| LkCa 15 | 3.51 / 2.31 | 2.94 / 2.21 | -0.010 | +0.012 |

The models land where they should -- the fitted stellar velocity agrees across
orders to 0.2-0.8 km/s and between the bands (GJ 281 6.6 and 6.7, LkCa 15 26.6
and 26.9 km/s) -- and take out a large part of the stellar structure, half of
GJ 281's K residual. But the bands' water disagreement **changes sign and keeps
its size**. So at the 1-2% level the target's water scale is limited by how well
its spectrum is known, whether that spectrum is left in the data or modelled
approximately; an approximate model moves the bias rather than removing it. A
target whose labels are actually fitted, or a stellar template learned from
several epochs, is what would be needed to do better.
Numbers: `docs/igrins_science_dct2018_model.json`.

## A dry-gas scale for high-airmass K

On McDonald 2017 the two K frames above airmass 2.4 transferred at 1.6-2.0 of
the noise with a per-frame water scale, while every other frame was under 0.8
and H at the same airmass was fine: the night's CO2 and CH4 columns rise with
airmass, and a water scale cannot absorb a dry-gas slant-path error.
`--dry-shift` adds per-frame CO2 and CH4 scales, measured the same way -- free
per order, median over orders. Tested by holding standards out
(`validate_igrins_transfer.py --dry-shift`, `docs/igrins_transfer_dry_*.json`),
correction against the full fit over the noise, absorbing orders:

| | water + velocity (S2) | + CO2, CH4 separately (S3) | + one tied dry scale (S3t) |
|---|---:|---:|---:|
| McDonald 2017 K | 0.96 | 0.45 | **0.46** |
| -- its airmass 2.96 frame | 2.00 | 0.42 | **0.41** |
| -- its airmass 2.48 frame | 1.56 | 0.31 | **0.31** |
| DCT 2018 K, no deficit | 0.38 | 0.40 | |
| McDonald 2017 H | 0.49 | 0.48 | |
| DCT 2018 K, Arcturus injected | 0.62 | 0.64 | **0.61** |

Where the deficit exists it halves the error and removes the high-airmass
outliers; where it does not, it costs nothing. The CO2 and CH4 shifts, measured
in different orders, agree to 0.007 or better on seven of the eight frames
(+0.081 and +0.080 on the airmass-2.96 frame, +0.074 and +0.072, -0.043 and
-0.039), which is what a shared column or path error does and
independent line-list errors would not; they are also not a function of
airmass (+0.07 at 2.48, -0.04 at 2.08), so the fix has to be per frame. But
freeing them separately lets a line-rich target pull CH4 by -3.3% and CO2 by
-0.8% -- CH4's orders share 2.3 um with the CO bandheads -- which is why the
default, `--dry-shift` with no value, **ties** them: CO2's shift, applied to
both. That keeps all of the benefit and none of the penalty.

**Later: what the shift was measuring.** The two frames are k Tau and HD 53205,
eight-exposure sequences of setting stars, 16 and 20 minutes long, whose
airmass the header gives for the first exposure only (`igrins_a0v.md`, "What
moves the well-mixed columns"; GitHub #1). Their sequence airmasses are 3.23 and
2.69 against the 2.96 and 2.48 used here, ln ratios of +0.089 and +0.082 -- the
+0.081 and +0.074 the shift measured. A dry-gas scale is a slant-path
correction in disguise, and it worked because one path factor scales every gas;
water's scale was absorbing its share. With `igrins_pointing.py --sequence`
it should be unnecessary. That is measured in GitHub #4, not assumed: until then
keep using it.

## Running it

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/download_rrisa_standard.py \
    --night 20181220 --objtype TAR --list
UV_CACHE_DIR=.uv-cache uv run python scripts/build_igrins_calibration.py \
    --run-dir data/corrected/igrins/ladder_a0v --output data/calibration/igrins_dct2018_h.h5
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_igrins_science.py \
    --spec data/igrins/20181220_0059/SDC?_20181220_0059.spec.fits \
    --calibration data/calibration/igrins_dct2018_h.h5 data/calibration/igrins_dct2018_k.h5 \
    --clip-sigma 3 --dry-shift --output-dir data/corrected/igrins/science_dct2018
UV_CACHE_DIR=.uv-cache uv run python scripts/summarize_igrins_science.py \
    --runs data/corrected/igrins/science_dct2018 --output docs/igrins_science_dct2018.json
# a target model in the source slot needs a starting stellar velocity: from 0,
# three of LkCa 15's H orders settle 40 km/s from the star
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_igrins_science.py \
    --spec data/igrins/20181220_0059/SDC?_20181220_0059.spec.fits \
    --calibration data/calibration/igrins_dct2018_h.h5 data/calibration/igrins_dct2018_k.h5 \
    --stellar data/stellar/lkca15_payne_zero_hk.npz --vsini-kms 13 --stellar-velocity-kms 26.6 \
    --clip-sigma 3 --output-dir data/corrected/igrins/science_dct2018_model \
    --record data/corrected/igrins/science_dct2018_model/record_0059.h5
```

To look at the result rather than the summary numbers,
`scripts/export_igrins_target_review.py` bundles these runs (plus the
three-standard sparse run and the PLP's own A0V-divided spectra) for
`docs/review_page/igrins_targets.html`; CLAUDE.md, *Review pages*, says how it
is published.

Each run's npz carries `clipped`, the pixels the clip set aside when it measured
the frame's shifts. It cannot be recomputed afterwards: it is cut against the
unclipped model, and the final continuum, refitted without those pixels, sits
higher and flags about 1.7x as many.

Fit both bands of an exposure in one run to get the band comparison printed;
the report computes it either way. A band costs about 15-25 minutes on one GPU
for a handful of frames, most of it per-order setup that every frame of the
night shares.
