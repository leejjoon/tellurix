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
disagreement beyond 0.04 in log column -- twice the clean scatter -- says one
band's water scale is absorbing something that is not water. The report
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
| 37 | Hubble 4 | 1.83 | between standards | 4.71 / 4.37 | -0.016 / -0.014 | -0.001 |
| 53 | V1075 Tau | 1.51 | between standards | 2.81 / 2.18 | +0.007 / +0.012 | -0.005 |
| 59 | LkCa 15 | 1.32 | between standards | 3.51 / 2.31 | -0.003 / +0.008 | -0.010 |
| 74 | UY Aur | 1.09 | between standards | 3.04 / 1.90 | +0.041 / +0.044 | -0.004 |
| 92 | GJ 281 | 1.43 | between standards | 8.06 / 6.30 | -0.002 / +0.020 | -0.022 |
| 142 | YY Gem | 1.16 | 2.5 h after the last | 2.07 / 1.82 | +0.153 / +0.138 | +0.015 |

The residuals are the targets' own lines, which a flat source leaves in; they
are not a measure of the correction. What measures it is the band agreement:
median 0.007, worst 0.022, none flagged. Clipping removes 1-4% of pixels and
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
GJ 281's bands agree to 0.022.

A residual cannot see this: the fit puts a slant-path error into the columns
and fits as well as before. A second band measuring the same water can.

## Not yet done

- The other K runs -- DCT 2016, McDonald 2017, Gemini South 2021, their ERA5
  variants, and DCT 2018's flat-source run -- predate the zenith fix and carry
  affected frames, among them McDonald's airmass-3 point. Their records, the
  airmass-ladder reports and `igrins_transfer_*_k.json` are stale until rerun.
- A night with fewer than five standards, which needs the master pattern.
- A target model in the source slot, which the injection test says removes the
  line-rich floor; `--stellar` accepts one but it is untested here.

## Running it

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/download_rrisa_standard.py \
    --night 20181220 --objtype TAR --list
UV_CACHE_DIR=.uv-cache uv run python scripts/build_igrins_calibration.py \
    --run-dir data/corrected/igrins/ladder_a0v --output data/calibration/igrins_dct2018_h.h5
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_igrins_science.py \
    --spec data/igrins/20181220_0059/SDC?_20181220_0059.spec.fits \
    --calibration data/calibration/igrins_dct2018_h.h5 data/calibration/igrins_dct2018_k.h5 \
    --clip-sigma 3 --output-dir data/corrected/igrins/science_dct2018
UV_CACHE_DIR=.uv-cache uv run python scripts/summarize_igrins_science.py \
    --runs data/corrected/igrins/science_dct2018 --output docs/igrins_science_dct2018.json
```

Fit both bands of an exposure in one run to get the band comparison printed;
the report computes it either way. A band costs about 15-25 minutes on one GPU
for a handful of frames, most of it per-order setup that every frame of the
night shares.
