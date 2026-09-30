# The blaze from the lamp flat

Numbers here come from `docs/igrins_blaze_dct2018.json`, written by
`scripts/compare_igrins_runs.py`.

## What the response pattern mostly is

The fitted continuum of an IGRINS order is a degree-9 Chebyshev in log flux. The
"instrument response pattern" the standards fits divide out (`igrins_a0v.md`) is
strongest in the last 100-150 pixels of every order -- a rise to +5-7% and a
fall to -8 to -10% where the blaze rolls off -- in both bands, with or without
absorption, the same on every night. RRISA ships each night's PLP calibrations
(catalog column `CAL_URL`), including 2-D lamp-on and lamp-off flats. Traced and
collapsed along the slit, the lamp's 1-D spectrum, fitted with the same
degree-9 continuum over the same pixels, leaves a residual that reproduces the
pattern at r = +0.74 to +0.89 (K80, K90, H108), red-edge rise and fall
included. So most of the pattern is the blaze's order-end roll-off, which a
degree-9 polynomial cannot bend to.

## The blaze as a fixed continuum shape

`tellurix.FlatBlaze` (`scripts/build_igrins_flat_blaze.py`) traces the orders in
the lamp flat -- bands ~61 rows tall, ~100 apart, along the columns -- collapses
them, names each by physical order by matching its lit column range to the
extracted spectra (to 9 columns in H, 5 in K), and smooths it into a blaze. The
fitting driver then divides flux and uncertainty by it, looked up by **detector
column** (`IGRINSOrder.pixel`), and the polynomial is left only the lamp-to-star
colour (`fit_igrins_standard.py --blaze`).

Three things the smoothing has to get right, each found by getting it wrong:

- **The lamp has telluric lines of its own** -- its light crosses air. Narrow
  dips are clipped and bridged; runs longer than 10 pixels are kept, because the
  order-end roll-off also sits below a lagging filter and is the thing being
  measured. Where the lamp's absorption is deep -- H98, H119-121 in the 1.49 um
  water band, K72-73, K89, K91-93 -- no smoothing separates it from the blaze, and
  those orders are refused (1st percentile of lamp/blaze below -6.5%, against
  -3% for noise). 20 of 24 traced H orders and 16 of 22 K orders remain; the
  bluest H orders and K71, K94-95 are cut by the detector edge and not traced.
- **A noiseless lamp would oscillate the clipping**: a dip has to be at least
  0.5% deep.
- **One window cannot serve both ends and middle.** The roll-off needs ~31
  pixels to follow; the interior carries the lamp's own 0.2-0.4% structure on
  ~15-pixel scales, which a 31-pixel blaze divides into the star -- measured, it
  raised H's interior residual on 16-pixel scales from 1.82 to 2.28 times white
  noise. The blaze is therefore the 31-pixel fit within 100 pixels of the order
  ends and a 151-pixel fit beyond 200, blended between (`edge_blended_blaze`).

## Results, DCT 2018-12-20

Ten standards, A0V model, over the orders with a usable blaze:

| | residual, no pattern | with pattern | red-edge rms | pattern still needed | Brackett v_star spread |
|---|---:|---:|---:|---:|---:|
| H, degree 9 (before) | 1.914 | 1.248 | 3.05 | 0.0116 | 28.0 km/s, 34% railed |
| H, blended blaze + degree 9 | **1.419** | **1.235** | **1.51** | **0.0037** | **21.9 km/s, 24%** |
| K, degree 9 (before) | 1.950 | **1.318** | 2.53 | 0.0132 | |
| K, blended blaze + degree 9 | **1.534** | 1.424 | **1.51** | **0.0040** | |

The block-averaged residual (1 = white noise) tells where it acts: in H the
blended blaze is at or below the old fit at every scale, interior and red end;
the red end falls from 7.09 to 2.78 at 16 pixels.

**H: better everywhere.** **K: much better without a pattern, 8% worse with the
night's own pattern.** K's loss is at 1-4 pixel scales, which a blaze smoothed
over 31-151 pixels cannot carry; it is in every frame (+3% to +15%) and does not
follow the PLP's flexure (r = -0.20), and the K fits with the blaze come out with
a slightly wider LSF (+0.04 km/s). It is not understood.

A lower continuum degree does not follow from the blaze: degree 5 is worse than
degree 9 in both bands (H 1.355, K 1.508 with the pattern) -- the lamp-to-star
colour is not that smooth.

The report's "blaze + degree 9" and "blaze + degree 5" runs used an earlier blaze
-- one 31-pixel window, before the dip-depth floor and the long-run rule -- whose
file was then rebuilt under the same name; their records' `blaze` path now points
at the blended one. Only the "blended blaze" run was fitted with the file that
is committed.

## Where to use it

Where the night's own pattern exists and K is being fitted, the old continuum
still wins by 8%. Everywhere else the blaze is the better base: H on any night,
and both bands where the pattern must be borrowed -- thin nights and science
targets, which is where the pattern matters most and is least well measured.
The blaze leaves a third of the pattern to correct, so a borrowed one has less
to get right. That test -- blaze plus master pattern against master pattern
alone on a thinned night -- needs the other nights' flats and is not done.

## Also in the bundle

`joined_flexure.csv` gives the PLP's per-frame H and K flexure. It does not
track the fitted velocity shifts (r = -0.42 in H, +0.15 in K; K's flexure moves
by 0.7 while its velocity barely does), so it is most likely a shift along the
slit. Not used.

## Running it

```bash
# the night's calibration bundle: the catalog's CAL_URL, one per night (~1 GB)
curl -sL <CAL_URL> | tar -xz -C data/igrins/cals/<night>
UV_CACHE_DIR=.uv-cache uv run python scripts/build_igrins_flat_blaze.py \
    --cal-dir data/igrins/cals/20181220 --band H \
    --spec data/igrins/20181220_0100/SDCH_20181220_0100.spec.fits \
    --output data/calibration/igrins_blaze_dct2018_h.h5
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_igrins_standard.py --spec ... \
    --blaze data/calibration/igrins_blaze_dct2018_h.h5
```
