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

Ten standards, A0V model, over the orders with a usable blaze. **Compare runs by
per-pixel z** -- the rms of residual / uncertainty on the pixels both runs call
reliable. The driver's `residual_rms_over_noise` divides the rms residual by the
*median* uncertainty, so it moves when a blaze reweights the order ends: it made
the blaze look 8% worse in K when per-pixel z says 10% better. The driver now
writes `residual_z_rms` beside it.

| | per-pixel z, whole order | interior only | driver metric, with / without pattern | red-edge rms | pattern still needed |
|---|---:|---:|---:|---:|---:|
| H, degree 9 (before) | 1.432 | 1.072 | 1.248 / 1.914 | 3.05 | 0.0116 |
| H, blended blaze + degree 9 | **1.149** | **1.071** | 1.235 / 1.419 | **1.51** | **0.0037** |
| K, degree 9 (before) | 1.576 | **1.257** | 1.318 / 1.950 | 2.53 | 0.0132 |
| K, blended blaze + degree 9 | **1.412** | 1.316 | 1.424 / 1.534 | **1.51** | **0.0040** |

**Better in both bands** -- 20% in H, 10% in K -- almost all of it at the order
ends, where the old continuum could not follow the roll-off (the red end's
block-averaged residual at 16 pixels falls from 7.09 to 2.78 times white noise
in H). In H the interior is unchanged. The Brackett-order stellar velocity
spread falls from 28 to 22 km/s and its rail rate from 34% to 24%.

**What is left in K's interior is a line-shape trade.** Its interior is 4.7%
worse, all of it inside telluric lines (continuum within 0.1-0.7%, transmission
below 0.8 +2-3%). In the line cores both fits miss by about 3 sigma: the old one
leaves the cores too deep (mean z +0.24) with a narrower Gaussian LSF, the blaze
run balances the cores (mean z -0.09) with a 1.7% wider one and pays slightly in
the flanks. A Gaussian cannot match an IGRINS line's core and wings at once, so
a change anywhere else -- here the continuum -- moves where the miss lands. It is
not the blaze and not flexure (r = -0.20 against the PLP's), and it is not the
old, larger pattern hiding line error: both patterns track absorption depth
equally weakly (r = +0.12, +0.13). The fix, if one is wanted, is a non-Gaussian
LSF.

A lower continuum degree does not follow from the blaze: degree 5 is worse than
degree 9 in both bands -- the lamp-to-star colour is not that smooth.

The report's "blaze + degree 9" and "blaze + degree 5" runs used an earlier blaze
-- one 31-pixel window, before the dip-depth floor and the long-run rule -- whose
file was then rebuilt under the same name; their records' `blaze` path now points
at the blended one. Only the "blended blaze" run was fitted with the file that
is committed.

## Where to use it

Everywhere a flat is available: it beats the old continuum in both bands and
leaves a third of the response pattern to correct, which matters most where the
pattern has to be borrowed -- thin nights and science targets. The test of that
-- blaze plus master pattern against master pattern alone on a thinned night --
needs the other nights' flats and is not done.

What the blaze and the pattern leave -- a 2.75-2.80 cm-1 fringe that the blaze
partly copies from the lamp into the star, bad pixels that recur between
nights, a detector patch that is noisy in some exposures, and a slowly drifting
response; below
the noise per pixel at S/N ~150, well above it at S/N 400-900 -- is in
`igrins_residual_structure.md`.

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
