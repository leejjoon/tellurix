# What is left after the blaze and the pattern

Numbers come from `docs/igrins_residual_structure_*.json`, written by
`scripts/analyze_igrins_residual_structure.py` from blended-blaze standards runs
(A0V model, `igrins_flat_blaze.md`) and each night's lamp flat:

| night | report | standards | per-pixel S/N (median; brightest frame) |
|---|---|---:|---|
| DCT 2018-12-20 | `igrins_residual_structure_{h,k}.json` | 10 | 165 H, 134 K (237) |
| McDonald 2015-12-03 | `..._mcd20151203_{h,k}.json` | 6 | 527 H, 433 K (893) |
| McDonald 2015-12-01 | `..._mcd20151201_{h,k}.json` | 7 | 445 H, 379 K (735) |

Every report also compares its night with the others and with the standards
runs of DCT 2016, McDonald 2017 and Gemini South 2021. Nothing was refitted: the
gains quoted are estimates made by subtracting a candidate term from the
residual.

**What a night shows depends on its signal to noise, and some of it on the
epoch, so nothing here is a property of IGRINS in general until more nights say
so.** At S/N ~150 (DCT 2018) the wiggles sit below the noise per pixel. At S/N
400-900 (McDonald 2015) they do not: line-free pixels fit at 2.4-4.6 sigma, and
three things carry that -- isolated detector pixels that misbehave the same way
on every night (mostly H), a patch of the detector that is noisy in some
exposures, and in K a fringe four to six times stronger than DCT 2018's whose
phase changes from one pointing to the next.

**Open, to revisit**: what the bad pixels and the noisy patch are (both need the
2-D frames), and whether the per-frame fringe term and a spike mask hold up
inside a refit rather than as subtractions from a residual.

| per-pixel z | DCT 2018 | McD 2015-12-03 | McD 2015-12-01 |
|---|---:|---:|---:|
| H, line-free pixels | 0.98 | 4.57 | 2.38 |
| H, pixels with lines | 1.60 | 6.12 | 3.85 |
| K, line-free pixels | 0.83 | 4.09 | 2.49 |
| K, pixels with lines | 1.55 | 4.60 | 3.64 |

"Line-free" means the 1 cm-1-smoothed model transmission is above 0.99; stellar
lines are not excluded.

## The fringe

The lamp flat, against a wide smooth (Savitzky-Golay 401, cubic), carries one
periodic component in wavenumber at **2.75-2.76 cm-1 in H and 2.78-2.80 in K**
on all three nights -- an optical thickness n*d of 1.8 mm, a window or filter.
It is an etalon and not a pixel artefact because the period holds in wavenumber
while its length in pixels runs 75-63 across H and 102-80 across K. It is close
to one sinusoid: the second harmonic is 0.035-0.05% against the fundamental, and
no second narrow peak appears between 0.12 and 12 cm-1 (DCT 2018).

| | lamp, median | stars | stars after the pattern | phase rms about one etalon, lamp |
|---|---:|---:|---:|---:|
| H, DCT 2018 | 0.20% | 0.14% | 0.08% | 105 deg |
| H, McD 2015-12-03 / -01 | 0.14 / 0.15% | 0.19 / 0.18% | 0.14 / 0.14% | 102 / 100 deg |
| K, DCT 2018 | 0.26% | 0.10% | 0.07% | 69 deg |
| K, McD 2015-12-03 / -01 | **0.98 / 0.98%** | **0.59 / 0.58%** | **0.30 / 0.31%** | 54 / 54 deg |

**H is much the same on both epochs; K is not.** In 2015 the K lamp's fringe
reaches 2.2% (K81) and the stars' 1%; by 2018 it is a quarter of that. Something
in the K beam changed between the two.

Phase does not carry across orders as one etalon at one angle would (the last
column), so each order needs its own phase. Within an order:

- **DCT 2018**: the stars' phase is stable across ten frames of five stars (13
  deg scatter where the fringe is strong, under 3 deg an hour of drift) but
  offset from the lamp's, so dividing by the lamp does not remove it.
- **McDonald 2015, H**: the phase is again common to every frame (e.g. H116
  -101 to -135 deg across six frames of four stars); only the amplitude differs.
- **McDonald 2015, K**: the phase changes **between pointings**. On 2015-12-01
  chi Tau's frames 0054 and 0070 sit at +31 to -48 deg while its 0094 and 0115,
  and three other stars, sit at -70 to -111 (`fringe_per_frame.own_fringe_by_order`);
  on 2015-12-03 chi Tau's three frames agree with each other and differ from the
  other stars. Time and airmass do not order it.

What removing it buys, on line-free pixels (`fringe_per_frame.clear_pixel_z`,
per-pixel z / rms of 15-pixel means, white noise reading 1 in both):

| frame | now | other frames' fringe | own fringe |
|---|---:|---:|---:|
| K 2015-12-03 0081 | 5.45 / 17.6 | 4.97 / 15.6 | **3.18 / 7.3** |
| K 2015-12-03 0106 | 3.32 / 10.8 | 2.52 / 7.5 | **2.00 / 5.1** |
| K 2015-12-01 0054 | 3.30 / 11.3 | 4.02 / 14.0 | **1.94 / 6.0** |
| K 2015-12-01 0094 | 2.30 / 7.6 | 2.16 / 7.0 | **1.47 / 4.0** |
| H 2015-12-03 0106 | 3.02 / 6.3 | 2.88 / 5.4 | 2.83 / 5.1 |
| H DCT 2018 0100 | 1.10 / 1.9 | 1.09 / 1.8 | 1.08 / 1.7 |

"Own fringe" is the frame's own sinusoid at the band's period, amplitude
quadratic along the order -- six parameters against ~1,000 pixels. **In K on the
2015 nights it has to be fitted per frame**: a template from the other frames
helps where the phase happens to agree and doubles the damage where it does
not. In H, and at DCT 2018's S/N, it is worth a few percent.

The blaze's own filters carry 17-32% of the lamp's fringe into the star (the
151-pixel interior filter; the 31-pixel ends carry nearly all of it). Fitting
and removing the fringe from the lamp before smoothing it would stop that.

## Bad pixels that recur between nights

The residual under 15 pixels (`pixel_scale`), on line-free pixels, is shared
between frames: each frame against the median of the others correlates at a
median **r = +0.86 and +0.88** in H on the two McDonald nights (+0.56 at DCT
2018, where the noise dilutes it); in K, +0.39 and +0.50 (+0.31).

**Most of what is shared is a set of isolated pixels.** Plotted on detector
columns, each night's template (the median over its frames) is flat apart from
single-pixel spikes that sit at the same columns night after night
(`nights.pixel_scale_spikes`, and the per-pair figures in
`nights.pixel_scale_template_correlation`):

- **The top 1% of a night's template pixels carry 49-51% of its variance** in H
  on the McDonald nights, 41% at DCT 2018; 4.1% of pixels lie beyond 5 robust
  sigma at McDonald 2015, 1.8% at DCT 2018 and DCT 2016, under 1% on the noisier
  McDonald 2017 and Gemini South 2021 templates -- how many are found depends
  on the template's own noise.
- **They recur.** 95% of 2015-12-03's spike pixels are spikes on 2015-12-01 too
  (within a pixel), and 92-94% of DCT 2018's are spikes on the 2015 nights.
  Against later nights the overlap falls: DCT 2016 25-56%, McDonald 2017
  17-22%, Gemini South 2021 4-11%.
- **The templates' correlation is mostly these pixels.** At zero lag, on
  detector columns, every pair of six H nights correlates (+0.26 to +0.91,
  against -0.05 to +0.01 for a 20-pixel shift); without the spike pixels that
  falls to +0.15 to +0.69 -- +0.69 between the two McDonald nights, +0.32 to
  +0.41 between them and DCT 2018. A continuous part is left, smaller.
- **K's barely travel**: +0.61 between the two 2015 nights (+0.43 without
  spikes), +0.03 to +0.18 against every other night.

They are not the lamp's pixel structure: on 2015-12-03 the night's pixel-scale
template correlates with the lamp's at r = -0.02 (a check not in the reports;
the spectra already carry the PLP's flat field). Nor telluric: on the same
night they correlate with the model's own pixel-scale absorption at +0.02 and
do not grow with airmass (-0.01 per unit, against +0.81 for an atmospheric
error; also a check, over the narrow 1.04-1.46 airmass range). Nor stellar:
they recur at zero lag between nights of different stars at different
velocities. **Most likely they are detector pixels the PLP does not mask.**

**Masking them is the fix, and it travels better than a template**
(`pixel_scale_z_rms_on_own_night`, pixel-scale z rms):

| | now | another night's template | another night's spike pixels masked | own night, leave-one-out |
|---|---:|---:|---:|---:|
| H McD 2015-12-03, from 2015-12-01 | 4.03 | 3.15 | **2.96** (4% of pixels) | 3.11 |
| H McD 2015-12-01, from 2015-12-03 | 1.99 | **1.11** | 1.15 (4%) | 0.93 |
| H McD 2015-12-01, from DCT 2018 | 1.99 | 1.60 | **1.44** (2%) | 0.93 |
| H DCT 2018, from McD 2015-12-03 | 0.86 | 0.77 | **0.71** (4%) | 0.71 |
| K McD 2015-12-03, from DCT 2018 | 2.26 | 2.59 | **2.24** (<0.5%) | 2.12 |
| K McD 2015-12-01, from 2015-12-03 | 1.19 | 1.10 | **1.01** (1%) | 1.06 |

A mask removes the pixels instead of guessing their values, so where a template
from another night makes things worse (K from DCT 2018, H from McDonald 2017) a
mask from the same night does not. It needs a night of the instrument's own
recent history, not the same epoch exactly: DCT 2018's pixels serve 2015.

## A detector patch that is noisy in some exposures

After the leave-one-out template, most frames are at the noise -- 2015-12-01's
0054 and 0070 at S/N 640-690 come out at 1.03 and 0.98 -- but three do not
(`pixel_scale.frames`):

| frame | S/N | z after template | robust width | top 1% of pixels' share of chi-squared |
|---|---:|---:|---:|---:|
| H 2015-12-03 0057 (s Tau) | 893 | 6.83 | 1.15 | 43% |
| H 2015-12-03 0081 (chi Tau) | 734 | 2.74 | 0.91 | 50% |
| H 2015-12-01 0115 (chi Tau) | 735 | 1.45 | 0.93 | 56% |

The bulk of these frames is at the noise. **The excess is a patch of the
detector** (`pixel_scale.frames.*.z_rms_by_order_and_column_block`): in 0057
the leave-one-out z is 15-26 at H columns from about 1536 to the order end, over
orders H106-H120, and about 1 everywhere else; 0081 shows the same patch
weaker, 0106 fainter still (1.5-3), and 2015-12-01's 0115 at the **same place**
(mainly H117-H120). In K, 0057 has a patch of its own at columns ~130-640 over
K80-K83. Within the patch, spikes are mostly positive (77-78% in 0057 and 0081),
and they rise at pixels that rise weakly in the night's other frames too.

Three explanations were checked and fail -- these checks are not in the reports:

- **Not count level.** 2015-12-01 0115 and 2015-12-03 0081 have the same
  extracted counts (99th percentile 368k and 373k per 300 s), and 0115 is the
  milder.
- **Not sky OH.** Spike pixels lie within a pixel of an OH line (AER/HITRAN
  positions, 3,958 lines in 5500-7000 cm-1) no more often than any clear pixel
  does: 21-28% against 19%.
- **Not over-predicted telluric lines.** The model's absorption at the spikes is
  that of any clear pixel (0.17-0.23%).

A region of the array that misbehaves in particular exposures -- persistence
from an earlier bright exposure, glow, a ghost -- fits the description; telling
needs the 2-D frames. Until then a per-frame check of this map, and masking or
down-weighting a noisy region, would contain it.

## The broader scales

The residual band-passed into 0.7-4 cm-1 and 4-15 cm-1 boxcars (`scales`),
correlated between frames of different stars more than three hours apart: the
short band is shared all night on every night (H: +0.53 DCT 2018, +0.58 and
+0.64 McDonald 2015, clear orders), the long band is not (-0.03, +0.01, +0.14):
a response that drifts over hours. Neither grows with airmass, and neither
follows the lamp except where the fringe is strong (K 2015, r = +0.33 and
+0.34, the fringe's period being in the band). A master from other nights does
not help the short band at DCT 2018 (0.230% to 0.227% in clear H orders, worse
in K). At DCT 2018's S/N none of this is visible per pixel.

## DCT 2018 alone

At S/N ~150 the picture is the one first measured: line-free pixels at z =
0.98 (H) and 0.83 (K), lines at 1.60 and 1.55, and every continuum-scale fix
estimated here worth at most 3% on line-free pixels and under 1% in lines
(`fixes_without_refit`). The wiggles in a plot of that night are real -- the
short band and the fringe -- but only binning lifts them above the noise.

## McDonald 2015 headers

Both 2015 nights needed two reader changes (`igrins.py`), and either would have
corrupted the slant path silently:

- **Weather units.** McDonald 2017 writes Fahrenheit and inches of mercury;
  2015-12-03 writes Celsius and station hectopascals (5.6 / 799.5, dewpoint
  -8.7, humidity 34%). `Site.alternatives` lists both, and `surface_conditions`
  keeps the convention the frame's own numbers support: the pressure has to
  match the site's hydrostatic expectation, and temperature, dewpoint and
  humidity have to agree (34% implied in Celsius, 51% in Fahrenheit). 2015-12-01
  writes -1 in all four cards, read as missing.
- **Pointing.** 12 of 13 frames carry a zenith distance that is -1 or 2-24 deg
  wrong (s Tau: 18.4 deg in the header, 42.6 by geometry), and the catalog's
  RA/Dec is empty or wrong for the same rows. `scripts/igrins_pointing.py`
  resolves the target by its SIMBAD name, computes the zenith distance over the
  exposure (`geometric_zenith_angle_deg`), and writes a `pointing.json` beside
  the frame, which `read_igrins_observation` then prefers (`zenith_source`). On
  DCT 2016/2018, McDonald 2017 and Gemini South 2021 geometry and header agree
  to 0.2 deg.

## Running it

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/igrins_pointing.py --spec data/igrins/20151203_*/SDCH_*.spec.fits   # check
UV_CACHE_DIR=.uv-cache uv run python scripts/igrins_pointing.py --spec ... --write all                            # override
UV_CACHE_DIR=.uv-cache uv run python scripts/analyze_igrins_residual_structure.py --band H \
    --run data/corrected/igrins/blazeB9_mcd20151203_h \
    --blaze data/calibration/igrins_blaze_mcd20151203_h.h5 \
    --cal-dir data/igrins/cals/20151203 \
    --spec data/igrins/20151203_0081/SDCH_20151203_0081.spec.fits \
    --night MCD1203=data/corrected/igrins/blazeB9_mcd20151203_h --own MCD1203 \
    --night MCD1201=... --night DCT18=... --night DCT16=... --night McD17=... --night GS21=... \
    --output docs/igrins_residual_structure_mcd20151203_h.json
# K: the same with the K runs, plus --velocity-run <the H run> for the stellar velocities
```

The McDonald 2015 standards were fitted exactly as DCT 2018's blended-blaze runs
(`--stellar data/stellar/a0v_payne_zero_hk.npz --vsini-kms 150 --blaze ...`),
with ERA5 profiles (`era5_site_profile.py`, 2.6 and 1.6 mm of water) and blazes
from each night's own flats.

The fit cache runs in **ascending wavenumber**, the extracted order in ascending
wavelength, so a cached array and `IGRINSOrder.pixel` run in opposite directions.
The script matches samples to detector columns by wavenumber and refuses to run
unless that mapping reproduces the blaze the driver saved.
