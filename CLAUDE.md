# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tellurix` (package `src/tellurix/`, repo directory `lblrtm/`) is a
differentiable JAX forward model for terrestrial (telluric) absorption in
high-resolution spectra such as IGRINS. It wraps ExoJAX opacity calculators
with a layered atmosphere, slant-path transmission, an instrument model, and a
bounded MAP fitter. LBLRTM 12.17 is the accuracy reference, not a runtime
dependency — LBLRTM is only executed offline to produce fixtures and
correction templates.

## Commands

```bash
UV_CACHE_DIR=.uv-cache uv sync --dev --extra gpu   # environment (.venv, Python 3.10)
# Plain `uv sync --dev` REMOVES the CUDA wheels and leaves jax CPU-only; the extra is
# what keeps the GPU working, and every fit driver defaults to --platform gpu.
UV_CACHE_DIR=.uv-cache uv run pytest          # full suite, ~70 s, no external data needed
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py::test_name   # one test
```

Always prefix with `UV_CACHE_DIR=.uv-cache`; the default cache location is not
used in this project. Set `JAX_PLATFORMS=cpu` to keep test/benchmark runs off
the GPU.

The Arcturus telluric-fitting pipeline (see `docs/arcturus_fit.md`). Needs the AER
line files and the MT_CKD file from `bootstrap_lblrtm.sh`, but not the LBLRTM binary:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/measure_atlas_ils.py          # FTS MOPD per page/epoch
UV_CACHE_DIR=.uv-cache uv run python scripts/make_site_profile.py          # site + epoch atmosphere CSV
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_arcturus_fit.py      # synthetic checks L0-L2
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_arcturus_page.py --stellar <npz>
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_arcturus_batch.py         # resumable, whole atlas
UV_CACHE_DIR=.uv-cache uv run python scripts/trim_atlas_summary.py         # -> docs/arcturus_atlas_summary.json
UV_CACHE_DIR=.uv-cache uv run python scripts/rebuild_arcturus_page.py --page ab5000_ --epoch summer --check
```

The batch driver writes its full summary to `data/corrected/atlas/summary.json`
(gitignored — 1.6 MB at atlas scale, mostly per-page timings and a per-stage
optimizer log). `trim_atlas_summary.py` reduces it to the 0.39 MB record under
`docs/`, keeping the window, the fitted parameters and the quality numbers, and
lifting per-page fields that never vary into `settings`.

Both fit drivers default to `--precompute-opacity` (see *What a fit costs*);
`--no-precompute-opacity` restores the exact-kernel-per-iteration path and is
the control to reach for when a fitted value looks wrong.

The committed Arcturus record is the **full-coverage** run: `page_windows`
no longer trims the ~5 cm-1 the atlas pages overlap by, so every pixel the atlas
ships is fitted (822,928 against 636,444) and adjacent pages overlap in every
product. `--trim-overlap-cm1 2.5` restores the old tiling and reproduces the
superseded record (`git show 06e5e3d:data/corrected/atlas/arcturus_atlas.h5`).
The residual median moves 4.22 -> 4.48 sigma, which is the page edges being
harder, and the at-bound rate improves 21.6% -> 19.4%. **Nothing in the products
tiles now**, so concatenating adjacent pages double-counts. The duplication is
the point: the *data* differs between two pages over the same wavenumbers by a
median 0.0128 against a 0.0045 pixel sigma, because each page carries its own
scalar normalization, so the two copies are two reductions of one measurement
rather than one measurement twice -- do not average them, use them as a
consistency check. It works: ab6250_ landed on `lsf_sigma_kms` = 1.278 km/s
against its neighbour's 0.110 with twice the residual, and only the overlap
showed it.

A sharded run writes one record per process; `scripts/merge_atlas_records.py`
joins them at the HDF5 dataset level rather than through `write_record`, because
the row dtype packs the column scales into `log_column_*` fields and unpacking
them is where a merge would put a value in the wrong column. Splitting the
598-row record three ways and merging it back is byte-identical. It warns, and
does not refuse, when shards disagree on `ils_velocity_kms`: that array is
**not** a per-run constant, since `ils_fingerprint` sizes its grid from each
page's own sinc first zero and `write_record` keeps only the last page's, so
every row's `ils_profile` is on its own grid while one grid is stored. That is
true of a single-process run too.

The stellar source is **already continuum-normalized** and the pipeline throws
that away. Payne Zero ships `flux = flux_total / flux_continuum` (exact to
6.7e-16), sitting at 1.0 where there is no line, and `prepare_stellar_source`
then divides by the *median*, which replaces a physical zero point with an
arbitrary one 2.8% off and is why the fitted continuum sits below the data's
envelope. The divisor is a scalar exactly degenerate with `continuum_0`, so
`--source-already-normalized` changes nothing measurable -- reduced chi2
identical to four decimals, columns to five -- and moves the continuum from
0.9554 to 0.9833 on ab6225_. Both committed npz files also store `flux_total`
and `flux_continuum` **reversed** against `wavenumber_cm1` (the generators
reordered only `flux`); the generators are fixed, and `StellarSpectrum.from_npz`
detects the orientation from the identity rather than assuming one, so it reads
both the broken and the fixed files. `resample_stellar_continuum` puts that
continuum on any grid and the drivers save it as `stellar_continuum`. Keep the
two apart: `continuum` is a free polynomial fitted jointly with fixed Payne Zero
gf values and is circular for anyone measuring line strengths, while
`stellar_continuum` is a prediction and carries the bound-free edges a
polynomial cannot represent -- the Brackett edge at 1458.8 nm is a 0.19% step
for Arcturus and **5.6%** for the A0V model, and the two IGRINS H orders
straddling it fit at a median 3.76 sigma against 1.79 for the other 25.
`corrected` itself is safe: it is exactly `observed / effective_transmission`,
so the source enters only through the convolution weighting, worth a median
0.00016 against 0.0137 of noise.

The batch driver's saved `transmission` is the **unconvolved** transmission at
pixel wavenumbers, not the operator the correction applied. That is
`model_flux / stellar_only`, the convolved effective transmission; the two
differ by up to 0.126 against 0.0059 of noise. `docs/arcturus_fit.md` lists
every saved array and the summary's `physics` block.

The IGRINS A0V standard pipeline (see `docs/igrins_a0v.md`). Same AER and
MT_CKD inputs; the spectra come from the RRISA reduced archive, not the atlas:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/download_rrisa_standard.py --catalog
UV_CACHE_DIR=.uv-cache uv run python scripts/download_rrisa_standard.py --list --facility DCT --night 20181220
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_igrins_standard.py --spec <SDCH_*.spec.fits>
UV_CACHE_DIR=.uv-cache uv run python scripts/summarize_igrins_fit.py --summary <*_summary.json>
```

Science frames (see `docs/igrins_science.md`) are corrected against a night's
calibration, built once from that night's standards run:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/download_rrisa_standard.py --night 20181220 --objtype TAR --list
UV_CACHE_DIR=.uv-cache uv run python scripts/build_igrins_calibration.py --run-dir <standards run> --output data/calibration/<night>_<band>.h5
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_igrins_science.py --spec <SDCH and SDCK of the frames> --calibration <H.h5> <K.h5> --clip-sigma 3 --dry-shift
UV_CACHE_DIR=.uv-cache uv run python scripts/summarize_igrins_science.py --runs <output dirs> --output docs/<report>.json
# a night with fewer than five standards borrows the instrument's response:
UV_CACHE_DIR=.uv-cache uv run python scripts/build_igrins_master_pattern.py --calibration <night calibrations> --output data/calibration/igrins_master_<band>.h5
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_igrins_standard.py --spec <the few standards> --response-pattern data/calibration/igrins_master_<band>.h5 ...
UV_CACHE_DIR=.uv-cache uv run python scripts/build_igrins_calibration.py --run-dir <that run> --master-pattern data/calibration/igrins_master_<band>.h5 --minimum-frames 3 --output ...
# a target model for the source slot, in Payne Zero's own environment (see the script):
.venv/bin/python scripts/generate_payne_zero_star.py --name <star> --teff <K> --logg <dex> --label-source <where from> --output <npz>
```

Three extensions, each measured (`docs/igrins_science.md`). **`--dry-shift`**
adds one per-frame dry-gas scale, CO2's, applied to CH4 as well: McDonald 2017's
K frames above airmass 2.4 need CO2 and CH4 7-8% up together, which a water scale
cannot absorb, and it takes them from 1.6-2.0 of the noise to 0.3-0.4 while
costing nothing elsewhere. Keep it **tied** -- freed separately, a line-rich
target pulls CH4 by 3% through the 2.3 um CO bandheads. **A master pattern**
(`MasterPattern`, the median of other nights') stands in for a thin night's own:
DCT 2018 thinned to three standards corrects the other eleven within 5-7% of the
full night's residual, with H and K still agreeing on water to 0.005.
`data/calibration/igrins_master_{h,k}.h5` use all four fitted nights. **A target
model** in the source slot halves GJ 281's K residual and places its lines to
0.2-0.8 km/s, but an approximate model (Payne Zero at literature labels, 4000 K
floor) flips the sign of the bands' 1-2% water disagreement rather than removing
it: that floor is how well the star is known. Calibrations store the pattern in
float32 (`night._store`), which moves it by at most 2.5e-8.

**Most of the "response pattern" is the blaze** (`docs/igrins_flat_blaze.md`).
The degree-9 log-Chebyshev continuum cannot follow the blaze's order-end
roll-off, and what it leaves is the pattern's red-edge rise and fall: the lamp
flat's own residual from the same continuum reproduces it at r = +0.74-0.89. Each
night's PLP calibrations are in RRISA (`CAL_URL`, ~1 GB, `flat_on`/`flat_off`
plus a sky wavelength solution and `joined_flexure.csv`, but no order traces).
`FlatBlaze` traces the orders in the lamp flat itself, names them by matching lit
column ranges to the extracted spectra, and smooths each into a blaze that
`fit_igrins_standard.py --blaze` divides out by **detector column**
(`IGRINSOrder.pixel`, new). Three smoothing rules, each learned the hard way:
the lamp has telluric lines of its own (clip narrow dips only -- the roll-off is
a long "dip" under a lagging filter and must be kept; refuse orders where the
lamp is absorbed below -6.5%); a dip must be at least 0.5% deep or a clean lamp
oscillates the clipping; and the window is 31 px near the order ends but 151 px
inside, because a 31-px blaze divides the lamp's own ~15-px structure into the
star. On DCT 2018 the blended blaze is better than the old continuum on every
measure in H (1.248 -> 1.235 sigma with the pattern, 1.914 -> 1.419 without, red
edge 3.05 -> 1.51, Brackett stellar-velocity spread 28 -> 22 km/s) and leaves a
third of the pattern; in K it is much better without a pattern (1.95 -> 1.53)
but 8% worse with the night's own, at 1-4 px scales, for a reason not yet found
(not flexure). Degree 5 on top of the blaze is worse than degree 9 in both bands.

`NightCalibration` (`night.py`) carries per physical order the standards'
median dry columns, LSF and velocity zero point, water as a time series, and the
response pattern taken over *all* the standards -- a science frame is none of
them, so no leave-one-out, but the 51-pixel smoothing guard still applies. The
science fit refits **one velocity shift and one water scale per frame**, the
median over orders, and never keeps per-order water, which chases a target's
lines. Held-out standards of DCT 2018 fit at 0.9-1.1 sigma that way. The
check that matters is **H against K**: both bands measure the same water, so
their frame shifts must agree -- to 0.005-0.006 rms on held-out standards
once the K airmass is right (the 0.018-0.021 first measured was mostly the
header defect), 0.009 at worst for four more standards, and 0.022 for the
worst of six line-rich targets, an M dwarf -- and a disagreement past 0.02 is
flagged. It caught the zenith-
angle header defect below before anything else did, because a fit absorbs a
slant-path error into the columns and its residual does not move.
`validate_igrins_transfer.py` builds its calibration through the same class,
so the transfer test tests the production code.

`fit_igrins_standard.py` takes **several `--spec` frames at once** and loops
orders outside, frames inside. That is worth 70% of the runtime and is specific
to this instrument: within a night the PLP uses one wavelength solution, so
order N covers a bit-identical wavenumber range in every frame (measured spread
across ten frames: 0.0000 cm-1), and across nights it moves at most 0.16 cm-1
against a 5 cm-1 grid margin. Everything expensive -- grid, lines, opacity,
precompute, and both XLA compilations -- therefore belongs to the order, not the
frame. Profiled before this: 13.1 s of a 31.7 s order-frame was compilation
(4.9 s gradient, 8.2 s Hessian), the Hessian half invisible because `fit_order`
computed the covariance unconditionally and the lazy `jax.hessian` compile
landed on stage 1. `OrderObjective` now takes flux, mask, uncertainty and the
zenith angle as jit **operands** rather than captured constants, so one
executable serves every frame; `rebind` checks the wavelength grid and source
match first, and `fit_order(covariance=False)` lets the intermediate stages skip
the Hessian entirely. Do **not** add warm-starting across frames: it would pull
each fit toward its neighbour and shrink the very frame-to-frame scatter the
airmass ladder measures.

Three things differ from the Arcturus driver and all three are load-bearing.
The instrument is the **default Gaussian**, not `BoxcarFTSInstrumentProfile` --
a grating spectrograph's LSF is the fitted `lsf_sigma_kms` itself, so there is
no MOPD to measure. `pixel_integration` is `"simpson"`, because IGRINS pixels
integrate while the FTS point-samples. And `igrins_spectral_order` **normalizes
the order** to `continuum_level`: the continuum's constant term is a log flux,
so raw PLP counts put it near 10, its bound has to span the counts scale, and
L-BFGS-B's rescaling onto that bound drives the continuum to zero (measured:
220 sigma against 2.7). The continuum degree is 9, not 3 -- an order spans
77-96 cm-1 and carries the blaze.

The dominant residual is a **repeatable instrument response**, not the line
spread function. |residual| in continuum units triples between the middle third
of an order and the last sixth, and by the same factor in orders with no
telluric absorption at all. Across a night it is the *same* pattern: 56-76% of
each frame's residual variance is common to all ten frames of five different
stars over airmass 1.07-2.50, its amplitude does not grow with airmass, and it
is fixed in detector coordinates (stars separated by tens of km/s could not
correlate at r~0.7 unshifted). `leave_one_out_patterns` (`igrins.py`) measures
it per order from a night's frames and the driver runs **two passes**: fit,
build each frame's pattern from the *others*, divide flux and uncertainty by
1 + pattern, refit. Dividing the data equals multiplying the model, so the
forward model is untouched. Worth 1.9 to 1.30 sigma on the H ladder, 1.8x in
runtime, and it needs at least five frames.

**Two guards, and the second is the one that is easy to miss.** Never build a
frame's pattern from that frame -- it would fit that frame's noise. And always
smooth the pattern (`--pattern-smooth-pixels`, default 51): leave-one-out does
nothing about a systematic *every* frame shares, and our own telluric model
error is exactly that, since every frame looks through the same sky with the
same line list. Measured: the unsmoothed pattern's high-frequency component
correlates with absorption depth at r = +0.44 and scales with how much
absorption an order has, while the smooth component correlates at +0.05.
Splitting orders by whether they have lines shows what each half is worth --
line-free orders go 2.16 -> 1.34 sigma smoothed and 0.80 unsmoothed, absorbing
orders 2.28 -> 1.94 -> 1.02. The extra in absorbing orders is the line list.
Running with `--pattern-smooth-pixels 0` reports 0.80 sigma instead of 1.30 and
makes the residual meaningless as a test of the telluric model.

Two things that looked like the cause and are not: the continuum coefficient
bound was binding on 218 of 248 order-frames yet loosening it changes the
residual by 0.01 sigma (it is 5.0 now so `at_bound` stays meaningful); and the
fitted LSF really does narrow by 10-33% along an order, but freeing it per
segment is worth a median 1.02. `throughput_floor` is back to 0.25 -- it used to
carry the red edge and no longer has to.

Fitting with `--stellar <a0v npz>` adds a **stellar stage** (`stages_for()`):
these are different A0V stars with radial velocities tens of km/s apart, so
`stellar_velocity_kms` must be fitted or the Brackett lines land in the wrong
place. Fitted velocities repeat per star to 1-2 km/s, which is a check the flat
source cannot give. The A0V ladder fits 41% more H pixels than flat+mask,
including the Brackett cores, and the residual moves by 0.03 sigma.

`mask_hydrogen_kms` is 600 and that is **too narrow**: measured against the A0V
model, a Brackett line is still 6-8% deep at that offset, and only reaches 1% by
±1800 km/s. It does not show in the residual -- the degree-9 continuum absorbs a
broad wing -- so it biases the continuum and the columns instead, by a few
percent. Widening it is not the fix: ±1800 km/s is ±880 pixels per line and
leaves 269 pixels of an order. Use `--stellar <a0v npz>`, which lifts the mask
entirely and keeps 60% more pixels at the same residual. The bias is common to
every frame of a night, so it cancels in an airmass slope but not in an absolute
column.

Three nights are fitted: DCT 2018-12-20 (~2 mm PWV), McDonald 2017-04-20
(~10 mm, and the only one that exercises the degF/inHg branch end to end) and
DCT 2016-12-08 (~7 mm). **Quote a slant-path bound from the scatter between
nights, not from one night's error bar**: the three give CH4 slopes of +0.010,
+0.046 and -0.027 per unit airmass, chi-squared 7.7 on 2 dof, so the 2 sigma
one night shows is a per-night systematic. CO2 does the same (8.2). On DCT 2016
both show a 3.5-sigma dependence on *time of night*, which neither can
physically have. The honest bound is the 3-4% night-to-night scatter. These are
the numbers after the zenith-angle header fix, which halved the within-night
CO2 scatter on DCT 2018 but left the between-night disagreement in place (7.1
and 6.8 before) -- the defect was not the systematic. Separately, the response
pattern **is** stable: DCT 2016 and 2018 agree at median r = +0.943 across 26
orders, their difference only 34% of the pattern, so nine tenths of it belongs
to the instrument rather than the night. The
wetter night fits about 1.4x worse at *every* transmission level, not only in
deep lines, which points at the water continuum and the weak-line forest rather
than at line depth alone. Two operational lessons from it: an order that is
opaque end to end (2.0 um CO2 at high water) is now skipped rather than writing
a NaN row (`minimum_reliable`); and the precipitable-water seed should be within
about a factor of two of the truth, because `precompute_opacity` linearizes
self-broadening about the profile's own water and a x1.7 scaling sits outside
the +-25% it is built for. (Reseeding did not improve this night's residual, so
that was not the limit here.) Do not choose it by eye: `igrins_site_profile.py`
reads a night's frames and fills in all three of `make_site_profile.py`'s
surface arguments, taking the column from the dewpoint via
`precipitable_water_mm` -- an exponential profile with an *effective* 1.4 km
scale height, which is not the 2.0 km the layer profile uses and is calibrated
on three nights. Worst case 1.6x against 2.0x by hand.

`scripts/era5_site_profile.py` replaces the analytic profile with a real one
from ERA5, writing the same CSV with the same layer edges so only T(z) and q(z)
change. It anchors at the **station pressure from the frame's own header** and
integrates upward, which is why the 0.25-degree cell's orography (1844 m where
DCT is at 2360 m) does not matter. Measured: ERA5's water column agrees with the
fitted one to 0.4% and 4.5% on the two DCT nights, and the real lapse rate is
4.5-8.3 K/km against the assumed 6.5. It **reduces CO2's inter-night
inconsistency to borderline** (chi-squared 8.2 to 5.7 on 2 dof, p 0.02 to 0.06;
it read 6.8 to 3.4 before the zenith-angle fix) but does not fix CH4 and does
not improve the residual at all. The strongest reason to use it is coverage: 31%
of the archive (Gemini South from 2020) carries no weather cards, and ERA5 needs
only a position and a time. Two backends -- ARCO-ERA5 on Google Cloud needs no
credentials and costs 150 MB a profile because its chunks are global; CDS needs
`~/.cdsapirc`, charges by fields (times x levels x variables) and is 4x faster
and 1.5 kB a profile once batched, with a limit near 8,000 fields a request.

The response pattern belongs to the **spectrograph, not the telescope or the
night**: IGRINS moved McDonald -> DCT -> Gemini South, its order centres agree to
0.04 nm across four nights spanning 2016-2021, and the patterns correlate at
median r = +0.89 to +0.95 for every pair, including McDonald 2017 against Gemini
South 2021. One master pattern captures 74-88% of each night's own. So a single
pattern measured once can serve nights with too few standards and the science
targets, which have none.

ERA5 also supplies the **station pressure** via `station_pressure_from_era5`,
matching the header's `BARPRESS` to 0.2% on the three nights that carry it.
That is what makes the no-weather nights work at all, since the column has to be
anchored somewhere; `--anchor auto` prefers the header and falls back.

An airmass ladder is confounded by its targets: a night observes few stars, each
over a limited airmass span. On the DCT night chi Cap is the only target above
airmass 1.82. `analyze_igrins_ladder.py` reports `leave_one_object_out` for this
reason. Quote a slant-path bound only over the range several stars cover.

`igrins.py` reads the RRISA reduced products. Two of its rules exist because
the archive is not self-describing. `surface_conditions` normalizes the weather
cards from the `TELESCOP` card -- McDonald is degF/inHg, DCT is
sea-level-reduced hPa, Gemini South is station hPa, and nothing in the file says
which -- then checks the result against the site's hydrostatic expectation and
raises past 8%, so a convention change fails instead of placing the observatory
at sea level. And the PLP's own `MASK` is **not** used (it flags 56% of H and
76% of K by its own flattening criterion); the throughput cut at a quarter of
peak, on the *smoothed* flux, is ours and is asymmetric because the blaze
roll-off is. `reduced_log.csv`'s `AM` column writes `-1` for missing and carries
impossible values -- the airmass comes from the header. And the header's *end*
cards are not always the exposure's: in 9 of 111 archive files, all K band,
`ZDEND`/`HAEND`/`DATE-END` describe another frame (`DATE-END` precedes
`DATE-OBS` in most), while the H file of the same exposure and every `ZDSTART`
are right. Averaging a foreign `ZDEND` moved the K slant path by up to 6% --
including the airmass-3 top of the McDonald ladder and a DCT 2018 calibrating
standard -- so `zenith_angle_deg` uses `ZDEND` only if the sidereal rate over
the exposure could have produced it, and `ZDSTART` alone otherwise. Every K run
fitted before this fix carries at least one such frame.

**An IGRINS order is named by its physical echelle order, never its row.**
`H109`, `K71`; `order(109)`; `--orders 109`; the record's `order_number`. Row
position means nothing across files: K ships 24, 25 or 26 rows depending on the
reduction, the PLP's `invert_order` reverses rows for wavelength-ascending
output, and a custom extraction range drops the WAT cards altogether.
`identify_orders` takes the order from each WAT2 `specN` entry's `beam` field,
matching entries to rows **by wavelength, not position**, checks every match
against `IGRINS_ORDER_CENTRES_UM` (centres repeat to 0.15 nm across 2014-2021;
neighbours are 11 nm or more apart), and falls back to that table alone when
the cards are missing or describe other rows -- which is not hypothetical: the
test fixture was cut from a 28-order frame, kept its WAT, and so read as orders
98-99 when its rows are 100 and 109. `order_source` records which path named
the rows. Products written before this named orders by row (`H05`,
`order_index`); `scripts/migrate_igrins_order_names.py` relabels them from each
frame's own header, checks every renamed npz against the order's wavelength,
and is idempotent. The PLP's own reader and writer for these cards are in the
untracked `igrins_aux/`, for reference only.

What the two bands can measure, from the peak vertical optical depth reached
anywhere in an exposure: H2O and CO2 and CH4 yes; CO and N2O peak at 0.115 and
0.125, just under the 0.15 the ladder analysis requires; **O2 is exactly zero in
both bands** and is not measurable with this instrument at all, its near-infrared
bands being at 0.76 and 1.27 um. `analyze_igrins_ladder.py` merges H and K of one
exposure, and merges order-shards of one band, by the date and frame number in
the filename -- sharding one band by order needs `--summary-suffix`, or every
shard writes the same per-frame summary and they overwrite each other.

`scripts/generate_payne_zero_a0v.py` synthesizes the A0V source, in Payne Zero's
own environment for the same reason the Arcturus one does. Its metadata reports
`atmosphere_converged: false`, and at 9,500 K the hydrogen lines are the whole
spectrum, so this was expected to be the weak link -- measured, it removes a
factor of six from the residual inside the Brackett windows and leaves 2.33
sigma, at the floor the rest of the band reaches. Use `--stellar flat` when the
point is to measure the atmosphere, since it depends on no stellar model at all;
use the A0V model to recover the 22% of H-band pixels the hydrogen mask discards.

**A science frame's telluric model can come from the night's standards, but not
by interpolation alone** (`scripts/validate_igrins_transfer.py`,
`docs/igrins_transfer.md`). Held out one at a time over five night-bands, a
standard given the others' per-order columns and LSF, and water linear in time,
fits within 2-9% of its own full fit in the median, but absorbing orders are
40-90% worse at the 90th percentile and the correction is off by 1.2-1.8x the
noise: water moves on timescales the standards do not sample (13% between
standards 42 and 78 minutes apart). Adding **one velocity shift and one water
scale per frame** -- the median over orders -- recovers nearly all of it: under 2%
in the median, 0.3-0.8x the noise in absorbing orders. The water error is the
frame's, not the order's (0.04-0.08 in log column before the frame shift,
0.009-0.014 after), and H and K measure the same shift independently at
r = +0.95-0.97, so it can be measured in whichever band the target's lines spoil
less. With a target's own lines injected (`--inject`, Arcturus at +37 km/s),
**per-order water becomes unsafe** -- each order's is pulled by its stellar
lines -- while the per-frame scale survives with a ~2% floor: biased -1.5% in H
and +1.8% in K, so the bands disagreeing by ~3% is the warning sign. Clipping
pixels 3 sigma below the model (`--clip-sigma`) fixes H and not K, whose CO
blends and weak-line forest never cross the threshold. Separately, the
per-order `stellar_velocity_kms` rails at +-60 in 42% of the *full* fits'
order-frames in a pattern set by the order, not the star: each order's offset
from its frame's median repeats across four nights (H115 +45, H119 +39, H106
-24 km/s) and follows whether its Brackett line sits at an edge of the usable
pixels, where one wing is fitted against a degree-9 continuum. The free shift is
absorbing a wing mismatch, so do **not** pin one velocity per frame to "fix" it
-- that moves the mismatch into the residuals. A star's velocity should come
from the central-line orders (H98, H103, H109, H113, H120); the real fix is the
wing, via a fixed master blaze.

`scripts/export_transmission_hdf5.py` writes the unconvolved transmission on the
model's own grid -- 4 samples per resolution element, no interpolation -- for a
consumer who wants the atmosphere as a multiplicand inside *their* synthesis
rather than our corrected spectrum. It rebuilds per **window**, not per row: both
epochs of a page share one (v1, v2), so 598 rows need 310 grids, line
selections, opacity backends and compilations. `--check` interpolates back to the
pixels and reproduces the cached `transmission` to 3e-16.

Three things the atlas ILS is not. **`mopd_cm` varying by a factor of 5 across
the atlas is correct, not a defect**: the compilation holds a constant resolving
power, so the path difference must track 1/nu, and it does -- corr(mopd, nu) =
-0.93, and the residual scatter about the constant-R law is 27% p16-p84, not the
factor of 3 the raw spread suggests. **The atlas's documented R = 100,000 is
right and the sinc alone does not measure it**: R from the measured MOPD is
115,700, but the fitted Gaussian carries the rest and the two in quadrature give
100,504. Quoting the sinc alone reports ~117,000 and looks like a discrepancy.
**The remaining +-12% scatter in the measured MOPD is what the per-page Gaussian
is absorbing**, which is why `lsf_sigma_kms` is the dominant at-bound parameter
(13.4% of converged pages, 76 of them railed at the 0.05 minimum, against 22.2%
at-bound overall). `fit_arcturus_page.py --sinc-resolving-power 115700` replaces
the per-page measurement with the atlas-wide law. Measured on 16 pages spanning
1868-10951 cm-1: the median residual moves by 0.2% (5.223 to 5.266 sigma, worst
page 1.8%) and `lsf_sigma_kms` at bound falls from 7/16 to 4/16. One fewer
free-floating per-page input at no cost in fit quality, so prefer it for new
runs; the committed record predates it.

`stellar_only / continuum` is **not** the source normalized to its continuum and
must not be read as one. `prepare_stellar_source` normalizes by the *median* over
the page, which is exactly degenerate with the Chebyshev's constant term and so
free to the fit, but it puts the unity level about 1.3% below the true continuum
where the star is blanketed. Measured over 1500-1540 nm: mean depth 0.0278 for
`stellar_only / continuum` against 0.0431 for the Payne Zero source against its
own `flux_continuum` -- but renormalizing that same source by its median gives
0.0304, so most of the gap is the zero point and only about 9% of it is anything
the continuum absorbed. The visible consequence is that ~16% of
`corrected / continuum` pixels sit above 1.02 at 1.5 um. Separately and still
unexplained: heavily blanketed pages retrieve 3-7% less water than lightly
blanketed ones, which survives a wavelength control (partial r = -0.19 summer,
-0.11 winter) but is **not** fixed by `--continuum-anchor 0.98` (median H2O shift
+0.4%, residual 1.1% worse), so the continuum-source degeneracy is not the cause.

`scripts/generate_payne_zero_arcturus.py` makes the stellar source and does **not**
run in this environment: Payne Zero needs Python >= 3.11 while this package is pinned
to 3.10 by `exojax==2.5.0`. It runs in Payne Zero's own venv and writes an npz that
`StellarSpectrum.from_npz` reads back; the script's docstring has the invocation.

Benchmarks and the offline LBLRTM workflows. All need `./scripts/bootstrap_lblrtm.sh`
to have been run first; `benchmark.py` and `validate_mixed_precision.py` use only the
AER line files it downloads, the rest also execute the LBLRTM binary:

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark.py --platform cpu --output benchmarks/results/cpu.json
UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark_fit.py   # staged-fit cost -> docs/precomputed_opacity_results.json
UV_CACHE_DIR=.uv-cache uv run python benchmarks/compare.py baseline.json candidate.json
UV_CACHE_DIR=.uv-cache uv run python scripts/run_lblrtm_reference.py      # rebuild tests/data fixture
UV_CACHE_DIR=.uv-cache uv run python scripts/build_lblrtm_correction.py   # rebuild data/corrections template
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_aer_co.py           # writes tests/data/aer_co_validation.json
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_mt_ckd.py           # writes docs/native_mt_ckd_validation.json
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_mixed_precision.py [--resume]
```

## Two toolchains

- `.venv/` — the `uv` Python environment for the package.
- `.conda-lblrtm/` — a conda-forge gfortran/netCDF-Fortran toolchain used only
  to build LBLRTM/LNFL. Never mix the two. `scripts/bootstrap_lblrtm.sh` creates
  it, clones and builds LBLRTM 12.17 + LNFL, downloads AER Line File 3.9, and
  runs LNFL to make `data/lblrtm/run_lnfl_igrins/TAPE3`.

Scripts hardcode the bootstrapped paths relative to the repo root:
`data/lblrtm/LBLRTM/lblrtm_v12.17_linux_gnu_sgl`,
`data/lblrtm/run_lnfl_igrins/TAPE3`,
`data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc`,
`data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule/`.
All of `data/lblrtm/`, `data/databases/`, `data/corrections/`, and
`benchmarks/results/*` are gitignored — only compact fixtures under
`tests/data/` and JSON reports under `docs/` are committed.

The package was `jax-telluric` until 0.2.0. The import is now `tellurix`, the
console script `tellurix-download-data`, and the environment variable
`TELLURIX_DATA`; `download.py` still honours `JAX_TELLURIC_DATA` and a non-empty
`~/.local/share/jax-telluric` so an existing download is not orphaned. Records
written before the rename carry `physics["jax_telluric"]` rather than
`physics["tellurix"]` -- that is the honest provenance of those runs and nothing
reads the key back, so it is left alone.

For installed users, `tellurix-download-data {mt-ckd,aer-lines,all}`
(`download.py`) fetches SHA-256-pinned copies into `$TELLURIX_DATA` or
`~/.local/share/tellurix`.

## Architecture

The forward model is composed from three duck-typed seams declared as
`Protocol`s in `model.py` — `OpacityBackend`, `ContinuumBackend`, and
`CorrectionBackend`. Anything satisfying them can be swapped in, which is how
tests run without spectroscopic data.

```
AtmosphereProfile (types.py)      layered, top-to-bottom, validated in __post_init__
        +
OpacityBackend                    ExoJAXOpacityBackend (exojax_backend.py) -> OpaDirect /
                                  SparseCoreDirect (direct.py) / OpaPremodit;
                                  ArrayOpacityBackend for fixtures
        + optional ContinuumBackend  MTCKDWaterContinuum (mt_ckd.py), ReferenceWaterContinuum
        + optional CorrectionBackend LBLRTMOpticalDepthCorrection (corrections.py)
        ->
TelluricModel (model.py)          .transmission()  = exp(-sum_layers tau / cos z)
                                  .predict(order, params) adds velocity/stretch, LSF
                                  convolution, Simpson pixel integration, Chebyshev
                                  log-continuum
        ->
fit_order (fit.py)                L-BFGS-B over jax.value_and_grad, bounded and rescaled
```

### What a fit costs

XLA compilation happens once per distinct jitted function and costs seconds,
while one evaluation costs milliseconds, so a staged fit is dominated by how
many times it compiles. Two seams exist for that, and both drivers use them:

- `OrderObjective(model, order, continuum_size)` compiles once over the *full*
  parameter vector and is passed to `fit_order(..., objective=...)` for every
  stage. Bounds, the free set, and the rescaling stay outside the compiled
  function, which is why they may change between calls. Without it `fit_order`
  builds a fresh `jax.jit` closure per call and recompiles the same graph.
- `TelluricModel.precompute_opacity()` evaluates the line-by-line kernel once
  and carries it as fixed arrays, expanded to first order in the
  self-broadening partial pressure (`LinearizedOpacityBackend`) -- the only
  route a fitted parameter takes into the kernel, and a weak one. Column scales
  stay free and differentiable. `self_broadening="frozen"` drops that term and
  is 70x less accurate; a fit using it must be repeated from its own result.

Measured together: 61 s to 3.3 s for a three-stage fit, 8.1x end to end
including the precompute, with fitted parameters agreeing to 1.8e-5.

Two traps follow from this. **Never call the forward model eagerly on a live
opacity backend** — an uncompiled kernel dispatches operation by operation and
one call costs more than the whole fit (measured: 37 s of a 57 s page). For
final products, call `precompute_opacity(fitted_parameters)` instead: refreezing
at those parameters is *exact* there, because the expansion's offset from its
own reference is zero, and it reuses the evaluator cached on the model. And
**`velocity_step_kms` is the same for every page** at a fixed resolving power
and sampling, which is what makes one grid spacing shared across the atlas.

### Two margins, not one

`igrins_wavenumber_grid(..., margin_cm1=25)` pads so that outside lines
contribute their wings — a property of the *line list*. The grid only has to
cover the window plus what the model reaches back for (LSF kernel, Doppler
shifts, instrument edge padding), under 2 cm-1 anywhere in this atlas. Sharing
one margin put 70% of every grid where there is no data, and 90% on the narrow
2 micron pages. Select lines over `(v1, v2)` with the wide margin, then cut the
grid with `trim_wavenumber_grid(grid, v1, v2, 5.0)`, which trims rather than
regenerating so the samples keep their spacing *and* phase. 3.3x fewer grid
points atlas-wide; maximum pixel-flux difference 2.5e-4 against 5.5e-3 noise.

`select_significant_lines(database, profile, species, budget)` is available and
off by default: it discards the weakest lines whose bounded contributions sum to
`budget` of optical depth, which bounds the transmission error by the same
amount. It buys little once the opacity is precomputed — see
`docs/performance.md` for when it does.

Observed data and the stellar source enter through two adapters: `atlas.py`
(`read_arcturus_page` / `arcturus_spectral_order`, a self-contained reader for the
Hinkle, Wallace & Livingston 1995 IR atlas) and `stellar.py` (`StellarSpectrum`,
`prepare_stellar_source` — resample onto the model grid, broaden, normalize).

A source spectrum may be given on the pixel grid (`SpectralOrder.source_flux`) or on
the model's high-resolution grid (`source_flux_model_grid`); only the latter can
represent structure narrower than a pixel, and only it activates the separate
`TelluricParameters.stellar_velocity_kms`. `InstrumentProfile` is a fourth seam:
the default is the built-in Gaussian, and `BoxcarFTSInstrumentProfile` applies an
unapodized FTS sinc by FFT (a truncated kernel is invalid for a sinc at any width).
`pixel_integration="point"` turns off Simpson pixel averaging, which is wrong for
point-sampled FTS data.

Line data reaches ExoJAX through `AERLineDatabase` (`aer.py`), a minimal
HITRAN-shaped adapter over AER 100-character per-molecule line files (it
exposes `nu_lines`, `logsij0`, `gamma_air/self`, `n_air`, `delta_air`,
`qr_interp`, `dbtype="hitran"`). The files carry no index, so it builds one per
file in numpy — offset, molecule id and wavenumber of every possible record —
memory maps the file and caches both by path, size and mtime. That index is a
**superset**: anything it cannot classify with certainty stays a candidate and
the ordinary per-line parser still makes every accept/reject decision, which is
what keeps it byte-identical. Keep that property if you touch it. Note AER
right-justifies the molecule field (`" 2"`, not `"02"`); reading only two-digit
forms silently matches nothing and is *slower* than no index at all. LBLRTM inputs/outputs live in `lblrtm.py`
(TAPE5 writer, subprocess runner) and `reference.py` (binary TAPE12 reader for
the GNU single-precision build, resolution degradation, `compare_transmission`
metrics). `io.py` loads CSV layer profiles and TelFit/RFM MIPAS level profiles.

### accuracy_mode

`TelluricModel(..., accuracy_mode=...)` selects the physics, and the
constructor enforces a strict combination matrix:

- `"fast"` (default) — lines only, plus an explicitly supplied `continuum`;
  a `correction` is rejected.
- `"mt_ckd"` — exactly one of `continuum` (runtime `MTCKDWaterContinuum`,
  differentiable across pressure/temperature/abundance) or `correction`
  (only its isolated H2O self/foreign terms via `mt_ckd_optical_depth`).
- `"lblrtm_corrected"` — requires a `correction` and forbids `continuum`;
  adds the continua plus per-species empirical line residuals and the fixed
  reference background. Valid only for the exact profile and grid the template
  was built for; `validate()` re-checks the profile and
  `validate_opacity()` refuses a backend without `pressure_shift=True` when the
  template needs it.

### Performance path

`SparseCoreDirect` subclasses `OpaDirect` and statically splits line/grid pairs
into a compact core list, keeping ExoJAX's exact formulas and branch selection.
Outside its configured temperature (default 150–400 K) or pressure bounds it
falls back to full Direct via `jax.lax.cond`.
Nothing of shape (line, grid) is stored: ExoJAX's dense offset matrix and the
core/wing test are exact functions of two 1-D vectors and are recomputed inside
the kernel, which cut what the calculators hold from 426 MB to 1.8 MB and the
opacity compile from 22 s to 4 s on the Arcturus window, bit-for-bit. Keep it
that way — `tests/test_direct.py` asserts no 2-D array survives construction.
The kernel is compute-bound, not bandwidth-bound, so steady state is unchanged.
XLA now fuses the whole wing sum into one reduction and never materializes it,
so **`layer_chunk_size` is obsolete**: peak memory is the same at every chunk
size while chunking multiplies compile time (40.8 s at chunk 2 against 8.0 s
unsplit). Both drivers default to no chunking; the option remains for a grid
fine enough to break that fusion. Do not reach for `vectorize_layers=False`
either — a Python layer loop unrolls the whole calculation per layer (24 s
chunked against ~400 s looped). A JVP through `xsvector` is worse than both:
`vmap` turns the out-of-range `lax.cond` into a select that evaluates the dense
fallback branch, which is why `precompute_opacity` uses a central difference.
`mixed_precision=True` evaluates only the asymptotic wing value and its
derivative in float32 — the JVP is algebraically rewritten to avoid the
catastrophic float32 cancellation in ExoJAX's identity. Coordinates, cores,
line physics, accumulation, and the whole instrument model stay float64.
`pressure_shift=True` adds HITRAN `delta_air` shifts scaled like LBLRTM's
RHORAT (`delta_air * P/1 atm * 296 K/T`); ExoJAX 2.5 omits these, so it is
opt-in to keep the default numerically identical to ExoJAX.

## Conventions

- `tellurix/__init__.py` enables `jax_enable_x64` at import. Import
  `tellurix` **before** ExoJAX so calculators are built in float64.
- Public data classes are frozen dataclasses/NamedTuples that validate and
  normalize in `__post_init__` (species names uppercased, arrays coerced,
  physical ordering checked) and raise `ValueError` with a short message.
  Follow that pattern rather than validating at call sites.
- Atmosphere arrays are ordered top-to-bottom: pressure edges increase,
  altitude decreases. Wavenumber grids are strictly increasing and evenly
  spaced in log wavenumber (constant velocity step) — `igrins_wavenumber_grid`
  builds them with padding for line wings.
- Units are in names: `_bar`, `_hpa`, `_cm1`, `_km`, `_kms`, `_k`, `cm2`.
- Comments explain *why* (a numerical or LBLRTM-compatibility reason), not what.
  Keep that density; the existing comments are load-bearing.
- Accuracy and performance claims are backed by a committed JSON report under
  `docs/` plus a prose companion (`docs/validation.md`,
  `docs/performance.md`, `docs/lblrtm_corrected_mode.md`,
  `docs/mixed_precision_validation.md`). `tests/test_lblrtm.py` asserts on
  `tests/data/aer_co_validation.json` and `docs/native_mt_ckd_validation.json`,
  so regenerating those reports requires a working LBLRTM build. Update the
  numbers in README/docs prose whenever a report is regenerated.
- `mt_ckd.py` carries an AER copyright notice; keep it with any derived code.
- `FitResult.covariance` now inverts the objective's actual Hessian over the
  parameters that were free and not on a bound (`_uncertainty` in `fit.py`); the
  old `hess_inv` value was a line-search byproduct and was out by two orders of
  magnitude. It is a *formal* covariance and still must not be quoted as an
  uncertainty — it assumes independent pixel errors, and this residual is
  dominated by correlated stellar error, so it comes out 3–9x too small
  (measured: 0.76% formal on the water column against 6.5–9% of sub-window
  scatter). Use `FitResult.correlation` for degeneracies, which survives a wrong
  noise model, and an empirical study for an actual error bar.
- A run's product is its HDF5 record (`src/tellurix/record.py`), not the
  `.npz` arrays, which are a cache. `docs/using_the_corrected_spectra.md` is the
  user-facing guide -- point a consumer there, not at the development records in
  `arcturus_fit.md` or `igrins_a0v.md`. `scripts/export_spectra_hdf5.py` packs a
  run into one self-contained file readable with h5py alone (12.4 MB for the whole
  atlas against 45.6 MB of npz); it is gitignored and produced on demand, and it
  names the correction operator `effective_transmission` so the unconvolved-
  transmission trap is hard to fall into from the file. `write_record` takes `key_fields` naming the
  columns that identify a row -- `("page", "epoch")` for the atlas, `("frame",
  "order")` for an IGRINS night -- and `extra_columns` for what the shared
  schema has no place for. A file written before that was configurable reads
  back as `("page", "epoch")`, so the committed `arcturus_atlas.h5` still
  works. Two traps when adding a pipeline: an order's parameter vector is not
  the run's when the species present depend on the window, so remap each row's
  sigma and correlation into the union rather than assuming one length; and
  `ils_fingerprint(None, ...)` is the built-in Gaussian branch, for a model with
  no `InstrumentProfile` object.
  `scripts/rebuild_arcturus_page.py --check` reconstructs a page from the record
  alone and agrees to 1e-13. It shares no code with the fitting driver on
  purpose — an independent reconstruction is the evidence; calling one function
  twice is not. Keep it that way.
- Commit subjects are short imperative lines with no body (`Add native JAX
  MT_CKD continuum`).
