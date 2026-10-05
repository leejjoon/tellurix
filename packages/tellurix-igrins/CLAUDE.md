# CLAUDE.md -- tellurix-igrins

Notes for the IGRINS pipelines: A0V standards, a night's calibration, and
science frames corrected against it. The package (`src/tellurix_igrins/`) holds
the reader, the calibration, the lamp-flat blaze and the per-order fit
(`standard.py`); the drivers that run them are in the repository's `scripts/`,
and the core's notes -- commands, architecture, what a fit costs, conventions --
are in the root `CLAUDE.md`.

## Standards, calibration and science frames

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
costing nothing elsewhere. Those frames are long sequences of setting stars, so this was
the sequence airmass seen from the science side (below); recheck it once the
calibrations use sequence airmasses. Keep it **tied** -- freed separately, a line-rich
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
star. On DCT 2018 the blended blaze is better in both bands -- per-pixel z
1.432 -> 1.149 in H, 1.576 -> 1.412 in K, almost all at the order ends -- and
leaves a third of the pattern. **Compare continuum models by per-pixel z**
(`residual_z_rms`, now in every summary and record), not by
`residual_rms_over_noise`: that divides by the *median* uncertainty, so a blaze
reweighting the order ends made it read K 8% *worse*. What is really left in K is
+2-3% inside deep lines, a Gaussian-LSF trade (the blaze fit balances the cores
with a 1.7% wider LSF; both miss cores by ~3 sigma) -- not the blaze, not
flexure, not pattern leakage. Degree 5 on top of the blaze is worse than 9.

**What is left after blaze and pattern depends on the night's S/N, and some of
it on the epoch -- do not generalize from one night**
(`docs/igrins_residual_structure.md`, `analyze_igrins_residual_structure.py`).
At S/N ~150 (DCT 2018) line-free pixels fit at z = 0.98 (H) and 0.83 (K) and the
wiggles show only after binning. At S/N 400-900 (McDonald 2015-12-01 and -03)
line-free pixels fit at 2.4-4.6 sigma, carried by three things. (1) **Bad
pixels that recur between nights**: under 15 pixels, frames share their
residual at r ~ +0.87 in H, and half of that is the top 1% of pixels -- isolated
columns that spike on every night (95% shared between the two 2015 nights,
92-94% with DCT 2018, fewer on later nights), not lamp, telluric or stellar.
**Mask them** with another night's spike pixels (`pixel_scale_spikes`): it does
as well as subtracting that night's template and never hurts where a template
does (K from DCT 2018). (2) **A detector patch noisy in some exposures**: H
columns >~1536 over H106-H120 at z 15-26 in one frame, at the same place on both
2015 nights; K columns ~130-640 over K80-K83 -- not count level, not OH, not
telluric; needs the 2-D frames. (3) **The 2.75-2.80 cm-1 etalon fringe** (n*d 1.8 mm): in H it is
~0.15% on both epochs with a phase common to all frames; in **K in 2015 it is
~1% (lamp to 2.2%), four to six times DCT 2018's, and its phase changes between
pointings**, so a template from other frames can double the damage and it must
be fitted per frame (K 0054: 3.30 -> 1.94 sigma). The blaze's filters copy
17-32% of the lamp's fringe into the star. **A fit cache (npz) runs in
ascending wavenumber, `IGRINSOrder.pixel` in ascending wavelength** -- map a
cached array to detector columns by wavenumber, never by position.

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
DCT 2016-12-08 (~7 mm). With first-exposure airmasses they gave CH4 slopes of
+0.010, +0.046 and -0.027 per unit airmass (chi-squared 7.7 on 2 dof), CO2 the
same, and a 3.5-sigma dependence on time of night; **that was the exposure
sequence's airmass** (below), and with sequence airmasses the three agree at
chi-squared 0.6 (CO2) and 2.0 (CH4), weighted slopes -0.0016 and +0.0006. Separately, the response
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

**A PLP spectrum is a sequence of exposures; its header is one of them**
(GitHub #1, `docs/igrins_a0v.md` "What moves the well-mixed columns",
`analyze_igrins_dry_systematic.py`). The catalog's `FILES` lists 4-10
exposures combined into one spectrum, up to 37 minutes, while `DATE-OBS`,
`DATE-END` and `ZDSTART/ZDEND` describe the first; at airmass 3 a setting star
gains 9% of air over the sequence. Uncorrected, every gas in the exposure is
scaled by that factor -- CO2 and CH4 in H and K moved together by up to +-6%,
which read as inconsistent airmass slopes between nights and a "time of night"
trend; the fit absorbs it into the columns and the residual never moves.
**Run `igrins_pointing.py --spec ... --sequence --write all` on every night
before fitting**: it writes `pointing.json` with the mean-airmass zenith
(`sequence_zenith_angle_deg`), the overhead taken from the catalog's `JD`, which
is the sequence midpoint (24-46 s per exposure; DCT 2018's JD is off by a
constant 241 s, so it falls back to the archive's 38 s). Predicted against the
measured factor: r = 0.995 and 0.992, slope 0.97 and 1.09, no free parameter.
Done for DCT 2018, McDonald 2017 and DCT 2016 (`seq_*` runs); everything else
-- ERA5 runs, calibrations, master pattern, transfer and science runs, Gemini
2021, McDonald 2015 -- still has first-exposure airmasses. `--dry-shift` was
this effect seen from the science side and may now be unnecessary; measure it.
Ruled out on the way, do not redo: blaze, fringe, veiling, a per-frame
atmosphere, nonlinearity, the response pattern, a water/dry trade
(`--fix-columns-from` pins species at a record's per-order night median).
That script reads only `record.h5`; **the K run
directories still hold the pre-zenith-fix shard summaries** (`*.s0_summary.json`)
beside the refit's merged ones, which `analyze_igrins_ladder.py` now skips --
anything else that globs `*_summary.json` must too.

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
fitted before this fix carries at least one such frame. **McDonald 2015 breaks
both rules further.** Its weather is Celsius and station hPa where 2017 is
Fahrenheit and inHg, so `Site.alternatives` lists both and `surface_conditions`
keeps whichever the frame's own numbers support (hydrostatic pressure, and
temperature/dewpoint/humidity agreeing by Magnus); four -1 cards are missing
weather. And 12 of its 13 frames carry a zenith distance that is -1 or 2-24 deg
wrong, with the catalog's RA/Dec empty or wrong too: `scripts/igrins_pointing.py`
resolves the target by SIMBAD name, computes the zenith distance from geometry
(`geometric_zenith_angle_deg`, 0.2 deg from the header on every later night) and
writes a `pointing.json` beside the frame, which the reader prefers and records
as `zenith_source`. Run it in check mode on any new night before fitting.

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

**`--line-coupling`** (`StandardFitSettings.line_coupling`, recorded in the
run's settings, physics and record config, restored from a calibration's config
by the science fit) applies AER's first-order CO2 line mixing. On DCT 2018 it
takes K89, the 2.0 um band centre, from z 1.97 to 1.68 in all ten frames, costs
1-3% in K86 and K92 at the band edges -- where tellurix still matches LBLRTM's
coupling to 0.03-0.06%, so the loss is AER's, not ours -- does nothing in H, and leaves every CO2
column and airmass slope within 0.0016 -- it is not the per-night systematic
(`docs/igrins_a0v.md`, "Line coupling on a real night"). Off by default.

## Science products: two things learned while building the targets page

- **The clip mask is saved** (`clipped` in each npz of a `--clip-sigma` run).
  It cannot be recomputed afterwards: it is cut against the unclipped model, and
  the final continuum, refitted without those pixels, sits higher and flags about
  1.7x as many (3.3% against 2.0%).
- **A science run is stale once its calibration is rebuilt.** The first
  `science_dct2018*` products predated the K calibration rebuilt after the
  zenith-angle fix; rerun, the water shifts moved by at most 4e-4 and the
  corrected spectra by a median 0.001 of the noise, but nothing had flagged it.
  Compare the run's mtime with its calibration's before trusting one. And with a
  target model in the source slot, pass `--stellar-velocity-kms` (now recorded in
  the settings): from 0, LkCa 15's H98, H102 and H123 settle 40 km/s from the
  star. `science_dct2018_model/` holds one record per target
  (`record_0059_{H,K}.h5`, `record_0092_{H,K}.h5`), because two runs into one
  `--record` path overwrite each other.
