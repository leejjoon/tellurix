# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`jax-telluric` (package `src/jax_telluric/`, repo directory `lblrtm/`) is a
differentiable JAX forward model for terrestrial (telluric) absorption in
high-resolution spectra such as IGRINS. It wraps ExoJAX opacity calculators
with a layered atmosphere, slant-path transmission, an instrument model, and a
bounded MAP fitter. LBLRTM 12.17 is the accuracy reference, not a runtime
dependency — LBLRTM is only executed offline to produce fixtures and
correction templates.

## Commands

```bash
UV_CACHE_DIR=.uv-cache uv sync --dev          # environment (.venv, Python 3.10)
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
nights, not from one night's error bar**: the three give CH4 slopes of +0.013,
+0.027 and -0.030 per unit airmass, chi-squared 7.1 on 2 dof, so the 2.5 sigma
one night showed was a per-night systematic. CO2 does the same. On DCT 2016 both
show a 4-sigma dependence on *time of night*, which neither can physically have.
The honest bound is the ~3% night-to-night scatter. Separately, the response
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
4.5-8.3 K/km against the assumed 6.5. It **halves CO2's inter-night
inconsistency** (chi-squared 6.8 to 3.4 on 2 dof) but does not fix CH4 and does
not improve the residual at all. The strongest reason to use it is coverage: 31%
of the archive (Gemini South from 2020) carries no weather cards, and ERA5 needs
only a position and a time. Two backends -- ARCO-ERA5 on Google Cloud needs no
credentials and costs 150 MB a profile because its chunks are global; CDS needs
`~/.cdsapirc`, charges by fields (times x levels x variables) and is 4x faster
and 1.5 kB a profile once batched, with a limit near 8,000 fields a request.

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
impossible values -- the airmass comes from the header.

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

For installed users, `jax-telluric-download-data {mt-ckd,aer-lines,all}`
(`download.py`) fetches SHA-256-pinned copies into `$JAX_TELLURIC_DATA` or
`~/.local/share/jax-telluric`.

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

- `jax_telluric/__init__.py` enables `jax_enable_x64` at import. Import
  `jax_telluric` **before** ExoJAX so calculators are built in float64.
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
- A run's product is its HDF5 record (`src/jax_telluric/record.py`), not the
  `.npz` arrays, which are a cache. `write_record` takes `key_fields` naming the
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
