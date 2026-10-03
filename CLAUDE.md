# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tellurix` (repo directory `lblrtm/`) is a
differentiable JAX forward model for terrestrial (telluric) absorption in
high-resolution spectra such as IGRINS. It wraps ExoJAX opacity calculators
with a layered atmosphere, slant-path transmission, an instrument model, and a
bounded MAP fitter. LBLRTM 12.17 is the accuracy reference, not a runtime
dependency — LBLRTM is only executed offline to produce fixtures and
correction templates.

The repository is a uv workspace of three packages under `packages/`, each
with its own `pyproject.toml`, source and tests:

- `tellurix` (`packages/tellurix/`) -- the core: forward model, opacity,
  continua, fitter, stellar source, run records, species scans, site profiles
  from the AFGL atmospheres it ships (`site_profile.py`, `afgl/`), the
  upper-bound quality rule (`quality.py`), and the LBLRTM reference tooling.
  It knows no instrument.
- `tellurix-fts` (`packages/tellurix-fts/`, import `tellurix_fts`) -- FTS
  atlases: `atlas.py` (Arcturus), `nso.py` (NSO Kitt Peak solar atlases),
  `ils.py` (MOPD measured from the spectrum), `window.py` (the solar window fit:
  `WindowSettings`, `prepare_window`, `fit_window`).
- `tellurix-igrins` (`packages/tellurix-igrins/`, import `tellurix_igrins`) --
  `igrins.py` (RRISA reader), `night.py` (night calibration), `flat.py`
  (lamp-flat blaze), `standard.py` (the per-order fit: `StandardFitSettings`,
  `build_order_context`, `fit_one`, and the order rule `ORDER_RULE`).

Code that more than one script needs lives in a package, never in `scripts/`:
no script imports another. A fit takes a frozen settings object and a
`tellurix.DataPaths` -- where the AER line files and MT_CKD are, either
`DataPaths.bootstrapped(checkout)` (what `bootstrap_lblrtm.sh` builds) or
`DataPaths.downloaded()` (what `tellurix-download-data` fetches) -- never an
argparse namespace or the repository root. The scripts are the command lines:
they parse arguments, build those two objects from this checkout, and write
the summaries and records. A run's order rule is its own copy
(`StandardFitSettings.order_rule`); restore a recorded run's rule by building
one from its `config`, never by mutating `ORDER_RULE`, which is read-only.

The root `pyproject.toml` is not a package (`tool.uv.package = false`): it
installs all three, editable, into one `.venv` for what stays at the top level
-- the pipeline `scripts/`, `benchmarks/`, `docs/`, `data/`, and `tests/` for
what spans packages. A package's tests read only their own fixtures, under `tests/data/` beside
them, so they run from any directory. Tests of the repository's own data or
scripts (the AFGL profiles, a script's defaults) belong in the root `tests/`.
Three profiles in `packages/tellurix/tests/data/profiles/` are copies of
`data/profiles/` and are inputs, not products; the two validation reports there
(`aer_co_validation.json`, `native_mt_ckd_validation.json`) are written in place
by `validate_aer_co.py` and `validate_mt_ckd.py`.

The direction is one way: an instrument package imports `tellurix`, never the
reverse, and neither instrument imports the other.
`tests/test_package_boundary.py` enforces it, including imports inside function
bodies. Put new instrument-specific code in its package, not in the core, and
import those names from their package (`from tellurix_igrins import
read_igrins_observation`) -- the core does not re-export them. A new
dependency goes in the `pyproject.toml` of the package whose code imports it.

## Commands

```bash
UV_CACHE_DIR=.uv-cache uv sync --dev --extra gpu   # environment (.venv, Python 3.10)
# Plain `uv sync --dev` REMOVES the CUDA wheels and leaves jax CPU-only; the extra is
# what keeps the GPU working, and every fit driver defaults to --platform gpu.
UV_CACHE_DIR=.uv-cache uv run pytest          # full suite, ~3 min on CPU, no external data needed
UV_CACHE_DIR=.uv-cache uv run pytest packages/tellurix-igrins   # one package
UV_CACHE_DIR=.uv-cache uv run pytest packages/tellurix/tests/test_model.py::test_name   # one test
```

Always prefix with `UV_CACHE_DIR=.uv-cache`; the default cache location is not
used in this project. Set `JAX_PLATFORMS=cpu` to keep test/benchmark runs off
the GPU.

## Instrument notes -- read the one you are working on

Most of what this project has learned is about particular data, and it lives
beside the package for that data rather than here. These files are not loaded
automatically when you work on a script, because the scripts are in the
top-level `scripts/`: read the relevant one before changing or running anything
it covers.

- `packages/tellurix-fts/CLAUDE.md` -- the Arcturus atlas and the NSO solar
  atlases: `fit_arcturus_*`, `fit_fts_*`, `measure_*_ils`, `rebuild_*`,
  `export_atlas_*`, `export_photatl_*`, `export_spectra_hdf5`,
  `export_transmission_hdf5`, `scan_*`, `generate_payne_zero_{arcturus,solar}`,
  `build_arcturus_notebook` and `docs/arcturus_walkthrough.ipynb`, the sibling
  project and the solar products it is asking for.
- `packages/tellurix-igrins/CLAUDE.md` -- IGRINS: `*igrins*`,
  `download_rrisa_standard`, `era5_site_profile`, `generate_payne_zero_{a0v,star}`,
  the night calibrations under `data/calibration/`.
- `docs/review_page/CLAUDE.md` -- the claude.ai review pages and their
  exporters, for both instruments.

Benchmarks and the offline LBLRTM workflows. All need `./scripts/bootstrap_lblrtm.sh`
to have been run first; `benchmark.py` and `validate_mixed_precision.py` use only the
AER line files it downloads, the rest also execute the LBLRTM binary:

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark.py --platform cpu --output benchmarks/results/cpu.json
UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark_fit.py   # staged-fit cost -> docs/precomputed_opacity_results.json
UV_CACHE_DIR=.uv-cache uv run python benchmarks/compare.py baseline.json candidate.json
UV_CACHE_DIR=.uv-cache uv run python scripts/run_lblrtm_reference.py      # rebuild the core test fixture
UV_CACHE_DIR=.uv-cache uv run python scripts/build_lblrtm_correction.py   # rebuild data/corrections template
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_aer_co.py           # writes packages/tellurix/tests/data/aer_co_validation.json
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_mt_ckd.py           # writes packages/tellurix/tests/data/native_mt_ckd_validation.json
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
each package's `tests/data/` and JSON reports under `docs/` are committed.

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

`constant_velocity_grid(..., margin_cm1=25)` pads so that outside lines
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

Observed data and the stellar source enter through two adapters: an instrument
reader such as `tellurix_fts.atlas`
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
that way — `packages/tellurix/tests/test_direct.py` asserts no 2-D array survives construction.
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
  `tellurix` **before** ExoJAX so calculators are built in float64. The
  instrument packages import `tellurix` first in their own `__init__` for the
  same reason.
- Public data classes are frozen dataclasses/NamedTuples that validate and
  normalize in `__post_init__` (species names uppercased, arrays coerced,
  physical ordering checked) and raise `ValueError` with a short message.
  Follow that pattern rather than validating at call sites.
- Atmosphere arrays are ordered top-to-bottom: pressure edges increase,
  altitude decreases. Wavenumber grids are strictly increasing and evenly
  spaced in log wavenumber (constant velocity step) — `constant_velocity_grid`
  builds them with padding for line wings.
- Units are in names: `_bar`, `_hpa`, `_cm1`, `_km`, `_kms`, `_k`, `cm2`.
- Comments explain *why* (a numerical or LBLRTM-compatibility reason), not what.
  Keep that density; the existing comments are load-bearing.
- Accuracy and performance claims are backed by a committed JSON report under
  `docs/` plus a prose companion (`docs/validation.md`,
  `docs/performance.md`, `docs/lblrtm_corrected_mode.md`,
  `docs/mixed_precision_validation.md`). `packages/tellurix/tests/test_lblrtm.py` asserts on
  `packages/tellurix/tests/data/aer_co_validation.json` and `packages/tellurix/tests/data/native_mt_ckd_validation.json`,
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
- A run's product is its HDF5 record (`packages/tellurix/src/tellurix/record.py`), not the
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
