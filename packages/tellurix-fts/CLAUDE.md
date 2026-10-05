# CLAUDE.md -- tellurix-fts

Notes for the Fourier-transform spectrometer pipelines: the Hinkle, Wallace &
Livingston Arcturus atlas and the NSO Kitt Peak solar atlases. The package
(`src/tellurix_fts/`) holds the readers and the solar window fit
(`window.py`); the drivers that run them are in the repository's `scripts/`, and
the core's notes -- commands, architecture, what a fit costs, conventions -- are
in the root `CLAUDE.md`.

## The Arcturus atlas pipeline

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

`docs/arcturus_walkthrough.ipynb` is the end-to-end explanation of that pipeline
-- every component on one page, then the same thing across all 598 -- and it is
**generated, not hand-edited**, so the prose and the code stay in one reviewable
file. Rebuild and re-execute it after changing anything it describes; the
committed copy carries its outputs so it renders without being run.

```bash
uv run --with nbformat python scripts/build_arcturus_notebook.py
CUDA_VISIBLE_DEVICES=0 uv run --with nbformat --with nbconvert --with ipykernel \
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 docs/arcturus_walkthrough.ipynb
quarto render docs/arcturus_walkthrough.ipynb --to html   # standalone page, gitignored
```

Quarto reads the stored outputs rather than re-executing, so render it *after*
`nbconvert --execute` or it will publish stale figures. Verified with Quarto
1.10.18: 4.06 MB self-contained HTML, all 20 figures embedded with their
`#| fig-cap:` captions, the table of contents and the MathJax equation intact,
no warnings.

It runs the real `fit_arcturus_page.py` on ab5000_ rather than reimplementing
it, which is what makes it a test as well as a document: the live fit reproduces
the committed record's 3.61 sigma to 3.62. Notebook tooling is deliberately not
a project dependency -- `--with` keeps it out of the lock, the same way
`era5_site_profile.py` takes `--with aiohttp`.

The batch driver writes its full summary to `data/corrected/atlas/summary.json`
(gitignored — 1.6 MB at atlas scale, mostly per-page timings and a per-stage
optimizer log). `trim_atlas_summary.py` reduces it to the 0.39 MB record under
`docs/`, keeping the window, the fitted parameters and the quality numbers, and
lifting per-page fields that never vary into `settings`.

Both fit drivers default to `--precompute-opacity` (see *What a fit costs* in the root `CLAUDE.md`);
`--no-precompute-opacity` restores the exact-kernel-per-iteration path and is
the control to reach for when a fitted value looks wrong.

## The sibling project, and the solar work it is asking for

`/home/jjlee/work/differentiable_stellar_spectroscopy` is where the observed
spectra come from and where this package's output goes. Its
`docs/solar_data_status.md` (2026-09-26) names **running this package on a solar
atlas as "the single highest-value piece of data work available"**, for a reason
that is this package's own result handed back: the NSO solar atlases ship a
telluric column *at atlas resolution*, which can only mask, and cannot be
multiplied inside a convolution. The deliverable would be the solar analogue of
`data/corrected/arcturus_transmission.h5`.

`docs/solar_atlases.md` here is the survey -- what the data is, the traps
measured on disk, and which of their documents to read for detail. Do not
duplicate their content into this repository; they are the authority. The
first thing to know supersedes much of the rest: **`photatl` is
`ftsspec_901218_5`**, rescaled per page as `gain * file 5 + offset` to 0.06
sigma of file 5's noise, so it has file 5's air mass (1.985) and refitting it is
refitting file 5. Its product, `photatl_corrected.h5`, is built from the file-5
fit by `scripts/export_photatl_from_ftsspec.py`; the one thing photatl adds is
the authors' zero level, which zeroes saturated cores and measurably improves
the fit only across 5298-5402 cm-1, whose four pages the product therefore takes from
a fit of photatl itself (`--patch`, `docs/solar_fit_plan.md` §4j). The telluric
lines also put the wavelength scale +141 m/s red, against the +260 m/s solar
lines gave them. **`niratl`** (8900-13600 cm-1) is independent data and is fitted
page by page: air mass 1.10 from the O2 A-band, its own sinc (FWHM 0.01859, not
photatl's 0.01753 -- pass `--fwhm-cm1`), the June 1983 profile and the `nir`
Payne Zero band; products `niratl_corrected.h5` and `niratl_transmission.h5`
(§4k). Its residual reads ~76 sigma only because its noise is 5x lower than
file 5's; in flux units the fit is as good. The 1983 raw pair
`ftsspec_830626_{2,3}` (8516-20735 cm-1, to 482 nm) is fitted too (§4l). Header air
masses are **Kasten & Young 1989** at the Sun's position -- recomputing them from the
UT times reproduces every header to 0.02 -- which is how `_3`, whose header prints
`?.??`, gets 3.53 -> 2.64 (pass `--airmass 3.085`). **Its B-band O2 columns are ~5%
high relative to its A-band, and that belongs to this file pair** (§4n, closed
2026-10-05): not the line list (AER's B-band is 2.4% *above* the lab CRDS intensities),
not widths or line shape (weak and saturated lines want the same excess), not the
temperature profile, water, zero level or fitted line-spread width; the IAG atlas and
Kitt Peak 1989 (`scripts/fit_atlas_o2.py`) both give the lab's weak-line B/A ratio. Do
not quote that pair's B-band columns. niratl's air mass, from the A-band, is 1.10 to
the ~2-3% the A-band line physics allow.
**O2's collision-induced continuum is ported from LBLRTM** (`tellurix.o2_cia`,
`--o2-cia`, validated to 0.13%, §4o). It barely moves columns -- the fitted Chebyshev
had been absorbing it; at 1.27 um, its strongest band, O2 moves by up to 8% but inside
its error -- but without it the corrected spectra keep its dimming, up to 9.3% at 1.27
um in photatl. **Every solar run and product was regenerated with it on 2026-10-05**
and promoted to the usual paths (`docs/solar_fit_plan.md`, "Done 2026-10-05"; the old
ones are in `data/corrected/superseded_pre_o2cia/`). "Open work" there lists the rest.
Ozone's Chappuis band and NO2 are cross-section absorbers AER does not carry: the
corrected visible spectra keep their broadband dimming. **A column at its
upper bound makes a window unusable** -- the fit is absorbing the solar model's error
into it, 5-13% of fake absorption -- while one at the lower bound is harmless; the
transmission files flag the former per row (`column_at_upper_bound`) and the
corrected files mark it unreliable (`tellurix.quality`, §4m). The survey's original four points, as it wrote them: `photatl` (1.11-5.41 um, 87% overlap with the
Arcturus pages already fitted) **interpolates pixels where the sky is opaque**;
its **ILS is the weakest input in their programme** -- R = 300,000 quoted once
for four atlases, no MOPD, no apodization; its **wavelength scale is +260 m/s**
off two independent ACE atlases that agree with each other to 2 m/s; and the NSO
telluric atlases have **no loader** anywhere. Their W4.1 also found that telluric
lines can measure an instrument profile where stellar lines cannot, which is a
lever on the `lsf_sigma_kms` railing documented below and has not been tried
here.

Their status document cites our export under a stale name
(`arcturus_transmission_full.h5`); the file is `arcturus_transmission.h5` and
comes from the promoted full-coverage record.

## Rerunning a whole solar run

How the 2026-10-05 regeneration was done; reuse it for any physics change.

- **The command is the record's own**: its `config` and `physics`, plus the change.
  Air masses the header lacks are passed (`_3` 3.085, file 4 4.73, niratl 1.10), the
  1983 pair takes `--v1 8530 --v2 20740` and `_3` `--scan-fwhm-cm1 0.04167`. Species
  scans are keyed on window, profile and scan settings, not on the fit physics, so a
  physics change reuses all of `data/scans/`; check a profile's sha256 has not
  changed first, or every scan reruns (~6 min a window).
- **Shards**: `--shard I/N` with a per-shard `--summary` and `--record`, one GPU each
  (`CUDA_VISIBLE_DEVICES`). Then `scripts/merge_fts_shards.py`, and the same command
  without `--shard` (on CPU is fine): `--resume` skips every window and writes one
  record. Two GPUs took 4 h for all five runs. Ask which GPUs are free -- the user
  shares the machine.
- **Check before promoting**: `scripts/compare_fts_runs.py old new --sigma`. Expect a
  few columns to move by percents with identical residuals in ill-conditioned windows;
  judge them against `sigma`.
- **Exports are CPU-bound** (a model rebuilt per window): a 1990 file's
  `export_transmission_hdf5.py` takes ~80 min on GPU or CPU alike. Run it on a GPU
  only if no fit worker holds that GPU -- a worker preallocates 75% and the export ran
  out of memory beside one. Exports record their source record's path, so export
  *after* moving the record to its final place.
- **Promote** by moving the old records, products and review bundles to
  `data/corrected/superseded_pre_<change>/` (gitignored, never packed) and the new ones
  to the canonical paths; then `results_archive.py check` should list exactly the
  replaced records, `pack --release results-<date>`, commit the manifest, push, and
  `gh release create` with the archive. The manifest must also name a Zenodo mirror
  (`tests/test_results_archive.py` asserts it): Zenodo's GitHub integration archives
  only the release's source zip, so the tarball has to be added to that Zenodo
  version by hand, with the user's account, and its file URL passed to `pack` as
  `--mirror` (or written into the manifest's `mirrors`). `git push` over SSH fails inside these
  sessions (no agent); push over HTTPS with `gh auth git-credential` as the helper.

## The Arcturus record and its products

The Arcturus atlas is **not fetched by this repository**. Runs read it from a
sibling project (`differentiable_stellar_spectroscopy/data/atlases/arcturus/ir`)
and the record pins it by that absolute path plus a per-page sha256 -- which
catches a changed file, not a moved directory. It is Hinkle, Wallace &
Livingston 1995 (ASP, ISBN 1-886733-04-X; PASP 107, 1042), whose README asks for
a citation in any publication. The `ftp://ftp.noao.edu/catalogs/arcturusatlas/`
URL every paper cites is dead; start from
https://noirlab.edu/science/data-services/other. README.md and
`docs/arcturus_fit.md` carry the full provenance.

The committed Arcturus record is the **full-coverage** run: `page_windows`
no longer trims the ~5 cm-1 the atlas pages overlap by, so every pixel the atlas
ships is fitted (822,928 against 636,444) and adjacent pages overlap in every
product. `--trim-overlap-cm1 2.5` restores the old tiling and reproduces the
superseded record (`data/corrected/atlas/arcturus_atlas_trimmed.h5`, in the results archive).
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
`corrected / continuum` pixels sit above 1.02 at 1.5 um. Separately, **fitted column scales are not comparable between pages.** The
atlas runs with `zenith_angle_deg = 0`, so air mass is absorbed wholly into the
column scale, and the atlas is not one exposure -- 0.92-5.36 um at R = 100,000
comes from at least five FTS configurations (five distinct sampling intervals,
each in its own spectral region) and its two epochs are two dates. A fitted
scale is `true column x air mass / profile column` and a single page cannot
separate them. The p16-p84 spread over 4000-9000 cm-1 is 1.34x/1.29x for H2O,
1.31x/1.55x for CO2, 1.81x/1.95x for CH4 (summer/winter); those bound air mass
*plus* model error and are not an accuracy. One feature survives: above
9000 cm-1 the water scale runs 2.4x/2.3x above the bulk in both epochs, on pages
that are better constrained than the bulk (median transmission 0.950 vs 0.981,
formal sigma 0.0059 vs 0.0071, residual 2.94 vs 3.34 sigma) -- but **CO2 and CH4
have no measurable band above 9000 cm-1**, so no second species shares the path
and air mass cannot be ruled out there. Do **not** attribute any of this to line
blanketing, as earlier notes did: the raw split reproduces (-4.3%/-6.4%) but
blanketing, page width and wavenumber are mutually confounded and the partial
correlation swings from -0.24 to +0.16 with the control set. The
continuum-source degeneracy is separately ruled out by `--continuum-anchor 0.98`
(median H2O shift +0.4%, residual 1.1% worse).

`scripts/generate_payne_zero_arcturus.py` makes the stellar source and does **not**
run in this environment: Payne Zero needs Python >= 3.11 while this package is pinned
to 3.10 by `exojax==2.5.0`. It runs in Payne Zero's own venv and writes an npz that
`StellarSpectrum.from_npz` reads back; the script's docstring has the invocation.

**`--line-coupling`** on `fit_fts_window.py` and `fit_fts_batch.py`
(`WindowSettings.line_coupling`) applies AER's first-order line mixing (CO2, CH4,
O2). The batch records it in `physics.line_coupling` and hashes `lncpl_lines` into
the inputs; `export_transmission_hdf5.py` and `export_solar_review.py` rebuild with
it from the record, a record without the key meaning off. No solar product has
been fitted with it yet; on IGRINS it matters at the 2.0 um CO2 band centre and
nowhere else measured (`docs/igrins_a0v.md`).
