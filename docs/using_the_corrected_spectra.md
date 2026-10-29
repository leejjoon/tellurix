# Using the corrected spectra

This is the short version, for someone who wants to *use* the output rather than
reproduce it. If you want to know **how** the correction works rather than how to
consume it, read `docs/arcturus_walkthrough.ipynb` instead: it takes one page
through every component with plots, then repeats it across all 598 -- of the
`fit_arcturus_batch` run superseded on 2026-10-29, whose method it still explains. `docs/arcturus_fit.md` and `docs/igrins_a0v.md` are the long
versions: they are development records, organised by how the work unfolded, and
you should not have to read either to get a spectrum out.

Two stellar datasets are described first; the Sun, fourteen corrected spectra
from Kitt Peak and Göttingen, has [its own section](#the-solar-spectra), which you can
read on its own once you know [which array is the correction](#read-this-first-which-array-is-the-correction).

| | Arcturus atlas | IGRINS A0V standards |
|---|---|---|
| what was fitted | Hinkle, Wallace & Livingston 1995, 598 page-epochs (592 with telluric lines) | 43 standards over 4 nights, 3 telescopes |
| keyed on | `(page, epoch)` | `(frame, order)` |
| record | `data/corrected/atlas/arcturus_atlas.h5` | `data/corrected/igrins/*/record.h5` |
| arrays | `data/corrected/atlas/*.npz` | `data/corrected/igrins/*/*.npz` |

The Arcturus record is the **solar-era refit** of 2026-10-29 (`fit_fts_batch.py
--arcturus`): the solar atlases' window fit, with telluric columns from
stellar-clean pixels, line coupling, O2 CIA, the line-strength table, one AFGL
1994 profile per epoch and a sinc at R = 115,700 (`docs/arcturus_fit.md`, "The
solar-era refit"). Every pixel the atlas ships is fitted and adjacent pages
overlap, as the atlas itself does. Against the `fit_arcturus_batch` record it
replaced, scored by flatness at telluric pixels as the solar spectra are below
(`docs/arcturus_solar_era_score.json`):

| | superseded | current | the atlas authors |
|---|---:|---:|---:|
| flatness, telluric pixels (340 page-epochs) | 2.33% | **2.08%** | 3.33% |
| flatness, T < 0.7 (269) | 4.26% | **4.00%** | 9.04% |
| residual, median | 4.22 sigma | 4.34 sigma | |

The residual rises because the telluric columns are now fitted on stellar-clean
pixels only. **Below 2200 cm-1 (4.5-5.4 um) the stellar model is the limit**: Payne
Zero's CO lines are about 25% too weak, the fitted continuum sits 1-5% below the
star's, and telluric CO or O3 is at its bound on a few pages there (`at_bound` in
the record) -- treat those pages' columns, and their correction, with care.

The superseded record (and the trimmed run before it, `arcturus_atlas_trimmed.h5`)
is in `data/corrected/superseded_pre_solar_era/atlas/`, out of the results archive
from results-2026-10-29 on; release results-2026-10-26 still carries both. IGRINS
residuals are 1.30-1.86 sigma median.

Three things can be handed to someone else, and which one you want depends
entirely on what you are going to do with it:

| you want | take | section |
|---|---|---|
| a corrected stellar spectrum | `corrected`, or the spectra export | [below](#if-you-just-want-a-file-of-spectra) |
| to put the atmosphere in your own synthesis | the transmission export | [below](#if-you-want-the-atmosphere-not-our-spectrum) |
| to reproduce or audit a fit | the record | [below](#what-is-in-the-record-and-why-it-is-the-real-product) |

If you are running your own forward model, it is the second one, and you should
not be reading `corrected` at all.

## Read this first: which array is the correction

**Do not divide by `transmission`.** The array called `transmission` is the
*unconvolved* atmospheric transmission on the model's own grid, interpolated to
the pixels. It is not the operator the correction applied, and using it will
leave a positive-then-negative spike beside every strong line.

The operator is

```python
effective_transmission = model_flux / stellar_only
```

They differ by up to 0.126 on the atlas against 0.0059 of noise, and by up to
0.48 in a deep IGRINS core. The difference is not a bug and not a misalignment:
convolution does not commute with multiplication, so `Conv(T x S) / Conv(T)` is
not `S`. Applying the naive division to the *model alone*, which contains no
data, reproduces 86% of the excursion.

The `corrected` array already does the right thing:

```python
corrected = (observed / model_flux) * stellar_only
```

which transfers the data onto the stellar model without ever dividing by a
convolved transmission. Measured peak-to-peak beside strong lines: 0.068 against
the naive division's 0.319, and the disagreement between two independent epochs
halves. **Use `corrected`, or build it with the formula above.**

## What is in a `.npz`

All arrays are on one **ascending-wavenumber** axis, which is *descending*
wavelength and, for IGRINS, descending detector column.

| array | meaning |
|---|---|
| `wavenumber_cm1` | vacuum wavenumbers of the pixels |
| `observed` | the input spectrum, NaN where masked |
| `corrected` | **the product** — telluric removed, stellar model retained |
| `model_flux` | the full forward model: continuum x source x transmission, convolved |
| `stellar_only` | the same model with the atmosphere removed |
| `transmission` | unconvolved transmission, interpolated to the pixels — *diagnostic only*, see above; the usable version is the [fine-grid export](#if-you-want-the-atmosphere-not-our-spectrum) |
| `continuum` | the **fitted** Chebyshev — instrument and normalisation, *not* the star's continuum |
| `stellar_continuum` | the stellar model's own physical continuum, where the model ships one |
| `residual` | `observed - model_flux` |
| `mask` | True where a pixel was fitted |
| `reliable` | `mask` and transmission above 0.15 — **use this to select pixels** |

**The Arcturus `.npz` is laid out differently**: it is `fit_window`'s cache, in
ascending *wavelength*, with `flux` and the model arrays normalized by the window's
99th percentile and `observed_raw` in the page's own units (`corrected` is
`observed_raw` over the effective transmission, so it is in page units too).
`tellurix_fts.window.arrays_in_page_units(np.load(path))` returns the table above's
arrays in ascending wavenumber and page units; the spectra export below is already
in that form. The superseded run's `.npz` carried the table's names directly, plus
`corrected_normalized` and the atlas's `atlas_telluric` and `atlas_ratioed`
columns. IGRINS adds `uncertainty` (the PLP's per-pixel sigma, a real
measurement), `plp_telluric` and `plp_continuum` (the pipeline's own, for
comparison only — see the caveat below), and `response_pattern` (the instrument
response that was divided out).

```python
import numpy as np
from tellurix_fts.window import arrays_in_page_units
a = arrays_in_page_units(np.load("data/corrected/atlas/ab5000__summer.npz"))
good = a["reliable"]
wavenumber = a["wavenumber_cm1"][good]
spectrum   = a["corrected"][good]
```

## If you just want a file of spectra

The record is the archival product, but regenerating arrays from it needs this
repository, the AER line files, the original data and a GPU. To hand someone
spectra instead:

```bash
uv run python scripts/export_spectra_hdf5.py \
    --record data/corrected/atlas/arcturus_atlas.h5 \
    --output data/corrected/arcturus_spectra.h5
```

One HDF5, **15.8 MB for the 592 Arcturus page-epochs with telluric lines** against
127 MB of `.npz`, readable with nothing but `h5py`. Rows are padded to the longest
spectrum with NaN, so every array is a plain 2-D `(row, pixel)` block. It
carries the arrays below, the fitted parameters, the formal sigmas and
correlations, and the whole `/config`, `/physics` and `/inputs` provenance
copied from the record, so it stands on its own.

```python
import h5py
with h5py.File("data/corrected/arcturus_spectra.h5") as f:
    i = list(f["key"].asstr()).index("ab5000_ summer")     # or "SDCH_20181220_0100 H108"
    good = f["reliable"][i]
    wavenumber = f["wavenumber_cm1"][i][good]
    spectrum = f["corrected"][i][good]
```

The file names the correction operator explicitly as
`effective_transmission`, beside `transmission_unconvolved` marked diagnostic
only, so the trap below is hard to fall into from the file alone.

These are gitignored: produce one when you need to ship spectra.

## If you want the atmosphere, not our spectrum

If you run your own synthesis, do not use `corrected` at all. Take the
transmission and put it inside your own convolution:

```
your_model(nu) = Conv_ILS[ your_continuum(nu) x your_source(nu) x T(nu) ]
```

fitted against the raw `observed` column. Nothing is ever divided by a convolved
quantity, so the non-commutation above does not arise; our stellar model never
enters your data; and your own continuum does the normalising.

For that you need T on a grid fine enough to convolve, which the `.npz` cache
and the spectra export do **not** give you -- their `transmission` is
interpolated to the atlas pixels, about 2.3 samples per resolution element, so
the lines are already undersampled before the interpolation runs.

```bash
uv run python scripts/export_transmission_hdf5.py \
    --record data/corrected/atlas/arcturus_atlas.h5 \
    --output data/corrected/arcturus_transmission.h5 --check

# an IGRINS night works the same way
uv run python scripts/export_transmission_hdf5.py \
    --record data/corrected/igrins/dct2016_h/record.h5 \
    --output data/corrected/dct2016_h_transmission.h5 --check
```

That writes T on the forward model's own grid -- 4 samples per resolution
element at the Arcturus run's R = 115,700 (a 0.648 km/s step), no interpolation
anywhere in the path. **10.9 MB for the 592 Arcturus page-epochs, about an hour on
one GPU.** Grids are shared where both epochs of a page sit on one window with the
same line-coupling choice, so the file holds 359 grids for 592 rows:

```python
import h5py
with h5py.File("data/corrected/arcturus_transmission.h5") as f:
    i = list(f["key"].asstr()).index("ab5000_ summer")
    g = f["grid_index"][i]
    n = f["grid_points"][g]
    nu = f["wavenumber_cm1"][g][:n]
    T  = f["transmission"][i][:n]
```

`/zenith_angle_deg` says what angle each row was evaluated at. For the Arcturus
atlas it is 0 throughout, so T is the **vertical** transmission and the fitted
column scales in `/parameters` have already absorbed any slant path. The same
script runs on an IGRINS record, where the angle is the frame's own -- 18.7 to
67.3 degrees on one night -- and T is then the slant transmission along that
line of sight. `/profile` carries the layered atmosphere the scales multiply
(pressure, temperature, altitude, air column, every VMR), so T can be
regenerated on any other grid from this file plus a line list. The Arcturus file
has one per epoch instead, `/profile_summer` and `/profile_winter`, and a row uses
its epoch's.

**Thirteen Arcturus rows have a column at its upper bound** (`column_at_upper_bound`;
`species_at_upper_bound` names it): eight below 2200 cm-1, where the column is
absorbing the stellar model's too-weak CO lines, four at 4000-4075 cm-1 and
ab6150_ winter. As for the solar windows below, treat those rows' transmission as
holding absorption the sky did not.

`--check` interpolates the fine grid back down to the pixels and compares
against the cached `transmission`:

| record | agreement | why |
|---|---|---|
| Arcturus atlas | 1.9e-7 | rebuilt from the record; T is stored as float32 (the superseded run's file agreed to 2.2e-16) |
| IGRINS night | 3.1e-6 | the driver saves T from the refrozen linearized backend, this script evaluates the exact kernel; 1600x below the noise |

### Stitching: nothing tiles, and that is deliberate

**Adjacent pages overlap, in every product. Do not concatenate them.**

The raw atlas pages overlap by about 5 cm-1 -- 282 of 287 adjacent summer pages
do. The run committed before this behaviour was configurable trimmed 2.5 cm-1
off each end so the fitted windows tiled, which cost **22.7% of the atlas's
pixels**. That trade is no longer made: `page_windows(..., trim_cm1=0.0)` is the
default and every pixel the atlas ships is fitted.
`fit_arcturus_batch.py --trim-overlap-cm1 2.5` restores the old behaviour and is
only there for reproducing the old record. The current run fits whole pages too.
The numbers below, and the `ab6250_` case, were measured on the superseded run.

So a page now shares about 250 pixels with each neighbour, and those pixels were
fitted twice, independently. Across the full run:

| | summer | winter |
|---|---|---|
| overlapping pairs | 279 | 300 |
| shared reliable pixels, median | 233 | 223 |
| median \|observed_a - observed_b\| | 0.01277 | 0.01260 |
| median rms(corrected_a - corrected_b) | 0.02045 | 0.01874 |
| p90 rms | 0.10259 | 0.09704 |

**The first of those rows is the one to understand.** The *data* differs between
two pages over the same wavenumbers, by a median of 0.0128 against a nominal
pixel sigma of 0.0045 -- nearly three times the noise. Each atlas page carries
its own scalar normalisation, so the two copies are not the same measurement
twice, they are two reductions of it. That is exactly why the original code
trimmed the overlap away, and it is why averaging the two copies is wrong.

What the duplication is good for is a **consistency check**. The corrected copies
agree to an rms of about 0.020, which is the same size as a page's own residual
(0.0199) -- so where two fits of identical photons disagree by much more than
that, one of them is wrong, and the parameters say which. A worked case: over
the 250 pixels `ab6225_` shares with `ab6250_`, the two differ by rms 0.027 with
a derivative-shaped spike at every telluric core. Undoing their 0.0386 km/s
velocity difference removes 2% of it. The cause is in the fit -- `ab6250_` landed
on `lsf_sigma_kms` = 1.278 km/s against its neighbour's 0.110, with twice the
residual. The overlap caught a bad ILS fit that nothing else in the products
would have flagged.

**The transmission export overlaps more**, because the model grid pads 5 cm-1
beyond the window on each side for line wings: adjacent rows share 9.99 cm-1.
Cut each row to its own fitted window, which `/parameters` carries:

```python
v1, v2 = f["parameters"]["v1"][i], f["parameters"]["v2"][i]
inside = (nu >= v1) & (nu <= v2)
```

The margin is not junk -- it is the same atmosphere extrapolated past the data
that constrained it, and where one page is inside its window while its neighbour
is in its margin the two agree to a median of 0.0009. But the worst point in
such an overlap is a median 0.027 across pairs, because a small difference in
fitted column becomes a large one in a line core. Use the margin for wings,
never as a substitute for the neighbour's own fit.

Three things this does not solve. The columns were retrieved with our continuum
and our source, so they carry whatever bias those imposed -- see the blanketing
caveat below, which is worth a few percent in H2O. T is the transmission at one
fitted state, not a function you can re-fit; to refit columns inside your own
synthesis, regenerate T from `/profile` rather than scaling this array. And the
ILS is still ours: T is unconvolved, which is the point, but the *columns* were
fitted through our instrument model, so an error there has already leaked into
them.

## What is in the record, and why it is the real product

The `.npz` arrays are a **regenerable cache**. The record — one HDF5 per run,
0.42 MB for the whole atlas against 127 MB of arrays — is what goes into the
results archive and what you should archive or ship.

```python
from tellurix import read_record, parameters_from_row
record = read_record("data/corrected/atlas/arcturus_atlas.h5")
row = record.row("ab5000_", "summer")        # IGRINS: record.row(frame, order)
row["residual_rms_over_noise"], row["median_transmission"]
parameters = parameters_from_row(row, ["H2O", "CO2", "CH4"])
```

Each row carries the fitted parameters, the window, the quality numbers, and the
sha256 of every input. `record.config`, `record.physics` and `record.inputs`
carry the run's settings, read from the code rather than retyped, so a record
cannot describe a run that did not happen. IGRINS rows add the airmass, zenith
angle, MJD, telescope, object and normalised surface conditions.

To rebuild the arrays from a record of the superseded run:

```bash
uv run python scripts/rebuild_arcturus_page.py --page ab5000_ --epoch summer --check
```

That script shares no code with the fitting driver on purpose. It agrees with
the cached arrays to 2.8e-7, which is the evidence that the record is
sufficient. It reads only `fit_arcturus_batch` records; for the current one the
independent reconstruction is `export_transmission_hdf5.py --check`, which
rebuilds every window's model from the record and lands on the cached
transmission to 1.9e-7.

## Caveats that will bite you

**Skip a solar window whose column is at its upper bound.** The solar transmission
files carry `column_at_upper_bound` (one boolean per row) and
`species_at_upper_bound`: there the fit drove a column to e^2 to soak up something
else, usually the solar model's error, and the transmission holds absorption the sky
does not -- 5-13% in the cases measured. The corrected-spectrum files already set
`reliable` False on those pages. A column at the *lower* bound is harmless and not
flagged. No window of the current solar runs is flagged (nine were, in runs before
the clean-pixel refits; `docs/solar_fit_plan.md` §4m), but read the flag anyway: it
cannot see a column that is far too high but short of the bound.

**The formal errors are not uncertainties.** `record.sigma` and
`FitResult.covariance` invert the objective's real Hessian, but they assume
independent pixel errors, and the residual is dominated by *correlated* model
error. Measured: 0.76% formal on the water column against 6.5-9% of sub-window
scatter — three to nine times too small. Use `record.correlation` for
degeneracies, which survives a wrong noise model, and an empirical study for an
error bar.

**The corrected spectrum is not normalised, and dividing by `continuum` does not
normalise it to the stellar continuum.** (This paragraph was measured on the
superseded Arcturus run, which normalised the source by its median. The current
one feeds Payne Zero's continuum-normalised flux, so the 1.3% zero-point offset
below is gone; what remains is below 2200 cm-1, where the continuum sits 1-5%
low because the stellar model's lines are too weak.) Values above 1 are expected: the
correction divides out the atmosphere, not the fitted continuum. Dividing by
`continuum` gets you close, and that is what `corrected_normalized` is, but the
zero point is the *median* of the stellar source over the page, not its true
continuum. `prepare_stellar_source` normalises by the median, which is exactly
degenerate with the Chebyshev's constant term and therefore free as far as the
fit is concerned -- but it puts the "1" level about 1.3% below the true continuum
on a blanketed page. Measured over 1500-1540 nm: the mean line depth of
`stellar_only / continuum` is 0.0278 where the Payne Zero source against its own
`flux_continuum` gives 0.0431, and normalising that same source by its median
reproduces 0.0304 of the 0.0431. So most of the gap is the choice of zero point,
not absorbed blanketing. The consequence for a user is concrete: about 16% of
`corrected / continuum` pixels sit more than 2% above 1 at 1.5 um, and that is
the offset, not noise. If you need a true continuum normalisation, renormalise
to a high percentile yourself, or use the transmission export and never touch
our continuum at all.

**`reliable` cuts at transmission > 0.15, which is far too permissive for line
work.** It is a floor for "this pixel carries information at all", not a
recommendation. Recut it yourself -- every file carries what you need:

```python
good = f["reliable"][i] & (f["effective_transmission"][i] > 0.8)
```

The floor the run used is on the dataset as `transmission_floor`.

**Do not compare fitted column scales between pages.** (The numbers in this caveat
are the superseded run's; the current one is also fitted at the zenith, so the
caveat holds as written.) The atlas was fitted
with `zenith_angle_deg = 0`, so the model divides optical depth by
cos(0) = 1 and **every bit of air mass ends up inside the fitted column scale**.
A page observed at air mass 1.8 returns a scale 1.8x larger than the same sky at
the zenith. The atlas is also not one exposure: it is 0.92-5.36 um at
R = 100,000 from an FTS, assembled from at least five instrument configurations
(the data carries five distinct sampling intervals, each confined to its own
spectral region), and its two epochs are two *dates* chosen for opposite
heliocentric shifts, not two exposures. So pages differ in air mass, and in the
case of water in the actual amount of water overhead.

A fitted scale is therefore `true vertical column x air mass / profile column`,
and the three factors are not separable from a single page. Measured p16-p84
spread over 4000-9000 cm-1 (1.11-2.50 um):

| | summer | winter |
|---|---|---|
| H2O | 1.34x | 1.29x |
| CO2 | 1.31x | 1.55x |
| CH4 | 1.81x | 1.95x |

Those numbers bound air mass plus model error together. They do not isolate
either, and **none of them should be read as an accuracy**.

**One feature does survive that caveat.** Above 9000 cm-1 (0.91-1.11 um) the
water scale runs 2.4x (summer) and 2.3x (winter) above the 4000-9000 cm-1
value, closely reproduced in both epochs. Those pages are not poorly
constrained -- they carry more absorption than the bulk (median transmission
0.950 against 0.981), a tighter formal error (0.0059 against 0.0071) and a
better residual (2.94 against 3.34 sigma). What cannot be checked there is
whether it is air mass: **CO2 and CH4 have no measurable band above 9000 cm-1**,
so no second species shares the path and the airmass-free cross-check does not
exist in that region. Treat water from the 0.91-1.11 um pages as carrying a
factor of about two of unexplained scale, of which an unknown part is air mass.

**Two attributions this is not.** Not line blanketing: the raw split reproduces
(-4.3% summer, -6.4% winter) but blanketing, page width and wavenumber are
mutually confounded -- each of the atlas's five page widths sits in its own
spectral region -- and the partial correlation swings from -0.24 to +0.16 with
the control set, which is instability rather than a measurement. And not the
continuum-source degeneracy, tested directly: `fit_arcturus_page.py
--continuum-anchor 0.98` moves the retrieved H2O by a median of +0.4% and makes
the residual 1.1% worse.

**The Arcturus residual is dominated by the stellar model, not the atmosphere.**
It is flat against transmission — 0.97 in deep absorption, 1.52 at the continuum
— which means the 4.22 sigma of the trimmed run, and the 4.48 of the full one,
are a K1.5 III line-list limit. Do not read either, or the current run's 4.34, as
telluric accuracy.

**In the published IGRINS records a frame's column scale carries its own
airmass error, up to +-6%.** A PLP spectrum combines 4-10 exposures, up to 37
minutes, but the records' `zenith_angle_deg` and airmass are the first
exposure's (GitHub #1, `igrins_a0v.md`, "What moves the well-mixed columns").
The fit absorbed the difference into the column scales, so the *slant*
transmission and the corrected spectra are right, and T rebuilt from a row's
own scales and angle is right too; what is off is the *vertical* column, by
ln(sequence airmass / recorded airmass), largest for long sequences of setting
or rising stars at high airmass. That is why airmass slopes differed in sign
between nights and the columns looked good only to about 3%. Refitted with
sequence airmasses the three nights agree to a few tenths of a percent per
airmass; those records replace these once GitHub #4 is done. Until then do not
compare vertical columns between frames, and do not quote a trend from one
night.

**IGRINS `plp_telluric` is not a cross-check.** The pipeline's own
`MODEL_TELTRANS` over-absorbs by 0.23-0.32 in transmission wherever there is
real absorption; it is a template, not a per-observation retrieval. It is kept
for comparison, not validation.

**Two IGRINS order-frames are pathological** — H107 and H104 of the DCT 2018 run,
at 115 and 33 sigma out of 516 — and are unexplained. Cut on
`residual_rms_over_noise` if you pool statistics.

## The solar spectra

Every solar spectrum here was fitted window by window -- 30 cm-1 windows on one tiling
shared by every run, so a window can be compared across products; niratl by its own
atlas pages -- and comes as three files: a corrected spectrum, a transmission, and the record.

| spectrum | where | um | air mass | windows fitted |
|---|---|---|---|---|
| Kitt Peak 1990-12-18 file 5, disc centre | `ftsspec_901218_5_*`, `photatl_corrected.h5` | 1.10-5.33 | 1.985 | 222 of 224 |
| Kitt Peak 1990-12-18 file 4, disc centre | `ftsspec_901218_4_*` | 1.10-5.33 | 4.73 | 215 of 223 |
| Kitt Peak 1983-06-26 file 2, disc centre | `ftsspec_830626_2_*` | 0.48-1.17 | 5.37 | 307 of 407 |
| Kitt Peak 1983-06-26 file 3, disc centre | `ftsspec_830626_3_*` | 0.48-1.17 | 3.085 | 287 of 406 |
| niratl (Wallace et al. 1993), disc centre | `niratl_*` | 0.735-1.124 | 1.10 | 164 of 188 pages |
| Wallace et al. 2011 flux, regions 1-3 | `wallace2011_reg{1,2,3}_*` | 0.44-1.00 | 1.5, 1.4, 1.23 | 100, 91, 25 |
| fluxatl (Kurucz et al. 1984) flux, scans 109-115 | `fluxatl_{109,111,113,115}_*` | 0.50-1.30 | each a co-add | 31, 86, 98, 70 |
| IAG flux (Reiners et al. 2016), VIS | `iag_vis_*` | 0.50-1.06 | nine-day co-add | 273 of 353 |
| IAG flux, NIR | `iag_nir_*` | 1.00-2.29 | six-day co-add | 170 of 187 |

`*` is `_corrected.h5` and `_transmission.h5`, both in `data/corrected/`; the records
are under `data/corrected/solar/`. A window not counted as fitted had no species above
the run's optical-depth cut (it is **not corrected**, not clean: lines up to ~1% deep may
remain), or was opaque, or was withheld. `docs/solar_coverage.png` draws all of it
against the corrections the atlases' authors and others made.

**The corrected files** (`export_atlas_corrected.py`, one row per window or atlas page)
hold `observed`, `corrected`, `effective_transmission`, `transmission_unconvolved`
(diagnostic only), `reliable`, and a per-pixel `status`:

```python
import h5py
with h5py.File("data/corrected/iag_vis_corrected.h5") as f:
    for i, (v1, v2) in enumerate(zip(f["v1"], f["v2"])):
        good = f["reliable"][i] & (f["status"][i] == 0)
        nu, spectrum = f["wavenumber_cm1"][i][good], f["corrected"][i][good]
```

`status` 0 is fitted; 1 not corrected (above); 2 not fitted; 3 a column at its upper
bound; 4 withheld because the atlas's saturated cores sit more than 5% of the continuum
above the model -- IAG NIR's 7216 and 7336 cm-1, a zero-level pedestal that the division
would turn into emission. The file's own dataset descriptions say the same. `reliable`
is the 0.15 transmission floor; recut it for line work as above.

**Which file for file 5.** `photatl_corrected.h5` is photatl's own 258 pages (the
atlas *is* file 5, rescaled per page as `gain * file 5 + offset`), corrected by the
file-5 fit, so it keeps the authors' zero level; the four 1.9 um band pages there come
from a fit of photatl itself. `ftsspec_901218_5_corrected.h5` is the raw file, window by
window. Use photatl's if you want the atlas's layout and zero level, the raw file if you
want the observation.

**The transmission files** are as [above](#if-you-want-the-atmosphere-not-our-spectrum),
with one difference for a co-add (IAG, fluxatl): `zenith_angle_deg` is NaN and `/coadd`
holds the days or scan intervals and their nodes; the transmission is the node-weighted
mean of the slant transmissions, which is what the atlas saw.

### How each was fitted

All of them share the physics: AER v3.9 lines with HITRAN's H2O pressure shifts, the per-line
strength table `data/line_strength/step3_iter1.npz` (8800-17600 cm-1), MT_CKD with O2
collision-induced absorption, and telluric columns fitted only from **solar-clean
pixels** -- where the Payne Zero solar model is under 1% deep -- so the solar model's
error does not leak into them. They differ where the data asked them to:

| | line width scale (H2O, O2) | CO2 line coupling | also |
|---|---|---|---|
| files 4, 5 | 1.00 | on | CO pinned at the 4.6 um median in the six 2.3 um windows (4156-4306), where telluric CO sits under solar CO |
| 1983 pair | 1.06 | off | file 3's air mass 3.085 is ours, its header reads `?.??` |
| niratl | 1.11 | off | air mass 1.10 from the O2 A band |
| Wallace 2011 | 1.00 | off | region 3: water, velocity and line spread pinned from the 571 nm band |
| fluxatl | 1.00 | on | scan 109: the same pins, from the same band |
| IAG VIS, NIR | 1.00 | off, on | |

The widths are measured: the 1983 campaign's spectra want lines 6-11% wider than AER
gives, the others do not (`docs/solar_fit_plan.md` §4r). Each record states every one of
these settings in `/physics`, with the SHA-256 of the strength table.

### How good they are

Scored by **flatness**: the corrected spectrum's scatter about a smooth curve at pixels
with telluric absorption but no solar line, as a percentage of the continuum, median over
windows (`scripts/score_telluric_correction.py`). "Telluric" pixels have T < 0.97, "deep"
T < 0.7. Where the atlas's authors (or someone else) corrected the same spectrum, theirs is
scored on the same pixels:

| | telluric, ours | theirs | deep, ours | theirs |
|---|---|---|---|---|
| file 5 vs photatl (Livingston & Wallace 1991) | 0.738% | 0.791% | 1.319% | 1.270% |
| file 4 | 1.240% | | 2.412% | |
| 1983 file 2 / file 3 | 0.943% / 0.833% | | 2.19% / 2.03% | |
| niratl vs the authors' solar column | 0.501% | 0.628% | | |
| Wallace 2011 reg 1 / reg 2, authors' pixels | 0.576% / 0.548% | 0.705% / 0.706% | | |
| Wallace 2011 reg 3, T < 0.99 | 0.42% | 0.52% | | |
| fluxatl 111 / 113 vs Kurucz 2005 | 0.534% / 0.766% | 0.500% / 0.741% | 0.942% / 1.353% | 1.123% / 1.123% |
| IAG VIS | 0.89% | | 1.97% | |
| IAG VIS against Baker et al. 2020's telluric-free Sun | 1.10% | | 2.84% | |
| IAG NIR | 1.19% | | 2.32% | |

Sources: `docs/kitt_peak_clean_1990_score.json`, `docs/niratl_correction_score.json`,
`docs/fluxatl_kurucz_score.json`, `docs/iag_correction_score.json`,
`docs/iag_baker_score.json`, `docs/iag_nir_correction_score.json`; the 1983 pair and
Wallace in `docs/solar_fit_plan.md`, "Done 2026-10-24". Read them as:

- **On telluric pixels ours is about as flat as the authors' or flatter**, everywhere
  they made one. On deep pixels it is mixed: photatl and Kurucz's scan 113 are flatter,
  Kurucz's scan 111 is not. Their corrections were tuned by hand and left out what they
  could not fix (photatl's interpolated pixels, Kurucz's straight stretches), which the
  score leaves out for both.
- **Kurucz smoothed his spectrum** by three times each scan's dispersion; ours is not.
  Smoothed the same way ours is 0.489% / 0.732% (telluric) and 0.638% / 1.300% (deep).
- **Baker's 1.10% is the honest number for the visible**: an independent telluric-free
  Sun, not a flatness. About half of it is line-list error every fit with this model
  shares (`docs/iag_residual_attribution.json`); a 10%-deep solar line on a telluric
  feature core can carry ~0.1 dex of it into a log gf.
- **IAG NIR is the worst score and the best correction per unit absorption**: Göttingen
  puts four times Kitt Peak's water on the same pixels, and the error scales with depth
  (`docs/iag_nir_file5_comparison.json`).
- **Do not quote the 1983 pair's B-band O2 columns**: about 5% high against its A band,
  a property of that pair and nothing else (§4n).

### What to take for what

- **Disc centre, 1.1-5.3 um**: file 5 (`photatl_corrected.h5`). File 4 is the same day
  at 2.4 times the air mass: a check, not a better spectrum.
- **Disc centre, 0.74-1.12 um**: niratl, the flattest; it combines spectra of the same
  1983 June campaign as the pair, which reaches down to 482 nm.
- **Flux, 0.50-1.06 um**: IAG VIS (resolution and S/N); Wallace 2011 and fluxatl are
  independent checks with different skies and different wavelength scales -- fluxatl's
  sits 0.5-0.7 km/s off in 501-1000 nm.
- **Flux, 1.0-2.3 um**: IAG NIR; avoid 7216 and 7336 cm-1.
- **Telluric-free, from orbit**: ACE-FTS (Hase et al. 2010; Bhusal et al. 2026) over
  2.25-5.47 um, near disc centre, not corrected here; it is the yardstick for 2.6-2.85
  and 4.2-4.5 um, where nothing on the ground sees the Sun.

## Regenerating from scratch

Needs the AER line files and the MT_CKD file from `scripts/bootstrap_lblrtm.sh`,
but not the LBLRTM binary.

```bash
# Arcturus, the whole atlas (sharded below)
uv run python scripts/fit_fts_batch.py --arcturus \
    --epoch-profile summer=data/profiles/kitt_peak_1994_spring_afgl.csv \
    --epoch-profile winter=data/profiles/kitt_peak_1994_winter_afgl.csv \
    --profile data/profiles/kitt_peak_1994_spring_afgl.csv \
    --scan-profile data/profiles/kitt_peak_1994_spring_afgl.csv \
    --o2-cia --line-coupling --telluric-from-clean-pixels \
    --line-strength-table data/line_strength/step3_iter1.npz --minimum-snr 20

# IGRINS: fetch a night, build its atmosphere, fit it
uv run python scripts/download_rrisa_standard.py --night 20181220 --min-snr 200
uv run python scripts/igrins_site_profile.py \
    --spec data/igrins/20181220_*/SDCH_*.spec.fits --output data/profiles/dct_2018.csv
uv run python scripts/fit_igrins_standard.py --profile data/profiles/dct_2018.csv \
    --stellar data/stellar/a0v_payne_zero_hk.npz --vsini-kms 150 \
    --spec data/igrins/20181220_*/SDCH_*.spec.fits
```

Pass the whole night's frames in one command, not one at a time: the driver
loops orders outside and frames inside, which shares the grid, the opacity and
the XLA compilations and is worth about two thirds of the runtime. It also needs
five frames to measure the instrument response.

Sharded, each process takes every Nth page (both epochs of a page stay together,
so a window's species scan runs once) and writes its own summary; the shards are
joined, and the same command without `--shard` writes one record from the saved
fits without refitting:

```bash
ARGS="--arcturus ..."          # the arguments above
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$((i % 2)) XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run python scripts/fit_fts_batch.py $ARGS --shard $i/6 \
      --summary data/corrected/atlas/shard${i}_summary.json \
      --record  data/corrected/atlas/shard$i.h5 &
done; wait
uv run python scripts/merge_fts_shards.py data/corrected/atlas arcturus \
    --shards shard{0,1,2,3,4,5}_summary.json
uv run python scripts/fit_fts_batch.py $ARGS      # --resume: writes arcturus_atlas.h5
```

Six shards on two GPUs took about 2 h for the 598 page-epochs.

The sinc is fixed atlas-wide at R = 115,700, the constant-resolving-power law the
atlas follows. The superseded run measured the path difference per page instead;
that measurement scattered +-12% about the law and the per-page Gaussian absorbed
it and railed (`lsf_sigma_kms` at a bound on 13.4% of pages). With the sinc alone
at 115,700 the fitted Gaussian carries the rest: in quadrature the two gave
**100,504, the R = 100,000 the atlas documents**. `fit_arcturus_batch.py` with
`merge_atlas_records.py` still reproduces the superseded run, into
`data/corrected/superseded_pre_solar_era/`.

For a night whose headers carry no weather — every Gemini South frame from 2020
— use ERA5 instead, which needs only a position and a time:

```bash
uv run --with aiohttp python scripts/era5_site_profile.py \
    --spec data/igrins/20210316_*/SDCH_*.spec.fits --output data/profiles/gemini_2021_era5.csv
```

## What this does not do

**It does not correct the sunspot atlas** (`spot1atl`, Wallace & Livingston 1992): the
umbra has its own water and other molecular lines, so there are no solar-clean pixels to
fit the sky from. Nor the mid-IR "telluric" spectra, which are ratios of two air masses,
not solar spectra (`docs/solar_atlases.md`).

**Cite the atlas if you publish.** The Arcturus data is Hinkle, Wallace &
Livingston 1995, *Infrared Atlas of the Arcturus Spectrum, 0.9-5.3 um*, ASP
(ISBN 1-886733-04-X); PASP **107**, 1042,
doi [10.1086/133660](https://doi.org/10.1086/133660). The README shipped with
the data asks for that citation explicitly. The README and the full provenance,
including where to find the atlas now that the FTP URL in every paper is dead,
are in `README.md` and `docs/arcturus_fit.md`.

For the Sun, cite the data you used as well as this work:

- photatl: Livingston & Wallace 1991, NSO Technical Report 91-001; with niratl, the
  refereed summary is Wallace, Livingston, Hinkle & Bernath 1996, ApJS 106, 165.
- niratl: Wallace, Hinkle & Livingston 1993, NSO Technical Report 93-001.
- The raw Kitt Peak files 1990-12-18 and 1983-06-26: the NSO/Kitt Peak FTS archive,
  McMath-Pierce telescope.
- Wallace 2011: Wallace, Hinkle, Livingston & Davis 2011, ApJS 195, 6,
  doi [10.1088/0067-0049/195/1/6](https://doi.org/10.1088/0067-0049/195/1/6).
- fluxatl: Kurucz, Furenlid, Brault & Testerman 1984, *Solar Flux Atlas from 296 to
  1300 nm*, NSO Atlas No. 1; Kurucz's 2005 re-reduction, which the score compares against,
  is Kurucz 2006, astro-ph/0605029.
- IAG: Reiners, Mrotzek, Lemke, Hinrichs & Reinsch 2016, A&A 587, A65; Baker et al.'s
  telluric-free Sun is Baker, Blake & Reiners 2020, ApJS 247, 24.

Where each was downloaded from, with checksums, is in the sibling project's
`data/atlases/*/PROVENANCE.md`.

**It does not correct science targets.** Every fit here is of a telluric
standard or of Arcturus. Taking a fitted atmosphere to a target observed at a
different airmass and time is a step that does not exist yet. If that is what
you came for, it is the next thing to build, not something to look for.
