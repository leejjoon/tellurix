# Using the corrected spectra

This is the short version, for someone who wants to *use* the output rather than
reproduce it. `docs/arcturus_fit.md` and `docs/igrins_a0v.md` are the long
versions: they are development records, organised by how the work unfolded, and
you should not have to read either to get a spectrum out.

Two datasets exist.

| | Arcturus atlas | IGRINS A0V standards |
|---|---|---|
| what was fitted | Hinkle, Wallace & Livingston 1995, 598 page-epochs | 43 standards over 4 nights, 3 telescopes |
| keyed on | `(page, epoch)` | `(frame, order)` |
| record | `data/corrected/atlas/arcturus_atlas.h5` | `data/corrected/igrins/*/record.h5` |
| arrays | `data/corrected/atlas/*.npz` | `data/corrected/igrins/*/*.npz` |
| residual | 4.22 sigma median | 1.30-1.86 sigma median |

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

Arcturus adds `corrected_normalized` (corrected divided by the fitted
continuum) and the atlas's own `atlas_telluric` and `atlas_ratioed` columns for
comparison. **`corrected_normalized` is not a stellar normalization** -- see the
caveat below before using it as one. IGRINS adds `uncertainty` (the PLP's per-pixel sigma, a real
measurement), `plp_telluric` and `plp_continuum` (the pipeline's own, for
comparison only — see the caveat below), and `response_pattern` (the instrument
response that was divided out).

```python
import numpy as np
a = np.load("data/corrected/atlas/ab5000__summer.npz")
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

One HDF5, **12.4 MB for all 598 Arcturus page-epochs** against 45.6 MB of
`.npz`, readable with nothing but `h5py`. Rows are padded to the longest
spectrum with NaN, so every array is a plain 2-D `(row, pixel)` block. It
carries the arrays below, the fitted parameters, the formal sigmas and
correlations, and the whole `/config`, `/physics` and `/inputs` provenance
copied from the record, so it stands on its own.

```python
import h5py
with h5py.File("data/corrected/arcturus_spectra.h5") as f:
    i = list(f["key"].asstr()).index("ab5000_ summer")     # or "SDCH_20181220_0100 H10"
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

That writes T on the forward model's own grid -- 4.00 samples per resolution
element at R = 100,000 (a 0.7494 km/s step), no interpolation anywhere in the
path. **8.0 MB for all 598 Arcturus page-epochs, about 50 minutes on one GPU.**
Grids are shared, because both epochs of a page sit on one window, so the file
holds 310 grids for 598 rows:

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
regenerated on any other grid from this file plus a line list.

`--check` interpolates the fine grid back down to the pixels and compares
against the cached `transmission`:

| record | agreement | why |
|---|---|---|
| Arcturus atlas | 2.2e-16 | same code path, machine precision |
| IGRINS night | 3.1e-6 | the driver saves T from the refrozen linearized backend, this script evaluates the exact kernel; 1600x below the noise |

### Stitching: the pixel products tile, the transmission grids overlap

The raw atlas pages overlap each other by about 5 cm-1 -- 282 of 287 adjacent
summer pages do. `page_windows()` trims 2.5 cm-1 off each end so the *fitted*
windows tile instead: **0 of 287 overlap**, with a median gap of 0.02 cm-1,
which is one atlas sample. Everything on the pixel grid inherits that, so the
`.npz` arrays and the spectra export can be concatenated directly. The cost is
2.64% of the raw union, lost at the outer edge of each of the atlas's ~38
disconnected coverage blocks, not between pages.

**The transmission export is the exception.** It is on the model grid, which
`trim_wavenumber_grid` pads by 5 cm-1 on each side so line wings are present, so
adjacent rows overlap by 9.99 cm-1 -- 282 of 287 pairs. Concatenating them
double-counts. Cut each row to its own fitted window first, which `/parameters`
carries:

```python
v1, v2 = f["parameters"]["v1"][i], f["parameters"]["v2"][i]
inside = (nu >= v1) & (nu <= v2)          # now the rows tile
```

The margin is not junk -- it is the same atmosphere, extrapolated past the data
that constrained it. Where one page is inside its window and its neighbour is in
its margin, the two transmissions agree to a median of 0.0009 against 0.0059 of
pixel noise. But the worst point in such an overlap is a median 0.027 across
pairs, because a small difference in fitted column becomes a large difference in
a line core. Use the margin for wings, never as a substitute for the neighbour's
own fit.

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
530 KB for the whole atlas against 47 MB of arrays — is what is kept in git and
what you should archive or ship.

```python
from jax_telluric import read_record, parameters_from_row
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

To rebuild the arrays from a record:

```bash
uv run python scripts/rebuild_arcturus_page.py --page ab5000_ --epoch summer --check
```

That script shares no code with the fitting driver on purpose. It agrees with
the cached arrays to 2.8e-7, which is the evidence that the record is
sufficient.

## Caveats that will bite you

**The formal errors are not uncertainties.** `record.sigma` and
`FitResult.covariance` invert the objective's real Hessian, but they assume
independent pixel errors, and the residual is dominated by *correlated* model
error. Measured: 0.76% formal on the water column against 6.5-9% of sub-window
scatter — three to nine times too small. Use `record.correlation` for
degeneracies, which survives a wrong noise model, and an empirical study for an
error bar.

**The corrected spectrum is not normalised, and `corrected_normalized` does not
normalise it to the stellar continuum.** Values above 1 are expected: the
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

**The retrieved H2O column depends on how blanketed the page is.** Splitting the
atlas at the median blanketing of the Payne Zero source over each page window,
the heavily blanketed half retrieves 3.1% (summer) and 7.1% (winter) less water
than the lightly blanketed half. Blanketing is confounded with wavelength
(r = -0.31), but the effect survives removing it: within eight narrow wavenumber
bands the partial correlation is still -0.19 (summer) and -0.11 (winter).

The sign is what a continuum-source degeneracy predicts -- `continuum x source
x T` lets a continuum placed too low be paid for by a transmission too high --
but that mechanism has been tested and does not account for it. Refitting the
twelve most blanketed page-epochs with `fit_arcturus_page.py --continuum-anchor
0.98`, which drops the 38% of pixels where the source model sits below 0.98 so
the Chebyshev is anchored only on near-continuum pixels, moves the retrieved
H2O by a median of +0.4% and makes the residual 1.1% worse. So the effect is
real, it is not the continuum degeneracy, and it is unexplained. Treat it as a
floor on an absolute column from a single page; slopes and ratios across pages
at similar blanketing are much safer. `--continuum-anchor` remains as an
experiment flag, not a default.

**Do not use `continuum` as the stellar continuum, and do not use it at all if you
are measuring line strengths.** It is a free degree-3 Chebyshev fitted *jointly
with* a source built from fixed Payne Zero oscillator strengths, so whatever
those got wrong in a smooth way, the polynomial absorbed. Deriving gf values
from anything divided by it is circular.

`stellar_continuum` is the other half: the model's own physical continuum,
interpolated to the pixels, a prediction rather than a free function. Its units
are the model's, so only its shape means anything. Splitting the fitted tilt by
it shows how little of the page-scale slope is actually stellar:

| page | fitted tilt | stellar | remainder |
|---|---|---|---|
| ab6225_ | +3.3% | −0.6% | +3.8% |
| ab6600_ | +25.1% | −0.2% | +25.3% |
| ab5825_ | +1.9% | −1.8% | +3.8% |
| ab8450_ | +0.1% | −0.5% | +0.6% |

**`stellar_continuum` carries bound-free edges, which no fitted polynomial can
represent.** The largest single-sample step in the Arcturus model sits at
1458.8 nm, the Brackett limit, at 0.19% — three hundred times any other step in
the array. In the A0V model the same edge is **5.6%**, because at 9500 K
hydrogen bound-free is the continuum opacity. It falls inside the two bluest
IGRINS H orders, and those two fit at a median 3.76 sigma against 1.79 for the
other 25, with H26 alone at 5.44. That is confounded with the band edge, so it
is a suspicion rather than a measurement — but it is the right place to be
suspicious.

**The correction itself does not carry our stellar model, though.** `corrected`
is exactly `observed / effective_transmission` (verified to 2.2e-16), so the
source enters only through the convolution weighting in
`Conv[T x S] / Conv[S]`. Recomputing that with a flat source moves it by a
median of 0.00016, p99 0.0031 and at worst 0.0097, against 0.0137 of pixel
noise. So `corrected` is safe to derive line strengths from; it is the
*continuum* that is not.

**The Arcturus residual is dominated by the stellar model, not the atmosphere.**
It is flat against transmission — 0.97 in deep absorption, 1.52 at the continuum
— which means the 4.22 sigma is a K1.5 III line-list limit. Do not read it as
telluric accuracy.

**The IGRINS well-mixed columns are good to about 3%**, set by night-to-night
scatter rather than any single night's error bar. Slopes against airmass differ
in *sign* between nights, so do not quote a trend from one night.

**IGRINS `plp_telluric` is not a cross-check.** The pipeline's own
`MODEL_TELTRANS` over-absorbs by 0.23-0.32 in transmission wherever there is
real absorption; it is a template, not a per-observation retrieval. It is kept
for comparison, not validation.

**Two IGRINS order-frames are pathological** — H09 and H06 of the DCT 2018 run,
at 115 and 33 sigma out of 516 — and are unexplained. Cut on
`residual_rms_over_noise` if you pool statistics.

## Regenerating from scratch

Needs the AER line files and the MT_CKD file from `scripts/bootstrap_lblrtm.sh`,
but not the LBLRTM binary.

```bash
# Arcturus, the whole atlas
uv run python scripts/fit_arcturus_batch.py

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

**For a new Arcturus run, fix the sinc globally.** The committed record measures
the FTS path difference per page, and that measurement scatters +-12% about the
constant-resolving-power law the atlas actually follows. The per-page Gaussian
then absorbs the scatter and rails: `lsf_sigma_kms` is at a bound on 13.4% of
converged pages, 76 of them pinned at the 0.05 km/s minimum, and it is the
single largest contributor to the 22.2% overall at-bound rate.
`fit_arcturus_page.py --sinc-resolving-power 115700` replaces the per-page
measurement with one atlas-wide value. Measured on 16 pages spanning
1868-10951 cm-1: the median residual moves by 0.2% (5.223 to 5.266 sigma, worst
page 1.8%) and `lsf_sigma_kms` at bound falls from 7/16 to 4/16. The committed
record predates this and still uses the per-page measurement.

Do not read the per-page MOPD spread as a defect, and do not take the sinc alone
as the resolution. `mopd_cm` runs 8.8 to 25.3 cm p16-p84 because the atlas holds
a constant resolving power over a 5.84x range in wavenumber, so the path
difference has to track 1/nu -- corr(mopd, nu) = -0.93. And R from the sinc alone
is 115,700, but the fitted Gaussian carries the rest: in quadrature the two give
**100,504, which is the R = 100,000 the atlas documents**.

For a night whose headers carry no weather — every Gemini South frame from 2020
— use ERA5 instead, which needs only a position and a time:

```bash
uv run --with aiohttp python scripts/era5_site_profile.py \
    --spec data/igrins/20210316_*/SDCH_*.spec.fits --output data/profiles/gemini_2021_era5.csv
```

## What this does not do

**It does not correct science targets.** Every fit here is of a telluric
standard or of Arcturus. Taking a fitted atmosphere to a target observed at a
different airmass and time is a step that does not exist yet. If that is what
you came for, it is the next thing to build, not something to look for.
