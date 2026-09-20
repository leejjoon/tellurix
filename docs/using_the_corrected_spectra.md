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
| `transmission` | unconvolved transmission — *diagnostic only*, see above |
| `continuum` | the fitted Chebyshev continuum |
| `residual` | `observed - model_flux` |
| `mask` | True where a pixel was fitted |
| `reliable` | `mask` and transmission above 0.15 — **use this to select pixels** |

Arcturus adds `corrected_normalized` (corrected divided by the fitted
continuum), and the atlas's own `atlas_telluric` and `atlas_ratioed` columns for
comparison. IGRINS adds `uncertainty` (the PLP's per-pixel sigma, a real
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

**The corrected spectrum is not normalised.** Values above 1 are expected: the
correction divides out the atmosphere, not the fitted continuum. Divide by
`continuum` if you want a normalised spectrum. On the atlas the continuum level
sits at 1.0011 median, but 10% of pages are off by more than 2%.

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
