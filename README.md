# tellurix

`tellurix` is an experimental differentiable forward model for terrestrial
absorption in high-resolution spectra. It wraps ExoJAX opacity calculators with
a layered Earth atmosphere, slant-path transmission, and an observation model
for wavelength corrections, continuum, and a Gaussian line-spread function.

The first validation target is LBLRTM 12.17 in representative IGRINS H- and
K-band intervals. See `docs/lblrtm_exojax_research.md` for the rationale and
`docs/validation.md` for the reference-data workflow.

The core is `tellurix` and knows no instrument. Readers and calibrations for
particular data are in `tellurix_fts` (the Arcturus and NSO solar FTS atlases)
and `tellurix_igrins` (IGRINS reduced spectra), which build on it.

## Environment

```bash
UV_CACHE_DIR=.uv-cache uv sync --dev
UV_CACHE_DIR=.uv-cache uv run pytest
```

JAX runs in 64-bit mode inside this package. Database downloads and LBLRTM
binaries belong under the ignored `data/` directories.

After installing from PyPI or GitHub, download only the runtime data you need:

```bash
# Small MT_CKD 4.3 coefficient file
tellurix-download-data mt-ckd

# AER line file 3.9 (approximately 406 MB compressed)
tellurix-download-data aer-lines

# Or both
tellurix-download-data all
```

The default location is `$TELLURIX_DATA`, when set, or
`~/.local/share/tellurix`. Use `--output /path/to/data` to choose another
location. Downloads use version-pinned official AER sources and are verified
with SHA-256 before use. The line archive is extracted with path and file-type
checks.

The resulting runtime paths are:

```text
~/.local/share/tellurix/mt_ckd/absco-ref_wv-mt-ckd.nc
~/.local/share/tellurix/aer_v_3.9/line_files_By_Molecule/
```

To reproduce the reference compilers, builds, and line-file download:

```bash
./scripts/bootstrap_lblrtm.sh
```

After bootstrapping, generate the committed-style LBLRTM reference fixture
from the bundled atmospheric profile with:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/run_lblrtm_reference.py
```

The high-resolution TAPE files remain under `data/lblrtm/run_reference/`; the
script writes an IGRINS-resolution spectrum and provenance JSON under
`tests/data/`.

Run the first end-to-end comparison, using AER 3.9 CO parameters in both
LBLRTM and ExoJAX, with:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_aer_co.py
```

The script records the error metrics in `tests/data/aer_co_validation.json`.
The `AERLineDatabase` adapter can likewise load the H2O, CO2, O3, N2O, CH4,
and O2 files extracted by the bootstrap script.

## Performance benchmark

The benchmark measures a six-layer, water-dominated 5000--5020 cm-1 sub-order,
including about 1,900 AER lines after wing padding, ExoJAX Direct opacity,
slant transmission, LSF convolution, and 512 detector pixels. It reports
compilation separately from synchronized steady-state calls:

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark.py \
  --platform cpu --output benchmarks/results/cpu.json
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark.py \
  --platform gpu --output benchmarks/results/gpu.json
```

See `docs/performance.md` for the measured CPU/GPU results and interpretation.

For faster terrestrial HITRAN/AER Direct opacity with the same line and wing
formulas, prepare the backend with:

```python
backend = ExoJAXOpacityBackend.prepare(
    databases, nu_grid, methods="direct_sparse", vectorize_layers=True
)
```

This selects a compact list of possible line-core pairs and batches the
atmospheric layers. Temperatures above 400 K automatically use original Direct.
Use `methods="direct"` to retain the reference path.

Add `mixed_precision=True` to this preparation call for float32 wing
calculations with float64 line centers, cores, accumulation, and instrument
model. On the tested full H2O order this gives another roughly 11x forward
and 8x gradient speedup with a maximum flux change of 3.5e-8. See the
mixed-precision section in `docs/performance.md` for scope and reproduction,
and [extended accuracy validation](docs/mixed_precision_validation.md) for
multi-species, atmospheric-state, spectral-derivative, and fit-recovery checks.
The extended checks pass, including paired mixed-precision and float64 fits.

## Optional pressure shifts, MT_CKD, and LBLRTM correction

The default `accuracy_mode="fast"` preserves the optimized behavior above.
For runtime MT_CKD 4.3 physics across pressure-temperature profiles, load the
official coefficient file distributed with LBLRTM:

```python
from tellurix import MTCKDWaterContinuum, TelluricModel, default_data_directory

continuum = MTCKDWaterContinuum.from_netcdf(
    default_data_directory() / "mt_ckd/absco-ref_wv-mt-ckd.nc", nu_grid
)
model = TelluricModel(
    profile, nu_grid, backend,
    accuracy_mode="mt_ckd", continuum=continuum,
)
```

This path evaluates the self and foreign continua from each layer's pressure,
temperature, H2O VMR, and molecular column in JAX. The prepared continuum can
therefore be supplied to models with different atmospheric profiles and remains
differentiable with respect to fitted H2O column scales. The coefficient file
itself remains an external AER/LBLRTM input with its original license terms.

For a fixed atmospheric profile and order grid, an offline LBLRTM correction
template can add MT_CKD H2O continuum, remaining continua, and empirical
per-species differences from LBLRTM while retaining JAX derivatives during
fitting. Prepare its sparse Direct backend with the same pressure-shift option
used by the builder:

```python
backend = ExoJAXOpacityBackend.prepare(
    databases, nu_grid, methods="direct_sparse", vectorize_layers=True,
    temperature_range_k=(profile.temperature_k.min(), profile.temperature_k.max()),
    maximum_pressure_bar=profile.pressure_layer_bar.max(),
    pressure_shift=True,
)
correction = LBLRTMOpticalDepthCorrection.load("data/corrections/order.npz")
model = TelluricModel(
    profile, nu_grid, backend,
    accuracy_mode="mt_ckd", correction=correction,
)
```

`mt_ckd` applies only the physics-derived H2O self and foreign continua. Use
`accuracy_mode="lblrtm_corrected"` with the same template to additionally add
the profile-calibrated line residuals and fixed background. In the tested
order, MT_CKD alone changed the 99th-percentile LBLRTM error from 0.0581 to
0.0569; the remaining line mismatch dominates.

Generate a template with `scripts/build_lblrtm_correction.py`. In the tested
5000--5020 cm-1 order, the correction reduced the 99th-percentile absolute
error against full LBLRTM from 0.0581 to 4.75e-5. The correction arrays have
negligible overhead; the pressure-shifted GPU forward path costs about 2 ms
more in the measured order. See [the corrected-mode guide](docs/lblrtm_corrected_mode.md)
for usage, assumptions, and reproduction.

## Observational data, and how to cite it

The package downloads its own spectroscopy (AER line files, MT_CKD) but **not**
the observed spectra it is validated against. Those come from elsewhere and
carry their own terms.

### The Arcturus infrared atlas

Hinkle, K., Wallace, L., & Livingston, W. 1995, *Infrared Atlas of the Arcturus
Spectrum, 0.9-5.3 um*, San Francisco: ASP (ISBN 1-886733-04-X). The accompanying
paper is PASP **107**, 1042, doi
[10.1086/133660](https://doi.org/10.1086/133660).

Ken Hinkle's README that ships with the data asks:

> We would appreciate a citation to the appropriate ASP atlas if this data is
> used in a publication.

**Where to get it.** The URL every paper cites,
`ftp://ftp.noao.edu/catalogs/arcturusatlas/ir/`, **is dead** -- NOAO became
NOIRLab and the FTP host no longer resolves. It is not in VizieR under an
Arcturus title, and not on the NSO archive, which is solar-only. Start instead
from NOIRLab's data-services page,
<https://noirlab.edu/science/data-services/other> ("High resolution spectral
atlas of Arcturus"); as retrieved on 2026-09-12 the files sat in the Google
Drive folder `1m7Vstoh6uTMmPb7b5FcflDA_OqWf19M9`, subfolder `ir`. The `ir`
volume is 282 files, 34 MB, one per page of the printed monograph.

This repository has no downloader for it, and a run record pins it by absolute
path plus a per-page sha256 -- so the hash will catch a changed file but not a
moved directory. `docs/arcturus_fit.md` has the rest of the provenance and what
the data's structure implies for interpreting a fit.

### Solar atlases

Not used yet. The NSO Kitt Peak FTS solar atlases live in the sibling project
`differentiable_stellar_spectroscopy`, which is asking for a solar telluric
retrieval from this package; `docs/solar_atlases.md` is the survey and points at
their documentation. Any publication using those data must carry the NSO
acknowledgement: *NSO/Kitt Peak FTS data used here were produced by NSF/NOAO.*

### IGRINS A0V standards

From the RRISA reduced archive; `scripts/download_rrisa_standard.py` fetches
them and records each file's sha256. See `docs/igrins_a0v.md`.
