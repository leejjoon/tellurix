# Optional MT_CKD and LBLRTM-corrected modes

The default `fast` mode evaluates ExoJAX/AER Voigt lines and any explicitly
supplied continuum backend. The `mt_ckd` mode accepts either the native
`MTCKDWaterContinuum` backend or a fixed optical-depth template. The native
backend evaluates MT_CKD 4.3 from the current pressure-temperature-abundance
profile and should be preferred for fitting across atmospheric states.

The fixed template modes generated offline by LBLRTM 12.17 are:

- `mt_ckd` with a correction adds only its isolated H2O self and foreign
  MT_CKD continua;
- `lblrtm_corrected` adds those continua plus empirical per-species line
  residuals and the fixed reference background.

LBLRTM is not executed during prediction or fitting.

## Native runtime MT_CKD

```python
from tellurix import MTCKDWaterContinuum, TelluricModel

continuum = MTCKDWaterContinuum.from_netcdf(
    "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc", nu_grid
)
model = TelluricModel(
    profile,
    nu_grid,
    backend,
    accuracy_mode="mt_ckd",
    continuum=continuum,
)
```

The implementation follows LBLRTM 12.17's `mt_ckd_h2o_module.f90`: it applies
the self-continuum temperature exponent, self and foreign collider densities,
the radiation factor, and the same four-point cubic interpolation. Coefficients
are loaded once and the fixed pressure-temperature spectral factors are
calculated when constructing `TelluricModel`; predictions perform no file I/O
or coefficient interpolation.
The layer pressure uses the hydrostatic column-weighted mean of its pressure
edges. Run `scripts/validate_mt_ckd.py` to reproduce direct comparisons with
LBLRTM over three pressure-temperature-water states.

Against isolated LBLRTM 12.17 continuum calculations at 5000--5020 cm-1, the
three cases give median relative optical-depth errors of 0.024%, 0.076%, and
0.029%. The largest 99th-percentile error is 0.107%; see
`packages/tellurix/tests/data/native_mt_ckd_validation.json` for the states and complete metrics.

The builder first uses the same pressure edges as the JAX layers and enables
the AER/HITRAN air-pressure line shifts that ExoJAX 2.5 Direct omits. LBLRTM
scales those shifts with density, so the JAX implementation uses
`delta_air * (P / 1 atm) * (296 K / T)`.

The resulting template contains two kinds of terms:

- MT_CKD 4.3 H2O self continuum, scaled with the square of the global H2O
  column multiplier;
- MT_CKD H2O foreign continuum, scaled linearly with H2O;
- an empirical line residual for every profile molecule, scaled linearly with
  that molecule. It is the measured difference between a species-only LBLRTM
  run and the selected JAX backend, rather than an attribution to one physical
  mechanism;
- the remaining full-atmosphere LBLRTM background at the reference profile.
  This includes other continua and small numerical decomposition residuals and
  is held fixed under abundance changes.

For this 5000--5020 cm-1 case, CO2 line coupling is outside the AER coupling
database's stated 597--2503 cm-1 range. LBLRTM also does not consume the AER
speed-dependence files used by MonoRTM. Those effects therefore do not explain
this order's correction. After fixing the profile extent, pressure shifts
reduce the fast 99th-percentile error from 0.107 to 0.0581. The residual
remains explicitly empirical; what it is made of is below.

### What the 5000--5020 cm-1 gap is

Measured by `scripts/attribute_lblrtm_gap.py` into
`docs/lblrtm_gap_attribution.json`, from the template and the LBLRTM runs that
built it:

- **Mostly, a different atmosphere.** The TAPE5 writer hands LBLRTM the
  6-layer profile as 7 levels, chosen so that adjacent levels average to the
  layer values. LBLRTM re-layers them into 18 and interpolates water
  exponentially between levels, which sags below that average: its water
  column is 2.143e22 cm-2 against tellurix's 2.440e22, 13.8% less. Air and CO2
  agree to 0.4%. tellurix's integrated H2O line optical depth is 11.7% high and
  CO2's 2.3% low.
- **Not LBLRTM's 25 cm-1 line cutoff.** LBLRTM truncates every line at
  25 cm-1 and subtracts its value there (`oprop.f90`, CONVF4; the CO2 chi
  factor is switched off in 12.17). tellurix keeps the full Voigt line. The
  difference is real but small: applying the cutoff moves the H2O residual's RMS
  from 0.0927 to 0.0926 in optical depth, and the fitted error below from
  0.0121 to 0.0120.
- **What a fit sees.** With the H2O and CO2 column scales and a linear
  continuum fitted to LBLRTM, as any fit would (H2O 0.92, CO2 1.02), the R=45,000
  transmission agrees to 0.15% median and 1.2% at the 99th percentile. That
  remainder still includes the two codes' different vertical distribution of
  water, so it is an upper bound on the physics difference.

### With identical layers

`scripts/compare_lblrtm_layers.py` hands LBLRTM tellurix's own layers as
IATM=0 layer input (`LBLRTMRunConfig(user_layers=True)`): each layer's
pressure, temperature and molecular columns, with the air-weighted mean
pressure for both codes. LBLRTM's TAPE6 echoes the columns exactly. What is left
is physics (`docs/lblrtm_identical_layers.json`), over 5000--5020 cm-1 at
R=45,000 against LBLRTM with its continua:

| | median | 99th percentile | max |
|---|---:|---:|---:|
| tellurix lines + MT_CKD, nothing fitted | 0.37% | 1.15% | 1.19% |
| H2O, CO2 scales and a linear continuum fitted | 0.05% | 0.69% | 0.76% |

The fitted scales are 1.001 for H2O and 1.000 for CO2: with the same
atmosphere, nothing is left for a column to absorb. Per species, tellurix's
integrated H2O line optical depth is 0.6% above LBLRTM's (residual RMS 0.0013,
down from 0.093 with LBLRTM's own layering), CO2's 1.5% above (RMS 0.013).
CO2 is now the larger line residual. The rest is continuum: LBLRTM's
continua average an optical depth of 0.0066 here, MT_CKD's water part 0.0028;
the remainder is smooth, and the fitted continuum absorbs it.

All terms are vertical optical depths. The normal model airmass calculation
therefore scales them with zenith angle. Templates are valid only for their
exact pressure-temperature-abundance profile and high-resolution grid; model
construction rejects a mismatch. Generate another template when either
changes materially.

## Build and use a template

After running `scripts/bootstrap_lblrtm.sh`, generate a template for an order:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/build_lblrtm_correction.py \
  --v1 5000 --v2 5020 \
  --profile data/profiles/example_midlatitude.csv \
  --output data/corrections/lblrtm_5000_5020.npz
```

The builder runs three full-atmosphere continuum calculations plus one
continuum-free LBLRTM calculation per profile molecule. It saves a compressed
template and provenance JSON, reloads the result, checks first derivatives,
and measures fast and corrected transmission against full LBLRTM.

To enable line shifts, prepare the opacity backend with `pressure_shift=True`:

```python
backend = ExoJAXOpacityBackend.prepare(
    databases,
    nu_grid,
    methods="direct_sparse",
    temperature_range_k=(profile.temperature_k.min(), profile.temperature_k.max()),
    maximum_pressure_bar=profile.pressure_layer_bar.max(),
    vectorize_layers=True,
    pressure_shift=True,
)
```

Then select the continuum-only mode:

```python
from tellurix import LBLRTMOpticalDepthCorrection, TelluricModel

correction = LBLRTMOpticalDepthCorrection.load(
    "data/corrections/lblrtm_5000_5020.npz"
)
model = TelluricModel(
    profile,
    nu_grid,
    backend,
    accuracy_mode="mt_ckd",
    correction=correction,
)
```

Change the mode to `lblrtm_corrected` when the calibrated residual terms are
desired. The same correction file supports both modes.

Version-2 correction files record the line baseline used to derive their
empirical residuals. Full corrected-mode construction rejects an unshifted
ExoJAX backend instead of silently applying the wrong residual. The `mt_ckd`
terms are independent of that line baseline, so they can also be combined
with another opacity backend.

Omitting `accuracy_mode` retains the existing `fast` behavior. The `mt_ckd`
mode does not add `reference_background_optical_depth` or any
`line_residual_optical_depth` arrays. Generated
templates live under ignored `data/corrections/` because they depend on the
chosen profile, order grid, LBLRTM build, and line database.

## Measured result

The example 5000--5020 cm-1 order uses a padded 4975--5045 cm-1 grid with
2,517 samples and the six-layer example atmosphere. Against full LBLRTM,
among samples with transmission above 0.05:

| Mode | Median absolute error | 99th percentile | Maximum |
|---|---:|---:|---:|
| Fast, pressure-shifted float64 | 3.904e-3 | 5.811e-2 | 7.893e-1 |
| Pressure shifts + MT_CKD only | 5.607e-3 | 5.689e-2 | 7.879e-1 |
| LBLRTM corrected | 4.80e-8 | 4.75e-5 | 3.78e-4 |

MT_CKD slightly improves the 99th-percentile error but increases the median
in this order. The fast comparison carries 13.8% more water than LBLRTM's (see
"What the 5000--5020 cm-1 gap is" above), so adding a known continuum on top
of it need not improve every aggregate agreement metric. The `mt_ckd` mode is the
physically attributable option; `lblrtm_corrected` is the close LBLRTM
emulator.

The same template was also checked against new LBLRTM runs away from its
calibration point:

| Case | Median absolute error | 99th percentile | Maximum |
|---|---:|---:|---:|
| H2O column 0.5x | 3.21e-4 | 7.24e-4 | 3.88e-3 |
| H2O column 2x | 5.08e-4 | 1.54e-3 | 3.19e-3 |
| Airmass 1.5 | 3.07e-4 | 7.96e-4 | 1.86e-3 |
| Airmass 2.5 | 8.98e-4 | 2.34e-3 | 2.95e-3 |

The builder requires 99th-percentile error below 1e-3 at the reference and
below 5e-3 for these four stress cases. Individual saturated-line pixels can
have larger errors under abundance changes, as shown by the maximum column.

On the RTX 5000 Ada, enabling pressure shifts changed the mixed-precision
H2O benchmark from 1.89 to 3.88 ms per forward call and from 3.31 to 3.56 ms
per objective-gradient call. Adding the precomputed correction arrays has
negligible cost relative to opacity evaluation. The bounds passed during
backend preparation must cover the fixed profile; wider bounds create more
sparse core pairs and cost more.

This test establishes calibrated, reference-profile accuracy, not universal
physical completeness under large abundance or pressure-temperature profile
changes. The H2O and per-species line terms have explicit scaling; the
empirical residual need not follow that linear approximation indefinitely and
the unattributed background remains fixed. Regenerate the
template outside the validated 0.5--2x water or 1--2.5 airmass range, when
changing the pressure-temperature profile, or when saturated-line accuracy
above the tabulated level is required.
