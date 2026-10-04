# Validation workflow

## Reference software and data

Use LBLRTM 12.17, the current LNFL release, MT_CKD 4.3, and AER Line File 3.9.
Store sources, binaries, databases, and full model output under the ignored
`data/lblrtm/` directory. Commit only compact `.npz` fixtures with provenance,
inputs, checksums, and the spectral samples needed by tests.

The repository includes one such fixture for 5000--5100 cm-1. Rebuild it from
`data/profiles/example_midlatitude.csv` after bootstrapping:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/run_lblrtm_reference.py
```

`tellurix.compare_transmission` reports the median, 99th-percentile, and
maximum absolute errors plus the global line displacement in resolution
elements. It interpolates the candidate spectrum onto the reference grid and
applies the transmission threshold below.

The Python package uses a local `uv` environment. LBLRTM 12.14 and newer also
needs netCDF-Fortran, so its compiler toolchain lives in `.conda-lblrtm/` and is
independent of `.venv`.

## Comparison ladder

1. Match a single layer with continuum disabled and identical retained lines.
2. Match a dry multilayer atmosphere.
3. Enable H2O self broadening.
4. Add H2O, CO2, CH4, O2, CO, and N2O.
5. Extract self and foreign MT_CKD H2O continuum terms and enable their
   differentiable abundance scaling.
6. Compare zenith angles corresponding to airmasses 1.0, 1.5, and 2.5.
7. Convolve and integrate onto an IGRINS-like R=45,000 detector grid.

The implemented first rung is a 4290--4310 cm-1 CO-only calculation with
continuum disabled. `scripts/validate_aer_co.py` runs LBLRTM and ExoJAX from
the same AER 3.9 line file and updates the recorded metrics. This case meets
all three acceptance thresholds below. The remaining rungs are explicitly
future validation work; their effects are already isolated in the reference
fixture where possible.

An optional fixed-profile implementation now covers the correction ladder for
the 5000--5020 cm-1 example order. `scripts/build_lblrtm_correction.py`
isolates H2O self/foreign continua and empirical per-species LBLRTM line
residuals, then stores the remaining reference background. The builder uses
LBLRTM-compatible pressure shifts in its JAX baseline. Its corrected
99th-percentile absolute error is 4.75e-5. Checks at 0.5--2x water abundance
and airmass 1--2.5 remain below 2.34e-3 at the 99th percentile. This validates that
profile and grid; broader pressure-temperature and wavelength coverage
remains future work.

`accuracy_mode="mt_ckd"` applies only the isolated physical H2O self and
foreign continua. Its reference 99th-percentile error is 0.0569 versus 0.0581
for pressure-shifted lines alone. That gap is mostly a difference in the
atmosphere the two codes were given, not in their physics; see "What the
5000--5020 cm-1 gap is" in `docs/lblrtm_corrected_mode.md`. Given identical
layers (`LBLRTMRunConfig(user_layers=True)`), the fitted transmission agrees
with LBLRTM to 0.05% median and 0.69% at the 99th percentile at R=45,000
(`docs/lblrtm_identical_layers.json`). That meets the median threshold below
(1e-3) and misses the 99th-percentile one (5e-3). That miss is CO2 line
coupling, which LBLRTM applies: with tellurix's own (`line_coupling=True`)
the fitted agreement is 0.010% median and 0.039% at the 99th percentile, inside
both thresholds.

## Layering

A layer's pressure, temperature and amounts can be sampled at its centre or
averaged over it. LBLRTM averages (`lblatm.f90`, ALAYER and FPACK): pressure
and temperature weighted by air density, each gas's amount integrated, with
pressure, density and number densities exponential in height between levels.
`tellurix.site_profile.weighted_layers` does the same, and since 2026-10 it is
the default of `build_site_profile`, `make_site_profile.py` and
`era5_site_profile.py` (`--layering weighted`); `--layering centre` reproduces
every earlier profile byte for byte. A weighted profile writes the air-weighted
pressure as a `pressure_bar` column, which `AtmosphereProfile.mean_pressure_bar`
carries to lines and continua alike. A profile without it keeps the geometric
mean of the edges for lines and the arithmetic mean for the continua, as
before, so old records reproduce.

`scripts/compare_layering.py` measures what the choice costs, into
`docs/layering_comparison.json`. From two
ERA5 nights (DCT, 2.35 mm; McDonald, 7.87 mm), the default 12 layers are
fitted to a 150-layer reference built from the same levels -- H2O, CO2 and
CH4 scales and a linear continuum, R=45,000, air mass 1.5 -- in four IGRINS
windows:

| | 99th percentile after the fit | before the fit |
|---|---:|---:|
| centre | 0.05--0.20% | 0.11--0.32% |
| weighted | 0.04--0.06% | 0.06--0.16% |

Both are under IGRINS noise, so layering does not limit a correction and the
products made with centre layers stand. Centre layering put 1.5% too much
water in the dry night's column; the fitted scale absorbs that. The pressure
matters as much as the amounts: weighted temperature and water with the
geometric-mean pressure are no better than centre layering in the CH4 window
(0.15% against 0.14%).

Use a water-dominated H interval and a mixed-species K interval. Within pixels
whose reference transmission exceeds 0.05, require median absolute error below
`1e-3`, 99th-percentile error below `5e-3`, and line-center displacement below
0.1 resolution element. Report continuum, line-data, and line-mixing residuals
separately rather than hiding them in a combined score.

## Gradient and recovery checks

Compare every fitted JAX derivative with a central finite difference away from
bounds. The relative difference target is `1e-4`. Synthetic noisy orders must
recover injected molecular scales, wavelength correction, LSF width, continuum,
and jitter within three estimated standard deviations.
