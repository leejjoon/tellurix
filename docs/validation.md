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

## Beyond LBLRTM: judging physics by the observations (open, to revisit)

*Recorded 2026-10-04 as the direction for later work; nothing below is done.*

Agreement with LBLRTM shows the implementation is right -- same lines, same
layers, same continuum, same answer, now to 0.04% at R=45,000 after a fit
(`docs/lblrtm_identical_layers.json`). It does not show the physics is right.
LBLRTM is a convention as much as a model: AER's line intensities rather than
HITRAN2020's, first-order line coupling from AER's own coefficients, a 25 cm-1
line cutoff less a pedestal, a Voigt shape with no speed dependence. For the
project's purpose -- correcting observed spectra -- the observations are the
judge, and LBLRTM agreement becomes a regression test that a change has not
broken the implementation.

The two can already be seen to disagree. CO2 line coupling improves the IGRINS
2.0 um band centre on DCT 2018 (K89, z 1.97 -> 1.68, ten of ten frames) and
costs 1-3% at the band edges (K86, K92, ten of ten), where tellurix still matches
LBLRTM to 0.03-0.06% -- LBLRTM would be "wrong" there the same way
(`docs/igrins_a0v.md`, "Line coupling on a real night").

**The trap.** A residual mixes telluric error with instrument error (the LSF's
shape -- 2-3% left in deep K lines from the Gaussian trading against the cores --
the blaze, the fringe) and stellar error (the ~1% solar-model floor). Freedom in
the line physics can quietly absorb either. A physics change counts as better
only if it

1. lowers the residual consistently across nights and instruments, not on one
   night and not chosen per order;
2. makes the retrieved columns more physical: column scales independent of
   airmass, H and K agreeing on water, water agreeing with ERA5, CO2 at the
   epoch's known mixing ratio -- tests that resist absorbing other errors in a
   way a residual does not;
3. survives out of sample, as the held-out transfer test does
   (`docs/igrins_transfer.md`).

**Which data for which question.** The NSO solar FTS atlases are the sharper
test of line physics: far higher resolution, a well-characterized instrument
and very high S/N show line-shape errors directly. IGRINS answers the practical
question of how good a correction is, and at its resolution the LSF and the
stellar model probably limit most orders more than line physics does -- worth
measuring before investing in more physics.

**Candidates where physics beyond LBLRTM may matter.**

- Speed-dependent line shapes (HITRAN2020 has parameters for CO2 and H2O); their
  signature is a W-shaped residual in line cores at the 1% level.
- HITRAN2020 against AER 3.9 intensities -- the unexplained O2 excess in the
  solar fits, ~5% in the A-band and ~9% in the B-band
  (`docs/solar_fit_plan.md`, §4n), is the first case.
- CO2 line mixing beyond first order, given the band-edge behaviour above.
- The continuum LBLRTM carries at 2 um beyond MT_CKD's water part: mean optical
  depth 0.0066 against 0.0028 at 5000-5020 cm-1, the difference not identified.

**Proposed method: a fixed scorecard.** A few IGRINS night-bands spanning dry to
wet, a few solar FTS windows, and the column-consistency checks above, scored
the same way every time. Each physics option -- coupling on or off, speed
dependence, HITRAN against AER, the cutoff -- is an A/B on that one benchmark,
with LBLRTM agreement run alongside as the regression test. The LBLRTM-equivalent
configuration stays the default until an option wins across the scorecard, not
in one order. A first, cheaper step is to measure how much of today's IGRINS and
solar residual is telluric at all, against instrument and stellar: if those
dominate, that is where the gains are.
