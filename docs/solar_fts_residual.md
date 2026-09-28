# What limits the solar FTS fit

The first fits of `ftsspec_901218_5` land at 3.7x the photon noise, and this
records what that residual is and is not. Every number here was measured on
2030-2060 cm-1 (4.9 um, the CO fundamental) unless stated, with
`docs/solar_fts_file5_2030.json` as the reference fit.

**Quote the absolute residual, not chi-squared.** This data's noise is 5-10x
better than the Arcturus atlas -- 0.00046 in continuum units against 0.005 --
so a fit reproducing Arcturus's absolute accuracy reports chi2 in the
thousands. Reduced chi2 here is 14; that is a statement about the noise, not
about the model.

## The window matters more than any parameter

| window | source | rms / noise |
|---|---|---|
| 6000-6030 cm-1 | Payne Zero | **67** |
| 6000-6030 cm-1 | flat | 75 |
| 4350-4380 cm-1 | Payne Zero | 5.3 |
| 2030-2060 cm-1 | Payne Zero | **3.7** |

At 6000 cm-1 the residual is 58% correlated with solar line depth and the
telluric model contributes nothing to it: in the 2,369 pixels of 3,166 where
transmission exceeds 0.99, the rms is 0.0238 against 0.0243 overall. Payne Zero
puts lines 30% deep where the Sun has lines 9% deep. The Payne Zero source has
mean line depth 0.020-0.057 around 6000-6350 cm-1 against 0.005 at 4450-4550,
and the residual follows that almost exactly.

**The stellar model also biases the columns, unevenly.** Swapping Payne Zero
for a flat source moves H2O by +36.3% at 6000 cm-1 but only +2.2% at 4350;
CH4 moves under 1% in both. So a water measurement at 6000 cm-1 is not
defensible with this source while a methane one is.

## What the residual at 2030-2060 is not

| tested | effect |
|---|---|
| velocity / wavelength scale | 0.0% of variance |
| LSF width | 1.7% |
| column scale | 0.0% |
| solar model | 2.6% |
| instrument FWHM, +-4% in MOPD | flat: rms 0.004737-0.004742 |
| continuum degree 3 -> 7 | 0.1% |
| vertical profile, ERA5 vs analytic | 1.5% of rms |
| layer count, 12 / 24 / 48 | **0.04%** -- converged |
| `lsf_sigma_kms` pinned vs railed | identical to four decimals |

The last two are worth keeping. **The 12-layer `DEFAULT_EDGES_KM` is
converged**: doubling and quadrupling it moves the residual by 0.04% and the
columns by under 0.7%. And **`lsf_sigma_kms` railing at 0.05 is cosmetic here**
-- at 2045 cm-1 the sinc FWHM is 2.57 km/s, so a Gaussian of 0.05 against
0.093 km/s adds 2-4% in quadrature and changes nothing.

**It is not in either spectrum.** Fitting `ftsspec_901218_4` over the same
window gives corr(residual_4, residual_5) = **+0.941** over 2,802 common
pixels, i.e. 88% shared variance, at air masses 4.730 and 1.985 two hours
apart. Detector noise, fringing and anything belonging to one scan are ruled
out. This is the same test `leave_one_out_patterns` applies to an IGRINS night.

## What it was: two missing species

**The residual was unmodelled opacity, not a defect in the physics.** Adding
two absorbers that were never in the species list:

| model | rms | x noise | reduced chi2 | jitter/sigma |
|---|---|---|---|---|
| H2O, CO2, CO | 0.004740 | 3.75 | 14.04 | 3.61 |
| + **OCS** | 0.002674 | 2.11 | 4.47 | 1.86 |
| + **O3** | **0.001863** | **1.47** | **2.17** | **1.08** |

2.5x end to end. Both come back at ordinary abundances -- 392 pptv of OCS
against a ~500 assumption, and 341 DU of ozone against a 300 DU profile --
nothing rails, the other columns move by under 3%, and the fitted jitter
converges onto the photon noise.

### OCS, found from Wallace's own line list

`telluric_near_ir/linelist_TOTAL.txt` is an empirical catalogue of telluric
lines measured from these very spectra. Over 2030-2060 it identifies 45 lines:
22 CO2, 6 H2O, and **17 OCS** -- a species the model did not carry at all. The
nu3 fundamental at 2062 cm-1 puts its P-branch through 2048-2060.

The residual before adding it, binned by what lies within 0.03 cm-1:

| | pixels | rms |
|---|---|---|
| an OCS line | 102 | **0.0813** |
| a CO2 or H2O line | 87 | 0.0137 |
| no catalogued line | 2763 | 0.0188 |

Six times worse at the OCS lines, mean residual **-0.0735** with **98% of
pixels negative** -- the data more absorbed than the model, at exactly those
wavenumbers. The window's high/low-wavenumber residual asymmetry, 2.16 before,
falls to 1.21 after: the OCS lines all lie above 2048.

### O3, found by scanning every molecule AER ships

Ranking all 46 AER molecules over 2030-2060 by peak line strength times a
nominal abundance puts H2O, CO2, CO and OCS on top -- the four now modelled --
and **O3 fifth with 20,143 lines**, from the nu1+nu3 band at 2110 cm-1.

**The first ranking missed it by 17x** because it used a tropospheric mixing
ratio. Ozone's column is stratospheric: the path-weighted effective VMR is
about 1e-6, not the 3e-8 at the surface. Ranking a species by its surface
abundance is wrong for anything that lives above the tropopause.

The next candidate below O3, N2O, is four orders of magnitude weaker, so this
window is likely complete at this level.

### What this revises

Several conclusions recorded here before the species were found rested on this
residual and were wrong:

- **"The model's lines are too broad and too shallow."** That came from binning
  the residual by distance from the nearest *modelled* line core. Seventeen
  unmodelled deep features sit mid-flank of nothing, and a missing line binned
  that way is indistinguishable from an over-broad neighbour. The signature was
  an artefact of the binning.
- **"The line physics does not explain it."** True of the physics and wrong as
  a conclusion: LBLRTM ran from the same profile with the same species, so it
  was missing OCS and O3 too, and the comparison was blind to this by
  construction.
- **The 5-10% inter-file column disagreement** was measured without these
  species; CO alone moved 10% when OCS was added. It needs re-measuring.

What survives unchanged is that the residual was **88% shared** between two air
masses -- now explained rather than merely observed, since both files look
through the same OCS and ozone.

### The species list was three lists

OCS was unreachable for longer than it should have been because the molecule
table existed in four places: `aer.py` whitelisted seven molecules, and
`fit_fts_window.py`, `rebuild_fts_window.py` and `build_lblrtm_correction.py`
each kept their own copy. Adding OCS to one left O3 "unknown species" in
another.

`aer.py` now carries **every molecule AER ships** -- 46 of the 47, all but
atomic oxygen, which is the only one HAPI's TIPS-2017 tables do not cover, and
the partition function is the only per-molecule input. It is exported as
`tellurix.AER_MOLECULE_IDS` and the scripts use it. Which species are worth
fitting stays a per-window question answered by `--species` and `at_bound`;
*considering* one is no longer a code change.

### The scan, checked on three windows

`scripts/scan_window_species.py` derives the species list from the profile's
own columns. Run on the three windows fitted by hand
(`docs/solar_scan_{2030,4350,6000}.json`):

| window | scan says | hand-picked | verdict |
|---|---|---|---|
| 2030-2060 | H2O, CO2, CO, **O3**, **OCS** | H2O, CO2, CO | two missing, worth **2.5x** |
| 4350-4380 | CH4, H2O, **N2O** | H2O, CH4 | one missing, worth **1.4%** |
| 6000-6030 | CH4, H2O, CO2 | CH4, H2O, CO2 | hand list was right |

So hand-picking got one window right and two wrong, and the cost of being
wrong ranged from a factor of 2.5 to almost nothing. The scan is the gate
because a guess cannot be distinguished from a check after the fact -- not
because every window hides an absorber.

Two calibration points. At 4350-4380 **N2O returns 0.915** of its AFGL 1990
abundance, an ordinary value, confirming the species is real even though it
only buys 1.4% -- its peak optical depth is 0.025 against CH4's 2.23. And O3
sits **1.4x below the cut** there, the closest call the scan has produced;
adding it rails at a bound and changes the residual by nothing, so the 1e-3
threshold is about right and the `headroom` field correctly flagged it as
worth one test rather than worth including.

**A correction.** When the 4350-4380 scan first came back this was recorded as
"not a measurement of anything, a fit missing an absorber, exactly like
2030-2060". That was too strong: adding N2O moves 5.28 to 5.20x noise. The
species list there was incomplete, not badly wrong.

### The stellar model dominates where the atmosphere is quiet

Decomposing each window's residual against the solar source and against
telluric depth:

| window | residual | \|1-source\| vs \|residual\| | \|1-transmission\| vs \|residual\| |
|---|---|---|---|
| 2030-2060 (4.9 um) | **1.47x** | 0.211 (4.4%) | 0.023 (0.1%) |
| 4350-4380 (2.3 um) | 5.20x | **0.503 (25.3%)** | 0.245 (6.0%) |
| 6000-6030 (1.7 um) | 67x | **0.760 (58%)** | -- |

The further into the near-infrared, the more Payne Zero dominates: the Sun has
more structure there and the atmosphere has less. **The windows best for
measuring the atmosphere are the ones worst for the stellar model**, and
4.9 um is where this data is most informative about tellurics -- which is why
the two missing species showed up there and nowhere else.

At 4350-4380 the residual is also **-0.0066 in the inner line wing with 22% of
pixels positive**, against near-zero in cores and between lines. Concentrated
in wings with a consistent sign is a shape error, and the scan says nothing is
missing, so that window's 5.2x is line shape plus the stellar model rather
than opacity that was left out.

### What is left, and it is not the solar model

At 1.47x noise the remaining residual correlates only weakly with the source:

| basis | corr | variance |
|---|---|---|
| 1 - solar source, signed | -0.324 | 10.5% |
| \|1 - solar source\| vs \|residual\| | +0.211 | 4.4% |
| **Payne Zero - ACE** (the model's measured error) | **+0.132** | **1.7%** |
| 1 - transmission | +0.023 | 0.1% |
| d(source)/dnu (solar velocity) | -0.009 | 0.0% |

The sharpest of these is the third, because ACE-FTS is an observation of the
same Sun: if the stellar model were driving the residual it would track the
measured model-minus-truth difference, and it barely does. For contrast the
same absolute test at 6000-6030 cm-1, where the solar model *does* dominate,
gives **+0.760**.

## Air mass: the model is right, the two files still disagree

`ftsspec_901218_{4,5}` are the same sky at air mass 4.730 and 1.985, which is
the test no other dataset in this package can make.

**The fold is exact.** Fitting at `zenith_angle_deg = 0` instead of the header
value must multiply the column scale by the air mass:

| | ratio | air mass | error |
|---|---|---|---|
| file 5, CO2 | 1.9850 | 1.985 | **-0.00%** |
| file 5, CO | 1.9848 | 1.985 | -0.01% |
| file 5, H2O | 1.9773 | 1.985 | -0.39% |
| file 4, CO2 | 4.7296 | 4.730 | **-0.01%** |
| file 4, CO | 4.7283 | 4.730 | -0.04% |
| file 4, H2O | 4.6759 | 4.730 | -1.14% |

`exp(-tau X)` is not `exp(-tau)^X` once a core saturates, so water is low and
the deficit grows with air mass -- 0.39% at X = 1.985 and 1.14% at 4.730. The
synthetic L3 check predicted 1.45% at X = 4.73 before any of this was fitted.
**Quote a slant-path result on CO or CO2, never on water.**

**Re-measured with the scanned five-species list** (`docs/solar_slant_*.json`,
2030-2060 cm-1, ERA5 temperature and water with the AFGL trace set), the fold
is exact on four species at both air masses:

| | file 5, X=1.985 | file 4, X=4.730 |
|---|---|---|
| CO | -0.00% | -0.01% |
| CO2 | -0.00% | -0.01% |
| O3 | +0.04% | +0.02% |
| OCS | -0.02% | -0.05% |
| H2O | -0.40% | **-1.18%** |

Water's deficit grows with air mass as saturation requires, and L3 predicted
1.45% at X = 4.73 before any real data was fitted.

**The files still disagree, and the missing species were not the reason.**

| | file 5 | file 4 | disagreement |
|---|---|---|---|
| OCS | 0.9160 | 0.8816 | -3.8% |
| CO2 | 1.0179 | 0.9777 | -3.9% |
| CO | 0.9696 | 0.8962 | -7.6% |
| O3 | 0.9255 | 0.8439 | -8.8% |
| H2O | 0.6751 | 0.4995 | -26.0% |

against -4.7% and -9.6% for CO2 and CO before OCS and O3 were added. The fit is
2.5x better, the species list is verified, and the disagreement barely moved.
Every species is low in file 4 while file 4's own fold is exact, so this is not
the slant path being mishandled -- it is file 4's **effective air mass being
below its header mean**, which is the scan average: 5.88 to 3.58 within one
40-minute exposure, biasing every column the same way. Predicted at 1-2% from
the header alone; measured at 4% on the well-mixed species.

**So the slant-path bound from this pair is ~4%**, and it is attributable
rather than mysterious. Note O3 and CO disagree by twice as much as CO2 and
OCS; a scalar air-mass error would move every species alike, so there is
structure there that is not understood and none of these should be quoted as a
column until it is.

### Superseded: the same comparison before the species were found

| | file 5 | file 4 | file 4 / file 5 |
|---|---|---|---|
| CO2 | 1.0207 | 0.9731 | -4.7% |
| CO | 1.0552 | 0.9539 | -9.6% |
| H2O | 0.6682 | 0.4903 | -26.6% |

ERA5 says the water column moved 2.5% between the two exposures and the
well-mixed species cannot move at all. File 4 is systematically low, the
direction over-broad lines predict. Its scan-average air mass -- 5.88 to 3.58
within one 40-minute exposure -- accounts for 1-2%, not the rest.

**So the honest slant-path bound from this pair is the ~5-10% the well-mixed
species disagree by, not the 0.01% the fold reproduces.**

## A fitted column is a column *times a profile*

Swapping the ERA5 profile for the analytic one moves the residual by 1.5% and
the water column by **12.2%** (0.668 to 0.750); CO moves 2.4% and CO2 1.4%.
A fitted scale is only meaningful against the profile that produced it, and
water is far more exposed than the well-mixed species. This is
`era5_site_profile.py`'s own argument, measured: one free scale absorbs an
error in the total and not in the distribution.

## The continuum is fitted, so a grey absorption is not measured

`--source-continuum` feeds the source as `flux_total`, carrying Payne Zero's
own predicted continuum instead of dividing it out. The degeneracy this
exposes is exact:

| window | stellar continuum tilt | absorbed by the Chebyshev's linear term |
|---|---|---|
| 2030-2060 | -5.686% | **+5.686%** |
| 4350-4380 | -2.546% | **+2.546%** |

Columns, velocities and residual are unchanged to five decimals. At 2-5 um
H- free-free is smooth and a cubic represents it exactly, so this buys nothing
measurable; it is there for the blue bands and `niratl`, where the bound-free
edges are steps a polynomial cannot represent.

What it does *not* fix is the grey degeneracy. The fitted effective
transmission never exceeds 0.9665 in this window -- even the clearest pixel is
3.4% absorbed by the water continuum and the weak-line forest -- and a grey
absorption is exactly degenerate with the Chebyshev's constant term. The fit
sees only the product, 0.201 x 0.9665 = 0.194. Our corrected spectrum sits
+0.46% above ACE-FTS in line-free pixels at 2030-2060 while Payne Zero agrees
to 0.06%, which is that split being decided by the model rather than measured.

## ACE-FTS as an external check

ACE-FTS 2026 is an observed, telluric-free, near-disc-centre solar spectrum
over 1829-4440 cm-1, so it can referee both our correction and Payne Zero.

| window | Payne Zero vs ACE | our corrected vs ACE |
|---|---|---|
| 2030-2060 | r = **0.974**, rms 0.0162 | r = 0.867, rms 0.0368 |
| 4350-4380 | r = 0.825, rms 0.0130 | r = 0.680, rms 0.0194 |

Both land on the same **-0.600 km/s** offset against ACE at 2030-2060, so that
is ACE's frame relative to ours rather than a defect in either solar model.

Two traps. **Pick the window for ACE's sake too**: its continuum wanders by
1.37% over 4350-4380, at the red edge of its range, against 0.15% at
3000-3030, so the 4350 comparison is ACE-limited and not a verdict on us. And
**mask the deep cores before dividing**: an unapodized sinc has negative side
lobes, the effective transmission reaches -0.0007, and `observed / effective`
diverges. A 0.5 floor keeps 76.8% of pixels here, 0.2 keeps 87.2%.

## Per-window species lists are required

A species with no signal in a window does not contribute nothing -- it absorbs
model error and rails. At 2030-2060 CH4 railed at its upper bound; removing it
pushed N2O to its lower bound; removing both changed the residual by nothing
at all. At 4350-4380 CO2 railed and moved 639% between stellar sources. The
batch driver needs a species list per window, and `at_bound` is the detector
for getting it wrong.

## LBLRTM could not run on a real site profile at all

`write_tape5` left `IBMAX = 0`, which asks LBLRTM to generate its own layer
boundaries with AUTLAY. On both 12-layer Kitt Peak profiles that fails: it
emits 600 boundaries at the same altitude and stops with *"THE NUMBER OF
GENERATED LAYER BOUNDARIES EXCEEDS THE DIMENSION IBDIM"*. It survives only
`example_midlatitude.csv`, the 6-layer profile every LBLRTM-backed product in
this repository was built on -- so `accuracy_mode="lblrtm_corrected"` was
unreachable for any site `make_site_profile.py` produces.

The writer now supplies the boundaries on record 3.3B. That fixes the crash,
and it removes a physics mismatch that was there even when AUTLAY succeeded:
the JAX model integrates over the profile's own edges, and LBLRTM was
integrating over different ones.

**The committed LBLRTM fixtures predate this** -- `tests/data/aer_co_validation.json`,
`docs/native_mt_ckd_validation.json`, `docs/lblrtm_corrected_results.json` and
the correction templates were all produced with LBLRTM choosing its own
layering. Regenerating them changes committed reference numbers that
`tests/test_lblrtm.py` asserts on, so it is a deliberate act and has not been
done here.

## Replacing the line physics does not move the residual

With the TAPE5 fix above, `accuracy_mode="lblrtm_corrected"` can be run on a
real site profile for the first time. Over 4350-4380 cm-1, against the
otherwise identical `mt_ckd` fit -- same window, species, stages, source,
grid and instrument, with the template built on the profile *pre-scaled to the
mt_ckd fit's own columns* so the correction is evaluated where it was derived:

| | `mt_ckd` | `lblrtm_corrected` |
|---|---|---|
| rms | 0.007315 | 0.007170 |
| x noise | 5.28 | **5.17** |
| p99 absolute | 0.02440 | 0.02445 |
| H2O scale | 0.6834 | 1.0212 |
| CH4 scale | 1.0147 | 0.9976 |

**The chain validates.** The corrected fit returns scales at 1.02 and 1.00 on
the pre-scaled profile, so the template is being applied where it was built;
multiplying through gives the same atmosphere to 2% (H2O 0.698 against 0.683).

**The residual moves 2%, and its structure not at all:**

| | core | inner wing | outer wing | between |
|---|---|---|---|---|
| `mt_ckd` | +0.00077 | -0.00627 | -0.00111 | +0.00076 |
| `lblrtm_corrected` | +0.00107 | -0.00620 | -0.00110 | +0.00075 |

Feature for feature the two residuals are the same spectrum with a small
scaling, not a residual that improved.

**Why, and it corrects the inference above.** A smooth, depth-dependent opacity
offset is close to degenerate with a free column scale, so the fit absorbs it
rather than the residual doing so. Converting the 0.003 transmission
difference into an expected residual improvement -- as an earlier revision of
this document did -- treats it as orthogonal to the fitted parameters, and it
is not. **The mt_ckd-against-LBLRTM difference biases the retrieved columns
(2% in water here) and does not set the residual floor.**

So `lblrtm_corrected` is worth its cost for column *accuracy*, not for
residual reduction -- and the cost is an LBLRTM build per window and per
profile.

Repeated at 2030-2060 with OCS included and a line file that covers the window,
the answer is the same: **+2.0% in rms**, 2.11x to 2.07x noise, and the residual
at the OCS lines moves 0.0171 to 0.0164. OCS was the one place the two codes
might have diverged, since its broadening parameters are less well determined
than water's; they do not. Read the column scales before trusting that number,
though -- the OCS scale shifts 36% between the two modes while the residual
agrees to 2%, and `lsf_sigma_kms` rails, so the two fits are exploring a flat
direction in a species constrained by 102 of 3,166 pixels.

### Five things in the LBLRTM path that had never been exercised

All three were found by running `lblrtm_corrected` on a real fit for the first
time; the machinery had only ever been used on its own defaults.

1. **`write_tape5` left `IBMAX = 0`**, so LBLRTM generated its own layering with
   AUTLAY -- which crashes on any 12-layer profile and, where it succeeds,
   makes the two codes integrate different atmospheres. It now supplies the
   boundaries on record 3.3B. Agreement with LBLRTM improves 1.6x from that
   alone (rms 0.00524 -> 0.00318).
2. **`build_lblrtm_correction` asked LBLRTM for exactly `[nu[0], nu[-1]]`** and
   then required strict coverage. LBLRTM lays its grid inside the request, so
   it returned a span 9.4e-05 cm-1 short and the build failed. The request is
   now padded by 0.5 cm-1.
3. **The template's grid must match the model's to `rtol=1e-12`**, so a rounded
   `--resolving-power` produces a 25-minute build that passes its own
   acceptance test and is then rejected at fit time. The builder takes
   `--fwhm-cm1` and derives the resolving power exactly as the fit driver does.
4. **Every script hardcoded `run_lnfl_igrins/TAPE3`**, which LNFL built for
   4000-6750 cm-1 with molecules 1-7. Pointing a 2030-2060 window at it gives
   LBLRTM no lines at all, and the failure surfaces as a missed acceptance
   threshold rather than as anything naming the cause. `--tape3` selects the
   line file; `data/lblrtm/run_lnfl_solar` covers 2005-2085 with molecules
   1-7 plus OCS (58,267 lines, of which 7,583 are OCS -- matching what
   `AERLineDatabase` reads independently).
5. **`write_tape5` declared seven molecules.** Record 3.6 is positional, so
   reaching OCS at 19 means declaring all nineteen with the ones we have no
   profile for at zero, and the abundances then wrap eight to a line because
   record 3.5 sets JLONG and 3.6 is read as (8E15.8).

One usage note: the template's species are the **profile's**, not the
`--species` subset, which restricts only the JAX side of the difference. A
molecule left out still gets a correction carrying LBLRTM's whole optical
depth. That is coherent, but the fit then sees every profile species and an
unconstrained one rails -- CO2 did here. `at_bound` is the detector.

## Reproduce

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_fts_fit.py
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_fts_window.py \
    --spectrum .../telluric_near_ir/ftsspec_901218_5.txt --v1 2030 --v2 2060 \
    --species H2O,CO2,CO --stages continuum,velocity,columns,stellar
UV_CACHE_DIR=.uv-cache uv run python scripts/rebuild_fts_window.py --check
```
