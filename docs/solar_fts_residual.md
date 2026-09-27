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

## What it is

**The model's lines are too broad and too shallow.** Signed residual by
distance from the nearest line core:

| file | region | mean (data - model) | fraction data > model |
|---|---|---|---|
| file 5 | core (<=2 px) | -0.00114 | 0.542 |
| file 5 | outer wing (9-25 px) | **+0.00578** | **0.696** |
| file 4 | inner wing (3-8 px) | **+0.00927** | **0.684** |
| file 4 | core (<=2 px) | -0.00133 | 0.494 |

Positive means the data is *less* absorbed than the model. The model
over-absorbs in the wings and under-absorbs in cores, in both files
independently and worse at the higher air mass -- overestimated pressure
broadening at fixed line strength. It also drives fitted columns low, which is
what the inter-file disagreement below shows.

The residual grows sub-linearly with air mass: rms ratio 1.708 and regression
slope 1.607 against an optical-depth ratio of 2.383, which is what a
transmission-space error does once lines saturate.

**About half of it is the mt_ckd-against-LBLRTM offset.** LBLRTM 12.17 over
4350-4380 cm-1 on the same Kitt Peak profile, the same zenith angle and every
column at 1.0 (`docs/solar_lblrtm_reference.json`, 30,293 monochromatic
points):

| | Kitt Peak profile | midlatitude, LBLRTM's own layers |
|---|---|---|
| median absolute | **0.00294** | 0.00393 |
| rms | **0.00318** | 0.00524 |
| p99 | 0.00709 | 0.01556 |
| mean signed | **-0.00291** | -0.00438 |

The second column is the same comparison run before the TAPE5 fix below, when
LBLRTM chose its own layering; the agreement is 1.6x better once both codes
integrate the same layers, which is most of the reason to make that fix.

The sign is uniform -- **this package transmits less than LBLRTM everywhere**:

| region | mean (tellurix - LBLRTM) |
|---|---|
| core | -0.00190 |
| inner wing | -0.00228 |
| outer wing | -0.00309 |
| between lines | -0.00306 |

Adopting LBLRTM's physics would raise our model flux by these amounts, so the
outer-wing excess against the data falls from +0.00578 to +0.0027 -- **about
half** -- while the core residual moves the wrong way, -0.00114 to -0.0030. So
the line treatment explains roughly half the systematic wing excess and none of
the core/wing sign reversal.

**Do not compare these rms against rms.** That conflates a systematic offset
with line-to-line scatter; done that way the line physics looks 6.5x too small
to matter, which is wrong.

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

**But the columns do not agree between files**, and they must:

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

### Three things in the LBLRTM path that had never been exercised

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
