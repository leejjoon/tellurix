# Telluric fitting of IGRINS A0V standards

Numbers here come from `docs/igrins_a0v_results.json`, written by
`scripts/summarize_igrins_fit.py`. Regenerate the report whenever the pipeline
changes and update this prose with it.

## Why an A0V standard, when the Arcturus atlas already works

The atlas run reaches a median residual of 4.22 sigma across 598 page-epochs,
and that residual is *flat against transmission* — 0.97 in deep absorption
against 1.52 at the continuum. A telluric model that is the limiting error
makes the residual grow where the absorption is; a flat curve says the error is
somewhere else, and on Arcturus it is the stellar model. The atlas is therefore
a good test of a K1.5 III line list and a poor one of an atmosphere.

An IGRINS A0V telluric standard removes all three of the atlas's handicaps at
once:

- **A real uncertainty.** The reduction ships a per-pixel `variance`, so
  `SpectralOrder.uncertainty` is measured. The atlas ships no error array and
  `robust_noise` has to estimate one from second differences.
- **A known slant path.** The header carries `ZDSTART`/`ZDEND`, so the zenith
  angle is measured. On the atlas it is pinned at zero and every airmass effect
  is absorbed into the fitted column scales.
- **Almost no star.** An A0V has no metal lines worth the name across H and K.
  The hydrogen recombination series is the whole of its spectrum there, and it
  is computable, localized and maskable — so a fit with `--stellar flat` and
  the series masked carries **no stellar model error at all**.

## The data

RRISA publishes raw and reduced IGRINS products
([igrinscontact.github.io](https://igrinscontact.github.io/)). Use the
**reduced** component: the raw one is 2D detector frames that would need the
PLP and a whole night of calibrations, and its Box folder only permits
whole-night downloads, while the reduced catalog carries a direct
per-observation URL.

`scripts/download_rrisa_standard.py` reads `reduced_log.csv` (17,730 rows) and
fetches a chosen standard. There are 4,859 `OBJTYPE=STD` observations — Gemini
South 1,815, McDonald 1,582, DCT 1,462 — of which 2,125 reach a signal to noise
per resolution element above 300 in H.

Per band the archive ships `spec.fits` (counts in the primary HDU, the
wavelength solution in microns in the first extension), `variance.fits`,
`sn.fits` and `spec_flattened.fits`, whose `MODEL_TELTRANS`, `FITTED_CONTINUUM`
and `A0V_NORM` extensions are read for comparison and never enter the fit.
SDCH is 28 orders over 1.4298–1.8347 um, SDCK 26 over 1.8473–2.5187 um, 2048
pixels each, 55–96 cm-1 per order.

**Do not trust `reduced_log.csv`'s `AM` column.** It writes `-1` for a missing
airmass and carries a few impossible values; 112 of the 4,859 STD rows are
unusable. The airmass that reaches the model comes from the FITS header.

### Three weather conventions, none declared

`AIRTEMP`, `BARPRESS`, `HUMIDITY` and `DEWPOINT` are present across the
archive in three mutually incompatible unit systems, with nothing in the file
saying which is in force:

| site | altitude | convention | a real value | as station pressure |
|---|---|---|---|---|
| McDonald | 2.077 km | °F, inHg, station | 78.0 / 23.6 | 799 hPa |
| DCT | 2.360 km | °C, hPa, **sea-level-reduced** | 10.6 / 1025.0 | 769 hPa |
| Gemini South | 2.722 km | °C, hPa, station | 11.0 / 730.0 | 730 hPa |

1025 hPa is impossible at 2360 m; read as a station pressure it would place the
observatory at sea level. `surface_conditions` normalizes all three to Kelvin
and station hectopascals from the `TELESCOP` card, then checks the answer
against the site's hydrostatic expectation and **raises** if it is more than 8%
off, so a future convention change fails instead of passing quietly.

The cards are also unreliably populated: Gemini frames from 2021 and 2023 keep
`HUMIDITY` and drop the other three, and 2015 McDonald frames carry none of the
four. Each value comes back as `None` when it is missing rather than as a
default that looks like a measurement.

## What differs from the Arcturus pipeline

- **Instrument.** The default Gaussian, with `lsf_sigma_kms` fitted, is the
  whole line spread function of a grating spectrograph. There is no
  `BoxcarFTSInstrumentProfile`, no `measure_atlas_ils.py`, and no `mopd_cm` to
  measure or record.
- **`pixel_integration="simpson"`.** IGRINS pixels integrate; the FTS the atlas
  came from point-samples, which is why that pipeline sets `"point"`.
- **The order is normalized** to `continuum_level`, a high percentile of its
  own finite flux. This matters more than it looks: the continuum's constant
  term is a *log* flux, so raw PLP counts put it near 10 and its bound has to
  span the counts scale — and L-BFGS-B rescales its variables onto that bound,
  so a first step of a fraction of a 60-unit range drives the continuum to
  zero. Measured on order H108 of the reference frame, leaving the order
  unnormalized gave a residual of 220 sigma against 2.7 once normalized.
- **Continuum degree 9, not 3.** An IGRINS order spans 77–96 cm-1 against an
  atlas page's 20 and carries the blaze. With the PLP's own telluric model
  divided out, a weighted Chebyshev fit of the blaze leaves 14.5x the photon
  noise at degree 5 and 7.9x at degree 9; past 9 the gain is small.
- **A throughput cut at a quarter of peak**, on the *smoothed* flux, separate
  from the saturation floor. The blaze does not roll off symmetrically: it
  falls below a quarter of peak across the first 350–700 pixels of an order
  while reaching almost to the last, so a fixed edge trim is the wrong shape.
  That region is not merely noisy — including it makes the blaze fit worse at
  every degree, which says the extraction misbehaves there.
- **The PLP `MASK` is not used.** It flags 56% of H pixels and 76% of K by its
  own flattening criterion, which is not ours.

## Results: one standard, H band

HIP 53433, Gemini South, 2018-04-03, airmass 1.039, signal to noise 318 per
resolution element. Flat source, hydrogen series masked at ±600 km/s, site
profile `data/profiles/gemini_south_2018.csv` seeded at 4 mm of precipitable
water. 24 of 28 orders fitted, 25,511 reliable pixels, about 30 s per order.
The four skipped orders are at the band edges, where the throughput cut and the
saturation floor together leave under 256 usable pixels.

**Median residual 2.57 sigma**, range 1.40–5.55 — against 4.22 on the atlas, at
far higher signal to noise.

### The residual against transmission

| effective transmission | pixels | rms/sigma | median bias / continuum |
|---|---:|---:|---:|
| 0.150–0.400 | 494 | 6.72 | −0.0023 |
| 0.400–0.600 | 856 | 5.13 | +0.0002 |
| 0.600–0.800 | 1814 | 3.78 | −0.0010 |
| 0.800–0.900 | 2223 | 2.77 | −0.0013 |
| 0.900–0.970 | 3477 | 2.59 | −0.0006 |
| 0.970–0.995 | 7351 | 2.60 | +0.0001 |
| 0.995–1.010 | 9285 | 2.62 | −0.0001 |

This is the curve the atlas could not produce, and it separates two errors that
were previously indistinguishable:

- **A floor of 2.6 sigma where there is no absorption at all.** That is not the
  atmosphere. It is the continuum model, the blaze, and the extraction.
- **A rise to 6.7 sigma in the deepest cores.** That *is* the telluric model,
  and it is now measured rather than assumed.

The bias column says the rise is scatter, not offset: even in the deepest bin
the median residual is 0.2% of the continuum. The model is not systematically
over- or under-absorbing; individual line positions and strengths are scattering
both ways.

Note that "effective transmission" means `model_flux / stellar_only`, the
operator the correction actually applies — **not** the saved `transmission`
array, which is unconvolved and differs from it by up to 0.48 in a deep core.
This is the same trap `docs/arcturus_fit.md` documents.

### Against the data, and against the PLP's own model

`observed / (continuum x source)` is the transmission the data itself shows.
Comparing both models against it:

| | median difference | rms |
|---|---:|---:|
| this fit, all pixels | +0.0003 | **0.0155** |
| this fit, T 0.40–0.60 | +0.0006 | 0.0212 |
| PLP `MODEL_TELTRANS`, all pixels | −0.0089 | 0.0990 |
| PLP `MODEL_TELTRANS`, T 0.40–0.60 | **−0.3160** | 0.3153 |

The PLP's model over-absorbs by 0.23–0.32 in transmission wherever there is
real absorption, and agrees only where there is none. It is evidently a
template rather than a per-observation retrieval, so **it is not a useful
independent check** — which is worth saying plainly, because it was the one
free cross-check this dataset appeared to offer.

That leaves a caveat on our own number: the model was *fitted* to this
spectrum, so reproducing it to 1.5% is a goodness of fit, not a prediction.
Sixteen free parameters against 25,511 pixels is not overfitting, but an
independent test needs a second observation — which is what the airmass ladder
below is for.

### The wavelength solution

The fitted telluric velocity is **−1.09 ± 0.18 km/s** across 23 orders, with a
coherent **+0.43 km/s per 1000 cm-1** trend and only 0.08 km/s of scatter about
it. The telluric rest frame is the observer's, so this is not astrophysics: it
is a real, measurable distortion in the PLP wavelength solution, about 0.6 km/s
end to end across H.

It also settles a question the plan raised. Air and vacuum wavelengths differ
by 82 km/s at 1.6 um; a fitted velocity near −1 km/s confirms the PLP solution
is **vacuum**, which is what `SpectralOrder` requires.

### Water

The fitted H2O column comes out at 1.17x the 4 mm seed, so about 4.7 mm of
precipitable water, with 0.35 dex of scatter between orders. That scatter is
large and is not an error bar on the water: orders with almost no water
absorption leave the column unconstrained, and only the absorbing orders carry
information. Treat the per-order values as a distribution to be cut on
absorption depth, not averaged.

## Results: an airmass ladder, and what it measures

DCT 2018-12-20, ten standards spanning **airmass 1.065 to 2.501** in one
night, H and K, 466 order-frames. Site profile `data/profiles/dct_2018.csv`,
flat source, hydrogen series masked. H and K are simultaneous readouts of one
pointing, so the analysis merges them per exposure.

The forward model already divides optical depth by cos(z) with the measured
zenith angle, so a fitted `log_column_scale` scales the **vertical** column
with the slant path removed. If the atmosphere model is right it does not
depend on where the telescope was pointing.

Two confounds have to be handled before any of that is readable.

**Time.** Airmass and time of night correlate at -0.788 here — targets rise —
so a trend fitted against airmass alone absorbs anything that varies with time.
Everything below is fitted against airmass, against time, and against both.

**Revised after a header defect was found.** Every number in the ladder sections
of this document was recomputed on 2026-09-30, after `zenith_angle_deg` stopped
averaging `ZDEND` cards that belong to another frame (9 of 111 archive files,
all K band; see CLAUDE.md). On this night it was one K frame, HD 53205 at
airmass 1.31, read as 1.37: the fit put the 3.8% slant-path error into every
column and its residual did not move, but it was this night's CO2 and CH4
outlier -- the frame-to-frame scatter of both halves without it.

**The stars themselves.** A night observes a handful of targets, each over a
limited span of airmass, so star identity and airmass are partly confounded.
On this night they are *completely* confounded at the top: chi Cap is the only
target above airmass 1.82.

| species | orders/frame | per airmass | per hour | joint d/dAM | frame scatter |
|---|---:|---:|---:|---:|---:|
| H2O | 39 | +0.1328 ± 0.0535 (2.5σ) | **-0.0337 ± 0.0056 (6.0σ)** | -0.0253 | 0.1148 |
| CO2 | 18 | -0.0010 ± 0.0044 (0.2σ) | +0.0013 ± 0.0006 (2.0σ) | +0.0087 | 0.0049 |
| CH4 | 19 | +0.0088 ± 0.0066 (1.3σ) | -0.0005 ± 0.0011 (0.5σ) | +0.0175 | 0.0063 |

### One target carries every slope

Refitting with each object dropped in turn:

| species | slope, all 10 frames | without chi Cap | shift |
|---|---:|---:|---:|
| H2O | +0.1328 | +0.3243 | 3.6σ |
| CO2 | -0.0010 | -0.0198 | 4.2σ |
| CH4 | +0.0088 | -0.0038 | 1.9σ |

Dropping chi Cap still moves the water slope by 3.6 sigma — but water genuinely
varies, so that is weather. For the well-mixed species it is not: CO2 moves by
4.2σ and CH4 by 1.9σ, because with the HD 53205 outlier gone the full-night
errors shrank and chi Cap's lever arm is what is left.

**The defensible statement is still narrower than the table alone suggests.**
chi Cap is the only target above airmass 1.82, so star identity and airmass stay
confounded at the top and no slope across that gap is purely atmospheric. With
every frame the well-mixed columns are flat -- CO2 −0.0010 ± 0.0044 and CH4
+0.0088 ± 0.0066 per unit airmass -- but without chi Cap CO2 reads
−0.0198 ± 0.0079, so the flatness at the top rests on one star.

An earlier version of this document quoted a 2.4% bound out to airmass 2.5 from
a fit with the same confounding, before the instrument response was corrected.
A ladder needs *several* stars at high airmass, not several frames of one; see
`leave_one_object_out`, which the analysis now reports by default.

**The water trend is the sky, not the model.** Against airmass H2O is 2.5σ;
against time it is 6.0σ, and the residual scatter nearly halves. In a joint fit
the airmass term collapses to -0.0253 while the time term survives at
-0.0391 per hour. The retrieved precipitable water falls monotonically
through the night — an ordinary drying night, recovered from the spectra alone.

**Repeatability.** chi Cap's two frames 3.7 minutes apart differ by CH4
+0.007, CO2 -0.005, H2O -0.023, and HD 53205's two, 42 minutes apart, by
CH4 +0.005 and CO2 +0.005 -- they differed by 3.4% and 3.6% before the header
fix, which is what the defect looked like from here. Half a percent is the
noise floor for the well-mixed species, and H2O's 11% frame-to-frame scatter
is real weather.

**Fit quality does not degrade with airmass**: +0.169 ± 0.129 sigma per unit
airmass, consistent with flat.

### What IGRINS can and cannot measure

Peak vertical optical depth reached anywhere in one exposure:

| species | H band | K band | orders above 0.15 |
|---|---:|---:|---:|
| H2O | 992 | 1973 | 39 |
| CO2 | 2.33 | 168 | 20 |
| CH4 | 1.53 | 3.26 | 19 |
| N2O | 0.002 | 0.125 | 0 |
| CO | 0.001 | 0.115 | 0 |
| O2 | **0.000** | **0.000** | 0 |

**O2 is not measurable with IGRINS at all** — its near-infrared bands are at
0.76 and 1.27 um, both blueward of the instrument, so its optical depth is
exactly zero across H and K. Earlier drafts of this document listed it among
the well-mixed species to test; that was wrong. CO and N2O peak just under the
0.15 threshold this analysis uses; admitting them at 0.10 gives
+0.0156 ± 0.0388 and +0.0353 ± 0.0241 per airmass — consistent with zero, but
with errors five to eight times CO2's, so they constrain nothing. CO2 and CH4
are the well-mixed test, and they are enough.

## Results: does a real A0V model beat masking the hydrogen lines?

The flat source plus a ±600 km/s hydrogen mask discards about 22% of the H-band
pixels. `scripts/generate_payne_zero_a0v.py` synthesizes an A0V (9,500 K,
log g 4.1, solar) across 1415-2545 nm; fitting with it and the mask lifted asks
whether that is worth it.

Two controls first. On orders with no Brackett line the A0V source changes
nothing: the residual outside the windows agrees with the flat-source fit to
0.022 sigma. And the A0V run reproduces the flat run's fitted velocities and
resolving powers.

Inside the hydrogen windows, with both residuals divided by the same real
uncertainty:

| order | line | pixels | no-star model | A0V model | gain |
|---|---|---:|---:|---:|---:|
| H103 | Br10 | 594 | 17.09 | 2.56 | 6.7 |
| H109 | Br12 | 576 | 2.88 | 2.14 | 1.4 |
| H113 | Br14 | 602 | 6.43 | 1.87 | 3.4 |
| H117 | Br18/19 | 912 | 20.95 | 2.66 | 7.9 |
| **pooled** | | 2684 | **13.97** | **2.33** | **6.0** |

The A0V model removes a factor of six, and leaves 2.33 sigma — slightly *better*
than the 2.6 sigma floor the rest of the band reaches, because the Brackett
lines sit mid-order where the blaze is flattest and the signal to noise is
highest. So the hydrogen lines stop being a limiting error and those 22% of
pixels come back.

That is a better outcome than expected. The synthesis reports
`atmosphere_converged: false`, meaning it rests on Payne Zero's learned
initializer rather than a converged solve, and at 9,500 K the hydrogen lines
are the entire spectrum — the reason for building the flat-source path first
was that this might not work. It does. The flat source remains the conservative
default because it depends on no stellar model at all, which is what makes the
residual-against-transmission curve above a clean measurement of the
atmosphere; use the A0V model when the masked pixels are worth recovering.

One caveat that survives: the model's Brackett lines are shallower than the
observed absorption at the same wavelengths (0.20 against 0.53 of the continuum
at Br10). That comparison is confounded, because the observed depth contains
telluric absorption and continuum-fit freedom as well as the star, so it is not
by itself evidence of a bad profile — but the residual is the metric that
matters, and it is fine.

## Cost

An order-frame took 31.7 s before this was profiled. The breakdown was not what
it looked like:

| | per order |
|---|---:|
| fitting, three stages | 16.3 s |
| opacity precompute | 7.4 s |
| gradient compile | 4.9 s |
| Hessian compile (hidden inside stage 1) | 8.2 s |
| steady gradient / Hessian call | 2.6 ms / 3.1 ms |

**13.1 s of it was XLA compilation**, and the Hessian half was invisible in the
stage log because `fit_order` computes the covariance unconditionally, so the
lazy `jax.hessian` compile landed on the first stage and made "continuum" look
like it cost ten times per iteration what the other stages did.

Two IGRINS-specific facts make almost all of that shareable. Within a night the
PLP uses one wavelength solution, so order N covers a **bit-identical**
wavenumber range in every frame (measured spread across ten frames: 0.0000
cm-1); across nights and sites it moves at most 0.16 cm-1 on an 80 cm-1 window,
far inside the 5 cm-1 grid margin. So the driver loops **orders outside,
frames inside**, and `OrderObjective` takes the per-frame flux, mask,
uncertainty and zenith angle as jit *operands* rather than captured constants,
so one compiled executable serves every frame. `OrderObjective.rebind` checks
the wavelength grid and the source match before reusing, and `fit_order` gained
`covariance=False` so only the final stage ever asks for the Hessian.

Measured on 3 orders x 5 frames: **141 s against 475 s, a 70% saving**, with the
first frame of an order costing 19-26 s and the rest 2.7-7.0 s. Results agree
with the old path to 1e-5 relative — not bit-identical, because XLA fuses a
different graph when arrays are operands rather than folded constants and
L-BFGS-B amplifies that over hundreds of iterations, but four orders of
magnitude below the 0.0155 rms the fit itself reaches.

What is deliberately **not** shared is the starting point. Warm-starting each
frame from the previous one would pull its fitted columns toward that neighbour
and shrink exactly the frame-to-frame scatter the ladder above exists to
measure. Every frame starts from the same neutral guess.

## Running it

```bash
uv run python scripts/download_rrisa_standard.py --catalog
uv run python scripts/download_rrisa_standard.py --list --facility DCT --night 20181220
uv run python scripts/download_rrisa_standard.py --night 20181220 --min-snr 200
uv run python scripts/make_site_profile.py --site "Gemini South" --site-altitude-km 2.722 \
    --surface-pressure-hpa 728.5 --surface-temperature-k 283.65 \
    --precipitable-water-mm 4.0 --epoch 2020 --output data/profiles/gemini_south_2018.csv
uv run python scripts/fit_igrins_standard.py \
    --spec data/igrins/20180402_0104/SDCH_20180402_0104.spec.fits
uv run python scripts/summarize_igrins_fit.py \
    --summary data/corrected/igrins/SDCH_20180402_0104_summary.json
```

`data/igrins/` and the per-order `.npz` are gitignored; the committed record is
`docs/igrins_a0v_results.json` and the two-order fixture under
`packages/tellurix-igrins/tests/data/igrins/`.

## What limits the residual, and the fix

The residual-against-transmission curve had a floor of a couple of sigma where
there is *no absorption at all*, which cannot be the atmosphere. Chasing it
found one cause and two dead ends.

### Two things that looked like causes and are not

**The line spread function does vary along an order, and it barely matters.**
`lsf_sigma_kms` is fitted per order but is a single constant within one.
Splitting orders into six x-segments, with five Chebyshev coefficients free per
segment so the blaze cannot leak into the width, the fitted R varies
monotonically with pixel: 29,400 to 40,400 across order H100 (33%), 42,600 to
47,200 across order H122 (10%), always narrower at high x. But freeing the width
per segment improves the residual by a median of 1.02, best 1.55. Velocity per
pixel is constant to 3.1% across the band, so a constant sigma in km/s is
already nearly a constant sigma in *pixels*. The calibration products cannot
settle it independently: `SKY_*.wvlsol_v1.fits` is only the wavelength array,
with no line table and no extracted sky spectrum.

**The continuum coefficient bound was binding and that did not matter either.**
218 of 248 order-frames had a Chebyshev coefficient pinned at ±1.5, coefficient
3 in 103 of them. Loosening it to ±8 releases them all and changes the residual
by 0.01 sigma; the basis is degenerate enough that a railed coefficient is
compensated by its neighbours. Worth loosening anyway — the ladder filters on
`at_bound` and 218 spurious flags pollute it — so the bound is now ±5.

### The cause: a repeatable instrument response

Across all 17 fitted H orders of one frame, |residual| in continuum units is
0.0044 in the middle third and 0.0141 in the last sixth — a **3.2×
degradation** — while the quoted sigma stays flat, so it is model error and not
underestimated noise. The same 3.2× appears in the 11 orders whose median
transmission exceeds 0.99, where there are no telluric lines to get wrong.

It is the same pattern in every frame:

- 56–76% of each frame's residual variance is **common across all ten frames**,
  five different stars spanning airmass 1.07 to 2.50.
- It is strongest in orders with no telluric absorption (H108, median T 0.998,
  71% common), so it is not the line list.
- Its amplitude is 0.13–0.21 with only 0.012–0.021 scatter, and the sign of its
  airmass dependence is **inconsistent between orders**. A telluric-model error
  would grow consistently with airmass.
- It is fixed in *detector* coordinates. Five stars with radial velocities
  differing by tens of km/s — several pixels — could not correlate at r ≈ 0.7
  without a shift.

That is the instrument, and it can be measured and divided out.

### The correction, and the guard that matters more than it

`leave_one_out_patterns` (in `igrins.py`) takes the fractional model error of
every frame of a night at one order and returns, for each frame, the median of
the **others**, smoothed on a 51-pixel boxcar. The driver runs two passes: fit,
build the patterns, divide the flux and its uncertainty by `1 + pattern`, refit.
Dividing the data is algebraically identical to multiplying the model, so the
forward model is untouched.

Two independent things have to be guarded, and only the first is obvious.

**Leave-one-out** stops the pattern absorbing the noise of the frame it
corrects. A pattern taken from that frame would fit its noise and flatter
everything downstream.

**Smoothing** stops it absorbing a systematic that *every* frame shares — and
our own telluric model error is exactly that, since every frame looks through
the same sky with the same line list. Leave-one-out does nothing about it. The
unsmoothed pattern's high-frequency component correlates with the absorption
depth at **r = +0.44** and with the transmission gradient at +0.39, and its
amplitude tracks how much absorption an order has: 0.013 of the continuum where
the median transmission is 0.67, 0.003 where it is 0.999 — the noise floor. The
smooth component, by contrast, correlates with depth at **+0.05**. The split is
clean, and the fine-scale half is the line list, not the instrument.

Splitting the orders by whether they contain lines at all settles what each half
is worth (leave-one-out throughout):

| | no correction | smoothed (51 px) | unsmoothed |
|---|---:|---:|---:|
| line-free orders, median T > 0.99 | 2.16σ | **1.34σ** | 0.80σ |
| absorbing orders | 2.28σ | **1.94σ** | 1.02σ |

In a line-free order there is nothing telluric to absorb, so the whole gain is
instrumental. In an absorbing order the unsmoothed correction buys far more —
and that extra is the model error we are trying to measure. A 51-pixel boxcar,
about fifteen resolution elements, keeps the broad response error (which runs
over hundreds of pixels at the red edge) and leaves telluric-scale structure in
the residual. On a synthetic test it recovers a broad injected component to
0.0012 while leaking 8.4% of a line-scale one, against 100% unsmoothed.

Over the whole ladder the median residual goes from about 1.9σ to **1.30σ** in H
and **1.29σ** in K. Running unsmoothed would report 0.80σ and 0.83σ, and that
number would be meaningless as a test of the telluric model.
`--pattern-smooth-pixels 0` disables the guard and is documented as such.

The median pattern amplitude is 2.1% of the continuum. With the pattern in place
the throughput floor goes back to **0.25** — its job is now only to drop dead
pixels — recovering the 16% of pixels and the one order per band that a 0.45
floor cost. The second pass costs about 1.8× in runtime.

**It needs at least five frames of a night** (`--fixed-pattern-min-frames`), and
it is per night; night-to-night stability is untested.

### What the high-frequency half is good for

It is a measurement, not a nuisance: the part of the residual that repeats
across every frame of a night and scales with absorption depth is an empirical
map of the telluric line list's errors at this resolution. Nothing here exploits
that yet, but it is the natural thing to compare against a different line list.

## How much of an order survives, and why

Coverage varies a lot between orders, and almost all of it is two cuts of ours.

| cut | pixels removed, of 2048 | varies between orders? |
|---|---|---|
| throughput floor (< 0.25 of peak) | 464–1002, typically ~500 | no; it hits both ends of every order |
| **hydrogen mask** | **0–1151** | **yes; this is what makes coverage vary** |
| saturation floor, T < 0.15, model < 0.2 continuum | 0–170 | mostly negligible |

An order with no Brackett line (H100, H101, H102, H104, H105, H108, H122, H123) loses
nothing to the mask and keeps 73–77% of its columns. An order with one loses
about 590, and H118, which holds Br19 and Br20, loses 1151 — 56% of the order,
leaving 14%.

That is arithmetic: ±600 km/s at 2.04 km/s per pixel is ±294 pixels per line.

### The ±600 km/s mask is too narrow, and widening it is not the answer

±600 was a guess when nothing better was available. Measured against the A0V
model now in hand, using a continuum defined far outside the line, it is much
too narrow:

| offset from line centre | Br10 | Br11 | Br12 |
|---|---:|---:|---:|
| ±600 km/s (the mask edge) | 7.7% | 7.6% | 6.4% |
| ±1200 km/s | 2.8% | 2.7% | 1.6% |
| ±1800 km/s | 1.1% | 1.0% | 0.2% |

At the mask edge the line is still 6–8% deep, so the flat-source fit is treating
strongly absorbed pixels as continuum. This does *not* show up in the residual —
just outside the mask it is −0.04% of the continuum — because the degree-9
Chebyshev absorbs a broad smooth wing. The bias goes into the continuum instead,
and through it into the columns.

Three treatments of the same two orders and the same ten frames:

| treatment | pixels kept | residual | ΔlogN(H2O) | ΔlogN(CO2) |
|---|---:|---:|---:|---:|
| flat source, ±600 km/s | 923 | 1.24σ | — | — |
| flat source, ±1800 km/s | **269** | 1.54σ | +0.017 | +0.047 |
| **A0V model, no mask** | **1499** | 1.31σ | −0.028 | −0.015 |

Masking properly costs almost the whole order. The A0V model keeps 60% more
pixels than the narrow mask *and* fits the Brackett cores, at the same residual.
Across the ladder, orders containing a Brackett line retrieve columns offset
from those that do not by −0.023 (H2O), +0.042 (CO2) and +0.034 (CH4) — though
that comparison is confounded, since the two populations sit at different
wavelengths.

The offset is common to every frame of a night, so it largely cancels in a slope
against airmass and the ladder conclusions stand. It biases the *absolute*
columns at the few-percent level, which is the same size as the precision quoted
there. The flat source remains the right default for measuring the atmosphere,
because it depends on no stellar model at all; but the ladder should be rerun
with the A0V model before any absolute column is quoted.


## The ladder rerun with the A0V model

The flat source plus a hydrogen mask keeps no stellar model in the loop, which
is what makes it the right default for *measuring* the atmosphere. But the mask
is too narrow to be clean (above), so the absolute columns it returns are biased
at the few-percent level. Rerunning the whole night with the Payne Zero A0V and
no mask fixes that, and costs about 1.4x in runtime.

Fitting with a real stellar source needs one more thing: a **stellar stage**.
These are five different A0V stars with radial velocities tens of km/s apart, so
`stellar_velocity_kms` has to be fitted or the Brackett lines land in the wrong
place. `stages_for()` adds it whenever `--stellar` is not `flat`. In order H109
(Br12) the fitted velocities are self-consistent per star -- chi Cap +58.9 and
+57.6, k Tau +42.7 and +43.8, HD 53205 +9.2 and +7.4 km/s -- which is a check
on the whole arrangement that the flat source cannot provide. **Most other
orders do not measure it:** 42% of order-frames rail at +-60 km/s, in a pattern
set by the order rather than the star (`docs/igrins_transfer.md`).

| | rows | reliable pixels | median residual |
|---|---:|---:|---:|
| H, flat + ±600 km/s mask | 248 | 261,053 | 1.30σ |
| **H, A0V model, no mask** | **270** | **367,810 (+41%)** | 1.33σ |
| K, flat + mask | 237 | 318,851 | 1.29σ |
| **K, A0V model** | **246** | **377,660 (+18%)** | 1.31σ |

Fitting 41% more pixels in H — including the Brackett cores, the hardest pixels
in the band — moves the residual by 0.03σ. That is the strongest statement
available that the A0V model is good enough to use.

| species | per airmass | per hour | joint d/dAM | leave-one-out shift |
|---|---:|---:|---:|---:|
| H2O | +0.1251 ± 0.0626 (2.0σ) | **-0.0348 ± 0.0066 (5.3σ)** | -0.0318 | 3.9σ (chi Cap) |
| CO2 | +0.0049 ± 0.0033 (1.5σ) | -0.0006 ± 0.0006 (1.0σ) | +0.0071 | 3.2σ (chi Cap) |
| CH4 | +0.0102 ± 0.0051 (2.0σ) | -0.0015 ± 0.0009 (1.7σ) | +0.0095 | **0.5σ (chi Cap)** |

Water is unchanged: 2.0σ against airmass, 5.3σ against time, the airmass term
collapsing in a joint fit. The sky dried; the model is fine.

**CO2 stays flat** at +0.0049 ± 0.0033 per unit airmass, 0.7% across the observed
range with a 2σ bound of 1.7%.

**CH4 is the one worth watching.** Its slope is much the same as the flat run's
(+0.0088 ± 0.0066 there, +0.0102 ± 0.0051 here) but the extra pixels shrink the
error, so it is 2.0σ -- and, unlike CO2, it does not rest on chi Cap: dropping
any single target moves it by 0.5σ. That is 1.5% across the range, 2.9% at 2σ.

Three reasons not to call that a detection. It is one of three species, so
the multiple-comparison penalty is real. CH4 shows a 1.7σ dependence on *time*
as well, which it cannot physically have, and airmass and time correlate at
-0.788 here; in a joint fit the airmass term is +0.0095. And it is one
night at one site. It is the most interesting thing in this dataset and the
clearest argument for a second night.

Both records are kept: `ladder/` and `ladder_k/` are the flat-source fits, which
depend on no stellar model; `ladder_a0v/` and `ladder_k_a0v/` are these.


## A second night, at a second site

DCT 2018-12-20 has one flaw that bounds every conclusion drawn from it: chi Cap
is the only target above airmass 1.82, so dropping it moves H2O by 3.9 sigma.
**McDonald 2017-04-20** was chosen to break that — 8 standards, airmass
1.020 to 2.991, 7 distinct stars, and **five different stars above
airmass 1.8**. It is also the first McDonald night fitted, so it exercises the
degF/inHg weather branch end to end: 23.4 inHg normalizes to 792.4 hPa station
against 787.4 expected at Mt Locke, 0.6% out.

| species | DCT (A0V) | McDonald (A0V) | joint d/dAM, McDonald | leave-one-out |
|---|---:|---:|---:|---:|
| H2O | +0.1251 ± 0.0626 (2.0σ) | +0.2350 ± 0.1735 (1.4σ) | -0.1005 | 0.9σ |
| CO2 | +0.0049 ± 0.0033 (1.5σ) | +0.0469 ± 0.0227 (2.1σ) | +0.0123 | 0.8σ |
| CH4 | +0.0102 ± 0.0051 (2.0σ) | +0.0456 ± 0.0226 (2.0σ) | +0.0168 | 0.9σ |

**CH4 is not settled.** McDonald's slope has the same sign, and this time it
does not rest on one star -- dropping any target moves it by 0.9 sigma. But its
error is four times DCT's and its value four times larger, so the two are
consistent at 1.5 sigma and neither settles the other. In a joint fit
against airmass *and* time, McDonald's airmass term falls to +0.0168 and DCT's
is +0.0095. Two nights is still not enough; the question is worth a third.

The header fix moved McDonald the other way from DCT. Its top frame, k Tau at
airmass 2.99, had been read as 3.16 in K; corrected, the CO2 and CH4 slopes
*steepen* from about 1.4σ to 2.0-2.1σ. The wrong airmass had been hiding part
of a real deficit at the top of the ladder -- one the transfer test
(`igrins_transfer.md`) also sees, as K frames above airmass 2.4 on this night
that a per-frame water scale cannot correct. (Later: the deficit was the
airmass of the frame's eight-exposure sequence, 3.23, which the defective 3.16
had been nearer than the corrected 2.99 -- "What moves the well-mixed columns"
below.)

**Water behaves the same way at both sites**: 1.4 sigma against airmass and
7.4 sigma against time here, the airmass term collapsing in a joint fit.
The sky changes; the model does not need to.

### A wetter site fits worse, everywhere

McDonald carries about 10 mm of precipitable water against DCT's 2, and its
median residual is 1.78 sigma in H and 2.37 in K against 1.33 and 1.31. Binned
by transmission, using the median rather than the rms because two of DCT's 516
order-frames are pathological (H107 at 115 sigma, H104 at 33, both unexplained):

| effective transmission | DCT, ~2 mm | McDonald, ~10 mm |
|---|---:|---:|
| 0.150–0.400 | 1.62 | 2.52 |
| 0.400–0.600 | 1.40 | 1.97 |
| 0.600–0.800 | 1.15 | 1.49 |
| 0.800–0.900 | 0.86 | 1.17 |
| 0.900–0.970 | 0.66 | 0.91 |
| 0.970–0.995 | 0.57 | 0.77 |
| 0.995–1.010 | 0.57 | 0.78 |

The *shape* is the same on both nights — the residual climbs monotonically with
absorption depth — but McDonald sits about 1.4x higher at **every** level,
including where there is no absorption at all. So it is not only that deeper
lines expose more line-list error: something about the wetter night degrades
even the clear pixels, which is what one would expect from the water continuum
and the forest of weak lines that touch every pixel at 10 mm.

Median |z| below 1 at the continuum on both nights is the same hint as before
that the PLP's variance is conservative.

### Two fixes this night forced

An order that is opaque end to end — the 2.0 um CO2 band at 10 mm of water —
used to produce a row with a NaN residual. It is now skipped and recorded as a
failure, which removed 11 such rows from K.

And the precipitable-water seed matters more than a pure column scaling would
suggest, because `precompute_opacity` linearizes the self-broadening about the
profile's own water content. Seeding 6 mm when the truth was 10 left the fitted
scale at x1.7, outside the +-25% the expansion is built for. Reseeding centred
it at 1.00. It did **not** improve the residual, so the linearization was not
what limited this night — but the seed should not be a guess.

### Seeding the water column from the header

`precipitable_water_mm` takes it from the surface dewpoint. For an exponential
water profile the column is the surface density times a scale height, and the
surface density follows from the dewpoint alone: `e(Td) / (R_v T) * H`. The
dewpoint is the right input because it *is* the vapour pressure; relative
humidity carries the same information with the temperature's error folded in.

The scale height here is an *effective* 1.4 km, not the 2.0 km over which the
layer profile distributes water — a different quantity, smaller because the real
profile falls off faster than exponential near the ground. Calibrated against
the three fitted nights it comes out at 1.4 km with 26% scatter:

| night | seeded by eye | from the dewpoint | fitted |
|---|---:|---:|---:|
| DCT 2018-12-20 | 4.0 (1.69x) | 3.8 (1.61x) | 2.36 mm |
| McDonald 2017-04-20 | 6.0 (0.60x) | 9.7 (0.98x) | 9.99 mm |
| DCT 2016-12-08 | 3.5 (0.51x) | 5.7 (0.83x) | 6.89 mm |

Worst case 1.6x against 2.0x by eye, and no longer a judgement call.
`scripts/igrins_site_profile.py` reads a night's frames, takes medians of
whatever weather cards they carry, estimates the column and calls
`make_site_profile.py` — so there is still one piece of code that writes a
profile, and none of its three numbers is chosen by hand. It refuses rather than
inventing a column when no frame carries a dewpoint or humidity, which is the
case for 2014 McDonald and recent Gemini frames.


## Three nights, and what they settle

A third night was run for one reason: to decide whether CH4's 2.5 sigma airmass
trend on DCT 2018-12-20 was real. **DCT 2016-12-08** was the best available
test — 11 frames, the most of any candidate, 10 distinct stars, the highest
median signal to noise of the archive's ladders, airmass to 2.59, and the same
site as the 2018 night two years earlier, which also makes it the night-to-night
stability test the response correction needed.

| night | PWV | frames | airmass | CH4 per airmass | CO2 per airmass |
|---|---|---:|---|---:|---:|
| DCT 2018-12-20 | ~2 mm | 10 | 1.07–2.50 | +0.0102 ± 0.0051 (2.0σ) | +0.0049 ± 0.0033 (1.5σ) |
| McDonald 2017-04-20 | ~10 mm | 8 | 1.02–2.99 | +0.0456 ± 0.0226 (2.0σ) | +0.0469 ± 0.0227 (2.1σ) |
| DCT 2016-12-08 | ~7 mm | 11 | 1.06–2.59 | **-0.0271 ± 0.0161 (1.7σ)** | **-0.0282 ± 0.0152 (1.9σ)** |

**The signs disagree, so the trend is not real.** For CH4 the three nights give
+0.010, +0.046, -0.027: chi-squared for a common value is 7.7 on two degrees
of freedom (p = 0.02). CO2 behaves the same way — +0.005, +0.047, -0.028,
chi-squared 8.2. The single night's slope was a per-night systematic. Correcting
the zenith angles did not remove it: before the fix the two chi-squareds were
7.1 and 6.8, so the header defect was not the systematic and, at McDonald, was
partly hiding it.

The same night says it twice over. On DCT 2016 both CO2 and CH4 show a **3.4-3.6
sigma** dependence on *time of night* (4.4 before the fix, whose one material
correction there was to the night's last frame), which neither can physically have — and
that night has the weakest airmass-time correlation of the three (-0.452, against
-0.788 and -0.681), so the two axes are best separated exactly where the
impossible trend is clearest. These systematics are real and are not a fitting
degeneracy between airmass and time.

(Superseded: the per-night systematic was the exposure sequence's airmass --
"What moves the well-mixed columns" below. With it the three nights agree at
chi-squared 0.6 and 2.0.) **So the slant-path bound is set by night-to-night
scatter, not by any one night's formal error.** Across three nights the well-mixed columns scatter by
0.0297 (CH4) and 0.0307 (CO2) per unit airmass, about **3-4%** over a
typical range, against the 1.7% a single night's error bar suggested. That is
the number to quote.

Water is consistent throughout: no night shows a credible airmass dependence,
and its time dependence is whatever the sky did.

### The response pattern is stable for two years

DCT 2016-12-08 and DCT 2018-12-20 are the same telescope 741 nights apart.
Aligned on detector column, their fitted response patterns agree at a **median
r of +0.943** across 26 orders. The amplitude is 0.0111 in 2018 and 0.0108 in
2016; the rms of their *difference* is 0.0038, only 34% of either.

So about nine tenths of the pattern's variance is a fixed property of the
instrument rather than of the night. That is stronger than the correction
needed, and it has a practical consequence: the pattern could be measured once
per instrument configuration and applied to nights carrying fewer than the five
standards `--fixed-pattern-min-frames` requires. The remaining third is the
genuinely per-night part — flexure, focus, and which frames were taken.

The orders where the two nights agree least (H100 at r = 0.52, H103 at 0.65) are
the low-throughput ones with the fewest shared pixels.


## Does a reanalysis atmosphere help?

The layer profile is analytic: a constant 6.5 K/km lapse rate to a 216.65 K
tropopause, and water falling exponentially with a 2 km scale height. The fit
gives each species one free scale factor on its whole column, which absorbs an
error in the *total* but not in the vertical *distribution* — and that is the
shape of the per-night systematic above. ERA5 tests it directly.

`scripts/era5_site_profile.py` writes the same CSV with the same layer edges, so
a fit against the two differs in T(z) and q(z) and in nothing else. The column is
anchored at the **station pressure from the frame's own header** and integrated
upward, which sidesteps the orography problem entirely: the 0.25 degree cell
containing DCT has a surface elevation of 1844 m against the telescope's 2360 m,
but ERA5's pressure-level data is a real atmospheric column whatever the model
thinks the ground is, and the levels it reports below the telescope are
extrapolated and discarded.

### First, ERA5 agrees with the fit about the water

| night | ERA5 | fitted (analytic profile) | from the dewpoint | ERA5 lapse rate |
|---|---:|---:|---:|---:|
| DCT 2018-12-20 | 2.35 mm | 2.36 mm | 3.8 | 5.14 K/km |
| McDonald 2017-04-20 | 7.87 | 9.99 | 9.7 | 8.29 K/km |
| DCT 2016-12-08 | 7.20 | 6.89 | 5.7 | 4.54 K/km |

Two independent methods agreeing to 0.4% and 4.5% on the two DCT nights is a
real check on both. And the true lapse rate is 4.5–8.3 K/km — off the assumed
6.5 by up to 30%, **in both directions, varying by night**, which is exactly the
sort of thing that produces inconsistent per-night systematics. The profiles
differ substantially: ERA5 is up to 3.9 K warmer in the lower troposphere, 13 K
colder in the stratosphere, and its water has a different *shape*, 0.33x the
analytic value at 3 km but 2.3x at 8.4 km.

Fitting against ERA5 leaves the water scale near unity (1.00, 1.22, 0.90 against
0.59, 1.00, 1.97), which is what a correct profile should do.

### But it only half works

| | analytic | ERA5 |
|---|---:|---:|
| **CO2** scatter between nights | 0.0307 | **0.0262** |
| CO2 chi-squared for a common value (2 dof) | 8.2 | **5.7** |
| **CH4** scatter | 0.0297 | 0.0279 |
| CH4 chi-squared | 7.7 | 7.8 |
| residual, H band | 1.33 / 1.78 / 1.89 | 1.30 / 1.76 / 1.86 |

(Later: the inconsistency both columns show here was the exposure
sequence's airmass, not the profile -- "What moves the well-mixed columns"
below. ERA5's part in it was small.) **CO2 improves, but less than it first
seemed.** Its chi-squared falls from
8.2 to 5.7, from inconsistent with a common value (p = 0.02) to borderline
(p = 0.06). Before the zenith-angle fix this read as 6.8 to 3.4, p = 0.03 to
0.18 -- the header defect had flattered ERA5. **CH4 does not move** and its
signs still disagree. And the **residual does not improve at all** — a 6.5 K/km
lapse rate and an exponential water profile fit the spectra just as well as the
real atmosphere does.

So the reanalysis fixes part of what it should and nothing it should not. The
asymmetry is suggestive: CO2's strongest lines are the saturated 2.0 um band,
which forms high in the column where the profile shape matters most, while CH4
is spread thinner across both bands. Whatever remains in CH4 is not the
temperature or water profile.

### Worth adopting, for a different reason

On accuracy alone this is a marginal call: better absolute columns, no better
residual. The stronger argument is coverage. **31% of the archive — every
Gemini South standard from 2020 on — carries no AIRTEMP, BARPRESS or DEWPOINT**,
so no profile can be built from its headers at all. ERA5 needs none of them; it
needs only a position and a time, which every frame has.

Both download paths work and the script supports either:

| | ARCO-ERA5 (Google Cloud) | Copernicus CDS |
|---|---|---|
| credentials | none | `~/.cdsapirc` |
| one profile | 20 s, 150 MB | 30 s, 0.05 MB |
| 140 profiles batched | 47 min, 21 GB | **11 min, 0.2 MB** |
| per profile in bulk | 20 s, 150 MB | **4.6 s, 1.5 kB** |

ARCO chunks are one timestep by global by all 37 levels, so extracting one
column always costs 150 MB. CDS charges by *fields* — times times levels times
variables — and serves a single grid point, which is four times faster and five
orders of magnitude less data once batched. Its limit is about 8,000 fields a
request, so bulk work batches in roughly ten-day blocks per site, and asking
only for the levels above the observatory halves the count for free. For the
whole archive that is ~100 requests and under 100 MB against ARCO's 730 GB.

Use ARCO for a one-off profile or without an account; use CDS for bulk.


## A Gemini South night, and what it shows

31% of the archive — every Gemini South standard from 2020 on — carries no
`AIRTEMP`, `BARPRESS` or `DEWPOINT`, so no profile can be built from its
headers. That ERA5 unblocks them was an argument until now; **2021-03-16**
demonstrates it.

The blocker is real: those 14 frames carry a humidity and nothing else, and
`igrins_site_profile.py` refuses them outright. ERA5 needs neither. It also
supplies the one thing the header was still being asked for — the station
pressure that anchors the column — by interpolating its own geopotential to the
telescope's altitude. Against the three nights that *do* carry `BARPRESS` that
agrees to **0.4-1.7 hPa, or 0.2%**, so `--anchor auto` prefers the header and
falls back to ERA5 without loss.

| night | site | H rows | H residual | K residual | water scale |
|---|---|---:|---:|---:|---:|
| DCT 2018-12-20 | DCT | 270 | 1.30 | 1.32 | 1.00 |
| McDonald 2017-04-20 | McDonald | 214 | 1.76 | 2.24 | 1.22 |
| DCT 2016-12-08 | DCT | 297 | 1.86 | 1.59 | 0.90 |
| **Gemini S 2021-03-16** | **Gemini South** | **378** | **1.76** | **1.64** | **1.20** |

A night with no usable weather at all fits as well as the ones that have it.
Gemini tracks near the zenith, so its airmass range (1.27-1.78) makes a poor
ladder by design — but with 14 standards it is the best-determined response
pattern in the set, which turned out to matter.

### The response pattern belongs to the spectrograph, not the telescope

IGRINS is one instrument that moved from McDonald to DCT to Gemini South. Its
order centres agree to 0.04 nm across all four nights, so the spectral format
survived the moves. So does the response:

| pair | orders | median r |
|---|---:|---:|
| DCT 2018 vs DCT 2016 | 26 | +0.942 |
| DCT 2018 vs Gemini S 2021 | 26 | +0.954 |
| DCT 2016 vs Gemini S 2021 | 26 | +0.954 |
| DCT 2016 vs McDonald 2017 | 25 | +0.925 |
| McDonald 2017 vs Gemini S 2021 | 24 | +0.920 |
| DCT 2018 vs McDonald 2017 | 24 | +0.894 |

McDonald 2017 against Gemini South 2021 is a different telescope on a different
continent four years later, and it correlates at +0.92.

A single master pattern — the median of the four nights, amplitude 0.0109 over
the 33,087 pixels all four measure — captures **74-88%** of each night's own
pattern by variance.

That is a bigger result than the per-night calibration it replaces. It means one
pattern, measured once from a handful of well-populated nights, could serve the
whole archive: the 46% of nights with fewer than five standards, and the 12,866
science targets, which have no standards of their own at all. The per-night
pattern stays the better choice where five standards exist; the master pattern
is what makes everything else possible.


## The run record

Per the repository's convention the record is the product and the `.npz` arrays
are a regenerable cache. `record.py` used to key its rows on a hard-coded
`page`/`epoch` pair; it now takes `key_fields`, so the Arcturus atlas keys on
`("page", "epoch")` and an IGRINS night on `("frame", "order")`, and
`extra_columns` carries what the shared schema has no place for — the band, the
airmass, the zenith angle, the MJD, the telescope and the normalized surface
conditions.

Two details were not obvious. Which molecules have lines depends on the window,
so an order's parameter vector is *not* the run's: the driver builds the union
of species across orders and remaps every row's sigma and correlation into it,
or the record would line one order's CH4 up against another's CO2. And the
instrument fingerprint had to learn the built-in Gaussian, since there is no
`InstrumentProfile` object on this path; `ils_fingerprint(None, ...)` pushes an
impulse through the same `_gaussian_convolve` the model calls, for the same
reason the FTS branch pushes one through the real `convolve`.

A record written before `key_fields` existed reads back as `("page", "epoch")`,
so the committed `arcturus_atlas.h5` is untouched and
`scripts/rebuild_arcturus_page.py --check` still reproduces its cached arrays to
2.8e-7.

For this night the two records come to 712 KB together, against 71 MB of `.npz`
for the same 485 order-frames.

Sharding one band across devices by order makes each shard write its own record
and its own per-frame summaries; `--summary-suffix` keeps them apart and
`merge_records` combines the shards afterwards. H and K cannot merge into one
record, because their `inputs` name different files — one record per band per
night.


## The stellar velocity belongs to the order, not the star

The A0V fits free `stellar_velocity_kms` per order, and it does not behave
like a star's velocity. In the full fits it rails at its +-60 km/s bound in 42%
of order-frames, and among orders with a Brackett line well inside them the
unrailed values of one frame scatter by 20-40 km/s. Frame medians repeat only
loosely per star (HIP 82560 -40.2 and -40.9, k Tau +4.2 and +4.8, but chi Cap
-12.8 and +8.9).

The scatter is not noise. Each order's offset from its frame's median repeats
across four nights, three telescopes and every star:

| order | line position | offset, km/s | frames |
|---|---|---:|---:|
| H98 | Br9 at 0.28 | +8 +- 5 | 29 |
| H103 | Br10 at 0.42 | -5 +- 3 | 43 |
| H106 | Br11 at -0.07 | **-24 +- 4** | 25 |
| H109 | Br12 at 0.23 | +13 +- 4 | 42 |
| H111 | Br13 at 0.14 | **-23 +- 5** | 11 |
| H113 | Br14 at 0.45 | **-25 +- 7** | 14 |
| H114 | Br15 at 0.15 | -15 +- 3 | 29 |
| H115 | Br15 at 1.02, Br16 at 0.06 | **+45 +- 4** | 32 |
| H116 | Br16 at 0.93, Br17 at 0.14 | **-21 +- 3** | 42 |
| H119 | Br20 at 1.03, Br21-22 inside | **+39 +- 5** | 21 |

Line position is where the line falls in the order's usable pixels, 0 at the
blue end and 1 at the red. The large offsets are almost all orders whose line
sits at or past an edge of the usable data, so that only one wing is fitted,
and the sign follows the edge: a line at the blue end is pushed blueward, one at
the red end redward -- in both cases *out* of the window, which lowers the
model's wing inside it. The likely reading is that the model's wings are too
strong for the continuum-normalized data where the degree-9 continuum cannot
tell a wing from the blaze, and a free shift is the cheapest way to weaken one.
H113, a central line at -25, does not fit that story.

Two consequences. Pinning one stellar velocity per frame is not neutral: it
takes away the freedom that is absorbing a wing mismatch and would move it into
those orders' residuals. And a star's velocity should come from orders with a
central line (H98, H103, H109, H113, H120), not from all of them. The actual fix
is the wing -- a fixed master blaze, so the continuum cannot trade against a
wing -- which is also what fitting vsini would need, since rotation shapes the
core and not the wings. None of this limits the telluric correction, whose
residual at the Brackett lines is already at the noise.

## Line coupling on a real night

LBLRTM applies AER's first-order line coupling to about half the CO2 lines in
both IGRINS CO2 bands, and with it tellurix agrees with LBLRTM to 0.04% where it
otherwise left 0.7% (`docs/lblrtm_corrected_mode.md`). On DCT 2018's ten
standards, refitted in the CO2 orders with `fit_igrins_standard.py
--line-coupling` and otherwise as the ERA5 run (`docs/igrins_line_coupling_dct2018.json`):

| order | CO2 peak optical depth | per-pixel z, without -> with | frames better |
|---|---:|---|---:|
| K89 (band centre) | 169 | 1.966 -> 1.683 | 10 of 10 |
| K87 | 36 | 1.732 -> 1.700 | 10 of 10 |
| K91 | 52 | 2.292 -> 2.278 | 8 of 10 |
| K86, K92 (band edges) | 36, 24 | 1.563 -> 1.616, 2.940 -> 2.956 | 0 of 10 |
| H111-H115 (1.6 um band) | ~1.9 | within 0.003 | -- |

Coupling is worth 14% at the 2.0 um band centre, a little in the orders beside
it, and costs 1-3% at the two band edges -- the edge orders' signs are
consistent across all ten frames, so that is a real mismatch, not noise. **It is
not the implementation**: over K86 and K92 themselves, with identical layers,
tellurix's coupling term matches LBLRTM's to an RMS of 0.00014 against terms of
0.004 and 0.002, and the fitted transmission agrees to 0.032% (K86) and 0.055%
(K92, with the 25 cm-1 cutoff) at the 99th percentile; K89 agrees to 0.15%
(`docs/lblrtm_identical_layers_k_orders.json`). LBLRTM would lose the same at the
band edges, so the loss is in AER's coupling coefficients or in first-order
coupling itself. The cost is small: the fit stages take the same time, and
refreezing the opacity for the products adds 3% to an H run and 8% to a K run. In H, where CO2 peaks near optical depth 2, it changes nothing at this
night's S/N. **It is not the CO2 per-night systematic**: the fitted CO2 columns
and their airmass slopes move by at most 0.0016 in every order but H124, whose
CO2 scale sits near 3.6 in both runs and is not a measurement. Coupling stays
off by default; turn it on for fits that lean on the 2.0 um CO2 band centre.

## What moves the well-mixed columns

GitHub issue #1, **resolved: a PLP spectrum combines a whole sequence of
exposures, up to 37 minutes long, and the airmass came from the first.** The
story in order -- what the systematic looked like, what it was not, and what it
was. `scripts/analyze_igrins_dry_systematic.py` rebuilds the
three-night table from the run records alone (`docs/igrins_dry_systematic.json`)
and then looks at the frames rather than the slopes. It uses one fixed order set
per species -- CO2 in H108-H109, H111-H117 and K85-K92; CH4 in H104-H110 and
K73-K83, the orders that pass the ladder's 0.15 optical-depth cut on every
frame of every night -- because a record does not carry the optical depth.
The table comes back as before: CO2 slopes +0.005, +0.048, -0.029 per airmass
(chi-squared 8.8 on 2 dof, against 8.2 from the summaries), CH4 +0.010, +0.048,
-0.027 (7.3 against 7.7); with ERA5, 4.5 and 7.4.

### It is one number per exposure

| night | PWV | frame scatter, CO2 / CH4 | per-frame error | H vs K, CO2 / CH4 | CO2 vs CH4 |
|---|---:|---:|---:|---:|---:|
| DCT 2018-12-20 | 2.4 mm | 0.004 / 0.006 | 0.005 / 0.008 | -0.71 / +0.03 | +0.61 |
| Gemini S 2021-03-16 (ERA5) | 5.2 mm | 0.008 / 0.008 | 0.004 / 0.009 | +0.74 / +0.56 | +0.73 |
| DCT 2016-12-08 | 6.9 mm | 0.024 / 0.026 | 0.007 / 0.007 | **+0.93 / +0.92** | **+0.98** |
| McDonald 2017-04-20 | 10.7 mm | 0.037 / 0.038 | 0.005 / 0.006 | **+0.91 / +0.98** | **+1.00** |

On the two wet nights the columns move by four to seven times their error, and
the H and K detectors -- different bands, blazes, fringes and fitted LSFs --
report the same movement, as do CO2 and CH4. McDonald's frame 0092 reads CO2
-0.053 in H and -0.051 in K; frame 0039, +0.046 and +0.059. Call the mean of the
four (two species, two bands, each about its night mean) the frame's **dry
factor**. On the dry night it is noise -- the bands do not even agree in sign --
and its size grows with the water across all four nights -- which turned out to
be a coincidence of how those nights were observed. Each night's airmass
and time slopes are that factor regressed against airmass and time, which is
why CO2's and CH4's agree to the third decimal on both wet nights.

Within a night it follows nothing recorded consistently: against time r = -0.73
at McDonald and +0.73 at DCT 2016, against surface temperature +0.86 and -0.63,
against the fitted water +0.56 and +0.13.

### It scales the optical depth, and only that

Every order moves with the frame (`order_response`): regressed on the dry
factor, the median order slope is 0.93-1.11 in each band and species on both
wet nights, with median r of 0.89-0.99. K89, the saturated 2.0 um band centre
at peak optical depth 169, sits at 0.97 and 0.95, the same as H-band CO2 orders
that peak near 2. And the residual against transmission is the same in the
frames at both ends of the factor (`residual_vs_transmission`, K band): within
0.1% everywhere for CO2 and within 0.2% for CH4, in frames whose columns differ by
11% (McDonald) and 10% (DCT 2016). The spectra are fitted by a scaled dry optical
depth, with nothing left over.

### What that rules out

- **Blaze and fringe.** Each belongs to one detector and acts through the
  continuum and the order ends. The effect is common to H and K at r =
  0.91-0.98 and has the same slope at optical depth 2 and 169. (That argument
  does not reach the LSF: seeing sets the slit illumination of both arms at
  once.)
- **An additive offset or veiling, and a broad LSF wing** (scattered light,
  sky residual, imperfect extraction background). Either dilutes line depth; a
  column scale absorbs that for weak lines, but not for strong ones. To make the
  observed spread it would need an offset of +-5% of the continuum, which leaves
  2.0-2.5% of residual at transmission 0.15-0.3 and 1.2-1.5% at 0.3-0.45, of
  opposite sign at the two ends of the factor (`veiling_signature`). The
  measured curves are flat to 0.2% and do not differ between the frames.
- **A per-frame atmosphere.** Replacing the night's whole analytic profile
  with ERA5 -- up to 3.9 K warmer in the lower troposphere, 13 K colder in the
  stratosphere, its water a different shape -- moves the frames' columns by
  0.1-0.7% rms about their mean shift (`profile_sensitivity`). A profile built
  per frame from the header would differ from the night's by 0.7-2.9 K at the
  surface and at most 2 hPa, so it cannot reach the 2.5-3.7% frame scatter.
- **Detector nonlinearity.** The same star 7 minutes apart on DCT 2016 (HD
  26512, frames 0123 and 0127) at 5.9 times the counts differs by +0.016, the
  wrong sign for a response that flattens at high counts.

The slant path had been ruled out before (the headers agree with geometry to
0.2 deg on both wet nights), and the frames seemed to agree: McDonald 0084 and
0151, at airmass 1.14 and 1.02, differ by 2.8%, where a zenith-angle error can
do almost nothing. Both conclusions were about the wrong exposure -- 0084 is
eight 250 s exposures spanning 37 minutes (below).

### Two refits

Two GPU refits of McDonald 2017, H104-H117 and K73-K92 (`refits_mcd2017`):

- **The response pattern is not it.** With the second pass off
  (`--no-fixed-pattern`) the dry factor is unchanged: frame scatter 0.0346
  against 0.0354, r = 0.997 with the full run.
- **Nor is a trade with the water.** With CO2 and CH4 pinned at each order's
  night median (`--fix-columns-from`, new), the water moves by at most 0.005 in
  log column. Instead the residual grows where the factor was large -- frame
  0039 by 3.5% (H) and 6.4% (K), frame 0092 by 1.5-2.1%, the frames near zero
  by under 0.1% -- and the LSF narrows or widens by 2-4% in step (r = -0.97 to
  -0.99) to deepen or fill the lines it can no longer scale. The spectra really
  do carry more or less dry absorption than the model's path allows.

### The cause: a frame is a sequence, and its header is one exposure

The catalog row of McDonald 0039 lists `FILES` 39-46: the PLP spectrum is the
combination of eight 90 s exposures, two ABBA sets. Its header's `DATE-OBS`,
`DATE-END` and `ZDSTART` describe the first of them, and that is the zenith
angle the fit used -- the one checked against geometry to 0.2 deg, correctly,
for that exposure. k Tau was setting at airmass 3, and over the 16 minutes of
the sequence its airmass rose by 9%. The combined spectrum saw the mean.

Frames on DCT 2018 are single AB pairs of 30-60 s, 2-5 minutes in all;
McDonald 2017 and DCT 2016 combine four to eight exposures of up to 300 s, 4-37
minutes. That, not the water, is what separated the nights. The catalog's `JD`
turns out to be the sequence midpoint (except on DCT 2018, a constant 241 s
off), so it measures the readout-and-nod overhead per exposure: 24-46 s on
these nights, against 36-39 s for thousands of back-to-back sequences across
the archive. The prediction for each frame is ln(mean sec z over the sequence /
the airmass used) (`sequence_prediction`):

| night | longest sequence | r, predicted vs dry factor | slope | scatter before | left after |
|---|---:|---:|---:|---:|---:|
| DCT 2018-12-20 | 5 min | 0.15 | 0.16 | 0.0042 | 0.0053 |
| McDonald 2017-04-20 | 37 min | **0.995** | **0.97** | 0.0354 | 0.0038 |
| DCT 2016-12-08 | 22 min | **0.992** | **1.09** | 0.0249 | 0.0038 |

No free parameter: the sequence length comes from the catalog. On DCT 2018 the
prediction is under 0.6% and the factor is noise, as it should be.

It also explains everything on the list above: one path factor for the whole
exposure, so common to both bands and both gases and uniform in line strength;
a pure optical-depth scale, so no residual signature; a setting star early in
the night and a rising one later, so a "dependence on time"; and a link to the
water only through which nights were observed with long sequences. Once more,
a fit absorbs a slant-path error into the columns without moving its residual.
The header check could not see it: geometry and header agree for the exposure
they both describe.

### The fix, and the three nights again

`sequence_zenith_angle_deg` (`tellurix_igrins.igrins`) returns the zenith
distance whose airmass is the mean of sec z over the sequence;
`igrins_pointing.py --sequence --write all` computes it per frame from the
catalog's `FILES`, `EXPTIME` and `JD` and writes `pointing.json` with source
`sequence`, which the reader prefers for both bands. Refitted with it, the same
frames, settings and orders as the analytic runs (`seq_*`, `*_sequence` in the
report):

| | CO2 per airmass, three nights | chi-squared | CH4 per airmass | chi-squared |
|---|---|---:|---|---:|
| first exposure's airmass | +0.005, +0.048, -0.029 | 8.8 | +0.010, +0.048, -0.027 | 7.3 |
| **sequence airmass** | **-0.001, -0.001, -0.005** | **0.6** | **+0.003, +0.003, -0.006** | **2.0** |

The per-frame dry scatter falls to 0.0033-0.0052 on all three nights, the
bands no longer agree with each other about it (r between -0.63 and +0.18), the
time-of-night slopes are within 1.4 sigma of zero, and the residuals do not
move (median rms/noise identical to 0.002 in every run, the same 1,457 rows
plus one). The weighted mean slope is -0.0016 for CO2 and +0.0006 for CH4 per
unit airmass: **the slant path is now consistent across nights to a few
tenths of a percent per airmass**, where the night-to-night scatter quoted
before was 3-4%.

What this does not cover: every other IGRINS product -- the ERA5 runs, the
night calibrations, the master pattern, the transfer and science runs -- was
fitted with first-exposure airmasses, and Gemini South 2021 and McDonald 2015
have no sequence sidecars yet. A science frame is a sequence too.

Two side findings. `--dry-shift` (`docs/igrins_science.md`) was this effect
caught from the other side: it was added because McDonald 2017's K frames above
airmass 2.4 needed CO2 and CH4 7-8% up together, and those are k Tau and HD
53205, the two long sequences of setting stars. With sequence airmasses it
should be unnecessary; that is to be measured, not assumed. And
`analyze_igrins_ladder.py` had begun double-counting: every K run refitted for
the zenith-angle fix kept its pre-fix shard summaries beside the new merged
ones, so a rerun read each K order twice, half of it at the defective angle.
It now drops a shard when a whole-frame summary exists, and reproduces the
committed reports exactly.


## What is not done yet

- **Science frames.** Whether a night's standards can set the telluric model of
  a frame they do not include is measured in `docs/igrins_transfer.md`: yes,
  with one velocity shift and one water scale fitted on the frame. Not yet on a
  target with lines of its own.
- **Refit the rest with sequence airmasses.** The per-night systematic
  (GitHub issue #1) was the airmass of a combined exposure sequence; three
  standards nights are refitted (`seq_*`). Still on first-exposure airmasses:
  the ERA5 runs, the night calibrations and master pattern, the transfer and
  science runs; Gemini South 2021 and McDonald 2015 need `igrins_pointing.py
  --sequence` first. Then measure whether `--dry-shift` is still needed.
- **Build and ship the master pattern.** The four nights above show it would
  capture 74-88% of each night's response and works across telescopes, so it is
  now a matter of measuring it from enough nights and storing it, not of
  establishing whether it is sound.
- **Two unexplained order-frames**, H107 and H104 of the DCT A0V run, at 115 and
  33 sigma out of 516. They are excluded from the pooled statistics above by a
  cut rather than understood.
- **A per-frame atmosphere.** Measured against the dry factor above: a whole
  ERA5 profile moves each frame's columns by 0.1-0.7% rms, so the 0.7-2.9 K the
  header's surface temperature moved within a night cannot matter. Not worth
  building.
- **`vsini` is fixed, not fitted.** The A0V run used 150 km/s for every
  standard. A0V rotation velocities range over roughly 100-250 km/s, and the
  Brackett lines are already hundreds of km/s wide from Stark broadening, so
  this should be second order — but it is assumed, not measured. Fitting it,
  or one stellar velocity per frame, waits on the wing mismatch above.
