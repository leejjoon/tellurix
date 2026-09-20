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
  zero. Measured on order 10 of the reference frame, leaving the order
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

**The stars themselves.** A night observes a handful of targets, each over a
limited span of airmass, so star identity and airmass are partly confounded.
On this night they are *completely* confounded at the top: chi Cap is the only
target above airmass 1.82.

| species | orders/frame | per airmass | per hour | joint d/dAM | frame scatter |
|---|---:|---:|---:|---:|---:|
| H2O | 39 | +0.1280 ± 0.0534 (2.4σ) | **-0.0342 ± 0.0058 (5.9σ)** | -0.0254 | 0.1164 |
| CO2 | 18 | -0.0014 ± 0.0059 (0.2σ) | +0.0013 ± 0.0009 (1.5σ) | +0.0078 | 0.0110 |
| CH4 | 19 | +0.0153 ± 0.0080 (1.9σ) | -0.0018 ± 0.0015 (1.2σ) | +0.0165 | 0.0111 |

### One target carries every slope

Refitting with each object dropped in turn:

| species | slope, all 10 frames | without chi Cap | shift |
|---|---:|---:|---:|
| H2O | +0.1280 | +0.3088 | 3.4σ |
| CO2 | -0.0014 | -0.0218 | 3.5σ |
| CH4 | +0.0153 | +0.0045 | 1.3σ |

Dropping chi Cap still moves the water slope by 3.4 sigma — but water genuinely
varies, so that is weather. For the well-mixed species the correction above
largely cured it: CH4's leave-one-out shift is 1.3σ and CO2's 3.5σ.

**The defensible statement is still narrower than the table alone suggests.**
chi Cap is the only target above airmass 1.82, so star identity and airmass stay
confounded at the top and no slope across that gap is purely atmospheric. What
the night supports is that the well-mixed columns are flat to
CO2 −0.0006 ± 0.0052 and CH4 +0.0030 ± 0.0071 per unit airmass, both under 0.5σ,
with a leave-one-out excursion no larger than the quoted error for CH4.

An earlier version of this document quoted a 2.4% bound out to airmass 2.5 from
a fit with the same confounding, before the instrument response was corrected.
A ladder needs *several* stars at high airmass, not several frames of one; see
`leave_one_object_out`, which the analysis now reports by default.

**The water trend is the sky, not the model.** Against airmass H2O is 2.0σ;
against time it is 5.2σ, and the residual scatter halves. In a joint fit the
airmass term collapses to -0.0320 while the time term survives at
-0.0411 per hour. The retrieved precipitable water falls monotonically
through the night — an ordinary drying night, recovered from the spectra alone.

**Repeatability.** chi Cap's two frames 3.7 minutes apart differ by CH4
+0.002, CO2 +0.001, H2O -0.027 — so a few tenths of a percent is the
noise floor for the well-mixed species, and H2O's 12% frame-to-frame
scatter is real weather.

**Fit quality does not degrade with airmass**: +0.111 ± 0.096 sigma per unit
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
| H05 | Br10 | 594 | 17.09 | 2.56 | 6.7 |
| H11 | Br12 | 576 | 2.88 | 2.14 | 1.4 |
| H15 | Br14 | 602 | 6.43 | 1.87 | 3.4 |
| H19 | Br18/19 | 912 | 20.95 | 2.66 | 7.9 |
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
`tests/data/igrins/`.

## What limits the residual, and the fix

The residual-against-transmission curve had a floor of a couple of sigma where
there is *no absorption at all*, which cannot be the atmosphere. Chasing it
found one cause and two dead ends.

### Two things that looked like causes and are not

**The line spread function does vary along an order, and it barely matters.**
`lsf_sigma_kms` is fitted per order but is a single constant within one.
Splitting orders into six x-segments, with five Chebyshev coefficients free per
segment so the blaze cannot leak into the width, the fitted R varies
monotonically with pixel: 29,400 to 40,400 across order 2 (33%), 42,600 to
47,200 across order 24 (10%), always narrower at high x. But freeing the width
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
- It is strongest in orders with no telluric absorption (H10, median T 0.998,
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

An order with no Brackett line (H02, H03, H04, H06, H07, H10, H24, H25) loses
nothing to the mask and keeps 73–77% of its columns. An order with one loses
about 590, and H20, which holds Br19 and Br20, loses 1151 — 56% of the order,
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
place. `stages_for()` adds it whenever `--stellar` is not `flat`. The fitted
velocities are self-consistent per star -- chi Cap +58.9 and +57.6, k Tau +42.7
and +43.8, HD 53205 +9.2 and +7.4 km/s -- which is a check on the whole
arrangement that the flat source cannot provide.

| | rows | reliable pixels | median residual |
|---|---:|---:|---:|
| H, flat + ±600 km/s mask | 248 | 261,053 | 1.30σ |
| **H, A0V model, no mask** | **270** | **367,810 (+41%)** | 1.33σ |
| K, flat + mask | 237 | 318,800 | 1.29σ |
| **K, A0V model** | **246** | **377,645 (+18%)** | 1.31σ |

Fitting 41% more pixels in H — including the Brackett cores, the hardest pixels
in the band — moves the residual by 0.03σ. That is the strongest statement
available that the A0V model is good enough to use.

| species | per airmass | per hour | joint d/dAM | leave-one-out shift |
|---|---:|---:|---:|---:|
| H2O | +0.1237 ± 0.0628 (2.0σ) | **-0.0348 ± 0.0067 (5.2σ)** | -0.0322 | 3.9σ (chi Cap V ) |
| CO2 | +0.0054 ± 0.0039 (1.4σ) | -0.0008 ± 0.0008 (1.0σ) | +0.0061 | 2.5σ (chi Cap V ) |
| CH4 | +0.0129 ± 0.0052 (2.5σ) | -0.0023 ± 0.0010 (2.4σ) | +0.0087 | **0.4σ (HR 1558)** |

Water is unchanged: 2.0σ against airmass, 5.2σ against time, the airmass term
collapsing in a joint fit. The sky dried; the model is fine.

**CO2 stays flat** at +0.0054 ± 0.0039 per unit airmass, 0.8% across the observed
range with a 2σ bound of 1.9%.

**CH4 changes status.** Its slope is much the same as the flat run's
(+0.0153 ± 0.0080 there, +0.0129 ± 0.0052 here) but the extra pixels halve the error, so
it is now 2.5σ — and, unlike before, it no longer rests on chi Cap: dropping
any single target moves it by 0.4σ. That is 1.8% across the range, 3.3% at 2σ.

Three reasons not to call that a detection yet. It is one of three species, so
the multiple-comparison penalty is real. CH4 shows a 2.4σ dependence on *time*
as well, which it cannot physically have, and airmass and time correlate at
-0.788 here; in a joint fit the airmass term falls to +0.0087. And it is one
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
| H2O | +0.1237 ± 0.0628 (2.0σ) | +0.2180 ± 0.2378 (0.9σ) | -0.1146 | 1.2σ |
| CO2 | +0.0054 ± 0.0039 (1.4σ) | +0.0390 ± 0.0273 (1.4σ) | -0.0003 | 1.4σ |
| CH4 | +0.0129 ± 0.0052 (2.5σ) | +0.0274 ± 0.0181 (1.5σ) | +0.0034 | 0.6σ |

**CH4 is not settled.** McDonald's slope has the same sign and a consistent
magnitude, and this time it does not rest on one star — dropping any target
moves it by 0.6 sigma. But its error is three and a half times DCT's, so it
neither confirms nor refutes: the two are consistent with each other and both
are consistent with zero at 1.5-2.5 sigma. In a joint fit against airmass *and*
time, McDonald's airmass term falls to +0.0034 and DCT's to +0.0087. Two nights is
still not enough; the question is worth a third.

**Water behaves the same way at both sites**: 0.9 sigma against airmass and
8.5 sigma against time here, the airmass term collapsing in a joint fit.
The sky changes; the model does not need to.

### A wetter site fits worse, everywhere

McDonald carries about 10 mm of precipitable water against DCT's 2, and its
median residual is 1.78 sigma in H and 2.37 in K against 1.33 and 1.31. Binned
by transmission, using the median rather than the rms because two of DCT's 516
order-frames are pathological (H09 at 115 sigma, H06 at 33, both unexplained):

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
scale at x1.7, well outside the +-25% the expansion is built for. Reseeding
centred it at 1.00. It did **not** improve the residual, so the linearization
was not what limited this night — but a seed within a factor of two is cheap
insurance and the site profiles now carry one.


## Three nights, and what they settle

A third night was run for one reason: to decide whether CH4's 2.5 sigma airmass
trend on DCT 2018-12-20 was real. **DCT 2016-12-08** was the best available
test — 11 frames, the most of any candidate, 10 distinct stars, the highest
median signal to noise of the archive's ladders, airmass to 2.59, and the same
site as the 2018 night two years earlier, which also makes it the night-to-night
stability test the response correction needed.

| night | PWV | frames | airmass | CH4 per airmass | CO2 per airmass |
|---|---|---:|---|---:|---:|
| DCT 2018-12-20 | ~2 mm | 10 | 1.07–2.50 | +0.0129 ± 0.0052 (2.5σ) | +0.0054 ± 0.0039 (1.4σ) |
| McDonald 2017-04-20 | ~10 mm | 8 | 1.02–2.99 | +0.0274 ± 0.0181 (1.5σ) | +0.0390 ± 0.0273 (1.4σ) |
| DCT 2016-12-08 | ~7 mm | 11 | 1.06–2.59 | **-0.0303 ± 0.0167 (1.8σ)** | **-0.0309 ± 0.0155 (2.0σ)** |

**The signs disagree, so the trend is not real.** For CH4 the three nights give
+0.013, +0.027, -0.030: chi-squared for a common value is 7.1 on two degrees
of freedom. CO2 behaves the same way — +0.005, +0.039, -0.031, chi-squared
6.8. The 2.5 sigma from one night was a per-night systematic.

The same night says it twice over. On DCT 2016 both CO2 and CH4 show a **4.4
sigma** dependence on *time of night*, which neither can physically have — and
that night has the weakest airmass-time correlation of the three (-0.452, against
-0.788 and -0.681), so the two axes are best separated exactly where the
impossible trend is clearest. These systematics are real and are not a fitting
degeneracy between airmass and time.

**So the slant-path bound is set by night-to-night scatter, not by any one
night's formal error.** Across three nights the well-mixed columns scatter by
0.0245 (CH4) and 0.0285 (CO2) per unit airmass, about **3%** over a
typical range, against the 1.9% a single night's error bar suggested. That is
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

The orders where the two nights agree least (H02 at r = 0.52, H05 at 0.65) are
the low-throughput ones with the fewest shared pixels.


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


## What is not done yet

- **Where the per-night systematic comes from.** Three nights agree that the
  well-mixed columns wander by about 3% per unit airmass in inconsistent
  directions, and that CO2 and CH4 can show a 4-sigma dependence on time of
  night, which is impossible. Something per-night moves them and it is not the
  slant path. That is now the most interesting open question here.
- **A pattern measured per instrument configuration** rather than per night,
  which the two-year stability above shows is defensible and would let nights
  with fewer than five standards be corrected at all.
- **Two unexplained order-frames**, H09 and H06 of the DCT A0V run, at 115 and
  33 sigma out of 516. They are excluded from the pooled statistics above by a
  cut rather than understood.
- **A per-frame atmosphere.** One profile is built per night from the median
  surface conditions, though the header gives T, P and humidity per frame and
  they moved by 3 K and 2 hPa across this night. The fitted column scales
  absorb most of that, and the airmass test above bounds what is left.
- **`vsini` is fixed, not fitted.** The A0V run used 150 km/s for every
  standard. A0V rotation velocities range over roughly 100-250 km/s, and the
  Brackett lines are already hundreds of km/s wide from Stark broadening, so
  this should be second order — but it is assumed, not measured.
