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
night, **H and K** (500 order-frames; 23 K orders skipped at the band edges).
Site profile `data/profiles/dct_2018.csv`, flat source, hydrogen series masked.
H and K are simultaneous readouts of one pointing, so the analysis merges them
per exposure.

The forward model already divides optical depth by cos(z) with the measured
zenith angle, so a fitted `log_column_scale` scales the **vertical** column
with the slant path removed. If the atmosphere model is right it does not
depend on where the telescope was pointing.

One trap first. Airmass and time of night correlate at **-0.788** on this
night — targets rise — so a trend fitted against airmass alone absorbs anything
that varies with time. Every trend below is fitted against airmass, against
time, and against both.

| species | orders/frame | per airmass | per hour | joint d/dAM | frame scatter |
|---|---:|---:|---:|---:|---:|
| H2O | 39 | +0.1206 ± 0.0665 (1.8σ) | **-0.0372 ± 0.0071 (5.2σ)** | -0.0258 | 0.1188 |
| CO2 | 18 | +0.0042 ± 0.0046 (0.9σ) | -0.0003 ± 0.0008 (0.3σ) | +0.0045 | 0.0100 |
| CH4 | 19 | +0.0037 ± 0.0070 (0.5σ) | -0.0004 ± 0.0011 (0.4σ) | +0.0020 | 0.0040 |

**The well-mixed species show no airmass dependence.** CH4 is flat to
+0.0037 ± 0.0070 per unit airmass (0.5σ) and CO2 to +0.0042 ± 0.0046
(0.9σ), with frame-to-frame scatters of 0.4% and 1.0%. Their abundances
are known and fixed, so this is not a fit succeeding — it is the slant-path
treatment and the assumed profile shape tested against a factor of 2.35 in path
length and not breaking. Taking either species' 2σ bound over the observed
airmass range puts any systematic slant-path error below about **2.4% out to
airmass 2.5**.

Adding K roughly doubles the orders that constrain CO2 and CH4 (9 to 18 and 8 to
19) and halves the slope error on CO2, from ±0.0091 to ±0.0046.

**The water trend is the sky, not the model.** Against airmass H2O looks like a
1.7σ effect; against time it is 5.1σ, and the residual scatter halves. In a
joint fit the airmass term collapses to -0.0249 while the time term survives at
-0.0400 per hour. The retrieved precipitable water falls monotonically from
2.7 mm at 01:00 UT to 2.0 mm at 07:47 — an ordinary drying night, recovered
from the spectra alone.

**Repeatability.** chi Cap was observed twice 3.7 minutes apart at airmass
2.46 and 2.50, where nothing about the sky had time to change. The columns
differ by CH4 +0.004, CO2 -0.010, H2O -0.025 — so around 1% is the noise floor for the
well-mixed species, and H2O's 12% frame-to-frame scatter is real weather.

**Fit quality does not degrade with airmass**: +0.291 ± 0.204 sigma per unit
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

- **A second night, and a second site.** Everything above is one night at DCT
  plus one Gemini South frame. The weather-convention normaliser is exercised
  on real DCT and Gemini headers but not yet on a McDonald one, where the
  degF/inHg branch lives.
- **A per-frame atmosphere.** One profile is built per night from the median
  surface conditions, though the header gives T, P and humidity per frame and
  they moved by 3 K and 2 hPa across this night. The fitted column scales
  absorb most of that, and the airmass test above bounds what is left.
- **`vsini` is fixed, not fitted.** The A0V run used 150 km/s for every
  standard. A0V rotation velocities range over roughly 100-250 km/s, and the
  Brackett lines are already hundreds of km/s wide from Stark broadening, so
  this should be second order — but it is assumed, not measured.
