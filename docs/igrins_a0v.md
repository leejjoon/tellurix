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

## What is not done yet

- **The K band.** The reader and driver handle it, but only H has been run.
  K carries Br-gamma at 2.1661 um and the Pfund crowd above 2.27 um, both
  already in `stellar_line_mask`, plus CO and CH4 that H does not constrain.
- **A real A0V model.** `--stellar <npz>` accepts one and lifts the hydrogen
  mask. Payne Zero's range reaches 9,500 K, but the only features that matter
  here are Stark-broadened hydrogen lines, and an emulator spanning 6,500 K may
  not reproduce those. The flat-source path does not depend on it, and
  `A0V_NORM` in the file is a free comparison.
- **The airmass ladder**, which is the test that makes all of this worth doing.
  DCT 2018-12-20 has 10 standards from airmass 1.07 to 2.49 in one night,
  including the same star three times. The retrieved H2O column should scale as
  sec(z); the retrieved CO2, CH4 and O2 column scales should be **constant**,
  because those abundances are known and fixed, so any airmass trend in them is
  a direct measurement of atmosphere-profile error. Nothing in the Arcturus
  atlas could measure that.
- **The run record.** These fits write `.npz` and a JSON summary.
  `src/jax_telluric/record.py` keys its rows on a hard-coded `page`/`epoch`
  pair; generalizing that would let one record module serve both pipelines.
