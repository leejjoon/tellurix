# Telluric correction of the solar atlases: the plan

The goal is a telluric correction for every solar atlas on disk, in the form
this package already delivers for Arcturus: an unconvolved transmission on the
model's own grid, plus the corrected spectrum, in an HDF5 record.

`docs/solar_atlases.md` is the survey of what the data is. This is the plan for
fitting it. The two disagree in one place, and the disagreement is resolved
here in favour of a measurement -- see *The interpolation trap is not ours*.

**Nothing in this document has a committed JSON companion yet.** Every number
below was measured in-session from files on disk, and the method is named so
each one can be reproduced and then formalized by the phase that owns it.
Numbers marked *(Phase 2)* become `docs/photatl_ils.json` and
`docs/solar_fts_ils.json`.

## Order of work, and why

1. `ftsspec_901218_5` -- the 1990-12-18 raw FTS solar spectrum at air mass
   2.17 -> 1.80.
2. `ftsspec_901218_4` -- the same sky, same day, at 5.88 -> 3.58.
3. `photatl`, then the rest.

The stellar source is a **fixed** Payne Zero solar spectrum throughout: labels
at literature values, not fitted. Only velocity, continuum and the telluric
columns are free.

Starting on the raw FTS spectra rather than on `photatl` buys three things that
`photatl` structurally cannot give:

- **Air mass in the header.** The Arcturus run fits at `zenith_angle_deg = 0`,
  so air mass is absorbed into the column scale and a fitted scale is
  `true column x air mass / profile column`, which a single page cannot
  separate. These two files can, because they are the same sky at two air
  masses forty minutes apart.
- **An exact UT**, so `era5_site_profile.py` can supply a real atmosphere
  instead of the analytic one (ARCO-ERA5 reaches back to 1940).
- **A measured instrument profile that transfers to `photatl`.** See below.

## What was measured, before any code was written

### `ftsspec_901218_{4,5}` and `photatl` are the same instrument configuration

| quantity | `ftsspec_901218_5` | `photatl` |
|---|---|---|
| sampling | 0.0095 cm⁻¹ | 0.0094771 cm⁻¹ |
| MOPD | 34.401, 34.339 cm | 34.414 cm (median of 257 pages) |
| sinc FWHM | 0.01754, 0.01757 cm⁻¹ | 0.017532 cm⁻¹ |
| apodization | transition/path-resolution ~71 | ~20.5 |

Method: `measure_atlas_ils.measure_mopd` on the FFT envelope of the spectrum,
unmodified from the Arcturus tool. The two apodization figures are not
comparable as printed -- they are ratios to the path-difference resolution,
which differs because the ftsspec measurement used 100 cm⁻¹ chunks and
`photatl` uses 29 cm⁻¹ pages. In absolute terms both transitions are
0.56-0.71 cm, i.e. the same unapodized sinc, against the ~L/3 a Norton-Beer
taper would give.

Wallace, Livingston, Hinkle & Bernath 1996 (ApJS 106, 165) says `photatl` "is
based on three spectra obtained by Livingston in 1990 December". These are two
spectra from 1990 December, at the same sampling and the same path difference.
Treat the lineage as very likely and not proven; if it holds, these headers
carry `photatl`'s missing air mass.

### The header's stated resolution is wrong, and the data is right

An FTS spectrum is the transform of a truncated interferogram, so the cut is a
measurement of the truncation and the header is a label. Measured per file:

| file | header says | interferogram says | factor |
|---|---|---|---|
| `ftsspec_901218_4` | 0.053 cm⁻¹ | 0.01754 | **3.02** |
| `ftsspec_901218_5` | 0.053 | 0.01757 | **3.02** |
| `ftsspec_830626_2` | 0.051 | 0.04167 | 1.22 |
| `ftsspec_830626_3` | 0.041 | 0.04102 | 1.00 |

One header is exact, one is 22% off and two are out by almost exactly three.
**The error is not consistent even within one observing programme**, so no
conversion recovers the truth from the label; the cut has to be measured.
`docs/solar_ils.md`

### `photatl` is a constant-MOPD atlas, where Arcturus is constant-R

Measured over all 258 pages with the Arcturus tool: *(Phase 2)*

| | `photatl` | Arcturus, for scale |
|---|---|---|
| MOPD p16-p84 | 34.207-34.552 cm, spread **1.010** | 8.82-25.32 cm, spread 2.87 |
| corr(MOPD, nu) | **-0.077** | -0.93 (the constant-R law) |
| sinc FWHM | **0.017532 cm⁻¹, constant** | tracks 1/nu |
| drop across the cut | 1.40 decades | 2.34 |

So R runs **114,075 at 2000 cm⁻¹, 285,188 at 5000, 513,338 at 9000**, and the
`R ~ 300,000` quoted once for the series in Wallace et al. 1996 is a mid-band
value rather than a constant. A handful of pages come back at 2.68 cm and are
failures, not measurements; keep the `mopd_measurable` guard that leaves 598 of
620 Arcturus page-epochs usable.

This retires what the sibling project calls the weakest input in its whole
programme. `dss/calib/lsf.py` gives `photatl` the widest LSF prior of any atlas
because `data/atlases/nso/PROVENANCE.md` concludes the absolute scale "is not
obtainable this way" -- the reasoning being that an FTS spectrum is zero-filled
by a factor its sampling does not reveal. That is true and it does not matter:
zero-filling extends the interferogram with zeros and does not move the cut.

It also removes a free parameter. `lsf_sigma_kms` is the dominant at-bound
parameter on Arcturus (13.4% of converged pages, 76 railed at the 0.05
minimum), and the per-page MOPD scatter is what it was absorbing. Here one
atlas-wide constant serves every window.

### The interpolation trap is not ours

`docs/solar_atlases.md` and the sibling project's `solar_data_status.md` both
lead with `photatl` interpolating 23.5% of its pixels where the sky is opaque.
That is real, and it applies **only to the `solar` column**.

Median |second difference| in the `atmospheric < dg` regions against the
transparent regions of the same page:

| page | column | opaque pixels | transparent pixels |
|---|---|---|---|
| wn5000 | solar | **1.11e-16** | 9.00e-04 |
| wn5000 | atmospheric | 3.32e-03 | 8.40e-04 |
| wn5000 | total | 3.06e-03 | 1.08e-03 |
| wn2000 | solar | **1.00e-05** | 7.25e-03 |
| wn2000 | total | 4.32e-03 | 7.23e-03 |

The solar column is exactly linear there, as the README warns. The atmospheric
and total columns are not -- they carry *more* structure in the opaque regions
than outside, which is what real absorption does. `total` is also not
reconstructible from the other two (rms 0.0054-0.032, max 0.078 over four
pages), so it is an independent measurement rather than a product.

**So the fit target is `total`**, the observed spectrum, measured at every
pixel, exactly as the Arcturus driver fits `observed`. The interpolation is a
trap for whoever uses the solar column, which is what `photatl`'s atmospheric
column was built to enable and what this work replaces.

For the record, the interpolated fraction of the `solar` column by band, from
the sibling project's own `coverage_report.json`: 5.8% over 1.11-1.67 um (120
pages), 10.9% over 1.67-2.5 um (80), 30.7% over 2.5-4.0 um (40), 25.0% over
4.0-5.4 um (18). Mean 23.5%, median 10.7%; 49 pages over half, 7 entirely.

### Both formats store a uniform grid at reduced precision, and neither says so

`photatl` writes **single-precision** wavenumbers. The spacings within a page
take two values exactly one float32 ulp apart -- measured ratio 1.01-1.15 at
1850, 5000 and 8975 cm-1 -- so the apparent 5% spacing modulation is
quantization, not sampling. The jitter about a uniform grid grows with
wavenumber:

| page | residual | as a fraction of the 0.017532 cm-1 FWHM |
|---|---|---|
| wn1850 | 6.1e-05 cm-1 | 0.4% |
| wn5000 | 2.6e-04 | 1.5% |
| wn8975 | 4.8e-04 | **2.8%** |

The ftsspec files store four decimals, which is a flat 5e-05 = 0.3% everywhere.

Fitting all N samples averages the quantization down by sqrt(N), so a
reconstructed grid is better than any stored sample and
`nso.uniform_wavenumber_grid` does that. Taking the spacing from the two
endpoints instead would inherit their quantization in full, and using the
stored values leaves a jitter that aliases into a fitted velocity.

**The fitted spacings agree to the eighth decimal**: 0.00947710 for every
`photatl` page and 0.00947709 for `ftsspec_901218_5`, against the 0.0094771 the
sibling project documents. That is the strongest single piece of evidence yet
that the two products come from the same observations.

### The data is 5-10x better than Arcturus, and that is the hard part

Robust noise in continuum units (`atlas.robust_noise` on 100 cm⁻¹ chunks):

| window | file 4 | file 5 |
|---|---|---|
| 2000-2100 | 0.0103 | 0.0073 |
| 4500-4600 | 0.00083 | 0.00053 |
| 6000-6100 | 0.00069 | 0.00046 |
| 8000-8100 | 0.00102 | 0.00066 |

Arcturus sits near 0.005. **The residual will be model-dominated from the first
fit.** A fit reproducing Arcturus's absolute accuracy would report 30-45 sigma
here. Quote absolute residual in continuum units as the primary metric; keep
reduced chi-squared as a diagnostic, not as a pass/fail.

### Air mass moves during the scan, and for file 4 that matters

The headers give start and stop air mass, not a constant:

| file | UT | air mass | mean |
|---|---|---|---|
| `ftsspec_901218_4` | 15:18:09-15:57:43 | 5.88 -> 3.58 | 4.730 |
| `ftsspec_901218_5` | 17:19:05-18:39:01 | 2.17 -> 1.80 | 1.985 |

The observed transmission is therefore `<exp(-tau X)>_t`, not
`exp(-tau <X>)`. For a linear `X(t)` the gap is exact:

| tau_vertical | file 5 | file 4 |
|---|---|---|
| 0.1 | +0.0000 | +0.0014 |
| 0.2 | +0.0002 | +0.0034 |
| 0.5 | +0.0005 | **+0.0053** |
| 1.0 | +0.0008 | +0.0021 |

Against the noise floor above, file 5's worst case is about 1 sigma and file
4's is 5-10 sigma. **File 5 is fittable with a single zenith angle. File 4 is
not**, and until it is treated it is a consistency check rather than a
measurement. This is the whole reason for the ordering.

### The usable range

Instrument response is dead below ~1880 cm⁻¹ and above ~9050 cm⁻¹ (filter
cuts), peaking at 0.999 near 6100. Two further regions pass essentially no
photons because the sky is opaque: **2200-2400 cm⁻¹** (CO2, 4.3 um) and
**3600-3900 cm⁻¹** (H2O/CO2, 2.7 um). Those are exactly `photatl`'s two
documented holes, which is further evidence of the lineage.

Usable: about 6,570 cm⁻¹, ~692,000 pixels per file, ~219 windows of 30 cm⁻¹.
The spectra go negative in the opaque regions, which is noise about zero and
must be masked rather than clipped.

## Phase 1 -- the Payne Zero solar source

`scripts/generate_payne_zero_solar.py`, on the `generate_payne_zero_arcturus.py`
pattern, run in Payne Zero's own environment (it needs Python >= 3.11; this
package is pinned to 3.10 by `exojax==2.5.0`).

Labels, all inside Payne Zero's stated support: Teff 5772 K, log g 4.438,
[M/H] 0.0, [alpha/M] 0.0, microturbulence 1.0 km/s.

**One synthesis serves the whole roster, but the roster is in two modes.**
Payne Zero returns Eddington flux and has no mu option -- there is no
`specific_intensity`, `mu_angle` or `emergent_intensity` anywhere in its API,
and `synthesize` returns `flux_total` and `flux_continuum` built from
`eddington_flux_*`. That is exactly the right quantity for the disc-integrated
atlases and a stated mismatch for the disc-centre ones:

| atlas | what was observed | Payne Zero flux is |
|---|---|---|
| `fluxatl`, `wallace2011_flux` | disc-integrated flux | **correct** |
| IAG `spvis`, `spnir` | disc-integrated flux | **correct** |
| `photatl`, `niratl` | disc-centre intensity | mismatched |
| `ftsspec_901218_{4,5}` (`CENT DISK`) | disc-centre intensity | mismatched |
| `ftsspec_830626_{2,3}` (`SUN CENTER, 1M OUT`) | disc centre, 1 m defocus | mismatched |
| `spot1atl` | umbral intensity, ~3800 K | out of range entirely |

So the flux atlases are the *easier* half for the stellar source, not the
harder one. They are harder in other ways -- air versus vacuum wavelengths, a
subjective pseudo-continuum in `fluxatl`, and a different species regime -- but
not here.

**Part of the mismatch is correctable and should be, because it is
broadening, not mode.** A disc-integrated spectrum carries the full solar
rotation; disc centre carries none, because there the rotation is transverse to
the line of sight. `prepare_stellar_source` applies this, so one synthesis
serves both with a different argument:

- disc centre: `vsini_kms = 0`, **not** the Arcturus default of 2.0
- disc integrated: `vsini_kms ~ 1.9`, and a radial-tangential macroturbulence
  rather than the disc-centre value

What remains after that is the limb-darkening weighting of line depth, which no
choice of broadening reproduces. It weakens toward the infrared and is not
zero. Quantify it once against `photatl` and `fluxatl` over their 0.9-1.3 um
overlap -- same Sun, same epoch-ish, two modes -- rather than carrying it as an
unbounded worry.

**Payne Zero is also 1-D LTE and has no convective blueshift**, while the IAG
convective-blueshift table shows it spans 415 m/s across line depth alone
(-514 m/s for the shallowest lines, -99 for the deepest). This one hits both
modes equally.

Neither limitation is fatal, for the reason the Arcturus work established: the
source enters the correction only through the convolution weighting, worth a
median 0.00016 there against 0.0137 of noise, because `corrected` is exactly
`observed / effective_transmission`.

**Therefore run `--stellar flat` as the control on every window**, as the
IGRINS A0V work does. It costs one flag and it is the only way to know what the
solar model bought. Where the two disagree, the flat fit is the one that
depends on no stellar model at all.

### Coverage and sampling

**Range is not a problem at either end.** Payne Zero's line data spans
19.8-99,967 nm and its continuum edge grid 1-500,000 nm, so the 296 nm blue
edge of `fluxatl` and the 5.4 um red edge of `photatl` are both well inside.
`data/stellar/arcturus_payne_zero_full.npz` already demonstrates 900-5400 nm in
1,075,056 points.

**Sampling is the constraint, and it is set per band by the atlas, not by the
Sun.** That file is r-grid 600,000 = 0.4997 km/s. `resample_stellar_source`
refuses a source coarser than the model grid as a resolution lie, so the rule
is

    r-grid  >=  4 * nu_max / FWHM_measured

with four samples per resolution element. **Take `FWHM_measured` from the
atlas's own interferogram, not from its documentation** -- every atlas here is
an FTS product, `measure_atlas_ils` applies to all of them unmodified, and the
ftsspec header has already been caught stating a resolution 2-3.7x away from
its own cut.

For the disc-centre IR atlases, where FWHM = 0.017532 cm⁻¹ is measured, this is
`scripts/generate_payne_zero_solar.py --band {blue,mid,red}`:

| band | nm | cm⁻¹ | R at nu_max | r-grid | step | points |
|---|---|---|---|---|---|---|
| `blue` | 1100-1510 | 6623-9091 | 518,532 | 2,074,130 | 0.1445 km/s | 657 k |
| `mid` | 1490-2610 | 3831-6711 | 382,809 | 1,531,237 | 0.1958 km/s | 858 k |
| `red` | 2590-5330 | 1876-3861 | 220,226 | 880,905 | 0.3403 km/s | 636 k |

The sources synthesized before the ILS was measured used 0.01727 cm⁻¹, which
asks for a *finer* grid than the table and so remains valid.

2.18 M points in all. The bands overlap by 20 nm and together cover
1100-5330 nm, containing ftsspec's usable 1105.0-5319.1 nm with margin, so a
model grid reaching past its fit window never extrapolates off an edge.

The blue atlases will be more demanding and their numbers are not yet
measured: if `niratl` and `ftsspec_830626` share the same path difference, they
reach R = 787,000 at 13,600 cm⁻¹ and 1.2 M at 20,735 cm⁻¹, needing r-grids of
3.1 M and 4.8 M. **Do not generate one universal source.** Synthesize per fit
band, at the r-grid that band's measured MOPD requires, and cache it.

## Phase 2 -- reader, ILS, site profile

`src/tellurix/nso.py` -- **built**, with `tests/test_nso.py` -- self-contained
in the way `atlas.py` is:

- `read_fts_spectrum(path)` -- parse the 15-line header (source name, comment,
  MST date, Julian day, UT/sidereal/hour-angle/air-mass start and stop, the
  `m`/`k`/`s` parameters, sample count, stated resolution), then two columns.
  Carry the air-mass pair and the UT pair as fields; do not average them at
  read time.
- `fts_spectral_order(spectrum, v1, v2, ...)` -- cut a window, reverse to
  ascending vacuum wavelength (`SpectralOrder` requires it, and getting the
  mirror wrong still looks plausible), mask non-finite, non-positive, and a
  response floor.
- `read_photatl_page` / `photatl_spectral_order` in the same module: drop the
  final row (the README says `dg` replaces the last point of `total`, so it is
  not data), carry `dg`, fit `total`, mask from the `atmospheric` column.

**Refuse an opaque window rather than fit it to noise.** A relative
saturation floor is not enough on its own: a window that passes no light has a
continuum made of noise, and every pixel clears a fraction of it, so the
relative test passes 99.9% of a dead window. The window's continuum over its
own noise separates the cases with room to spare, measured on
`ftsspec_901218_5`:

| window (cm⁻¹) | continuum/noise | what it is |
|---|---|---|
| 6000-6030 | 2702 | clean, peak response |
| 2200-2230 | 188 | real signal, heavily absorbed |
| 3700-3730 | 11 | the 2.7 um hole |
| 2340-2370 | 6 | the CO2 band core |
| 1700-1730 | 5 | below the filter cut |

`minimum_continuum_snr` defaults to 30 and raises. That is the
`minimum_reliable` lesson from IGRINS: skip the order, do not write a NaN row.

**Three header traps, all now covered by tests.** The clock components are
right-justified, so a single-digit second arrives as `15:18: 9.0` -- with a
space inside the token, which a `\S+` match drops silently and did.
`ftsspec_830626_3` writes `?.??` for air mass, which must read as None rather
than fall back to the free-text comment's "EAST 3. AIRMASSES". And the header's
`number of samples` is the transform length, not the number of spectral points:
they agree for the 1990 files at 831,488 and disagree for the 1983 pair, where
the header says 745,472 and 1,048,576 against 540,672 rows on disk.

**ILS -- measured**, `scripts/measure_solar_ils.py`, reported in
`docs/photatl_ils.json`, `docs/solar_fts_ils.json` and `docs/solar_ils.md`. Pin
**one** MOPD of 34.41 cm (FWHM 0.017532 cm⁻¹) rather than fitting per window:
the page-to-page spread of 1.010 is inside the FFT bin, so a per-page value
would be fitting noise. This is the `--sinc-resolving-power`
lesson from Arcturus, on firmer ground here because the MOPD really is
constant.

**Grid spacing is per window, not shared.** This is the one architectural
difference from Arcturus. There `velocity_step_kms` is a single number for the
whole atlas because the resolving power is constant. Here the *resolution
element* is constant, so set `resolving_power = nu_mid / 0.017532` per window
and let `constant_velocity_grid` follow. A single atlas-wide step would be set
by 9050 cm⁻¹ and oversample 1880 cm⁻¹ by 4.5x.

**Site profile -- built.** `make_site_profile.py` gained a `"1990"` epoch (NOAA
GML global annual means: CO2 354.4 ppm, CH4 1.714, N2O 0.3085, against the 1994
column's 357/1.72/0.310) and `era5_site_profile.py` gained two more ways to say
where and when: `--fts`, which reads an NSO spectrum and takes the **midpoint**
of its exposure, and `--latitude/--longitude/--altitude-km/--time`. Only
`--spec` could ever supply a station pressure from a header, so these anchor on
ERA5's own geopotential, which is what `station_pressure_from_era5` is for.

ERA5 for 1990-12-18 at Kitt Peak:

| | file 4, 16:00 UT | file 5, 18:00 UT |
|---|---|---|
| station pressure | 788.7 hPa | 789.1 hPa |
| precipitable water | 3.19 mm | 3.11 mm |
| lapse over the first 3 km | 3.60 K/km | 3.54 K/km |

Three things follow.

**The slant-path test's premise holds, and this is an independent check on it.**
The two exposures are 2h20m apart, and the whole design assumes they look
through the same atmosphere. ERA5 says the water column moved by **2.5%** and
the surface pressure by 0.05% between them; the well-mixed species are
identical by construction. So a disagreement in the fitted columns larger than
a few percent is the model's, not the sky's.

**The assumed lapse rate is nearly a factor of two wrong here.** 3.5-3.6 K/km
against the analytic profile's 6.5 -- a colder-than-standard winter morning
inversion, and further off than the 4.5-8.3 K/km the IGRINS nights showed.

**The distribution is what ERA5 buys, not the total.** Holding the column fixed
at 3.11 mm and comparing the analytic profile against ERA5 layer by layer, the
analytic one is up to 8 K too cold between 3 and 6 km, up to 7 K too warm above
10 km, and puts **2-4x too much water in the upper troposphere** while putting
12% too little near the ground. A fit has one free scale per species, which
absorbs an error in the total and not in the shape, and line strength depends
on the pressure and temperature where the absorption happens.

The ERA5 cell's orography is 955 m against the telescope's 2096 m, which does
not matter for the reason the script's docstring gives: the column is anchored
at the site pressure and integrated upward.

Written: `data/profiles/kitt_peak_19901218_file4.csv`,
`kitt_peak_19901218_file5.csv`, and `kitt_peak_1990_analytic.csv` as the
control.

## Phase 3 -- fit file 5, and make the slant-path measurement

`scripts/fit_fts_window.py` is `fit_arcturus_page.py` with the reader swapped:
`accuracy_mode="mt_ckd"`, `OrderObjective`, `--precompute-opacity`, staged
`continuum,velocity,columns`, degree-3 Chebyshev over 25-30 cm⁻¹ windows
(~3,160 pixels, the same size as an atlas page).

**Species**: H2O, CO2, CH4, N2O, CO, and **O2**. The 1.27 um band sits at
7874 cm⁻¹, inside the usable range. O2 is exactly zero in both IGRINS bands and
has never been constrained by this package; check the AER line file carries the
magnetic-dipole band before promising it.

**Velocities.** `velocity_kms` free -- it absorbs any multiplicative error in
the FTS wavenumber scale. `stellar_velocity_kms` measured in a handful of clean
windows and then **pinned**, per the fixed-stellar rule. It is not zero: the
solar gravitational redshift is +636 m/s, convective blueshift is around
-400 m/s and depth-dependent, and Earth's orbital component adds a few hundred
m/s. At R = 300,000 that separation from the telluric rest frame is resolved,
which is the point.

**Each file is fitted on its own.** No joint objective, no parameters shared
between them, and -- as in the IGRINS driver -- no warm-starting one from the
other. Three reasons, and the third is the one that makes it not merely
convenient:

- The test below needs two *independent* measurements of the same columns.
  Sharing the column scales between the files would build the answer into the
  fit and leave nothing to check.
- It is already the project's rule, for the same reason. `CLAUDE.md` says of
  the IGRINS driver: do not warm-start across frames, because it "would pull
  each fit toward its neighbour and shrink the very frame-to-frame scatter the
  airmass ladder measures". Two air masses on one night is that ladder, with
  two rungs.
- Their parameter sets are not the same anyway. File 4 carries an effective air
  mass that file 5 does not need, so a joint fit would have to special-case one
  of them.

The shared work is the *window*, not the file: grid, line selection, opacity
backend, precompute and both XLA compilations depend only on the wavenumber
range, which is identical between the two files. So loop windows outside and
files inside, exactly as `fit_igrins_standard.py` loops orders outside and
frames inside, and pass flux, mask, uncertainty and zenith angle to
`OrderObjective` as jit operands rather than captured constants. That is where
the 70% saving lives, and it costs nothing in independence.

### The validation, which is the reason to start here

Run every window twice:

1. `zenith_angle_deg` from the header mean -- fitted column scales must be
   **equal** between files 4 and 5.
2. `zenith_angle_deg = 0`, the Arcturus configuration -- fitted scales must
   differ by exactly **4.730 / 1.985 = 2.383**.

Same sky, same instrument, same day, two air masses. Nothing else in this
programme can make that test, and it is a direct check on the
air-mass-inside-the-column-scale caveat the Arcturus record has to carry as an
untested assumption.

`scripts/validate_fts_fit.py` does this synthetically first, over
6000-6020 cm-1 with H2O and CH4 at the measured 0.00046 noise
(`docs/solar_fts_validation.json`). All four levels pass, and **L3 found that
the prediction above is exact only for an unsaturated species**:

| | at the header zenith | at zenith 0 |
|---|---|---|
| CH4 | recovered to +0.02% | **4.7305x truth against an air mass of 4.730** -- exact to 1 part in 10^4 |
| H2O | recovered to -0.008% | 4.799x, i.e. **1.45% high** |
| reduced chi2 | 1.06 | 1.66 |
| rms/noise | 1.03 | 1.29 |

`exp(-tau X)` is not `exp(-tau)^X` once a line saturates, so the fold is exact
where the line is optically thin and not where it is not. H2O has saturated
cores in this window and CH4 does not. Two consequences:

- **Quote the ratio test on a well-mixed species, not on water.** CH4 and CO2
  are the clean rungs; a 1.45% miss on H2O at X = 4.73 is the model behaving
  correctly, not a failure.
- **Fitting at zenith 0 is not a reparameterization, it is a worse fit** --
  chi2 1.66 against 1.06 here. The Arcturus configuration is lossy in a way the
  Arcturus data could never reveal, because it has no second air mass to
  compare against.

L0 recovers the injected columns to 0.07% and 0.02% with reduced chi2 0.978;
L1 shows the grid converged, the full and double grids agreeing to 0.026%; L2
prices the Gaussian at 4.9% of residual and under 1e-4 of column bias, much
cheaper here than on Arcturus because this sinc is narrow against the fitted
Gaussian.

`scripts/rebuild_fts_window.py --check` reconstructs the window from the report
alone and compares against the driver's own products. As with
`rebuild_arcturus_page.py`, it shares no code with the driver -- an independent
reconstruction is the evidence; calling one function twice is not. It rebuilds
from the JSON report rather than an HDF5 record, because the record arrives
with the batch driver in Phase 4; point it at the record once that exists.

## What the first fits found

Phase 3 is built and run; `docs/solar_fts_residual.md` is the record and
`docs/solar_lblrtm_reference.json` the machine-readable companion. Five things
change the plan below.

**The window matters more than any parameter.** Same code, same night, same
source: 67x noise at 6000-6030 cm-1 against 3.7x at 2030-2060. The residual
tracks how much solar structure Payne Zero has to get right, so a batch run
must choose windows on solar line content, not only on instrument response.

**Per-window species lists are required.** A species with no signal in a window
rails at a bound and absorbs model error rather than contributing nothing --
CH4 and N2O at 2030-2060, CO2 at 4350-4380. `at_bound` is the detector.

**A fitted water column is a column times a profile.** ERA5 against the
analytic profile moves H2O by 12.2% and the residual by 1.5%. The well-mixed
species move 1-2%.

**The slant-path test works and gives a weaker bound than hoped.** The air-mass
fold is exact to 0.01% on CO2 and CO within each file, and L3's predicted water
deficit shows up at 1.14% against a predicted 1.45%. But the two files still
disagree by 4.7% (CO2) and 9.6% (CO) at their own air masses, so **the honest
slant-path bound is that 5-10%, not the 0.01% the fold reproduces.**

**The residual was two missing species.** At 2030-2060 cm-1, adding OCS (nu3
at 2062) and O3 (nu1+nu3 at 2110) takes the fit from 3.75x to **1.47x** noise,
reduced chi2 14.0 to 2.2, at 392 pptv and 341 DU -- both ordinary. OCS came
from Wallace's own empirical line list for these spectra; O3 from ranking every
molecule AER ships. **Rank a species by its path-weighted column, not its
surface abundance**: ranking ozone tropospherically understated it 17x.

Nine things were eliminated before this -- velocity, LSF, column scale, solar
model, instrument FWHM, continuum degree, vertical profile shape, layer count
and line physics -- and every one of those tests was sound and beside the
point, because they all assumed the species list was right. A batch run must
scan for missing absorbers per window before trusting any residual.

## Phase 4 -- the rest of the atlases

Ordered by what each needs beyond Phases 1-3.

| # | atlas | range | needs | note |
|---|---|---|---|---|
| 1 | `ftsspec_901218_5` | 1880-9050 cm⁻¹ | -- | Phase 3 |
| 2 | `ftsspec_901218_4` | same | scan-average air mass | 5-10 sigma bias; after 1 |
| 3 | **`photatl`** | 1850-9000, 258 pages | page reader | fit `total`; per-page intensity scale absorbed by the continuum |
| 4 | `niratl` | 8900-13600, 188 pages | source to 735 nm; mask the -1.0 fill | ships an observed column already at air mass 1.0; O2 A-band |
| 5 | `ftsspec_830626_{2,3}` | 8516-20735 cm⁻¹ | visible source | `_2` is air mass 6.66 -> 4.08; `_3`'s header reads `?.??` |
| 6 | IAG `spnir`, `spvis` | 1000-2300, 405-1065 nm | **re-download**; `vsini_kms ~ 1.9` | not on disk -- only `PROVENANCE.md` survives. Disc-integrated flux, so the source mode is correct. `spnir` has tellurics, ships no telluric column, and is what the sibling project's H band actually needs; its wavelength scale is anchored on telluric O2, which this work would make circular unless the fit is judged on other species |
| 7 | `telluric_mid_ir` spec1-10 | TBD | header inspection | telluric-dominated; likely the best pure validation target, with 11 per-molecule line lists to check against |
| 8 | `fluxatl`, `wallace2011_flux` | 296-1300 nm, 2958-9250 A | air -> vacuum; `vsini_kms ~ 1.9`; blue r-grid | disc-integrated, so Payne Zero's Eddington flux is the right quantity here. `fluxatl`'s pseudo-continuum is subjective and its scale is distorted by 10-100 m/s (Reiners+2016) |
| 9 | `spot1atl` | 1970-8640 cm⁻¹ | `--stellar flat` | Payne Zero floors at 4000 K; an umbra is ~3800 K and molecule-dominated. Or reuse a transmission fitted at matched air mass |

Each writes an HDF5 record through `write_record` with `key_fields` chosen per
atlas -- `("file", "window")` for the FTS spectra, `("page",)` for `photatl`,
which is single-epoch and so does not take Arcturus's `("page", "epoch")`.
Then `export_transmission_hdf5.py` per window. `docs/using_the_corrected_spectra.md`
is the user-facing guide and should gain a solar section rather than a second
document.

Note the trap `record.py` already carries: an order's parameter vector is not
the run's when the species present depend on the window, so remap each row's
sigma and correlation into the union. That trap is now the normal case rather
than an edge case -- see below.

### 4a and 4b -- built

`scripts/extract_afgl_profiles.py` reads the six AFGL standard atmospheres out
of LBLRTM's own `src/lblatm.f90` -- 47 molecules, 50 levels, 0-120 km -- so the
abundances come from the same source LBLRTM uses. `make_site_profile.py
--afgl-model` interpolates them onto the site's layers in the log, scaling the
trended gases to the epoch by their surface ratio and leaving water to the
site's own precipitable-water argument.

`scripts/scan_window_species.py` then ranks every molecule by peak vertical
optical depth from the profile's own column. Over 2030-2060 cm-1 it returns

    fit:      H2O, CO2, CO, O3, OCS
    strongest rejected: NO2 at 1.72e-05, 58x below the cut

which is the list that took a day of elimination and a hand cross-match against
Wallace's line list to find. There are **four orders of magnitude** between the
last species in (OCS, 1.21e-01) and the first out, so the threshold is not a
close call. `docs/solar_scan_2030.json` is the record.

Two costs to fix before a batch run. The scan compiles once per species per
window and needs `--layer-chunk-size 1` for a line-rich species like O3
(38,011 lines within the 25 cm-1 margin), so it is slow; one reused evaluator
across species, as `OrderObjective` does across stages, is the fix. And
`DEFAULT_EDGES_KM` stops at 30 km above the site while **ozone peaks at 38
km** -- the AFGL profile knows this and our layer grid truncates it, so the
fitted O3 scale is partly compensating for a column we do not integrate.

### 4a. The species list is computed, not chosen

**This is the change the OCS and O3 result forces.** The batch driver cannot
take a hand-written species list, because a hand-written list is what hid two
absorbers worth a factor of 2.5 in residual, and no amount of residual analysis
found them -- nine other explanations were eliminated first, each test sound
and each beside the point.

`scripts/scan_window_species.py`, run per window before any fit:

1. For every molecule in `tellurix.AER_MOLECULE_IDS`, select lines over the
   window plus the wing margin. Molecules with no lines drop out.
2. For each, compute the **peak vertical optical depth from the profile's own
   column**, not from a nominal mixing ratio. This is the step that has to be
   got right: ranking ozone by its surface abundance understated it by 17x,
   because its column is stratospheric. The profile carries the column; use it.
3. Emit every species down to an evaluation floor, together with the ranked
   list of what was rejected and by how much.
4. The window's fit re-thresholds that list for **its own air mass**. The cut
   is 1e-3, roughly a tenth of this data's noise, on the *slant* column; the
   scan ranks the *vertical* one, and a secant multiplies every species alike.
   This is what lets one scan serve file 4 at air mass 4.73 and file 5 at 1.99,
   and it is not cosmetic: at 6270-6300 file 5 fits CO2 and H2O while file 4
   additionally clears CO and N2O. The floor must therefore sit below the cut
   divided by the largest air mass asked for -- 2e-4 covers air mass 5 --
   because rejecting on the bound is the one cut a scan cannot undo. The
   rejected list is recorded, so a surprising residual can be checked against
   what was nearly included.

It is **per window**: at 2030-2060 the answer is H2O, CO2, CO, OCS, O3, while
at 6000-6030 O2 has no lines at all and CH4 dominates.

**It is not cheap, and it does not have to be.** This was planned as cheap on
the grounds that it needs one optical-depth evaluation rather than a fit;
measured, a 30 cm-1 window costs 5m50s, nearly all of it in the two line-rich
species (O3 ships 38,011 lines in that window, CO2 10,694). A bound-based
pre-filter now rejects 16 of the 23 candidates without evaluating them at all
and leaves the verdict unchanged, but the cost lives in the species it cannot
reject, so it stays.

Three cheaper schemes were measured and rejected, and each failed for a reason
worth keeping:

- **Rank by `line_optical_depth_bound` alone**, skipping the evaluation. The
  bound's conservatism spans 1.3x to 1.05e8x across species, so it reorders O3
  and CO at 2030-2060. It is sound as a *reject* test against a threshold --
  which is what it is used for -- and useless as a ranking.
- **Coarsen the grid.** `constant_velocity_grid` refuses fewer than 2 samples
  per resolution element, which is the Nyquist guard, and the scan is already
  at 2.
- **Lower the resolving power instead.** Stratospheric lines are Doppler-limited
  at about 0.002 cm-1 against an 0.0175 cm-1 instrument element, so a coarser
  grid preferentially loses exactly the species whose column is stratospheric.
  That is O3 -- the species this whole section exists because of.

What makes the cost affordable is that **a scan is a cacheable artefact**. It is
a function of the window and the atmosphere alone: no spectrum, no observation
date, no fitted parameter enters it. One entry therefore serves file 4 and file
5, every refit, and any later re-analysis. `tellurix.ScanIdentity` keys an entry
on the window, the profile's *contents* (a sha256, not its path -- a profile
rebuilt with a different AFGL model keeps its filename), the threshold, the line
budget, the wing margin, the ILS width, the sampling and the line-file version;
`save_scan` / `load_scan` store them under `data/scans/`, and `read_scan`
re-checks the recorded identity against the filename so a hand-copied or
hand-edited entry is refused rather than trusted. Entries are committed: 4.5 kB
each, six minutes each to regenerate. Measured on 2040-2042: 1m15s cold, 1.0s
warm. An entry is not *byte*-reproducible -- the GPU kernel's reduction order
varies run to run and moves an optical depth in its last ulp -- but at 1e-16
against a 1e-3 threshold the verdict is, so two entries differing in a final
digit mean nothing changed.

At about six minutes a window, 220 windows is ~22 hours, or ~11 across the two
GPUs -- paid once for the whole atlas, not once per fit.

### 4b. The profile has to carry the species before the scan can rank them

`make_site_profile.py` currently emits six gases. A scan can only rank what the
profile knows about, so the generator needs a full trace-gas set with
**vertical distributions, not single mixing ratios**:

- well mixed through the troposphere and falling above it: CO2, CH4, N2O, CO,
  OCS
- **stratospheric**: O3 above all, and the species whose columns sit above the
  tropopause
- constant: O2, N2

The AFGL standard atmospheres that LBLRTM ships are the obvious source, scaled
to the epoch for the gases that have trended. Until that exists, a scan will
silently rank a missing species at zero -- which is exactly the failure being
designed out, so **4b blocks 4a**.

### 4c. Windows are chosen, and the choice matters more than any parameter

Same code, same night, same source: 67x noise at 6000-6030 cm-1 against 1.47x
at 2030-2060. Measured across three windows, the stellar model's share of the
residual variance runs 4% at 4.9 um, 25% at 2.3 um and 58% at 1.7 um -- **the
windows best for measuring the atmosphere are the ones worst for the stellar
model.** Choosing on instrument response alone would pick the opposite ones,
since the ftsspec response peaks at 6100 cm-1.

Choose on three measured quantities, all already available:

- `nso.window_continuum_snr` above 30 -- is there light at all
- the Payne Zero source's mean line depth over the window -- the residual
  tracks how much solar structure the stellar model has to get right
- the scan's rejected list -- how close the next unmodelled species came

### 4d. Acceptance, per window

A row is only written if:

- the scan's strongest rejected species is well below the cut
- no fitted parameter is `at_bound` -- a species with no signal does not
  contribute nothing, it absorbs model error and rails, which is how CH4, N2O
  and CO2 were each caught in the wrong window
- `jitter_over_uncertainty` is near 1; at 2030-2060 it went 3.61 -> 1.08 as the
  species list was completed, and it is the cleanest single indicator that the
  model no longer needs excess variance

Rows that fail are kept with their flags rather than dropped, as the IGRINS
`minimum_reliable` rule does -- but they must not be quoted as measurements.

### 4e. Re-measure what was measured without the missing species

The slant-path result and the inter-file column comparison were both taken with
H2O, CO2 and CO only. CO moved 10% when OCS was added, so **the 5-10%
disagreement between files 4 and 5 is not a result** until it is redone with a
scanned species list on both.

### 4f. The batch driver, and what one window actually costs

`scripts/fit_fts_batch.py` chains scan -> fit -> record. It **calls**
`fit_fts_window.py`'s `fit_window` rather than reimplementing it, which is the
one place this driver deliberately departs from `fit_arcturus_batch.py`: a
second copy of the fit was affordable there because the species list was a
six-molecule constant, and it is not affordable here, where the list is a
per-window result and a second implementation would be free to compute it
differently. `fit_fts_window.py` was split into a callable plus a CLI to make
that possible; the split reproduces the committed 4350 fit to 4e-10 in reduced
chi-squared, which is the GPU's own run-to-run scatter.

**The record is assembled from disk, not from memory.** Each window's `.npz`
carries its own `sigma`, `correlation` and ILS fingerprint, and `record_row`
reads them back at the end. So a run that is interrupted and resumed still
writes a record covering every window -- verified byte-identical across a
resume. `fit_arcturus_batch.py` does not have this property: it pops the record
block out of each row before writing the summary, so a resumed run writes a
record covering only its final pass. Worth fixing there.

**Measured costs**, on one RTX 5000 Ada:

| | per window |
|---|---|
| scan, 1876-2206 cm-1 (O3 carries 20,000+ lines) | 1.45 min |
| scan, whole band, average | 1.26 min |
| fit, including both XLA compilations | ~20 s |

224 windows at 30 cm-1 cover 1876-9091 cm-1, the range both the FTS filters and
the Payne Zero bands reach. That is about 2h20m of scanning per shard, ~2 hours
wall clock across the two GPUs, paid **once** for both files; then about 75
minutes of fitting per file.

An earlier estimate of ~22 hours in this document's history was wrong by an
order of magnitude. It extrapolated from a single 5m50s measurement of the
2030-2060 window, which was taken while the process was under memory pressure
and rematerializing heavily. Do not extrapolate a GPU cost from one sample.

**Two bugs this shook out, both about the difference between a warm cache and a
cold one, and both worth remembering:**

- `cached_scan` returned the raw computation on a miss and the stored copy on a
  hit, and only the stored copy carried the identity block. Every consumer
  therefore worked on every cache hit and failed on every cold window -- which
  is the entire first pass over the atlas. Developing against a warm cache hides
  exactly the half that has to work unattended.
- The scan accumulated compiled executables across windows and died of a 25 GiB
  allocation three windows into a batch, on a window that ran fine alone. Every
  species has its own line count and so its own shape, so nothing is ever reused
  and the JAX cache was pure retention; `jax.clear_caches()` after each species
  fixes it. A layer-chunk back-off ladder is kept as insurance and has not yet
  had to fire.

**What the scan finds that no hand-written list would.** N2 is a fitted species
at 2146-2206 cm-1; O3 and OCS run through the whole 4-5 um region; CH4 appears
and disappears across 30 cm-1 steps. The species list changes from window to
window, which is the thing 4a exists to capture.

### 4g. The first full run, and the bound that made a fifth of it meaningless

Both files fitted in one pass each, 224 windows, ~40 s a window, about two
hours wall clock on the two GPUs with the scans already cached. The first run's
numbers are kept here because what it found is more useful than the numbers
were.

**The FTS spectra are not normalized and the fitted continuum's constant term
is a log flux.** Its bound is +-2. The instrument's filters put the raw flux at
0.98 near 6000 cm-1 and **0.017 at 9060**, whose log is -4.07, so across the
whole 8656-9076 cm-1 edge the continuum railed at exp(-2) and the fit was
meaningless -- 14 windows of 224 in file 5, 15 in file 4. Measured on
9046-9076: rms/noise **293.30 -> 2.00** once the window is normalized, with
`continuum_0` off the rail at -0.118 and H2O and the velocity off their bounds
as well. It turns the worst windows in the run into some of the better ones.

This is the same failure `igrins_spectral_order` already normalizes away, for
the same reason and with the same fix -- `fts_spectral_order(normalize=True)`
divides by `fts_continuum_level`, a scalar exactly degenerate with the
continuum's constant term, so the fit stays scale free everywhere the bound was
not binding. Verified: windows near 6000 cm-1, where the raw flux is 0.985,
are unchanged.

**The lesson is about where to look.** Nine explanations were eliminated before
the missing species were found, and the residual at the blue edge would have
supported another nine. `at_bound` named it immediately: `continuum_0` at
exactly -2.000 in 14 windows, all adjacent, all at one end of the band. Read
the bound report before the residual.

Two other things the first run showed, both still open:

- `lsf_sigma_kms` is at a bound in 36% of file 5's windows and 50% of file 4's,
  overwhelmingly at the 0.05 km/s floor and concentrated in 6700-9100 cm-1
  (45/79 and 63/79). The ILS here is a measured sinc and this Gaussian is an
  addition to it, so railing at zero means the sinc alone already accounts for
  the width -- the same reading as the earlier 0.05-against-0.093 result. A
  dozen windows per file rail at the 4.0 km/s *ceiling* instead, and those are
  worth looking at.
- The residual's wavelength trend is the one already measured and it survives
  the continuum fix elsewhere in the band: about 2x the noise at 4.0-5.3 um,
  4-8x at 2.0-4.0, and 25-36x at 1.5-2.0 um. The windows best for the
  atmosphere remain the worst for the stellar model.

### 4h. The corrected run, and which windows to throw away

Both files refitted with the window normalized. 223 and 222 windows, one failure
each (1876-1906 cm-1, no solar source). Median residual over the noise:

| band | file 5 (X = 1.99) | file 4 (X = 4.73) |
|---|---|---|
| 4.0-5.3 um | 2.03 | 1.78 |
| 2.8-4.0 | 5.28 | 4.04 |
| 2.0-2.8 | 8.26 | 6.85 |
| 1.5-2.0 | 36.02 | 25.27 |
| 1.1-1.5 | **11.95** (was 15.73) | **8.86** (was 10.66) |
| all | 10.29 (was 12.31) | 7.55 (was 8.63) |

88% and 86% of windows are identical to the unnormalized run to 1e-6, which is
the degeneracy holding; the whole gain is at the blue edge, where the continuum
had been railed.

**Do not use a window with `reliable == 0`.** Two in file 5 and eight in file 4
have no pixel whose modelled transmission exceeds 0.15 -- no telluric-free flux
anywhere -- so the continuum and the column are exactly degenerate and the
continuum runs up until `continuum_0` hits +2. Every railed window is one of
these; the rail is the symptom and `reliable` is the cause, so filter on
`reliable`, which is in the record. Widening the bound (the IGRINS driver uses
5.0 for its continuum coefficients) would hide this rather than fix it: e^5 is
148, and no bound recovers information the data do not contain.

**One window also exposed a floor in its saturated cores.** At 5326-5356 cm-1
*zero* of 3166 pixels are masked, yet 1772 have modelled transmission below
0.01 and an observed flux sitting at 0.066 of the window's continuum.
`_saturation_mask` misses it because it tests flux against a **local**
continuum estimate, and where almost everything is saturated that estimate
collapses onto the floor itself, so "above 2% of continuum" passes;
`minimum_continuum_snr = 30` does not catch it either, the window's SNR being
140. Section 4i measures how general this is -- the answer is that it is not,
and that this window is an outlier by a factor of 40.

Other quality flags in the record: a stage failed to converge in 12 windows of
file 5 and 16 of file 4, and 4 and 7 rows carry no covariance at all.

### 4i. The floor in saturated cores is instrumental contrast, not a baseline

A line core the model says is opaque should read zero, and it does not. The
question is whether to subtract that floor or to refuse to quote those pixels.
Measured over the 100 windows that have at least 60 deeply saturated pixels
(modelled transmission below 0.005), as a fraction of each window's own
continuum:

| | |
|---|---|
| observed floor | 0.282% |
| what the model's own sinc puts there | 0.012% |
| unexplained leftover | **+0.158%** (p16 -0.04%, p84 +0.33%) |

So the unapodized sinc filling the core from neighbouring continuum -- a real
effect this model already carries -- accounts for almost none of it.

**It is not an additive baseline.** The floor is 0.286% of the continuum in the
faint half of the band and 0.282% in the bright half: identical, so it scales
with the light rather than sitting at a fixed level, and a detector or
transform zero-point would do the opposite. It is also **negative in 11 of 100
windows**.

**Fitting it as veiling buys nothing.** If a fraction `eps` of the continuum
leaks into every pixel then `observed - model = eps * (continuum - model)`,
which is a one-parameter model that can simply be fitted. Over the 221 windows
with enough deep lines to constrain it:

- `eps` median **4.0e-4**, p16 **-4.1e-4**, p84 **+1.3e-3**
- **negative in 63 of 221 windows** -- a veiling fraction cannot be
- residual 7.0 -> 7.0 sigma, median improvement **1.00x**; exactly one window
  improves by more than 1.2x

A term whose fitted amplitude changes sign between neighbouring windows and
improves nothing is not a correction, it is noise being absorbed. And the bias
it would remove is small where it matters least: under veiling the apparent
line depth is `(1 - eps)` times the true one, so `eps` = 4e-4 biases every
column low by 0.04%, against 3-8% of file-to-file scatter.

**So: mask, do not correct.** Where the floor does matter is the *corrected*
spectrum, which divides by a modelled transmission heading to zero -- and that
is handled by declining to quote those pixels (`reliable`, and the review
page's transmission cut), not by a correction.

**The exception is 5326-5356 cm-1**, whose floor is 6.6% -- forty times the
median -- and whose mask keeps 100% of its opaque pixels. That is not
instrumental contrast, it is something specific to that window, and it is
already flagged `reliable == 0`.

### 4j. photatl is file 5, and its zero level is the one correction that works

Numbers here are in `docs/photatl_identity.json`: every page's gain, offset and
residual, the air-mass pilot, and the paired test page by page.

**photatl's observed column is `ftsspec_901218_5`.** On all 258 pages, `total =
gain * ftsspec_5 + offset` with one gain and one offset per page, on the same
samples to 1.58e-5 cm-1 (photatl's single precision). What the two numbers
leave is at most **0.0625 sigma** of file 5's own noise, median 0.0155; a scale
alone leaves a median 0.218 and up to 4.9. **This is the documented method, not a
discovery**: `telluric_near_ir/README.pdf` (NSO TR 2014-01, §2) says the
1840-8600 cm-1 telluric spectrum came from ratioing 1990/12/18 #4 against #5,
"rescaled to fit the telluric component in the observed 2 airmass spectrum which
was then divided out", and ships both as "the 1.16-5.43 um center-of-disk
unratioed solar spectra". That README was not read before the pilot, which is
why the pilot found it the hard way. What the measurement adds is the per-page
gain and offset, and the demonstration that nothing else was changed.

It was found by the air-mass pilot this section was meant to be. photatl records
no air mass, so 13 pages were fitted at zenith 0 and each well-mixed species'
column divided by file 5's (vertical) column over the same wavenumbers:

| species | pages | X against file 5 | X against file 4 |
|---|---|---|---|
| CO2 | 6 | 1.986 | 2.084 |
| CH4 | 4 | 2.014 | 2.085 |
| N2O | 2 | 2.008 | 2.084 |
| O2 | 3 | 1.992 | 2.048 |
| **H2O** | 13 | **1.991** | 2.671 |

File 5's header says 1.985. The mixed gases say only that the slant-path model
is consistent; **water** says it is the same sky at the same time, because the
water column changes from hour to hour and against file 4 it gives 2.67. The
~4% by which the mixed gases disagree between the two files is the file-4
slant-path inconsistency, not a photatl property.

Consequences:

- **photatl has an air mass: file 5's, 2.17 -> 1.80, mean 1.985.** The Arcturus
  caveat that a column scale is `true column x air mass` does not apply to it.
- **Fitting photatl is refitting file 5.** It is not an independent
  measurement, and the two must not be averaged or quoted as agreeing.
- **The wavelength scale runs +141 m/s red**, measured by telluric lines, which
  carry no solar physics: the fitted telluric velocity is -141 m/s (p16-p84 -172
  to -105, 143 windows) in file 5, the same in file 4, and on every pilot page.
  The sibling project's +260 m/s is from solar lines against ACE, and the
  difference could be solar -- disc-centre against the ACE geometry changes the
  convective line shifts. That is a hypothesis; nothing here tests it.

**The offset is the atlas authors' zero level, and it zeroes saturated cores.**
It is negative on 239 of 258 pages, in continuum units:

| cm-1 | pages | median | most negative |
|---|---|---|---|
| 1848-2200 | 14 | -6.0e-4 | -1.3e-3 |
| 2400-3500 | 44 | -3.4e-4 | -4.2e-3 |
| 4000-5000 | 40 | -3.0e-3 | -4.2e-3 |
| 5000-5600 | 24 | -3.1e-3 | **-1.3e-2** |
| 5600-7000 | 56 | -1.1e-3 | -1.9e-3 |
| 7000-9002 | 80 | -3.3e-4 | -1.5e-3 |

Median data in cores the model calls opaque (effective transmission < 0.005),
as a fraction of the page continuum level:

| page | with photatl's offset | file 5 as observed |
|---|---|---|
| wn4950 | +0.0001 | +0.0033 |
| wn4975 | +0.0000 | +0.0030 |
| wn5300 | +0.0005 | +0.0056 |
| wn5325 | +0.0007 | **+0.0139** |
| wn5350 | +0.0004 | +0.0057 |
| wn5375 | +0.0008 | +0.0076 |

**Whether that correction is right** was tested by fitting 17 pages twice, at
air mass 1.985 on identical windows: photatl as published, and photatl with the
offset undone -- which is file 5's data to 0.023 sigma. The cores are masked in
both, so the comparison is made on pixels a fit uses, which the correction was
not constructed to fix:

- **Outside the 1.9 um band it does nothing.** rms residual ratio 0.997-1.003
  on 13 pages; every well-measured column moves by 0.9% or less.
- **Inside it, it helps in proportion to the offset**: 0.976 on wn5300, 0.983
  on wn5350, 0.985 on wn5375, 0.704 on wn5325 (32 reliable pixels only), where
  CO2 moves 7.5% and H2O 6%.

This refines §4i without contradicting it. Band-wide the floor is small and
correcting it buys nothing, which is what §4i measured and what the 13 pages
repeat. Across 5298-5402 cm-1 (wn5300-wn5375) it is 0.6-1.4% of continuum, and there a zero-level
correction measurably helps -- and photatl's authors have already made one.
(§4i's 6.6% for the 5326-5356 window is against that window's own continuum
reference, not the page's, so the two figures are not in conflict.)

A first comparison, photatl's fit against file 5's windows, was dominated by
something else: photatl's 29 cm-1 pages and file 5's 30 cm-1 grid are
different windows, and on them the residual ratio scattered 0.62-1.18 with no
relation to the offset, while the columns of well-measured species agreed to
1-3%. That spread is the window-choice systematic §4c warned about, measured.

**The product is `photatl_corrected.h5`, built from the file-5 fit for 254
pages and from a fit of photatl itself for the four band pages**
(`scripts/export_photatl_from_ftsspec.py --patch
data/corrected/solar/photatl_band/photatl_band_summary.json`). Every photatl
pixel is a file-5 pixel, so the effective transmission transfers sample for
sample with no interpolation, and `corrected = total / effective_transmission`
in the page's own units. 258 pages, 10.6 MB, 84.9% of pixels reliable. `--check`
divides photatl with its offset undone and compares against the file-5 fit's own
corrected spectrum: worst pixel 0.223 sigma, which is what the per-page
relation's 0.0625 sigma scatter reaches at its extreme; the patched pages
reproduce their own fit's corrected spectrum to 2.8e-16. The unconvolved
transmission is `ftsspec_901218_5_transmission.h5` and is not repeated -- so
for those four pages it is the file-5 one, the one inconsistency left.

**Why only four pages are refitted.** Dividing file 5's transmission into
photatl's floor-corrected data is inconsistent where the floor matters: against
a photatl refit the corrected spectrum differs by a median 10, 84 and 22 sigma
on wn5300, wn5325 and wn5350 (-1.1%, +10%, -2.2% of flux). Elsewhere a refit
moves it by 0.05-1.4 sigma, and none of that is an improvement: part is the
change of windows, and part is the offset being absorbed into continuum and
columns at no cost to the fit -- on wn4950 removing it moves the transmission
by 2.8 sigma while the residual changes by 0.2%. The data cannot choose between
those answers, so a refit would be different, not better. The four pages carry
2,308 of the product's 670,608 reliable pixels (0.34%).

### 4k. niratl: 188 pages at air mass 1.10, and a fit as good as file 5's

niratl (Wallace, Hinkle & Livingston 1993, 8900-13600 cm-1) is independent
data, not a copy of a raw spectrum: smoothed to the 1983 June files' resolution
it misses both by 23-58% of the signal wherever there are telluric lines. What
its documentation says, and what was measured before fitting, is in
`docs/solar_atlases.md`. Inputs:

- **Reader** `read_niratl_page`: a line of continuum levels, then 4096 rows;
  -1.0 fill only in the `solar` and `atmospheric` columns; the grid
  reconstructed (step 0.0073265 cm-1 on every page).
- **ILS** an unapodized sinc, **FWHM 0.01859 cm-1** constant (MOPD 32.46 cm),
  from the 61 pages with a sharp interferogram cut. `docs/niratl_ils.json`,
  whose `sharp_cut` block is the number used; its headline `all` block mixes in
  the soft-cut failures and reads 0.0191.
- **Profile** `kitt_peak_19830626_era5_afgl.csv`: ERA5 at 1983-06-26 19:00 UT
  (4.69 mm of water, 7.24 K/km over the first 3 km), AFGL midlatitude-summer
  trace gases, a new `"1983"` epoch (CO2 from NOAA; CH4 and N2O extrapolated,
  stated in `make_site_profile.py`). The observation's date is "1983 June" and
  its time is not recorded; noon is where an air mass of 1.1 puts it.
- **Source** a new Payne Zero band `nir`, 725-1110 nm, 69 minutes to generate.

**Air mass 1.10, from the O2 A-band.** O2's mixing ratio is known, so fitted
at zenith 0 its column scale is the air mass. The eight A-band pages with
100-3000 clean O2 pixels give **1.080-1.121, median 1.10**; the README's "1.0"
is rounded. The 1.07 um a1-Delta band was expected to give a second, independent
value and does not: its lines are 2-3% deep, and under the solar residual its
O2 scale wanders from 0.93 to 2.05, as does the A-band's on its weak outer pages.

**The fit.** 188 of 188 pages ran; 164 fitted and 24 had every species below
the cut. The species present are few -- H2O on 148 pages, O2 on 19, CO2 on 2,
CH4 on 1 -- against photatl's 19. The residual is a median **76 sigma**, and
that is the noise, not the fit:

| | noise | rms residual | telluric lines | solar lines | clean continuum | reliable |
|---|---|---|---|---|---|---|
| niratl, 164 pages | **0.018%** | 1.20% | 1.04% | 3.82% | 0.46% | 97.5% |
| ftsspec_5, 222 windows | 0.097% | 1.03% | 0.84% | 4.69% | 0.48% | 84.0% |

The same ~1% floor, set by the solar model (the residual follows stellar depth
at r = -0.45 to -0.73 and telluric depth at r ~ 0), against five times less
noise. The noise estimate is sound: the power spectrum is flat from the MOPD to
Nyquist, so the pages were not resampled, though 4096 = 2^12 per page invited
the suspicion.

**The species cut stays at 1e-2, measured rather than assumed.** 48 pages had a
species below the cut at slant optical depth above 1e-3 -- up to 21 sigma at the
90th percentile, 112 at worst, at niratl's noise. Refitted at 2e-3 (about 10
sigma): the residual ratio is a median 1.000 (best 0.979), and **21 of the 42
species it added rail at a bound** (H2O 12 of 20, O2 6 of 9). It is the §4g
result again: the detection limit is the solar model's, not the noise's, so five
times less noise does not move it. The 24 pages the cut leaves unfitted carry
unfitted absorption up to optical depth 9.9e-3, and are exported as **not
corrected** rather than as clean.

**Products.** `data/corrected/solar/niratl/niratl.h5` and `niratl_summary.json`
are the record. `niratl_transmission.h5` (7.7 MB, 164 windows, `--check`
exactly 0) is the unconvolved transmission on the model grid;
`niratl_corrected.h5` (8.2 MB, 188 pages, 85.1% of pixels reliable, each page
reproducing its own fit to 3.0e-16) is the corrected spectrum, written by the
new `scripts/export_atlas_corrected.py` in `photatl_corrected.h5`'s layout.

**A provenance bug this run found.** The batch driver recorded
`physics["fwhm_cm1"]` from a constant, not from `--fwhm-cm1`, so niratl's first
record named photatl's 0.01753 and `export_transmission_hdf5.py` rebuilt every
window on the wrong grid (9130 points against at most 8610). Every earlier run
used the default, which is why nothing showed. The driver now records what it
ran, and niratl's record and summary were rewritten from the saved fits.

## One decision left before writing code

**How much of file 4 to salvage.** The scan-average bias is analytic for a
linear `X(t)`, so a three-node average inside `transmission()` would fix it
exactly -- but that is a change to the forward model serving one file and the
830626 pair. Fitting an effective air mass instead costs nothing and measures
how large the effect really is. Do that first.

## Provenance and acknowledgement

The data is not fetched by this repository. It is read from
`/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/nso/`,
retrieved 2026-08-30 from <https://nso.edu/data/historical-archive/>; that
project's `data/atlases/nso/PROVENANCE.md` is the authority on the retrieval
route and the formats.

Both NSO READMEs impose one condition on any publication using these data:

> NSO/Kitt Peak FTS data used here were produced by NSF/NOAO.

The citable primary source for the instrument and the atlas series is Wallace,
L., Livingston, W., Hinkle, K., & Bernath, P. 1996, ApJS, 106, 165.
