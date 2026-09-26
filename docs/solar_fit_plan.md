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
| MOPD | 34.24-34.64 cm | 34.93 cm (median of 258 pages) |
| sinc FWHM | 0.01742-0.01762 cm⁻¹ | 0.01727 cm⁻¹ |
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

The ftsspec header says `resolution= 0.053cm-1`, which would be a 9.43 cm MOPD
read as `1/(2L)` or 18.87 cm read as `1/L`. The interferogram cut is at 34.4 cm.
An FTS spectrum is the transform of a truncated interferogram, so the cut is a
measurement of the truncation and the header is a label. Trust the cut.
*(Phase 2)*

### `photatl` is a constant-MOPD atlas, where Arcturus is constant-R

Measured over all 258 pages with the Arcturus tool: *(Phase 2)*

| | `photatl` | Arcturus, for scale |
|---|---|---|
| MOPD p16-p84 | 34.156-35.180 cm, spread **1.030** | 8.82-25.32 cm, spread 2.87 |
| corr(MOPD, nu) | **-0.084** | -0.93 (the constant-R law) |
| sinc FWHM | **0.01727 cm⁻¹, constant** | tracks 1/nu |
| drop across the cut | 1.40 decades | 2.34 |

So R runs **113,600 at 2000 cm⁻¹, 291,800 at 5000, 502,255 at 9000**, and the
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

| page | residual | as a fraction of the 0.01727 cm-1 FWHM |
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

For the disc-centre IR atlases, where FWHM = 0.01727 cm⁻¹ is measured, this is
`scripts/generate_payne_zero_solar.py --band {blue,mid,red}`:

| band | nm | cm⁻¹ | R at nu_max | r-grid | step | points |
|---|---|---|---|---|---|---|
| `blue` | 1100-1510 | 6623-9091 | 526,400 | 2,105,596 | 0.1424 km/s | 667 k |
| `mid` | 1490-2610 | 3831-6711 | 388,600 | 1,554,467 | 0.1929 km/s | 871 k |
| `red` | 2590-5330 | 1876-3861 | 223,600 | 894,269 | 0.3352 km/s | 645 k |

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

**ILS.** Run the measurement per window, confirm 34.4 cm, then **pin one
MOPD** rather than fitting per window. This is the `--sinc-resolving-power`
lesson from Arcturus, on firmer ground here because the MOPD really is
constant.

**Grid spacing is per window, not shared.** This is the one architectural
difference from Arcturus. There `velocity_step_kms` is a single number for the
whole atlas because the resolving power is constant. Here the *resolution
element* is constant, so set `resolving_power = nu_mid / 0.01727` per window
and let `constant_velocity_grid` follow. A single atlas-wide step would be set
by 9050 cm⁻¹ and oversample 1880 cm⁻¹ by 4.5x.

**Site profile.** `make_site_profile.py` already defaults to Kitt Peak
(2.096 km, 786 hPa), the same mountain as the Arcturus atlas. Add a `"1990"`
entry to `EPOCH_DRY_VMR` (CO2 ~354 ppm against the 1994 column's 357) and write
`data/profiles/kitt_peak_1990_12_18.csv`. Then a real profile:
`era5_site_profile.py` currently takes `--spec` IGRINS FITS frames, so add an
entry point taking position and time directly -- 31.9583 N, 111.5967 W,
1990-12-18, 17:19-18:39 UT.

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

Before any of it: `scripts/validate_fts_fit.py` on the
`validate_arcturus_fit.py` L0-L2 synthetic pattern, and
`scripts/rebuild_fts_window.py --check` reconstructing a window from the record
alone. As with `rebuild_arcturus_page.py`, it must share no code with the
driver -- an independent reconstruction is the evidence; calling one function
twice is not.

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
sigma and correlation into the union.

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
