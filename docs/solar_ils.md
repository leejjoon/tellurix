# The instrument line shape of the NSO solar atlases, measured from the data

`photatl` documents no resolution at all: the only instrumental number in the
refereed description of the series is **R ≈ 300,000 quoted once for four
atlases** (Wallace, Livingston, Hinkle & Bernath 1996, ApJS 106, 165), with no
MOPD and no apodization. The raw `ftsspec_*` spectra do carry a `resolution=`
header, and it disagrees with their own data.

Both are recoverable, because an FTS spectrum is the Fourier transform of an
interferogram truncated at the maximum optical path difference: transforming a
page returns that interferogram, whose envelope is flat out to the MOPD and
then cuts.

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/measure_solar_ils.py --format photatl
UV_CACHE_DIR=.uv-cache uv run python scripts/measure_solar_ils.py --format ftsspec
```

Machine-readable results are in `docs/photatl_ils.json` (per page) and
`docs/solar_fts_ils.json` (per 100 cm⁻¹ window). The method lives in
`tellurix.ils` and is shared with `scripts/measure_atlas_ils.py`; moving it
there reproduces all 598 committed Arcturus page-epoch measurements
bit-for-bit.

## Method

Each spectrum is mean-subtracted, Hann-windowed and transformed with a real
FFT. The MOPD is located at the steepest sustained drop of the smoothed log
envelope rather than at the last sample above a threshold, because the stored
precision leaves a floor beyond the true cut that a threshold detector locks
onto.

Two things specific to these atlases:

**Reconstruct the grid first.** Both formats store a uniform grid at reduced
precision — `photatl` in single precision, the ftsspec files in four decimal
places — and an FFT assumes even sampling, so the stored jitter smears the cut.
Measuring on the stored `photatl` grid gives MOPD 34.93 cm, FWHM 0.01727 cm⁻¹
and a page-to-page spread of 1.030; measuring on the least-squares grid gives
34.41 cm, 0.01753 cm⁻¹ and a spread of **1.010**. The tightening is the
evidence that the reconstruction is the right move: removing grid noise cannot
make a real spread smaller.

**Refuse an opaque window.** The interferogram of a window that passes no light
is noise, and it returns a confident-looking MOPD that means nothing — the
first pass on `ftsspec` produced 2.57 cm from a dead window this way. Windows
below 30× continuum-to-noise are skipped (`tellurix.nso.window_continuum_snr`).
That costs 1 of 258 `photatl` pages (`wn2100`, at 23.9σ) and 15 of 400 ftsspec
windows.

**A flat envelope still returns a MOPD.** The steepest drop of a flat envelope
is somewhere, so noise yields a confident-looking number — white noise here
gives 7.21 cm. What gives it away is that the envelope does not *step*: the
in-band level and the tail level come out at -0.551 and -0.556, a difference of
0.005 decades. Across 1,240 real measurements spanning `photatl`, the ftsspec
spectra and the Arcturus atlas the smallest step is **1.14 decades**, so
`describe_truncation` requires 1.0 and reports `envelope_step_decades`.
Applying it rejects nothing real: the counts stay at 257 of 258 pages and 385
of 400 windows.

The original `mopd_measurable` test does not cover this. It asks whether the
MOPD is shorter than the sampling can express, which is a different failure,
and it passes white noise.

## `photatl`: a constant-MOPD atlas

Measured over 257 pages spanning 1862–8988 cm⁻¹:

| quantity | value |
|---|---|
| MOPD | **34.414 cm**, MAD 0.103 |
| p16–p84 | 34.207–34.552 cm, spread **1.010** |
| corr(MOPD, ν) | **−0.077** |
| sinc FWHM | **0.017532 cm⁻¹** |
| samples per FWHM | 1.85 |
| transition / path-difference resolution | 20.0 |
| drop across the cut | 1.40 decades |

**This is the opposite design from the Arcturus atlas**, which holds the
resolving power constant so its MOPD must track 1/ν — corr −0.93, spread 2.87.
Here the path difference is held constant instead, so the resolution element is
constant in wavenumber and the resolving power rises with it:

| ν (cm⁻¹) | R |
|---|---|
| 2000 | 114,075 |
| 5000 | 285,188 |
| 9000 | 513,338 |

So the single "R ≈ 300,000" in the literature is a **mid-band value, not a
constant**: it is right near 5,100 cm⁻¹ and wrong by a factor of 2.5 at either
end of the atlas.

**What this retires.** The sibling project's `data/atlases/nso/PROVENANCE.md`
concludes that the shape R ∝ σ is solid but "the absolute scale is not
obtainable this way", on the grounds that an FTS spectrum is zero-filled by a
factor its sampling does not reveal, and `dss/calib/lsf.py` therefore gives
`photatl` the widest LSF prior of any atlas. The premise is true and the
conclusion does not follow: appending zeros to an interferogram does not move
where it was truncated. The scale is measured above.

## `ftsspec_*`: two configurations, and headers that misreport

Per file, in 100 cm⁻¹ windows:

| file | windows | MOPD (cm) | spread | FWHM (cm⁻¹) | ν range (cm⁻¹) |
|---|---|---|---|---|---|
| `ftsspec_901218_4` | 70 | 34.401 ± 0.065 | 1.007 | 0.01754 | 1880–9080 |
| `ftsspec_901218_5` | 71 | 34.339 ± 0.093 | 1.007 | 0.01757 | 1880–9180 |
| `ftsspec_830626_2` | 122 | 14.480 ± 0.030 | 1.006 | 0.04167 | 8566–20666 |
| `ftsspec_830626_3` | 122 | 14.710 ± 0.000 | 1.001 | 0.04102 | 8566–20666 |

**Do not pool them.** The combined figure is MOPD 14.71 cm with a spread of
2.377 and corr(MOPD, ν) = −0.771, all three of which are artefacts of mixing a
34 cm configuration covering the infrared with a 14 cm one covering the
visible. The script prints every group for this reason.

**The 1990 pair matches `photatl` to 0.2%** — 34.401 and 34.339 against 34.414
— on top of a sampling that agrees to the eighth decimal (0.00947709 against
0.00947710). Wallace et al. 1996 says `photatl` is "based on three spectra
obtained by Livingston in 1990 December"; these are two spectra from 1990
December, at the same path difference and the same sampling.

**The stated resolution is not the instrument profile, and the error is not
even consistent:**

| file | header says | interferogram says | factor |
|---|---|---|---|
| `ftsspec_901218_4` | 0.053 cm⁻¹ | 0.01754 | **3.02** |
| `ftsspec_901218_5` | 0.053 | 0.01757 | **3.02** |
| `ftsspec_830626_2` | 0.051 | 0.04167 | 1.22 |
| `ftsspec_830626_3` | 0.041 | 0.04102 | 1.00 |

One file's header is exact, one is 22% off and two are out by almost exactly a
factor of three. Whatever convention produced them, it cannot be inverted
without knowing which file it was applied to — so measure the cut.
`tellurix.nso.FTSSpectrum` records the value as `stated_resolution_cm1` and
nothing consumes it.

## Apodization

Unapodized, in every case. The transition from in-band to floor spans 0.690 cm
on `photatl` and 0.710 cm on `ftsspec_901218_5`, against the ~L/3 ≈ 11.5 cm a
Norton–Beer taper would give. As a multiple of the path-difference resolution
`photatl` reads 20.0, identical to the Arcturus atlas's 20.0.

`BoxcarFTSInstrumentProfile` is therefore the right instrument model for these
atlases, as it is for Arcturus, and it must be applied by FFT — a truncated
kernel is invalid for a sinc at any width.

## What a fit should use

One atlas-wide constant, **FWHM = 0.01753 cm⁻¹** (`tellurix.nso.MEASURED_FWHM_CM1`),
equivalently MOPD 34.41 cm, for `photatl` and for the 1990 `ftsspec` pair. Not
a per-page measurement: the page-to-page spread of 1.010 is within the FFT bin,
so a per-page value would be fitting noise. That also removes the free
parameter that rails `lsf_sigma_kms` on 13.4% of Arcturus pages, where the
per-page MOPD scatter was real and had to be absorbed.

## Acknowledgement

> NSO/Kitt Peak FTS data used here were produced by NSF/NOAO.
