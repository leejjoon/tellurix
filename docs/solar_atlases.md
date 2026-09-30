# The solar atlases, and what is being asked of this package

A survey, not a plan. Everything here was measured from files on disk or quoted
from the sibling project's own documentation; where the two disagree the
disagreement is stated rather than resolved.

**The details are not duplicated here.** The authority is the sibling project,
`differentiable_stellar_spectroscopy`, whose documents are listed at the bottom.
This file exists so that a session working in *this* repository knows the solar
work is being asked for, knows the traps before touching the data, and knows
where to read.

## What is being asked

From `docs/solar_data_status.md` in the sibling project, written 2026-09-26:

> **There is no solar equivalent.** Running `jax-telluric` on a solar atlas is
> the single highest-value piece of data work available here: the tooling
> exists, the problem is already solved once, and it would turn masking into
> forward-modelling.

The reason is this package's own result, quoted back at it:

> A column at atlas resolution can only *mask*. It cannot be multiplied inside
> our convolution, which is what the Arcturus work established as the correct
> treatment -- never divide by a convolved quantity.

So the deliverable is the solar analogue of
`data/corrected/arcturus_transmission.h5`: unconvolved transmission on a fine
grid, from a fit to a solar atlas.

## What the data is

Under `differentiable_stellar_spectroscopy/data/atlases/nso/` (296 MB,
retrieved 2026-08-30 from <https://nso.edu/data/historical-archive/>):

| directory | what | coverage |
|---|---|---|
| `photatl/` | Livingston & Wallace 1991, NSO TR 91-001, McMath-Pierce FTS, solar **disc centre** | 1848-9002 cm-1 = 1.11-5.41 um, 258 pages |
| `niratl/` | Wallace, Hinkle & Livingston 1993, NSO TR 93-001 | 0.735-1.124 um |
| `telluric_near_ir/` | Wallace telluric near-IR: four raw FTS solar spectra | see below |
| `telluric_mid_ir/` | Wallace telluric mid-IR: 10 spectra, 11 per-molecule line lists | |
| `fluxatl/`, `spot1atl/`, `wallace2011_flux/` | solar flux, sunspot | 0.296-1.30 um |

`photatl` is the one that matters: it spans 87% of the Arcturus page-epochs this
package has already fitted (520 of 598), on the same mountain and the same
instrument lineage. Its four columns are wavenumber, **solar component,
atmospheric component, total**, 25 cm-1 per page plus 2 cm-1 of overlap, 3061
points each.

### Traps, measured

- **`photatl` is `ftsspec_901218_5`**, rescaled page by page: `total = gain *
  ftsspec_5 + offset` on every page, on the same samples, to 0.06 sigma of that
  file's noise. It is not independent data, it inherits file 5's air mass
  (mean 1.985), and its offset is a zero-level correction that puts saturated
  cores at zero. `solar_fit_plan.md` §4j has the measurement; the traps below
  that call photatl's air mass unknown or its ILS unmeasured predate it.

- **The atmospheric column is at atlas resolution.** It is a mask, not an
  operator. This is the whole reason for the request.
- **Pixels are interpolated where the sky is opaque.** The sibling project
  reports 23.5% of `photatl` pixels are linear interpolation rather than
  measurement. A crude second-difference test here reproduces only 0.3%, so the
  criteria differ and theirs should be trusted -- but the mechanism is
  confirmed: of the 10.2% of points where the atmospheric column is below 0.1,
  **54% have a locally linear solar component**. Filled in under opaque sky.
- **Two holes**, verified exactly: 2501-2856 nm and 4170-4541 nm.
- **The ILS is the weakest input in the whole programme.** R = 300,000 is quoted
  once for all four NSO atlases, with **no MOPD and no apodization stated**.
  After what the Arcturus work found about the FTS sinc and the 1/nu MOPD law,
  this is the input most likely to be wrong and least likely to announce it.
- **The wavelength scale is offset by +260 m/s** against two independent ACE
  atlases that agree with each other to -1.9 +/- 4.2 m/s. Unresolved there.
- **The NSO telluric atlases have no loader** -- about 190 MB the sibling project
  lists as unreachable from code.

### One thing worth a second look

`telluric_near_ir/` carries raw FTS solar spectra with **air mass in the
header**, which is the quantity the Arcturus run structurally cannot constrain
(it fits at `zenith_angle_deg = 0`, so air mass lives inside the column scale --
see `arcturus_fit.md`):

| file | date | air mass, start -> stop | range |
|---|---|---|---|
| `ftsspec_901218_4` | 1990-12-18 | 5.88 -> 3.58 | 1.5-5 um |
| `ftsspec_901218_5` | 1990-12-18 | 2.17 -> 1.80 | 1.5-5 um |
| `ftsspec_830626_2` | 1983-06-26 | 6.66 -> 4.08 | 0.5-1 um |

The two 1990 files are the same sky, the same day and the same instrument at air
mass ~4.7 against ~2.0. That is a cleaner slant-path test than the IGRINS
airmass ladder could give, where per-night systematics left the honest bound at
~3% night-to-night scatter. Not mentioned in the sibling project's status
document, so treat it as an observation from here rather than an agreed plan.

## A finding of theirs that bears on a problem of ours

The sibling project's W4.1 work reports that **telluric lines can measure an
instrument profile where stellar lines provably cannot** -- 2 dlog L of about
1300 against 0.57 nats over a 5.6x range in resolving power. This package has an
open ILS problem: `lsf_sigma_kms` is at a bound on 13.4% of converged Arcturus
pages, and the only remedy tried here is fixing the sinc globally
(`--sinc-resolving-power`, worth 0.2% of residual for 43% of the at-bound
cases). Their result suggests the telluric lines themselves are the lever.

## Housekeeping

Their `solar_data_status.md` cites this package's output as
`data/arcturus_newly_reduced/arcturus_transmission_full.h5`. **That path does not
exist in the sibling project**, and the name is stale here too: the file is
`data/corrected/arcturus_transmission.h5`, regenerated from the promoted
full-coverage record. Anything they hold under the old name predates that run.

## Where to read, in their repository

Paths are relative to `/home/jjlee/work/differentiable_stellar_spectroscopy`.

| file | what it is |
|---|---|
| `docs/solar_data_status.md` | the map: coverage, modes, what is ready and what is missing |
| `plans/phase4_calibration_and_linelist.md` | section 3 is the authority on every dataset and its traps; section 5 is the acceptance protocol |
| `data/atlases/nso/PROVENANCE.md` | retrieval route, the required NSO acknowledgement, per-directory contents |
| `data/atlases/arcturus/PROVENANCE.md` | the Arcturus atlas this package already fits |
| `docs/w41_status.md`, `docs/w41_open_questions.md` | what the Arcturus path established and left open |
| `CLAUDE.md` | their method rules |

## Acknowledgement

The NSO READMEs impose one condition on any publication using these data:

> NSO/Kitt Peak FTS data used here were produced by NSF/NOAO.
