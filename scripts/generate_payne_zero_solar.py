#!/usr/bin/env python
"""Synthesize the fixed solar stellar spectrum used as the telluric fit's source.

Payne Zero needs Python >= 3.11 and PyTorch, while this package is pinned to
3.10 through ``exojax==2.5.0``, so this script does not run in the project
environment. It runs in Payne Zero's own environment and writes an npz that
:class:`tellurix.StellarSpectrum` reads back:

    cd /home/jjlee/work/payne-zero
    export PAYNE_ZERO_DATA_ROOT=$PWD/source_data_files
    export PAYNE_ZERO_SYNTHESIS_CACHE_DIR=$PWD/.cache/payne-zero/synthesis
    export NUMBA_CACHE_DIR=$PWD/.cache/payne-zero/numba-atmosphere
    .venv/bin/python /home/jjlee/work/lblrtm/scripts/generate_payne_zero_solar.py --band red
    .venv/bin/python /home/jjlee/work/lblrtm/scripts/generate_payne_zero_solar.py --band mid
    .venv/bin/python /home/jjlee/work/lblrtm/scripts/generate_payne_zero_solar.py --band blue
    .venv/bin/python /home/jjlee/work/lblrtm/scripts/generate_payne_zero_solar.py --band red_edge
    .venv/bin/python /home/jjlee/work/lblrtm/scripts/generate_payne_zero_solar.py --band nir --fwhm-cm1 0.01859

The spectrum is a *fixed* input: the stellar labels are held at literature
values and are not fitted. See docs/solar_fit_plan.md for where it is used.

Two properties of the output that decide how it may be used:

**Payne Zero returns Eddington flux and has no mu option.** There is no
``specific_intensity`` or ``mu_angle`` in its API. That is the correct quantity
for the disc-integrated atlases (``fluxatl``, ``wallace2011_flux``, both IAG
settings) and a stated mismatch for the disc-centre ones (``photatl``,
``niratl``, the four ``ftsspec`` spectra), where what was observed is I(mu=1).

**Do not chase ``atmosphere_converged: false``.** The synthesis rests on the
learned initializer atmosphere and reports that flag, as the Arcturus and A0V
sources do. It was measured on Arcturus and is a red herring: solving the
physical atmosphere and re-synthesizing changed the spectrum by an rms 0.0005
at 1.04, 2.00 and 3.06 um, 50-100x below the residuals it would have to
explain, with the deep-line counts unchanged and refits agreeing to 0.03 sigma
(docs/arcturus_fit.md, "The stellar limit"). A converged structure adds no
lines; the stellar error is in the line list. The check was made at 4286 K
rather than at solar temperature, so that is an expectation here and not a
measurement.

**No broadening is applied here**, by design -- ``prepare_stellar_source`` does
it, and the right value depends on the atlas rather than on the Sun. Disc
centre takes ``vsini_kms = 0``, because there solar rotation is transverse to
the line of sight; disc-integrated takes ~1.9 km/s. Passing the Arcturus
default of 2.0 to a disc-centre fit would broaden a spectrum that carries no
rotation at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


_C_KMS = 299792.458

# The Sun. All inside Payne Zero's stated five-label support of 4,000-10,500 K,
# 0.7-5.3 in logg, -2.5-0.5 in [M/H], -0.1-0.5 in [alpha/M], and 0.5-4.0 km/s
# in microturbulence.
SOLAR_LABELS = {
    "effective_temperature": 5772.0,
    "log_surface_gravity": 4.438,
    "metallicity": 0.0,
    "alpha_enhancement": 0.0,
    "microturbulence_km_s": 1.0,
}
SOLAR_LABEL_SOURCE = "IAU 2015 Resolution B3 nominal solar values; [M/H] and [alpha/M] zero by definition"

# The sinc FWHM measured from the interferogram cut of photatl (257 of 258
# pages, 0.01753) and of both 1990 ftsspec spectra (0.01754, 0.01757). It is
# constant across those atlases -- they hold the path difference fixed, not the
# resolving power -- so one number sizes the whole source. Measure it per atlas
# before reusing this default: the ftsspec headers state a resolution that is
# 3.02x their own cut for the 1990 pair and 1.00-1.22x for the 1983 pair, so
# the documentation is not a substitute and is not even self-consistent.
# docs/solar_ils.md. (The sources synthesized before this was measured used
# 0.01727, which asks for a *finer* grid and so remains valid.)
DEFAULT_FWHM_CM1 = 0.01753

# Bands exist because r_grid is set by the *bluest* wavenumber a band contains,
# and one grid fine enough for 9050 cm-1 would oversample 1880 cm-1 by 4.5x.
# They overlap by ~10 nm so that resampling onto a model grid -- which reaches
# past its fit window by the wing margin plus the LSF and Doppler reach -- never
# extrapolates off an edge.
BANDS = {
    "blue": (1100.0, 1510.0),   # 6623-9091 cm-1
    "mid": (1490.0, 2610.0),    # 3831-6711 cm-1
    # 5355 rather than 5330. The first fitted window, 1876-1906 cm-1, reaches
    # 5330.5 nm, but what has to be covered is the model *grid*, which carries
    # the wing margin: 1870.998 cm-1 = 5344.8 nm. Sizing this from the window
    # instead of the grid is why the first attempt still failed.
    "red": (2590.0, 5355.0),    # 1867-3861 cm-1
    # Only for photatl's first page, wn1850 (1848-1877 cm-1), whose grid reaches
    # 1843 cm-1 = 5426 nm. A separate band rather than a wider "red" because
    # widening it would change the source of every ftsspec window already
    # fitted; solar_source_for takes the first band in sorted order that
    # covers a window, and "red" sorts first wherever both do.
    "red_edge": (5300.0, 5445.0),   # 1837-1887 cm-1
    # niratl, 8900-13600 cm-1 plus its grid margin. Generate it with niratl's
    # own sinc, --fwhm-cm1 0.01859 (docs/niratl_ils.json): 6% wider than
    # photatl's, and this band's sampling is set by its 725 nm end. It sorts
    # after "mid" and before "red", so where it overlaps "blue" (1100-1110 nm)
    # "blue" still wins and no ftsspec window changes source.
    "nir": (725.0, 1110.0),     # 9009-13793 cm-1
}


def required_r_grid(
    wavelength_start_nm: float, fwhm_cm1: float, samples_per_resolution: float
) -> int:
    """Intrinsic sampling density the model grid will demand of this source.

    ``resample_stellar_source`` refuses a source coarser than the grid it is
    being put on, as a resolution lie. The grid is built at
    ``samples_per_resolution`` samples across the instrument's resolution
    element, so the finest step any window in this band will ask for is set by
    the band's largest wavenumber -- its shortest wavelength.
    """

    if not (wavelength_start_nm > 0.0 and fwhm_cm1 > 0.0 and samples_per_resolution > 0.0):
        raise ValueError("wavelength, FWHM and sampling must all be positive")
    wavenumber_max_cm1 = 1.0e7 / wavelength_start_nm
    return int(math.ceil(samples_per_resolution * wavenumber_max_cm1 / fwhm_cm1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--band", choices=sorted(BANDS), default=None,
                        help="preset wavelength range; see BANDS. Overridden by "
                             "--wavelength-start-nm/--wavelength-end-nm.")
    parser.add_argument("--wavelength-start-nm", type=float, default=None)
    parser.add_argument("--wavelength-end-nm", type=float, default=None)
    parser.add_argument("--fwhm-cm1", type=float, default=DEFAULT_FWHM_CM1,
                        help="measured instrument sinc FWHM; sizes the intrinsic grid")
    parser.add_argument("--samples-per-resolution", type=float, default=4.0,
                        help="must match the fit driver's grid, or the source is too coarse")
    parser.add_argument("--r-grid", type=int, default=None,
                        help="override the density derived from --fwhm-cm1")
    # "auto" prefers CUDA. Synthesis is parallel over wavelength and these
    # bands run to millions of points, so the GPU is the production path.
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.wavelength_start_nm is None or args.wavelength_end_nm is None:
        if args.band is None:
            parser.error("give --band, or both --wavelength-start-nm and --wavelength-end-nm")
        start_nm, end_nm = BANDS[args.band]
        args.wavelength_start_nm = args.wavelength_start_nm or start_nm
        args.wavelength_end_nm = args.wavelength_end_nm or end_nm
    if not args.wavelength_start_nm < args.wavelength_end_nm:
        parser.error("the wavelength range must be increasing")

    needed_r_grid = required_r_grid(
        args.wavelength_start_nm, args.fwhm_cm1, args.samples_per_resolution
    )
    r_grid = args.r_grid or needed_r_grid
    if r_grid < needed_r_grid:
        parser.error(
            f"--r-grid {r_grid} is coarser than the {needed_r_grid} this band needs at "
            f"{args.wavelength_start_nm} nm; resample_stellar_source would refuse it"
        )
    if args.output is None:
        label = args.band or f"{args.wavelength_start_nm:.0f}_{args.wavelength_end_nm:.0f}nm"
        args.output = Path(f"/home/jjlee/work/lblrtm/data/stellar/solar_payne_zero_{label}.npz")

    from payne_zero_synthesis import synthesize_from_labels

    spectrum = synthesize_from_labels(
        **SOLAR_LABELS,
        wavelength_start_nm=args.wavelength_start_nm,
        wavelength_end_nm=args.wavelength_end_nm,
        r_grid=r_grid,
        device=args.device,
    )

    wavelength_nm = np.asarray(spectrum.wavelength_nm, dtype=float)
    normalized = np.asarray(spectrum.normalized_flux, dtype=float)
    if np.any(~np.isfinite(normalized)) or np.any(normalized <= 0.0):
        raise SystemExit("synthesis returned non-positive or non-finite normalized flux")

    # tellurix works in ascending vacuum wavenumber.
    wavenumber_cm1 = 1.0e7 / wavelength_nm
    order = np.argsort(wavenumber_cm1)

    # Check what was delivered rather than what was asked for: r_grid is a
    # request, and finding out that the source is too coarse when a fit refuses
    # it costs the whole synthesis again.
    velocity_step_kms = float(np.median(np.diff(np.log(wavenumber_cm1[order]))) * _C_KMS)
    finest_needed_kms = _C_KMS * args.fwhm_cm1 / (
        args.samples_per_resolution * 1.0e7 / args.wavelength_start_nm
    )
    if velocity_step_kms > finest_needed_kms:
        raise SystemExit(
            f"delivered {velocity_step_kms:.4f} km/s but this band needs "
            f"{finest_needed_kms:.4f} km/s at {args.wavelength_start_nm} nm"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        wavenumber_cm1=wavenumber_cm1[order],
        flux=normalized[order],
        # Every array goes in the same ascending-wavenumber order. Saving
        # flux_total and flux_continuum unordered leaves them reversed against
        # wavenumber_cm1, which is what the committed Arcturus and A0V npz
        # files have.
        wavelength_vacuum_nm=wavelength_nm[order],
        flux_total=np.asarray(spectrum.flux_total, dtype=float)[order],
        flux_continuum=np.asarray(spectrum.flux_continuum, dtype=float)[order],
    )

    metadata = {
        "star": "Sun",
        "labels": SOLAR_LABELS,
        "label_source": SOLAR_LABEL_SOURCE,
        # Eddington flux, not I(mu=1): correct for the disc-integrated atlases,
        # a stated mismatch for the disc-centre ones. See the module docstring.
        "emergent_quantity": "eddington_flux",
        "broadening_applied": None,
        "band": args.band,
        "wavelength_nm": [args.wavelength_start_nm, args.wavelength_end_nm],
        "instrument_fwhm_cm1": args.fwhm_cm1,
        "samples_per_resolution": args.samples_per_resolution,
        "r_grid": r_grid,
        "r_grid_required": needed_r_grid,
        "points": int(wavelength_nm.size),
        "velocity_step_kms": velocity_step_kms,
        "velocity_step_required_kms": finest_needed_kms,
        "normalized_flux_range": [float(normalized.min()), float(normalized.max())],
        "initializer_family": getattr(spectrum, "initializer_family", None),
        # False means the spectrum rests on the learned initializer atmosphere
        # rather than a converged solve. Recorded because it bounds how far the
        # stellar model can be trusted, and it is not visible in the arrays.
        "atmosphere_converged": bool(getattr(spectrum, "atmosphere_converged", False)),
        "atmosphere_closure_required": bool(
            getattr(spectrum, "atmosphere_closure_required", False)
        ),
        "seconds": float(getattr(spectrum, "seconds", float("nan"))),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
