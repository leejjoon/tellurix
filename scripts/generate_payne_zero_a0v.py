#!/usr/bin/env python
"""Synthesize the A0V stellar source for the IGRINS telluric standard fits.

Like ``generate_payne_zero_arcturus.py`` this does **not** run in the project
environment: Payne Zero needs Python >= 3.11 and PyTorch while this package is
pinned to 3.10 by ``exojax==2.5.0``. It runs in Payne Zero's own environment and
writes an npz that :class:`jax_telluric.StellarSpectrum` reads back:

    cd /home/jjlee/work/payne-zero
    export PAYNE_ZERO_DATA_ROOT=$PWD/source_data_files
    export PAYNE_ZERO_SYNTHESIS_CACHE_DIR=$PWD/.cache/payne-zero/synthesis
    export NUMBA_CACHE_DIR=$PWD/.cache/payne-zero/numba-atmosphere
    .venv/bin/python /home/jjlee/work/lblrtm/scripts/generate_payne_zero_a0v.py

Read the caveat before trusting the result. Across H and K an A0V has no metal
lines worth the name -- the hydrogen Brackett and Pfund series is the whole of
its spectrum there. Those lines are shaped by Stark broadening, and Payne Zero
is an emulator spanning 4,000-10,500 K whose atmosphere here does not converge
(the metadata records ``atmosphere_converged``). So this source is a hypothesis
to be tested against the flat-source fit, not a better starting point assumed
in advance: ``scripts/fit_igrins_standard.py --stellar flat`` masks the series
and depends on none of this, and the difference between the two fits on the
masked orders is the measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


# A0V. Teff and logg are the standard main-sequence values for the type rather
# than any one star's measurement; the standards used here are a heterogeneous
# set of catalogued A0Vs, so a type-average is the honest choice. Vega itself
# would be the wrong anchor -- it is a pole-on rapid rotator and metal weak.
# All inside Payne Zero's stated support of 4,000-10,500 K, 0.7-5.3 in logg,
# -2.5-0.5 in [M/H], -0.1-0.5 in [alpha/M], 0.5-4.0 km/s in microturbulence.
A0V_LABELS = {
    "effective_temperature": 9500.0,
    "log_surface_gravity": 4.1,
    "metallicity": 0.0,
    "alpha_enhancement": 0.0,
    "microturbulence_km_s": 2.0,
}
A0V_LABEL_SOURCE = (
    "spectral-type average for A0V (Teff 9500 K, log g 4.1); not a fit to any "
    "individual standard, and not Vega, which is a pole-on rapid rotator"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # IGRINS covers 1.4298-1.8347 um in H and 1.8473-2.5187 um in K; the extra
    # few nanometres at each end keep resampling from ever extrapolating.
    parser.add_argument("--wavelength-start-nm", type=float, default=1415.0)
    parser.add_argument("--wavelength-end-nm", type=float, default=2545.0)
    # The model grid runs at 1.67 km/s (R = 45,000 at four samples per
    # resolution element), so the source must be at least that fine or
    # resample_stellar_source refuses it as a resolution lie. 600,000 is
    # 0.5 km/s, with room for a finer sampling later.
    parser.add_argument("--r-grid", type=int, default=600_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("/home/jjlee/work/lblrtm/data/stellar/a0v_payne_zero_hk.npz"),
    )
    args = parser.parse_args()

    from payne_zero_synthesis import synthesize_from_labels

    spectrum = synthesize_from_labels(
        **A0V_LABELS,
        wavelength_start_nm=args.wavelength_start_nm,
        wavelength_end_nm=args.wavelength_end_nm,
        r_grid=args.r_grid,
        device=args.device,
    )

    wavelength_nm = np.asarray(spectrum.wavelength_nm, dtype=float)
    normalized = np.asarray(spectrum.normalized_flux, dtype=float)
    if np.any(~np.isfinite(normalized)) or np.any(normalized <= 0.0):
        raise SystemExit("synthesis returned non-positive or non-finite normalized flux")

    # jax_telluric works in ascending vacuum wavenumber.
    wavenumber_cm1 = 1.0e7 / wavelength_nm
    order = np.argsort(wavenumber_cm1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        wavenumber_cm1=wavenumber_cm1[order],
        flux=normalized[order],
        wavelength_vacuum_nm=wavelength_nm,
        flux_total=np.asarray(spectrum.flux_total, dtype=float),
        flux_continuum=np.asarray(spectrum.flux_continuum, dtype=float),
    )

    # Where the hydrogen lines landed and how deep they came out, so the one
    # thing this spectrum is for can be checked without reopening it.
    rydberg = 1.09677583e7
    lines = {}
    for lower, label in ((4, "Br"), (5, "Pf")):
        for upper in range(lower + 1, 26):
            line_nm = 1.0e9 / (rydberg * (1.0 / lower**2 - 1.0 / upper**2))
            if not args.wavelength_start_nm + 1 < line_nm < args.wavelength_end_nm - 1:
                continue
            near = np.abs(wavelength_nm - line_nm) < 1.5
            if near.any():
                lines[f"{label}{upper}"] = {
                    "wavelength_nm": round(float(line_nm), 4),
                    "minimum_normalized_flux": round(float(normalized[near].min()), 5),
                }

    metadata = {
        "star": "A0V (spectral-type average)",
        "labels": A0V_LABELS,
        "label_source": A0V_LABEL_SOURCE,
        "wavelength_nm": [args.wavelength_start_nm, args.wavelength_end_nm],
        "r_grid": args.r_grid,
        "points": int(wavelength_nm.size),
        "velocity_step_kms": float(
            np.median(np.diff(np.log(wavenumber_cm1[order]))) * 299792.458
        ),
        "normalized_flux_range": [float(normalized.min()), float(normalized.max())],
        "hydrogen_lines": lines,
        "initializer_family": getattr(spectrum, "initializer_family", None),
        # False means the spectrum rests on the learned initializer atmosphere
        # rather than a converged solve. Recorded because it bounds how far the
        # stellar model can be trusted, and it is not visible in the arrays.
        # At 9,500 K, where the hydrogen lines are the entire spectrum, this is
        # the single most important caveat on the file.
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
