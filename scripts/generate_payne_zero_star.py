#!/usr/bin/env python
"""Synthesize a science target's spectrum across IGRINS H and K with Payne Zero.

The same recipe as ``generate_payne_zero_a0v.py``, with the labels given on the
command line: a science frame fitted with its own star in the source slot is
the A0V case again, and the injection test says that is what removes the floor
a line-rich target puts under the per-frame water scale.

Like the other generators this does **not** run in the project environment --
Payne Zero needs Python >= 3.11 and PyTorch, this package is pinned to 3.10 --
so run it in Payne Zero's own:

    cd /home/jjlee/work/payne-zero
    export PAYNE_ZERO_DATA_ROOT=$PWD/source_data_files
    export PAYNE_ZERO_SYNTHESIS_CACHE_DIR=$PWD/.cache/payne-zero/synthesis
    export NUMBA_CACHE_DIR=$PWD/.cache/payne-zero/numba-atmosphere
    .venv/bin/python /home/jjlee/work/lblrtm/scripts/generate_payne_zero_star.py \\
        --name "LkCa 15" --teff 4370 --logg 3.9 --metallicity 0.0 \\
        --label-source "..." --output /home/jjlee/work/lblrtm/data/stellar/lkca15_payne_zero_hk.npz

Payne Zero supports 4,000-10,500 K, 0.7-5.3 in log g, -2.5-0.5 in [M/H]; the
labels are refused outside that rather than extrapolated. Rotation is not
applied here -- the fitting drivers take ``--vsini-kms`` -- so one file serves
any rotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SUPPORT = {"effective_temperature": (4000.0, 10500.0), "log_surface_gravity": (0.7, 5.3),
           "metallicity": (-2.5, 0.5), "alpha_enhancement": (-0.1, 0.5),
           "microturbulence_km_s": (0.5, 4.0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--teff", type=float, required=True)
    parser.add_argument("--logg", type=float, required=True)
    parser.add_argument("--metallicity", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--microturbulence-kms", type=float, default=1.5)
    parser.add_argument("--label-source", required=True,
                        help="where the labels came from; recorded, because a model is only "
                             "as good as them and nothing in the arrays says so")
    # IGRINS covers 1.4298-1.8347 um in H and 1.8473-2.5187 um in K; the extra
    # few nanometres at each end keep resampling from ever extrapolating.
    parser.add_argument("--wavelength-start-nm", type=float, default=1415.0)
    parser.add_argument("--wavelength-end-nm", type=float, default=2545.0)
    # 0.5 km/s, finer than the model grid's 1.67 km/s, which
    # resample_stellar_source requires.
    parser.add_argument("--r-grid", type=int, default=600_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = {
        "effective_temperature": args.teff,
        "log_surface_gravity": args.logg,
        "metallicity": args.metallicity,
        "alpha_enhancement": args.alpha,
        "microturbulence_km_s": args.microturbulence_kms,
    }
    for key, (low, high) in SUPPORT.items():
        if not low <= labels[key] <= high:
            raise SystemExit(f"{key} = {labels[key]} is outside Payne Zero's {low}-{high}")

    from payne_zero_synthesis import synthesize_from_labels

    spectrum = synthesize_from_labels(
        **labels,
        wavelength_start_nm=args.wavelength_start_nm,
        wavelength_end_nm=args.wavelength_end_nm,
        r_grid=args.r_grid,
        device=args.device,
    )
    wavelength_nm = np.asarray(spectrum.wavelength_nm, dtype=float)
    normalized = np.asarray(spectrum.normalized_flux, dtype=float)
    if np.any(~np.isfinite(normalized)) or np.any(normalized <= 0.0):
        raise SystemExit("synthesis returned non-positive or non-finite normalized flux")

    # tellurix works in ascending vacuum wavenumber, every array alike.
    wavenumber_cm1 = 1.0e7 / wavelength_nm
    order = np.argsort(wavenumber_cm1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        wavenumber_cm1=wavenumber_cm1[order],
        flux=normalized[order],
        wavelength_vacuum_nm=wavelength_nm[order],
        flux_total=np.asarray(spectrum.flux_total, dtype=float)[order],
        flux_continuum=np.asarray(spectrum.flux_continuum, dtype=float)[order],
    )
    metadata = {
        "star": args.name,
        "labels": labels,
        "label_source": args.label_source,
        "wavelength_nm": [args.wavelength_start_nm, args.wavelength_end_nm],
        "r_grid": args.r_grid,
        "points": int(wavelength_nm.size),
        "velocity_step_kms": float(np.median(np.diff(np.log(wavenumber_cm1[order]))) * 299792.458),
        "normalized_flux_range": [float(normalized.min()), float(normalized.max())],
        "fraction_of_pixels_deeper_than_5_percent": float(np.mean(normalized < 0.95)),
        "initializer_family": getattr(spectrum, "initializer_family", None),
        # False means the spectrum rests on the learned initializer atmosphere
        # rather than a converged solve; not visible in the arrays.
        "atmosphere_converged": bool(getattr(spectrum, "atmosphere_converged", False)),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
