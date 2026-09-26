#!/usr/bin/env python
"""Synthetic checks on the NSO solar FTS fit, before any real data is trusted.

Four questions, each answered by generating data from a known truth and fitting
it back:

L0  Can the fitter recover its own parameters at this data's noise level?
L1  Is the internal grid fine enough, or does it bias the retrieved columns?
L2  How much does the Gaussian line spread function cost against the measured
    unapodized FTS sinc?
L3  Does air mass fold into the column scale exactly, as the slant-path test
    assumes?

L1-L3 generate one reference spectrum on the finest grid with the sinc, then fit
it with each candidate configuration, so the comparison isolates the model
choice rather than re-deriving the truth each time.

L3 has no counterpart in the Arcturus checks and is the reason this file exists
separately. The whole point of starting on ``ftsspec_901218_{4,5}`` is that they
are the same sky at two air masses, so the fitted columns must differ by exactly
their ratio when the fit is run at ``zenith_angle_deg = 0`` and must agree when
it is run at the header's own air mass. That is a property of the forward model
before it is a property of the data, and it is cheaper to check here.

    UV_CACHE_DIR=.uv-cache uv run python scripts/validate_fts_fit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from tellurix import (
    AERLineDatabase,
    BoxcarFTSInstrumentProfile,
    ExoJAXOpacityBackend,
    MTCKDWaterContinuum,
    SpectralOrder,
    TelluricModel,
    TelluricParameters,
    constant_velocity_grid,
    fit_order,
    load_atmosphere_csv,
    trim_wavenumber_grid,
    zenith_angle_deg_for_airmass,
)
from tellurix.nso import MEASURED_FWHM_CM1

# 1.67 um: H2O against the CH4 2nu3 band, which is the strongest absorber here
# (max logSij0 -48.3 against H2O's -55.4). A different pair from the Arcturus
# checks on purpose -- this window is where the ftsspec response peaks.
MOLECULE_IDS = {"H2O": 1, "CH4": 6}
TRUTH = {"H2O": np.log(1.20), "CH4": np.log(1.15)}
TRUTH_VELOCITY_KMS = 0.30
TRUTH_LSF_KMS = 0.40
TRUTH_CONTINUUM = np.array([-0.02, -0.03, 0.02, -0.03])
# Measured on ftsspec_901218_5 over 6000-6100 cm-1, in continuum units. Ten
# times better than the Arcturus atlas, which is what makes this data hard.
NOISE = 0.00046
# The interferogram cut, constant across photatl and the 1990 ftsspec pair.
MOPD_CM = 1.20671 / (2.0 * MEASURED_FWHM_CM1)
# L3 injects at file 4's mean air mass and asks for it back.
TRUTH_AIRMASS = 4.730


def build(profile, root, v1, v2, margin, grid_margin, resolving_power, samples,
          gaussian_ils, chunk):
    grid = constant_velocity_grid(
        1.0e7 / v2, 1.0e7 / v1, resolving_power=resolving_power,
        samples_per_resolution=samples, margin_cm1=margin,
    )
    # Lines reach in from the wide margin; the grid only has to cover the
    # window plus what the LSF and the Doppler shifts reach back for.
    grid = trim_wavenumber_grid(grid, v1, v2, grid_margin)
    line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    databases = {}
    for species, molecule_id in MOLECULE_IDS.items():
        name = f"{molecule_id:02d}_{species}"
        databases[species] = AERLineDatabase(
            line_root / name / name, species, (v1, v2), margin_cm1=margin
        )
    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse",
        temperature_range_k=(float(np.min(profile.temperature_k)), float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        vectorize_layers=True, mixed_precision=True, pressure_shift=True,
        layer_chunk_size=chunk,
    )
    continuum = MTCKDWaterContinuum.from_netcdf(
        root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc", grid
    )
    instrument = None if gaussian_ils else BoxcarFTSInstrumentProfile(
        mopd_cm=MOPD_CM, wavenumber_center_cm1=float(0.5 * (v1 + v2)), max_residual_sigma_kms=4.0
    )
    model = TelluricModel(
        profile, grid, opacity, continuum=continuum, accuracy_mode="mt_ckd",
        max_lsf_sigma_kms=4.0,
        # The FTS point-samples; it does not integrate over a pixel.
        pixel_integration="point",
        instrument=instrument,
    )
    return model, grid


def truth_parameters(species):
    return TelluricParameters(
        log_column_scales={name: TRUTH[name] for name in species},
        velocity_kms=TRUTH_VELOCITY_KMS,
        wavelength_stretch=0.0,
        lsf_sigma_kms=TRUTH_LSF_KMS,
        continuum_coeffs=TRUTH_CONTINUUM,
        log_jitter=np.log(1.0e-5),
    )


def fit_bounds(species, pinned_jitter):
    bounds = {name: (-2.0, 2.0) for name in species}
    bounds.update({
        "velocity_kms": (-5.0, 5.0),
        # An FTS wavenumber scale is exactly linear, so the only physical
        # freedom is a multiplicative factor, which is already the velocity.
        "wavelength_stretch": (0.0, 0.0),
        "lsf_sigma_kms": (0.05, 4.0),
        "log_jitter": (pinned_jitter, pinned_jitter),
    })
    for index in range(len(TRUTH_CONTINUUM)):
        bounds[f"continuum_{index}"] = (-2.0, 2.0) if index == 0 else (-0.5, 0.5)
    return bounds


def run_fit(model, wavelength, flux, uncertainty, zenith_angle_deg=0.0):
    order = SpectralOrder(wavelength, flux, np.full(wavelength.size, uncertainty),
                          zenith_angle_deg=zenith_angle_deg)
    start = TelluricParameters(
        log_column_scales={name: 0.0 for name in model.species},
        velocity_kms=0.0, wavelength_stretch=0.0, lsf_sigma_kms=0.5,
        continuum_coeffs=np.zeros(len(TRUTH_CONTINUUM)), log_jitter=np.log(uncertainty),
    )
    return fit_order(model, order, start, fit_bounds(model.species, float(np.log(uncertainty))))


def summarize(result, model, uncertainty):
    names = list(model.species)
    residual = np.asarray(result.residuals)
    return {
        "success": bool(result.success),
        "message": result.message,
        **{f"{name}_scale": float(np.exp(result.parameters.log_column_scales[name])) for name in names},
        "velocity_kms": float(result.parameters.velocity_kms),
        "lsf_sigma_kms": float(result.parameters.lsf_sigma_kms),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "rms_over_noise": float(np.sqrt(np.mean(residual**2)) / uncertainty),
        "reduced_chi2": float(np.mean((residual / uncertainty) ** 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1", type=float, default=6000.0)
    parser.add_argument("--v2", type=float, default=6020.0)
    parser.add_argument("--margin-cm1", type=float, default=25.0)
    parser.add_argument("--grid-margin-cm1", type=float, default=5.0)
    parser.add_argument("--pixels", type=int, default=2110,
                        help="the real sampling over this window is 0.0094771 cm-1")
    parser.add_argument("--profile", type=Path,
                        default=Path("data/profiles/kitt_peak_19901218_file5.csv"))
    parser.add_argument("--layer-chunk-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=19901218)
    parser.add_argument("--output", type=Path, default=Path("docs/solar_fts_validation.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    profile = load_atmosphere_csv(root / args.profile)
    wavelength = np.linspace(1.0e7 / args.v2, 1.0e7 / args.v1, args.pixels)
    rng = np.random.default_rng(args.seed)
    started = time.time()

    # The resolving power this window actually needs: the resolution element is
    # constant in wavenumber for this instrument, so R is not a free choice.
    centre = 0.5 * (args.v1 + args.v2)
    needed_r = centre / MEASURED_FWHM_CM1
    common = dict(v1=args.v1, v2=args.v2, margin=args.margin_cm1,
                  grid_margin=args.grid_margin_cm1, chunk=args.layer_chunk_size or None)

    reference_model, _ = build(profile, root, resolving_power=2.0 * needed_r, samples=4.0,
                               gaussian_ils=False, **common)
    truth = truth_parameters(reference_model.species)
    zenith = zenith_angle_deg_for_airmass(TRUTH_AIRMASS)
    blank = SpectralOrder(wavelength, np.ones(args.pixels), np.full(args.pixels, NOISE))
    slanted = SpectralOrder(wavelength, np.ones(args.pixels), np.full(args.pixels, NOISE),
                            zenith_angle_deg=zenith)
    clean = np.asarray(reference_model.predict(blank, truth))
    noisy = clean + rng.normal(0.0, NOISE, clean.size)
    slant_clean = np.asarray(reference_model.predict(slanted, truth))
    slant_noisy = slant_clean + rng.normal(0.0, NOISE, slant_clean.size)
    print(f"reference on {reference_model.wavenumber_cm1.size} points "
          f"({reference_model.velocity_step_kms:.4f} km/s), R needed {needed_r:,.0f}, "
          f"MOPD {MOPD_CM:.3f} cm")

    report = {
        "window_cm1": [args.v1, args.v2],
        "pixels": args.pixels,
        "noise": NOISE,
        "profile": str(args.profile),
        "instrument": {"mopd_cm": MOPD_CM, "fwhm_cm1": MEASURED_FWHM_CM1,
                       "resolving_power_at_centre": needed_r},
        "truth": {
            **{f"{name}_scale": float(np.exp(value)) for name, value in TRUTH.items()},
            "velocity_kms": TRUTH_VELOCITY_KMS,
            "lsf_sigma_kms": TRUTH_LSF_KMS,
            "airmass": TRUTH_AIRMASS,
        },
        "cases": [],
    }

    configurations = [
        ("L1_half", 0.5 * needed_r, 4.0, False),
        ("L1_full", needed_r, 4.0, False),
        ("L1_double", 2.0 * needed_r, 4.0, False),
        ("L2_gaussian", needed_r, 4.0, True),
    ]
    for label, resolving_power, samples, gaussian in configurations:
        model, grid = build(profile, root, resolving_power=resolving_power, samples=samples,
                            gaussian_ils=gaussian, **common)
        case_started = time.time()
        result = run_fit(model, wavelength, noisy, NOISE)
        case = {
            "case": label,
            "resolving_power": resolving_power,
            "samples_per_resolution": samples,
            "points": int(grid.size),
            "velocity_step_kms": model.velocity_step_kms,
            "instrument": "gaussian" if gaussian else "boxcar FTS sinc",
            "seconds": round(time.time() - case_started, 1),
            **summarize(result, model, NOISE),
        }
        report["cases"].append(case)
        print(f"[{label}] {case['points']:6d} pts  " + "  ".join(
            f"{name} {case[f'{name}_scale']:.4f}" for name in TRUTH) +
            f"  rms/noise {case['rms_over_noise']:.3f}  ({case['seconds']} s)")

    by_label = {case["case"]: case for case in report["cases"]}

    # L0: the finest grid with the correct instrument must recover the truth.
    # The criterion is a fractional error, not a pull: FitResult.covariance is a
    # formal covariance that assumes independent pixel errors and comes out
    # 3-9x too small on real data, so it must not be used as a test threshold.
    l0 = by_label["L1_double"]
    l0_checks = {"note": "formal errors are not quotable uncertainties; see fit.py"}
    for name, value in TRUTH.items():
        l0_checks[f"{name}_fractional_error"] = float(
            (l0[f"{name}_scale"] - np.exp(value)) / np.exp(value))
    l0_checks["velocity_error_kms"] = float(l0["velocity_kms"] - TRUTH_VELOCITY_KMS)
    l0_checks["lsf_error_kms"] = float(l0["lsf_sigma_kms"] - TRUTH_LSF_KMS)
    l0_checks["reduced_chi2"] = l0["reduced_chi2"]
    l0_checks["threshold_fractional_error"] = 0.01
    l0_checks["passed"] = bool(
        all(abs(l0_checks[f"{name}_fractional_error"]) <= 0.01 for name in TRUTH)
        and abs(l0["reduced_chi2"] - 1.0) <= 0.15
        and abs(l0_checks["velocity_error_kms"]) <= 0.1
    )
    report["L0_injection_recovery"] = l0_checks

    # L1: the two finest grids must agree to a fixed fraction.
    fine, finest, coarse = by_label["L1_full"], by_label["L1_double"], by_label["L1_half"]
    tolerance = 0.005
    l1 = {"threshold_fractional_shift": tolerance}
    for name in TRUTH:
        shift = abs(fine[f"{name}_scale"] - finest[f"{name}_scale"]) / finest[f"{name}_scale"]
        l1[name] = {
            "fractional_shift_full_vs_double": float(shift),
            "converged": bool(shift <= tolerance),
            "fractional_shift_half_vs_double": float(
                abs(coarse[f"{name}_scale"] - finest[f"{name}_scale"]) / finest[f"{name}_scale"]),
        }
    l1["half_rms_over_noise"] = coarse["rms_over_noise"]
    l1["passed"] = bool(all(l1[name]["converged"] for name in TRUTH))
    report["L1_grid_convergence"] = l1

    # L2: a Gaussian standing in for the measured sinc.
    gaussian_case, sinc_case = by_label["L2_gaussian"], by_label["L1_full"]
    report["L2_instrument_profile"] = {
        "gaussian_rms_over_noise": gaussian_case["rms_over_noise"],
        "sinc_rms_over_noise": sinc_case["rms_over_noise"],
        "gaussian_penalty": float(gaussian_case["rms_over_noise"] / sinc_case["rms_over_noise"]),
        **{f"{name}_bias_vs_sinc": float(
            (gaussian_case[f"{name}_scale"] - sinc_case[f"{name}_scale"]) / sinc_case[f"{name}_scale"])
           for name in TRUTH},
        "threshold_rms_over_noise": 1.2,
        "passed": bool(gaussian_case["rms_over_noise"] <= 1.2),
    }

    # L3: the slant-path identity the whole ftsspec plan rests on. Data
    # generated at air mass X, fitted at the same zenith angle, must return the
    # truth; fitted at zero, it must return the truth times X.
    model, _ = build(profile, root, resolving_power=needed_r, samples=4.0,
                     gaussian_ils=False, **common)
    at_zenith = summarize(run_fit(model, wavelength, slant_noisy, NOISE, zenith), model, NOISE)
    at_zero = summarize(run_fit(model, wavelength, slant_noisy, NOISE, 0.0), model, NOISE)
    l3 = {
        "airmass": TRUTH_AIRMASS,
        "zenith_angle_deg": zenith,
        "fitted_at_header_zenith": {k: at_zenith[k] for k in at_zenith if k != "message"},
        "fitted_at_zero_zenith": {k: at_zero[k] for k in at_zero if k != "message"},
        "threshold_fractional_error": 0.01,
    }
    for name, value in TRUTH.items():
        recovered = at_zenith[f"{name}_scale"]
        folded = at_zero[f"{name}_scale"]
        l3[name] = {
            "recovered_at_header_zenith": recovered,
            "fractional_error": float((recovered - np.exp(value)) / np.exp(value)),
            "recovered_at_zero_zenith": folded,
            "ratio_to_truth": float(folded / np.exp(value)),
            # exp(-tau * X) is not exp(-tau) ** X through a convolution and a
            # saturated line, so the fold is only exact where the line is
            # optically thin. This records how close it comes.
            "ratio_over_airmass": float(folded / np.exp(value) / TRUTH_AIRMASS),
        }
    l3["passed"] = bool(all(abs(l3[name]["fractional_error"]) <= 0.01 for name in TRUTH))
    report["L3_air_mass"] = l3

    report["runtime_seconds"] = round(time.time() - started, 1)
    (root / args.output).parent.mkdir(parents=True, exist_ok=True)
    (root / args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key in ("L0_injection_recovery", "L1_grid_convergence",
                "L2_instrument_profile", "L3_air_mass"):
        print(f"{key}: passed={report[key]['passed']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
