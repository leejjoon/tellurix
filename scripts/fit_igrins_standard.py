#!/usr/bin/env python
"""Fit the telluric absorption in IGRINS A0V standards, order by order.

This is the Arcturus batch driver's structure applied to a different kind of
data, and the differences are the point:

* the reduction ships a per-pixel ``variance``, so the uncertainty is measured
  rather than estimated from second differences;
* the header carries the standard's own zenith distance, so the slant path is
  known and a fitted column scale measures the atmosphere instead of absorbing
  the airmass;
* an A0V is featureless across H and K once the hydrogen series is masked, so
  ``--stellar flat`` carries no stellar model error at all. On the atlas the
  residual was flat against transmission, which said the error was in the star.
  Here that plot is a measurement of the telluric model.

    uv run python scripts/fit_igrins_standard.py \\
        --spec data/igrins/20180402_0104/SDCH_20180402_0104.spec.fits --orders 2

Give several ``--spec`` paths to fit a whole night at once, which is far cheaper
per frame than running them one at a time -- see below. ``--orders`` takes a
comma-separated list; leave it out for all of them.

Why the loops are this way round
--------------------------------
The outer loop is over echelle orders and the inner one over frames, which is
the opposite of the obvious nesting and is worth about two thirds of the
runtime.

Within a night the PLP uses one wavelength solution, so order N covers a
*bit-identical* wavenumber range in every frame (measured spread across ten
frames: 0.0000 cm-1); across nights and sites it moves by at most 0.16 cm-1 on
an 80 cm-1 window, far inside the 5 cm-1 grid margin. Everything expensive
therefore belongs to the order and not to the frame: the grid, the line
selection, the opacity calculators, the precomputed kernel, and -- once the
per-frame arrays are passed as operands rather than captured as constants --
the two XLA compilations, which were 13.1 s of the 31.7 s an order used to cost
(4.9 s for the gradient, 8.2 s for the Hessian, against 2.6 ms and 3.1 ms to
run them).

What is deliberately *not* shared is the starting point. Warm-starting each
frame from the previous one would pull its fitted columns toward that
neighbour and shrink exactly the frame-to-frame scatter an airmass ladder
exists to measure. Every frame starts from the same neutral guess.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

MOLECULE_IDS = {"H2O": 1, "CO2": 2, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
STAGES = ("continuum", "velocity", "columns")

# Everything about the physics that is the same for every order. Named here and
# read from here, so the summary records what ran rather than a second
# description of it that can drift away from the code.
PHYSICS = {
    "accuracy_mode": "mt_ckd",
    "opacity_method": "direct_sparse",
    "mt_ckd_file": "absco-ref_wv-mt-ckd.nc",
    "pressure_shift": True,
    "mixed_precision": True,
    "vectorize_layers": True,
    # IGRINS pixels integrate the spectrum across their own width; the FTS the
    # Arcturus atlas came from point-samples it, which is why that pipeline
    # sets "point" and this one must not.
    "pixel_integration": "simpson",
    # A grating spectrograph's line spread function is the fitted Gaussian, not
    # a residual on top of an FTS sinc, so there is no BoxcarFTSInstrumentProfile
    # here and no MOPD to measure.
    "instrument": "gaussian",
    "max_lsf_sigma_kms": 10.0,
    "macroturbulence_kms": 0.0,
    "limb_darkening": 0.6,
    "normalize_stellar_source": True,
    "stages": list(STAGES),
}

# How an extracted order becomes a fittable one.
ORDER = {
    "saturation_floor": 0.02,
    "throughput_floor": 0.25,
    "continuum_percentile": 95.0,
    "mask_hydrogen_kms": 600.0,
    "minimum_pixels": 256,
}


def stage_bounds(stage, model_species, free_species, parameters, degree, fit_stellar):
    """Which parameters each stage frees, and within what range."""

    free = {
        "continuum": {"continuum", "log_jitter"},
        "velocity": {"continuum", "log_jitter", "velocity_kms", "lsf_sigma_kms"},
        "columns": {"continuum", "log_jitter", "velocity_kms", "lsf_sigma_kms", "species"},
        "stellar": {"continuum", "log_jitter", "velocity_kms", "lsf_sigma_kms", "species",
                    "stellar_velocity_kms"},
    }[stage]
    pinned = lambda value: (float(value), float(value))  # noqa: E731
    bounds = {}
    for species in model_species:
        bounds[species] = (-2.0, 2.0) if ("species" in free and species in free_species) else (0.0, 0.0)
    bounds["velocity_kms"] = (-8.0, 8.0) if "velocity_kms" in free else pinned(parameters.velocity_kms)
    bounds["wavelength_stretch"] = (0.0, 0.0)
    # The whole line spread function is this Gaussian: R = 45,000 puts it near
    # 2.8 km/s, and the slit and the focus move it by tens of percent, not by
    # the factor the atlas's residual-broadening term needed.
    bounds["lsf_sigma_kms"] = (1.0, 6.0) if "lsf_sigma_kms" in free else pinned(parameters.lsf_sigma_kms)
    for index in range(degree + 1):
        value = float(np.asarray(parameters.continuum_coeffs)[index])
        if "continuum" not in free:
            bounds[f"continuum_{index}"] = pinned(value)
        else:
            # The order is normalized to its own continuum level, so the
            # constant term sits near zero. The blaze shape the rest describe
            # runs over a factor of a few across an order, which is wider than
            # the atlas needed but far short of the counts scale.
            bounds[f"continuum_{index}"] = (-2.0, 2.0) if index == 0 else (-1.5, 1.5)
    bounds["log_jitter"] = ((np.log(1e-5), np.log(0.5)) if "log_jitter" in free
                            else pinned(parameters.log_jitter))
    bounds["stellar_velocity_kms"] = ((-60.0, 60.0) if ("stellar_velocity_kms" in free and fit_stellar)
                                      else pinned(parameters.stellar_velocity_kms))
    return bounds


def build_order_context(observation, index, args, root, profile, stellar):
    """Everything about one echelle order that no frame can change.

    The window comes from one frame's wavelength solution, which within a night
    is every frame's wavelength solution. Across nights it moves far less than
    the grid margin, so a context built on one night still covers another -- but
    the objective compiled against it will refuse to rebind onto a different
    pixel grid, which is what keeps that assumption honest rather than silent.
    """

    import jax.numpy as jnp
    from jax_telluric import (
        AERLineDatabase, ExoJAXOpacityBackend, MTCKDWaterContinuum, StellarSpectrum,
        TelluricModel, igrins_wavenumber_grid, prepare_stellar_source, trim_wavenumber_grid,
    )

    timing, started = {}, time.time()

    def mark(name):
        nonlocal started
        timing[name] = round(time.time() - started, 2)
        started = time.time()

    extracted = observation.order(index)
    v1, v2 = extracted.wavenumber_range_cm1

    # Two different margins. Lines are selected over the full reach of their
    # wings; the grid only has to cover the window plus what the LSF, the
    # Doppler shifts and the instrument profile reach back for.
    grid = trim_wavenumber_grid(
        igrins_wavenumber_grid(1.0e7 / v2, 1.0e7 / v1,
                               resolving_power=args.resolving_power,
                               samples_per_resolution=args.samples_per_resolution,
                               margin_cm1=args.margin_cm1),
        v1, v2, args.grid_margin_cm1)

    line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    databases, absent = {}, []
    for species, molecule_id in sorted(MOLECULE_IDS.items()):
        name = f"{molecule_id:02d}_{species}"
        try:
            databases[species] = AERLineDatabase(
                line_root / name / name, species, (v1, v2), margin_cm1=args.margin_cm1)
        except ValueError as exc:
            if not str(exc).startswith(f"no {species} lines found"):
                raise
            absent.append(species)
    if not databases:
        raise RuntimeError(f"no molecular lines in order {index}")
    mark("line_files")

    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid,
        temperature_range_k=(float(np.min(profile.temperature_k)),
                             float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        methods=PHYSICS["opacity_method"], vectorize_layers=PHYSICS["vectorize_layers"],
        mixed_precision=PHYSICS["mixed_precision"], pressure_shift=PHYSICS["pressure_shift"])
    mark("opacity_prepare")

    continuum_backend = MTCKDWaterContinuum.from_netcdf(
        root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc", grid)
    # No `instrument=`: the default Gaussian is the whole line spread function
    # of a grating spectrograph.
    model = TelluricModel(profile, grid, opacity, continuum=continuum_backend,
                          accuracy_mode=PHYSICS["accuracy_mode"],
                          max_lsf_sigma_kms=PHYSICS["max_lsf_sigma_kms"],
                          pixel_integration=PHYSICS["pixel_integration"])
    mark("continuum")

    spectrum = StellarSpectrum.flat(grid) if stellar is None else stellar
    source = prepare_stellar_source(
        spectrum, model, vsini_kms=args.vsini_kms,
        limb_darkening=PHYSICS["limb_darkening"],
        macroturbulence_kms=PHYSICS["macroturbulence_kms"],
        normalize=PHYSICS["normalize_stellar_source"])
    mark("stellar_source")

    fit_model = (model.precompute_opacity(self_broadening=args.self_broadening)
                 if args.precompute_opacity else model)
    mark("precompute")

    # Decide which species the data can actually measure. Everything stays in
    # the model; only what is constrained is freed. This depends on the profile
    # and the grid, not on the frame, so it belongs here too.
    pressure = jnp.asarray(profile.pressure_layer_bar)
    partial = {s: pressure * jnp.asarray(profile.vmr[s]) for s in model.species}
    cross_sections = fit_model.opacity.cross_sections(
        jnp.asarray(profile.temperature_k), pressure, partial)
    air_column = np.asarray(profile.air_column_cm2)
    optical_depth = {
        s: float(np.max(np.sum(np.asarray(cross_sections[s])
                               * (air_column * np.asarray(profile.vmr[s]))[:, None], axis=0)))
        for s in model.species
    }
    mark("species_scan")

    return {
        "index": index, "v1": v1, "v2": v2, "grid": grid, "model": model,
        "fit_model": fit_model, "source": source, "profile": profile,
        "databases": databases, "absent": absent, "optical_depth": optical_depth,
        "free_species": sorted(s for s, t in optical_depth.items() if t >= args.min_optical_depth),
        "mt_ckd_version": continuum_backend.version,
        "setup_timing": timing,
        "setup_seconds": round(sum(timing.values()), 2),
    }


def fit_one(context, observation, args, objective):
    """Fit one order of one frame, reusing whatever the context already built.

    Returns the row and the objective -- the compiled one when it could be
    rebound onto this frame, a fresh one when it could not.
    """

    from jax_telluric import (
        ArrayOpacityBackend, OrderObjective, TelluricModel, TelluricParameters,
        chebyshev_continuum, continuum_level, fit_order, igrins_spectral_order,
    )

    timing, started = {}, time.time()

    def mark(name):
        nonlocal started
        timing[name] = round(time.time() - started, 2)
        started = time.time()

    index = context["index"]
    model, fit_model = context["model"], context["fit_model"]
    profile, grid = context["profile"], context["grid"]
    extracted = observation.order(index)
    level = continuum_level(extracted, ORDER["continuum_percentile"])
    order = igrins_spectral_order(
        extracted, source_flux_model_grid=context["source"],
        saturation_floor=ORDER["saturation_floor"],
        throughput_floor=ORDER["throughput_floor"],
        continuum_percentile=ORDER["continuum_percentile"],
        # With a flat source the hydrogen series is unmodelled and has to go.
        # With a real A0V model it is the thing being tested, so it stays.
        mask_hydrogen_kms=ORDER["mask_hydrogen_kms"] if args.stellar == "flat" else None)
    if int(np.count_nonzero(order.mask)) < ORDER["minimum_pixels"]:
        raise RuntimeError(
            f"order {index} keeps {int(np.count_nonzero(order.mask))} pixels, "
            f"below the {ORDER['minimum_pixels']} this driver requires")
    mark("order")

    degree = args.continuum_degree
    usable = np.asarray(order.flux)[np.asarray(order.mask)]
    parameters = TelluricParameters(
        log_column_scales={s: 0.0 for s in model.species},
        velocity_kms=0.0, wavelength_stretch=0.0,
        # R = 45,000 in FWHM is 6.66 km/s, so 2.83 km/s in sigma.
        lsf_sigma_kms=299792.458 / (args.resolving_power * 2.3548200),
        continuum_coeffs=np.concatenate([[float(np.log(np.median(usable)))], np.zeros(degree)]),
        log_jitter=float(np.log(np.median(np.asarray(order.uncertainty)[np.asarray(order.mask)]))),
        stellar_velocity_kms=0.0)

    reused = False
    if objective is not None:
        try:
            objective = objective.rebind(order)
            reused = True
        except ValueError:
            objective = None
    if objective is None:
        objective = OrderObjective(fit_model, order, degree + 1)
        # Force the compilation here so the stage timings measure fitting, not
        # the one-time XLA cost that would otherwise land on the first stage.
        objective(objective.codec.pack(parameters))
    mark("compile_objective")

    stages = []
    for position, stage in enumerate(STAGES):
        began = time.time()
        result = fit_order(
            fit_model, order, parameters,
            stage_bounds(stage, model.species, context["free_species"], parameters, degree,
                         fit_stellar=args.stellar != "flat"),
            objective=objective,
            # Only this call ever compiles the Hessian, and that compilation
            # costs 8.2 s against 3.1 ms to run it. The intermediate stages'
            # covariances are thrown away, so only the last stage asks.
            covariance=(position == len(STAGES) - 1) and not args.no_covariance)
        parameters = result.parameters
        stages.append({"stage": stage, "success": bool(result.success),
                       "objective": result.objective, "iterations": result.iterations,
                       "seconds": round(time.time() - began, 1)})
    mark("stages")

    star_only = TelluricModel(
        profile, grid,
        ArrayOpacityBackend({s: np.zeros((len(profile.temperature_k), grid.size))
                             for s in model.species}),
        accuracy_mode="fast", max_lsf_sigma_kms=PHYSICS["max_lsf_sigma_kms"],
        pixel_integration=PHYSICS["pixel_integration"])
    stellar_only_pixels = np.asarray(star_only.predict(order, parameters))
    mark("star_only")

    # Everything saved is re-evaluated without the precomputation's
    # approximation. Refreezing at the fitted parameters is exact there and
    # reuses the compilation, so it costs milliseconds.
    exact = model.precompute_opacity(parameters)
    wavenumber = 1.0e7 / np.asarray(order.wavelength_vacuum_nm)
    axis = np.argsort(wavenumber)
    nu = wavenumber[axis]
    mask = np.asarray(order.mask)[axis]
    observed = np.asarray(order.flux)[axis]
    model_flux = np.asarray(exact.predict(order, parameters))[axis]
    star = stellar_only_pixels[axis]
    transmission = np.interp(nu, np.asarray(model.wavenumber_cm1),
                             np.asarray(exact.transmission(parameters, order.zenith_angle_deg)))
    # predict()'s own continuum axis runs over [-1, 1] with ascending
    # wavelength, which is descending wavenumber, so it reverses with the rest.
    continuum = np.asarray(chebyshev_continuum(
        parameters.continuum_coeffs,
        np.linspace(-1.0, 1.0, len(order.wavelength_vacuum_nm))))[axis]
    reliable = mask & (transmission >= args.min_transmission)
    corrected = (observed / np.maximum(np.abs(model_flux), 1e-6)) * star

    plp_telluric = (np.asarray(extracted.telluric_model)[axis]
                    if extracted.telluric_model is not None else None)
    comparison = {}
    if plp_telluric is not None:
        both = reliable & np.isfinite(plp_telluric)
        if int(both.sum()) >= 64:
            difference = transmission[both] - plp_telluric[both]
            comparison = {
                "plp_pixels": int(both.sum()),
                "plp_median_difference": float(np.median(difference)),
                "plp_rms_difference": float(np.sqrt(np.mean(difference**2))),
                "plp_max_difference": float(np.max(np.abs(difference))),
            }

    residual = observed - model_flux
    sigma = np.asarray(order.uncertainty)[axis]
    pixel_sigma = float(np.median(sigma[mask]))
    residual_rms = float(np.sqrt(np.mean(residual[reliable] ** 2))) if reliable.any() else float("nan")
    mark("products")

    stem = observation.path.name.split(".")[0]
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            wavenumber_cm1=nu, observed=np.where(mask, observed, np.nan),
            model_flux=model_flux, transmission=transmission, corrected=corrected,
            stellar_only=star, continuum=continuum, residual=residual,
            uncertainty=sigma, mask=mask, reliable=reliable)
        if plp_telluric is not None:
            arrays["plp_telluric"] = plp_telluric
        if extracted.plp_continuum is not None:
            arrays["plp_continuum"] = np.asarray(extracted.plp_continuum)[axis]
        np.savez_compressed(args.output_dir / f"{stem}_{extracted.name}.npz", **arrays)

    names = list(objective.codec.names)
    deviation = (np.sqrt(np.clip(np.diag(result.covariance), 0.0, None))
                 if result.covariance is not None else np.zeros(len(names)))
    row = {
        "order": index,
        "band": extracted.band,
        "name": extracted.name,
        "v1": context["v1"], "v2": context["v2"],
        "pixels": int(mask.size), "grid_points": int(grid.size),
        "kept": int(mask.sum()), "reliable": int(reliable.sum()),
        "continuum_level_counts": float(level),
        "zenith_angle_deg": float(order.zenith_angle_deg),
        "airmass": float(1.0 / np.cos(np.radians(order.zenith_angle_deg))),
        "lines": {k: int(np.asarray(v.nu_lines).size) for k, v in context["databases"].items()},
        "absent_species": context["absent"],
        "free_species": context["free_species"],
        "max_optical_depth": {k: round(v, 4) for k, v in context["optical_depth"].items()},
        "log_column_scales": {s: float(parameters.log_column_scales[s]) for s in model.species},
        "sigma": {n: float(d) for n, d in zip(names, deviation)},
        "at_bound": list(result.at_bound or ()),
        "condition_number": (float(result.condition_number)
                             if result.condition_number is not None else None),
        "velocity_kms": float(parameters.velocity_kms),
        "stellar_velocity_kms": float(parameters.stellar_velocity_kms),
        "lsf_sigma_kms": float(parameters.lsf_sigma_kms),
        "resolving_power_fitted": float(
            299792.458 / (2.3548200 * float(parameters.lsf_sigma_kms))),
        "log_jitter": float(parameters.log_jitter),
        "continuum_coeffs": [float(c) for c in np.asarray(parameters.continuum_coeffs)],
        "pixel_sigma": pixel_sigma,
        "residual_rms": residual_rms,
        "residual_rms_over_noise": residual_rms / pixel_sigma,
        "median_transmission": float(np.median(transmission[mask])),
        "all_stages_converged": all(s["success"] for s in stages),
        "mt_ckd_version": context["mt_ckd_version"],
        "reused_compilation": reused,
        "stages": stages,
        "timing": timing,
        **comparison,
    }
    return row, objective


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True, nargs="+",
                        help="one or more PLP .spec.fits files; giving a whole night at once "
                             "shares the grid, the opacity and the compilations across frames")
    parser.add_argument("--orders", default=None,
                        help="comma-separated order indices; default every order")
    parser.add_argument("--profile", type=Path, default=Path("data/profiles/gemini_south_2018.csv"))
    parser.add_argument("--stellar", default="flat",
                        help="'flat' for a featureless source with the hydrogen series masked, "
                             "or the path to a stellar npz")
    parser.add_argument("--vsini-kms", type=float, default=0.0,
                        help="rotational broadening of the stellar source; A0V standards are "
                             "fast rotators, so set this when --stellar is a model")
    parser.add_argument("--resolving-power", type=float, default=45_000.0)
    parser.add_argument("--samples-per-resolution", type=float, default=4.0)
    parser.add_argument("--margin-cm1", type=float, default=25.0,
                        help="line-selection margin: how far outside lines still reach in")
    parser.add_argument("--grid-margin-cm1", type=float, default=5.0,
                        help="grid margin: what the LSF and Doppler shifts reach back for")
    parser.add_argument("--continuum-degree", type=int, default=9,
                        help="an IGRINS order spans four times an atlas page and carries the blaze; "
                             "measured against the PLP telluric model, degree 5 leaves twice the "
                             "residual of degree 9, and past 9 the gain is small")
    parser.add_argument("--min-optical-depth", type=float, default=0.02)
    parser.add_argument("--min-transmission", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=root / "data/corrected/igrins")
    parser.add_argument("--precompute-opacity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-broadening", choices=("linear", "frozen"), default="linear")
    parser.add_argument("--no-covariance", action="store_true",
                        help="skip the formal errors, and with them the 8.2 s Hessian compile")
    parser.add_argument("--compilation-cache", default=str(root / ".jax-cache"),
                        help="pass an empty string to disable")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()

    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Path("") is Path("."), which is truthy -- check the string, not the path,
    # or an empty argument silently writes a cache into the repository root.
    if args.compilation_cache:
        import jax
        jax.config.update("jax_compilation_cache_dir", str(Path(args.compilation_cache)))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    from jax_telluric import StellarSpectrum, load_atmosphere_csv, read_igrins_observation

    observations = [read_igrins_observation(path) for path in args.spec]
    profile = load_atmosphere_csv(root / args.profile)
    stellar = None if args.stellar == "flat" else StellarSpectrum.from_npz(args.stellar)

    counts = {o.orders for o in observations}
    if len(counts) != 1:
        raise SystemExit(f"these frames disagree about how many orders they have: {counts}")
    indices = (list(range(counts.pop())) if args.orders is None
               else [int(value) for value in args.orders.split(",")])

    print(f"{len(observations)} frame(s), {len(indices)} orders, "
          f"source {'flat' if stellar is None else args.stellar}")
    for observation in observations:
        print(f"  {observation.path.name.split('.')[0]:22s} {observation.object_name[:16]:16s} "
              f"{observation.telescope:14s} {observation.date_obs[:19]}  "
              f"airmass {1.0 / np.cos(np.radians(observation.zenith_angle_deg)):.3f}")
    print()

    results = {id(o): [] for o in observations}
    failures = {id(o): [] for o in observations}
    started = time.time()
    setup_total = 0.0

    for index in indices:
        context, objective = None, None
        for observation in observations:
            if context is None:
                try:
                    context = build_order_context(observation, index, args, root, profile, stellar)
                    setup_total += context["setup_seconds"]
                except (RuntimeError, ValueError) as exc:
                    # The window belongs to the order, so a failure here is
                    # usually shared; try the next frame anyway, because a bad
                    # extraction in one frame is not a property of the order.
                    failures[id(observation)].append({"order": index, "error": str(exc)})
                    print(f"  order {index:2d}  {observation.path.name[5:18]}  skipped: {exc}")
                    continue
            began = time.time()
            try:
                row, objective = fit_one(context, observation, args, objective)
            except (RuntimeError, ValueError) as exc:
                failures[id(observation)].append({"order": index, "error": str(exc)})
                print(f"  order {index:2d}  {observation.path.name[5:18]}  skipped: {exc}")
                continue
            row["seconds"] = round(time.time() - began, 1)
            results[id(observation)].append(row)
            tag = "reused" if row["reused_compilation"] else "built "
            plp = (f"  plp d={row['plp_rms_difference']:.4f}" if "plp_rms_difference" in row else "")
            print(f"  order {index:2d}  {observation.path.name[5:18]}  "
                  f"T={row['median_transmission']:.3f}  rms/sig={row['residual_rms_over_noise']:6.2f}  "
                  f"R={row['resolving_power_fitted']:6.0f}  v={row['velocity_kms']:+5.2f}  "
                  f"{tag} {row['seconds']:5.1f}s{plp}")

    elapsed = time.time() - started
    for observation in observations:
        rows = sorted(results[id(observation)], key=lambda r: r["order"])
        stem = observation.path.name.split(".")[0]
        summary = {
            "observation": {
                "path": str(observation.path), "object": observation.object_name,
                "object_type": observation.object_type, "band": observation.band,
                "telescope": observation.telescope, "date_obs": observation.date_obs,
                "mjd": observation.mjd, "exposure_time_s": observation.exposure_time_s,
                "zenith_angle_deg": observation.zenith_angle_deg,
                "sha256": dict(observation.sha256), "surface": dict(observation.surface),
            },
            "settings": {
                "profile": str(args.profile), "stellar": args.stellar,
                "vsini_kms": args.vsini_kms,
                "resolving_power": args.resolving_power,
                "samples_per_resolution": args.samples_per_resolution,
                "margin_cm1": args.margin_cm1, "grid_margin_cm1": args.grid_margin_cm1,
                "continuum_degree": args.continuum_degree,
                "min_optical_depth": args.min_optical_depth,
                "min_transmission": args.min_transmission,
                "precompute_opacity": args.precompute_opacity,
                "self_broadening": args.self_broadening,
                "covariance": not args.no_covariance,
                "frames_in_run": len(observations),
            },
            "physics": dict(PHYSICS), "order_rule": dict(ORDER),
            "results": rows, "failures": failures[id(observation)],
        }
        path = args.output_dir / f"{stem}_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        if rows:
            ratio = np.array([r["residual_rms_over_noise"] for r in rows])
            print(f"\n{stem}: {len(rows)} orders, median rms/sigma {np.median(ratio):.2f}, "
                  f"range {ratio.min():.2f}-{ratio.max():.2f}")

    fitted = sum(len(r) for r in results.values())
    reused = sum(1 for rows in results.values() for r in rows if r["reused_compilation"])
    print(f"\n{fitted} order-frames in {elapsed:.0f} s ({elapsed / max(fitted, 1):.1f} s each); "
          f"{reused} reused a compilation; {setup_total:.0f} s of that was per-order setup")


if __name__ == "__main__":
    main()
