"""Fitting the telluric absorption of an IGRINS order, frame by frame.

``scripts/fit_igrins_standard.py`` runs a night of A0V standards through this
module; the science-frame driver, the transfer validation and the review export
build their order contexts here too, so all of them describe one fit.

This is the Arcturus pipeline's structure applied to a different kind of data,
and the differences are the point:

* the reduction ships a per-pixel ``variance``, so the uncertainty is measured
  rather than estimated from second differences;
* the header carries the standard's own zenith distance, so the slant path is
  known and a fitted column scale measures the atmosphere instead of absorbing
  the airmass;
* an A0V is featureless across H and K once the hydrogen series is masked, so a
  flat source carries no stellar model error at all.

Everything expensive belongs to the order, not the frame: within a night the
PLP uses one wavelength solution, so order N covers a bit-identical wavenumber
range in every frame. :func:`build_order_context` builds it once and
:func:`fit_one` fits each frame against it, rebinding one compiled objective.
What is deliberately *not* shared is the starting point: warm-starting a frame
from its neighbour would shrink exactly the frame-to-frame scatter an airmass
ladder exists to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from types import MappingProxyType
from typing import Mapping

import numpy as np

from tellurix.download import DataPaths

from .igrins import continuum_level, igrins_spectral_order

MOLECULE_IDS = {"H2O": 1, "CO2": 2, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
STAGES = ("continuum", "velocity", "columns")
# With a real stellar source the star's own velocity matters and has to be
# fitted: these standards are different A0V stars with radial velocities tens of
# km/s apart, and the Brackett lines land wherever the star puts them. A flat
# source has no lines, so the parameter is meaningless there and stays pinned.
STELLAR_STAGE = "stellar"


def stages_for(stellar: str) -> tuple[str, ...]:
    return STAGES if stellar == "flat" else STAGES + (STELLAR_STAGE,)


# Everything about the physics that is the same for every order. Named here and
# read from here, so the summary records what ran rather than a second
# description of it that can drift away from the code. Read-only: a caller that
# wants a variation copies it.
PHYSICS = MappingProxyType({
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
})

# How an extracted order becomes a fittable one. The default rule; a run that
# overrides part of it carries its own copy in StandardFitSettings.order_rule,
# and a run's record stores the copy it used.
ORDER_RULE = MappingProxyType({
    "saturation_floor": 0.02,
    # The floor's job is only to drop genuinely unusable pixels. It used to do
    # more: without the fixed-pattern correction, the red edge of every order
    # corrupted the whole-order continuum, and raising this to 0.45 took the
    # residual on a fixed middle-third reference set from 1.86 to 1.67 for 16%
    # fewer pixels. The pattern handles that far better -- at 0.25 it takes one
    # order from 2.59 to 0.89 -- and with it in place 0.25 and 0.45 land within
    # 0.08 sigma of each other, so the pixels are worth keeping.
    "throughput_floor": 0.25,
    "continuum_percentile": 95.0,
    "mask_hydrogen_kms": 600.0,
    "minimum_pixels": 256,
    "minimum_reliable": 64,
    # How far a Chebyshev coefficient above the constant may move. The blaze
    # spans a factor of ten or more in log flux across an order, so this has to
    # be generous: at 1.5 it was binding on 218 of 248 order-frames, with
    # coefficient 3 sitting at its bound in 103 of them, which strangles the
    # continuum rather than regularising it.
    "continuum_bound": 5.0,
})


@dataclass(frozen=True)
class StandardFitSettings:
    """How each order is fitted. The defaults are the command line's."""

    # 'flat' for a featureless source with the hydrogen series masked, or the
    # path of a stellar npz -- recorded as given.
    stellar: str = "flat"
    vsini_kms: float = 0.0
    resolving_power: float = 45_000.0
    samples_per_resolution: float = 4.0
    # Line selection: how far outside lines still reach in.
    margin_cm1: float = 25.0
    # Grid: what the LSF and Doppler shifts reach back for.
    grid_margin_cm1: float = 5.0
    # An IGRINS order spans four times an atlas page and carries the blaze;
    # degree 5 leaves twice the residual of degree 9, and past 9 the gain is small.
    continuum_degree: int = 9
    min_optical_depth: float = 0.02
    min_transmission: float = 0.15
    precompute_opacity: bool = True
    self_broadening: str = "linear"
    # The formal errors, and with them the 8.2 s Hessian compile.
    covariance: bool = True
    # AER's first-order line coupling, which LBLRTM applies to about half the
    # CO2 lines in both the 2.0 and the 1.6 um bands. Off reproduces every run
    # made before it existed.
    line_coupling: bool = False
    order_rule: Mapping = field(default=ORDER_RULE)

    def __post_init__(self):
        unknown = set(self.order_rule) - set(ORDER_RULE)
        if unknown:
            raise ValueError(f"unknown order-rule keys: {', '.join(sorted(unknown))}")
        object.__setattr__(self, "order_rule",
                           MappingProxyType({**ORDER_RULE, **self.order_rule}))


def stage_bounds(stage, model_species, free_species, parameters, degree, fit_stellar,
                 continuum_bound=ORDER_RULE["continuum_bound"]):
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
            bounds[f"continuum_{index}"] = ((-2.0, 2.0) if index == 0
                                            else (-continuum_bound, continuum_bound))
    bounds["log_jitter"] = ((np.log(1e-5), np.log(0.5)) if "log_jitter" in free
                            else pinned(parameters.log_jitter))
    bounds["stellar_velocity_kms"] = ((-60.0, 60.0) if ("stellar_velocity_kms" in free and fit_stellar)
                                      else pinned(parameters.stellar_velocity_kms))
    return bounds


def build_order_context(observation, number, settings: StandardFitSettings, paths: DataPaths,
                        profile, stellar):
    """Everything about one echelle order that no frame can change.

    The window comes from one frame's wavelength solution, which within a night
    is every frame's wavelength solution. Across nights it moves far less than
    the grid margin, so a context built on one night still covers another -- but
    the objective compiled against it will refuse to rebind onto a different
    pixel grid, which is what keeps that assumption honest rather than silent.
    """

    import jax.numpy as jnp
    from tellurix import (
        AERLineDatabase, ExoJAXOpacityBackend, MTCKDWaterContinuum, StellarSpectrum,
        TelluricModel, constant_velocity_grid, prepare_stellar_source, trim_wavenumber_grid,
    )

    timing, started = {}, time.time()

    def mark(name):
        nonlocal started
        timing[name] = round(time.time() - started, 2)
        started = time.time()

    extracted = observation.order(number)
    v1, v2 = extracted.wavenumber_range_cm1

    # Two different margins. Lines are selected over the full reach of their
    # wings; the grid only has to cover the window plus what the LSF, the
    # Doppler shifts and the instrument profile reach back for.
    grid = trim_wavenumber_grid(
        constant_velocity_grid(1.0e7 / v2, 1.0e7 / v1,
                               resolving_power=settings.resolving_power,
                               samples_per_resolution=settings.samples_per_resolution,
                               margin_cm1=settings.margin_cm1),
        v1, v2, settings.grid_margin_cm1)

    databases, absent = {}, []
    for species, molecule_id in sorted(MOLECULE_IDS.items()):
        try:
            databases[species] = AERLineDatabase(
                paths.line_file(species, molecule_id), species, (v1, v2),
                margin_cm1=settings.margin_cm1,
                line_coupling=paths.line_coupling if settings.line_coupling else None)
        except ValueError as exc:
            if not str(exc).startswith(f"no {species} lines found"):
                raise
            absent.append(species)
    if not databases:
        raise RuntimeError(f"no molecular lines in order {number}")
    mark("line_files")

    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid,
        temperature_range_k=(float(np.min(profile.temperature_k)),
                             float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        methods=PHYSICS["opacity_method"], vectorize_layers=PHYSICS["vectorize_layers"],
        mixed_precision=PHYSICS["mixed_precision"], pressure_shift=PHYSICS["pressure_shift"],
        line_coupling=settings.line_coupling)
    mark("opacity_prepare")

    continuum_backend = MTCKDWaterContinuum.from_netcdf(paths.mt_ckd, grid)
    # No `instrument=`: the default Gaussian is the whole line spread function
    # of a grating spectrograph.
    model = TelluricModel(profile, grid, opacity, continuum=continuum_backend,
                          accuracy_mode=PHYSICS["accuracy_mode"],
                          max_lsf_sigma_kms=PHYSICS["max_lsf_sigma_kms"],
                          pixel_integration=PHYSICS["pixel_integration"])
    mark("continuum")

    spectrum = StellarSpectrum.flat(grid) if stellar is None else stellar
    source = prepare_stellar_source(
        spectrum, model, vsini_kms=settings.vsini_kms,
        limb_darkening=PHYSICS["limb_darkening"],
        macroturbulence_kms=PHYSICS["macroturbulence_kms"],
        normalize=PHYSICS["normalize_stellar_source"])
    mark("stellar_source")

    fit_model = (model.precompute_opacity(self_broadening=settings.self_broadening)
                 if settings.precompute_opacity else model)
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
        "number": number, "v1": v1, "v2": v2, "grid": grid, "model": model,
        "fit_model": fit_model, "source": source, "profile": profile,
        "stellar": spectrum if stellar is not None else None,
        "databases": databases, "absent": absent, "optical_depth": optical_depth,
        "free_species": sorted(s for s, t in optical_depth.items()
                               if t >= settings.min_optical_depth),
        "mt_ckd_version": continuum_backend.version,
        "setup_timing": timing,
        "setup_seconds": round(sum(timing.values()), 2),
    }


def fit_one(context, observation, settings: StandardFitSettings, objective, *, response=None,
            write_arrays=True, blaze=None, output_dir=None):
    """Fit one order of one frame, reusing whatever the context already built.

    ``response`` is a fractional instrument-response correction for this order,
    measured from the *other* frames of the night (see
    :func:`leave_one_out_patterns`). Dividing the flux and its uncertainty by
    ``1 + response`` is algebraically identical to multiplying the model by it,
    and needs no change to the forward model. ``blaze`` is a
    :class:`FlatBlaze` to divide out first. With ``output_dir`` the order's
    arrays are cached there as an npz.

    Returns the row and the objective -- the compiled one when it could be
    rebound onto this frame, a fresh one when it could not.
    """

    from tellurix import (
        ArrayOpacityBackend, OrderObjective, TelluricModel, TelluricParameters, SpectralOrder,
        chebyshev_continuum, fit_order, ils_fingerprint, resample_stellar_continuum,
    )

    rule = settings.order_rule
    timing, started = {}, time.time()

    def mark(name):
        nonlocal started
        timing[name] = round(time.time() - started, 2)
        started = time.time()

    number = context["number"]
    model, fit_model = context["model"], context["fit_model"]
    profile, grid = context["profile"], context["grid"]
    extracted = observation.order(number)
    level = continuum_level(extracted, rule["continuum_percentile"])
    order = igrins_spectral_order(
        extracted, source_flux_model_grid=context["source"],
        saturation_floor=rule["saturation_floor"],
        throughput_floor=rule["throughput_floor"],
        continuum_percentile=rule["continuum_percentile"],
        # With a flat source the hydrogen series is unmodelled and has to go.
        # With a real A0V model it is the thing being tested, so it stays.
        mask_hydrogen_kms=rule["mask_hydrogen_kms"] if settings.stellar == "flat" else None)
    if int(np.count_nonzero(order.mask)) < rule["minimum_pixels"]:
        raise RuntimeError(
            f"order {number} keeps {int(np.count_nonzero(order.mask))} pixels, "
            f"below the {rule['minimum_pixels']} this fit requires")
    blaze_pixels = None
    if blaze is not None:
        # The lamp's blaze, by detector column. Dividing it out leaves the
        # continuum only the lamp-to-star colour -- the steep order-end
        # roll-off a degree-9 polynomial cannot follow is most of what the
        # response pattern was absorbing (tellurix_igrins.flat).
        blaze_pixels = blaze.blaze_on(number, extracted.pixel)
        if blaze_pixels is None:
            raise RuntimeError(f"order {number} has no usable blaze in the {blaze.band} "
                               "flat (not traced, or the lamp's own absorption is too deep)")
        usable_blaze = np.isfinite(blaze_pixels) & (blaze_pixels > 0.02)
        blaze_pixels = np.where(usable_blaze, blaze_pixels, 1.0)
        order = SpectralOrder(
            order.wavelength_vacuum_nm, np.asarray(order.flux) / blaze_pixels,
            np.asarray(order.uncertainty) / blaze_pixels,
            mask=np.asarray(order.mask) & usable_blaze,
            zenith_angle_deg=order.zenith_angle_deg,
            source_flux_model_grid=order.source_flux_model_grid)
    if response is not None:
        scale = 1.0 + np.clip(np.asarray(response, dtype=float), -0.8, 5.0)
        order = SpectralOrder(
            order.wavelength_vacuum_nm, np.asarray(order.flux) / scale,
            np.asarray(order.uncertainty) / scale, mask=order.mask,
            zenith_angle_deg=order.zenith_angle_deg,
            source_flux_model_grid=order.source_flux_model_grid)
    mark("order")

    degree = settings.continuum_degree
    usable = np.asarray(order.flux)[np.asarray(order.mask)]
    parameters = TelluricParameters(
        log_column_scales={s: 0.0 for s in model.species},
        velocity_kms=0.0, wavelength_stretch=0.0,
        # R = 45,000 in FWHM is 6.66 km/s, so 2.83 km/s in sigma.
        lsf_sigma_kms=299792.458 / (settings.resolving_power * 2.3548200),
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
    schedule = stages_for(settings.stellar)
    for position, stage in enumerate(schedule):
        began = time.time()
        result = fit_order(
            fit_model, order, parameters,
            stage_bounds(stage, model.species, context["free_species"], parameters, degree,
                         fit_stellar=settings.stellar != "flat",
                         continuum_bound=rule["continuum_bound"]),
            objective=objective,
            # Only this call ever compiles the Hessian, and that compilation
            # costs 8.2 s against 3.1 ms to run it. The intermediate stages'
            # covariances are thrown away, so only the last stage asks.
            covariance=(position == len(schedule) - 1) and settings.covariance)
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
    _stellar_continuum = (None if context.get("stellar") is None
                          else resample_stellar_continuum(context["stellar"], nu))
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
    reliable = mask & (transmission >= settings.min_transmission)
    if int(reliable.sum()) < rule["minimum_reliable"]:
        # A band that is opaque end to end -- the 2.0 um CO2 band at high water,
        # say -- is a legitimate outcome, but it has nothing to report. Skipping
        # it keeps a NaN row out of the record.
        raise RuntimeError(
            f"order {number} leaves {int(reliable.sum())} pixels above a transmission of "
            f"{settings.min_transmission}; nothing to measure")
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

    # The fractional model error on the order's own axis, which is what the
    # night's other frames are averaged over to find the fixed pattern. Only
    # where the model is well above zero: in a saturated core the ratio is
    # meaningless and would dominate a median.
    native_model = np.asarray(exact.predict(order, parameters))
    native_continuum = np.asarray(chebyshev_continuum(
        parameters.continuum_coeffs, np.linspace(-1.0, 1.0, len(order.wavelength_vacuum_nm))))
    deep = native_model > 0.2 * native_continuum
    fractional = np.where(
        np.asarray(order.mask) & deep,
        (np.asarray(order.flux) - native_model) / np.maximum(native_model, 1e-12), np.nan)

    stem = observation.path.name.split(".")[0]
    if output_dir is not None and write_arrays:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            wavenumber_cm1=nu, observed=np.where(mask, observed, np.nan),
            model_flux=model_flux, transmission=transmission, corrected=corrected,
            # The A0V model's own continuum. This band carries the Brackett
            # bound-free edge at 1458.8 nm, a 5.6% step at 9500 K, which no
            # fitted polynomial can represent and which lands inside the two
            # bluest H orders.
            **({} if _stellar_continuum is None else {"stellar_continuum": _stellar_continuum}),
            stellar_only=star, continuum=continuum, residual=residual,
            uncertainty=sigma, mask=mask, reliable=reliable)
        if plp_telluric is not None:
            arrays["plp_telluric"] = plp_telluric
        if extracted.plp_continuum is not None:
            arrays["plp_continuum"] = np.asarray(extracted.plp_continuum)[axis]
        if response is not None:
            arrays["response_pattern"] = np.asarray(response)[axis]
        if blaze_pixels is not None:
            arrays["blaze"] = np.asarray(blaze_pixels)[axis]
        np.savez_compressed(output_dir / f"{stem}_{extracted.name}.npz", **arrays)

    names = list(objective.codec.names)
    deviation = (np.sqrt(np.clip(np.diag(result.covariance), 0.0, None))
                 if result.covariance is not None else np.zeros(len(names)))
    row = {
        "order": number,
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
        # Per pixel, residual over its own uncertainty. residual_rms_over_noise
        # divides by the *median* uncertainty, so it moves when anything
        # reweights the order -- dividing out a blaze raised it 8% in K on
        # DCT 2018 while this fell 10%. Compare runs with different continuum
        # models by this one.
        "residual_z_rms": (float(np.sqrt(np.mean((residual[reliable] / sigma[reliable]) ** 2)))
                           if reliable.any() else float("nan")),
        "median_transmission": float(np.median(transmission[mask])),
        "all_stages_converged": all(s["success"] for s in stages),
        "mt_ckd_version": context["mt_ckd_version"],
        "reused_compilation": reused,
        "stages": stages,
        "timing": timing,
        **comparison,
    }

    # The record is the run's product; the .npz above is a regenerable cache.
    # Keyed on frame and order, because that is what identifies a row here --
    # the atlas keyed on page and epoch.
    correlation = (result.correlation if result.correlation is not None
                   else np.zeros((len(names), len(names))))
    ils_velocity, ils_profile = ils_fingerprint(
        None, float(parameters.lsf_sigma_kms), model.velocity_step_kms)
    row["_record"] = {
        "frame": stem, "order": extracted.name,
        "page_sha256": observation.sha256["spec"],
        "log_column_scales": row["log_column_scales"],
        "continuum_coeffs": np.asarray(parameters.continuum_coeffs, dtype=float),
        "v1": context["v1"], "v2": context["v2"], "mopd_cm": 0.0,
        "pixels": row["pixels"], "grid_points": row["grid_points"],
        "reliable": row["reliable"],
        "velocity_kms": row["velocity_kms"],
        "stellar_velocity_kms": row["stellar_velocity_kms"],
        "wavelength_stretch": 0.0, "lsf_sigma_kms": row["lsf_sigma_kms"],
        "log_jitter": row["log_jitter"], "pixel_sigma": pixel_sigma,
        "residual_rms": residual_rms,
        "residual_rms_over_noise": row["residual_rms_over_noise"],
        "residual_z_rms": row["residual_z_rms"],
        "reduced_chi2": 0.0, "median_transmission": row["median_transmission"],
        "continuum_level": row["continuum_level_counts"], "continuum_level_pixels": 0,
        "condition_number": row["condition_number"] or 0.0,
        "all_stages_converged": row["all_stages_converged"],
        "negligible_telluric": not context["free_species"],
        "free_species": "+".join(context["free_species"]),
        "at_bound": "+".join(row["at_bound"]),
        "sigma": deviation, "correlation": correlation,
        "ils_velocity_kms": ils_velocity, "ils_profile": ils_profile,
        # Columns the atlas had no use for.
        "band": extracted.band, "order_number": number,
        "order_source": observation.order_source,
        "airmass": row["airmass"], "zenith_angle_deg": row["zenith_angle_deg"],
        "mjd": observation.mjd, "telescope": observation.telescope,
        "object": observation.object_name, "date_obs": observation.date_obs,
        "surface_temperature_k": observation.surface["temperature_k"] or 0.0,
        "surface_pressure_hpa": observation.surface["pressure_hpa"] or 0.0,
        "surface_humidity_percent": observation.surface["relative_humidity_percent"] or 0.0,
        "reused_compilation": reused,
    }
    row["_parameter_names"] = tuple(names)
    row["_species"] = tuple(model.species)
    row["_fractional_residual"] = fractional
    row["response_pattern_rms"] = (
        float(np.sqrt(np.nanmean(np.asarray(response)[np.asarray(order.mask)] ** 2)))
        if response is not None else 0.0)
    row["_record"]["response_pattern_rms"] = row["response_pattern_rms"]
    return row, objective


def run_provenance(driver: Path, paths: DataPaths, settings: StandardFitSettings, observations,
                   profile_path: Path) -> tuple[dict, dict]:
    """What produced this run, and the identity of everything that went into it.

    Paths alone are not provenance: a line list can be replaced under the same
    name. The hashes are what let a rebuild say whether it is looking at the
    same inputs. ``driver`` is the script that ran; it and this module are both
    hashed, because the driver delegates the fit here and a change in either
    changes the result.
    """

    import jax
    import tellurix
    from tellurix import file_sha256

    driver = Path(driver).resolve()
    fitter = Path(__file__).resolve()
    inputs = {
        "profile": str(profile_path), "profile_sha256": file_sha256(profile_path),
        "stellar": settings.stellar,
        "frames": [str(o.path) for o in observations],
        "frame_sha256": [o.sha256["spec"] for o in observations],
    }
    if settings.stellar != "flat" and Path(settings.stellar).exists():
        inputs["stellar_sha256"] = file_sha256(settings.stellar)
    if paths.mt_ckd.exists():
        inputs["mt_ckd"] = str(paths.mt_ckd)
        inputs["mt_ckd_sha256"] = file_sha256(paths.mt_ckd)
    inputs["aer_line_root"] = str(paths.line_root)
    for species, molecule_id in sorted(MOLECULE_IDS.items()):
        path = paths.line_file(species, molecule_id)
        if path.exists():
            inputs[f"aer_{species}_sha256"] = file_sha256(path)
    run = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "driver": driver.name, "driver_sha256": file_sha256(driver),
        "fitter": f"{fitter.parent.name}/{fitter.name}", "fitter_sha256": file_sha256(fitter),
        "tellurix": getattr(tellurix, "__version__", ""),
        "jax": jax.__version__,
    }
    return run, inputs
