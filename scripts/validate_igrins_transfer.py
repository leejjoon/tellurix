#!/usr/bin/env python
"""Can a night's standards set the telluric model for a frame they do not include?

The science use of this pipeline is a target with no standard of its own: its
telluric parameters have to come from the night's A0V standards, carried over to
the target's airmass and time, with as little as possible refitted on the target
itself -- the target's own lines are what would contaminate a refit.

This measures that transfer on the standards, where the answer is known. Each
standard in turn is held out; the others form the calibration; the held-out
frame is then fitted with progressively more freedom and compared with its own
full fit from the run's record:

    L0  columns, velocity and LSF from the calibration; only the continuum
    L1  + the velocity zero point, per order
    L2  + the water column, per order
    L3  + the LSF width, per order
    S1  one velocity shift for the whole frame, the median of L2's
    S2  S1 plus one water scale for the whole frame, the median of L2's

The shared levels are what a science frame could afford: two numbers per frame
constrained by every order, instead of per-order freedom that the target's own
lines can pull on.

How the calibration is built, per order, from the other frames only:

* dry columns (CO2, CH4, N2O, CO, O2) and the LSF -- the median. The model's
  columns are vertical and the slant path comes from the zenith angle, so a
  well-mixed gas needs no airmass interpolation;
* water -- linear in time between the bracketing standards, held flat outside
  them. Per order, not per frame: orders disagree about the water column by
  10-20% within a frame but each order's offset repeats to 1-2% across frames,
  so the per-order value carries that offset over for free;
* velocity -- the median;
* the instrument response -- the pattern the run already applied to this frame,
  which :func:`leave_one_out_patterns` built from the other frames.

The stellar velocity is freed at every level. It is the star's, not the sky's,
and its analogue for a science target is the target's own model.

    uv run python scripts/validate_igrins_transfer.py \\
        --run-dir /path/to/data/corrected/igrins/ladder_a0v \\
        --data-root /path/to/lblrtm --output docs/igrins_transfer_dct2018_h.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("L0", "L1", "L2", "L3")
SHARED = ("S1", "S2")
# With --clip-sigma: L2 and S2 again, on the pixels S2's fit does not put far
# below the model -- the unmodelled lines of a target, if it has any.
CLIPPED = ("L2c", "S2c")
# With --dry-shift: L2 plus the well-mixed columns free per order (L4), and the
# frame-wide medians of all four shifts (S3). A water scale cannot absorb a
# dry-gas slant-path error, which is what McDonald's high-airmass K frames show.
DRY = ("CO2", "CH4")
# S3t ties the two: one dry scale, CO2's, applied to both. McDonald shows them
# moving together (+0.081 and +0.080 on its airmass-3 frame), and CO2 is the one
# a line-rich target biases least -- CH4 shares 2.3 um with the CO bandheads.
DRY_LEVELS = ("L4", "S3", "S3t")


def load_driver(root: Path):
    """The fitting driver's order rule and context builder, so both agree."""

    spec = importlib.util.spec_from_file_location(
        "fit_igrins_standard", root / "scripts/fit_igrins_standard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_run(run_dir: Path, record: Path | None = None):
    import h5py

    record = record or run_dir / "record.h5"
    with h5py.File(record, "r") as handle:
        pages = handle["pages"][:]
        attrs = lambda group: {k: (v.decode() if isinstance(v, bytes) else v)  # noqa: E731
                               for k, v in handle[group].attrs.items()}
        inputs, config, physics = attrs("inputs"), attrs("config"), attrs("physics")
        species = [s.decode() for s in handle["species"][:]]
    text = lambda value: value.decode() if isinstance(value, bytes) else str(value)  # noqa: E731
    rows = {}
    for page in pages:
        if "order_number" not in page.dtype.names:
            raise SystemExit(f"{record} names orders by row position; run "
                             "scripts/migrate_igrins_order_names.py on it first")
        rows[(text(page["frame"]), int(page["order_number"]))] = page
    return rows, inputs, config, physics, species


def calibration(night, number, mjd):
    """This order's telluric parameters, from a night calibrated without the frame.

    ``night`` is the :class:`~tellurix_igrins.NightCalibration` a science frame would
    get, built with the held-out standard excluded -- so this tests the code a
    science fit runs, not a copy of it.
    """

    try:
        order = night.order(number)
    except KeyError:
        return None
    return {
        "columns": order.log_columns_at(mjd),
        "velocity_kms": order.velocity_kms,
        "lsf_sigma_kms": order.lsf_sigma_kms,
        "frames": len(order.frames),
    }


def injected_lines(spectrum, wavelength_nm, velocity_kms, sigma_kms, step_kms=0.25):
    """A second star's normalized spectrum as this order's pixels would see it.

    Multiplying the *observed* flux by an already-convolved spectrum is not the
    same as convolving the product, but the difference is the cross term the
    Arcturus run measured at a median 0.00016 against 0.0137 of noise. Pixel
    integration is left out for the same reason: it acts on both alike.
    """

    c_kms = 299792.458
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    # Rest-frame wavelengths the pixels see, with room for the kernel.
    reach = 1.0 + 8.0 * sigma_kms / c_kms
    low = wavelength_nm.min() / (1.0 + velocity_kms / c_kms) / reach
    high = wavelength_nm.max() / (1.0 + velocity_kms / c_kms) * reach
    source_nm = 1.0e7 / np.asarray(spectrum.wavenumber_cm1)
    order = np.argsort(source_nm)
    grid = np.exp(np.arange(np.log(low), np.log(high), step_kms / c_kms))
    flux = np.interp(grid, source_nm[order], np.asarray(spectrum.flux)[order])
    half = int(np.ceil(5.0 * sigma_kms / step_kms))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * step_kms / sigma_kms) ** 2)
    kernel /= kernel.sum()
    # Edge-padded, so the order's ends are not dragged toward zero.
    smoothed = np.convolve(np.pad(flux, half, mode="edge"), kernel, mode="valid")
    return np.interp(wavelength_nm / (1.0 + velocity_kms / c_kms), grid, smoothed)


def native_axis(sorted_values, wavelength_nm):
    """Undo the driver's sort into ascending wavenumber."""

    axis = np.argsort(1.0e7 / np.asarray(wavelength_nm))
    native = np.empty_like(np.asarray(sorted_values))
    native[axis] = sorted_values
    return native


# Every order carries enough water to free it, so splitting on that says
# nothing; how much absorption the order has is what separates easy from hard.
ABSORBING = 0.9


def summarize(selection):
    out = {"order_frames": len(selection)}
    if not selection:
        return out
    full = np.array([r["full"]["residual_rms_over_noise"] for r in selection])
    out["full_median_residual"] = float(np.median(full))
    for level in [k for k in LEVELS + SHARED + CLIPPED + DRY_LEVELS
                  if k in selection[0]["levels"]]:
        residual = np.array([r["levels"][level]["residual_rms_over_noise"] for r in selection])
        dt = np.array([r["levels"][level]["transmission_rms_difference"] for r in selection])
        noise = np.array([r["levels"][level]["noise_continuum_units"] for r in selection])
        out[level] = {
            "median_residual": float(np.median(residual)),
            "median_residual_ratio_to_full": float(np.median(residual / full)),
            "p90_residual_ratio_to_full": float(np.percentile(residual / full, 90)),
            "max_residual_ratio_to_full": float(np.max(residual / full)),
            "median_transmission_rms_difference": float(np.median(dt)),
            "median_transmission_difference_over_noise": float(np.median(dt / noise)),
        }
    return out


def rows_path(path: Path) -> Path:
    """Where a report's rows live: gitignored, beside the runs they came from."""

    return ROOT / "data/corrected/igrins/transfer" / f"{path.stem}_rows.json"


def write_report(path: Path, report: dict) -> None:
    """The summaries to ``path``, the rows beside the runs.

    The rows are what a later question will need, but at 2 KB each they are
    0.6 MB a night, so only the summaries are the committed report. Six decimal
    places, not six significant figures: the latter truncates an MJD to the day.
    """

    rows = report.pop("rows")
    absorbing = lambda r: r["full"]["median_transmission"] < ABSORBING  # noqa: E731
    report["summary"] = {
        "all": summarize(rows),
        "absorbing_orders": summarize([r for r in rows if absorbing(r)]),
        "clear_orders": summarize([r for r in rows if not absorbing(r)]),
        "absorbing_below_median_transmission": ABSORBING,
    }
    report["per_frame"] = {name: summarize([r for r in rows if r["frame"] == name])
                           for name in report["frames"]}

    def rounded(value):
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rounded(v) for v in value]
        return value

    report["rows_file"] = str(rows_path(path).relative_to(ROOT))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rounded(report), indent=1) + "\n")
    rows_path(path).parent.mkdir(parents=True, exist_ok=True)
    rows_path(path).write_text(
        "[\n" + ",\n".join(json.dumps(rounded(r), separators=(",", ":")) for r in rows) + "\n]\n")


def resummarize(path: Path, run_dir: Path, record: Path | None = None) -> None:
    """Rewrite a finished report's summaries without refitting anything."""

    report = json.loads(path.read_text())
    if "rows" not in report:
        report["rows"] = json.loads(rows_path(path).read_text())
    rows, *_ = read_run(run_dir, record)
    for entry in report["rows"]:
        page = rows[(entry["frame"], entry["order"])]
        entry["full"]["median_transmission"] = float(page["median_transmission"])
        report["frames"][entry["frame"]]["mjd"] = float(page["mjd"])
        # The frame table carries these once; early reports repeated them per row.
        entry.pop("object", None)
        entry.pop("airmass", None)
    write_report(path, report)
    print(json.dumps(report["summary"]["all"], indent=1))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="a fit_igrins_standard.py output: record.h5 plus the per-order npz "
                             "cache, which holds the response pattern each frame was given")
    parser.add_argument("--record", type=Path, default=None,
                        help="the run's record, when it is not RUN_DIR/record.h5 -- e.g. a "
                             "committed record beside a cache that lives in another checkout")
    parser.add_argument("--data-root", type=Path, default=root,
                        help="where the record's relative frame and stellar paths resolve")
    parser.add_argument("--profile", type=Path, default=None,
                        help="default: the record's own")
    parser.add_argument("--orders", default=None,
                        help="comma-separated physical echelle orders; default every order "
                             "in the record")
    parser.add_argument("--frames", default=None, help="comma-separated held-out frames")
    parser.add_argument("--inject", type=Path, default=None,
                        help="a stellar npz whose normalized spectrum is multiplied into every "
                             "held-out frame and left out of the model: a line-rich target's "
                             "lines, as contamination the transfer has to survive")
    parser.add_argument("--inject-velocity-kms", type=float, default=0.0)
    parser.add_argument("--clip-sigma", type=float, default=None,
                        help="after S2, drop pixels more than this many sigma BELOW the model "
                             "and measure the frame's shifts again (levels L2c, S2c). One-sided "
                             "because a target's unmodelled lines absorb.")
    parser.add_argument("--dry-shift", action="store_true",
                        help="also free CO2 and CH4 per order (L4) and apply their frame "
                             "medians with the velocity and water (S3)")
    parser.add_argument("--clip-pixels", type=int, default=2,
                        help="how far each clipped pixel's exclusion reaches either side: a "
                             "line's wings are below threshold but still pull")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resummarize", action="store_true",
                        help="rewrite --output's summaries from its rows and the run's record; "
                             "fits nothing")
    parser.add_argument("--compilation-cache", default=str(root / ".jax-cache"))
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if args.resummarize:
        resummarize(args.output, args.run_dir, args.record)
        return

    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.compilation_cache:
        import jax
        jax.config.update("jax_compilation_cache_dir", str(Path(args.compilation_cache)))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    driver = load_driver(root)
    from tellurix import (
        ArrayOpacityBackend, OrderObjective, SpectralOrder, StellarSpectrum, TelluricModel,
        TelluricParameters, fit_order, load_atmosphere_csv,
    )
    from tellurix_igrins import NightCalibration, igrins_spectral_order, read_igrins_observation

    rows, inputs, config, physics, run_species = read_run(args.run_dir, args.record)
    for key in ("continuum_bound", "continuum_percentile", "saturation_floor",
                "throughput_floor", "mask_hydrogen_kms", "minimum_pixels", "minimum_reliable"):
        if key in config:
            driver.ORDER[key] = config[key]
    stellar_path = config["stellar"]
    stellar = (None if stellar_path == "flat"
               else StellarSpectrum.from_npz(args.data_root / stellar_path))
    injected = None if args.inject is None else StellarSpectrum.from_npz(args.inject)
    profile_path = args.profile or Path(inputs["profile"])
    profile = load_atmosphere_csv(profile_path if profile_path.is_absolute()
                                  else root / profile_path)
    context_args = argparse.Namespace(
        resolving_power=config["resolving_power"],
        samples_per_resolution=config["samples_per_resolution"],
        margin_cm1=config["margin_cm1"], grid_margin_cm1=config["grid_margin_cm1"],
        vsini_kms=config["vsini_kms"], precompute_opacity=True, self_broadening="linear",
        min_optical_depth=config["min_optical_depth"], stellar=stellar_path)
    degree = int(config["continuum_degree"])

    observations = {}
    for relative in inputs["frames"]:
        observation = read_igrins_observation(args.data_root / relative)
        observations[observation.path.name.split(".")[0]] = observation
    mjd = {name: o.mjd for name, o in observations.items()}
    frames = sorted(observations, key=mjd.get)
    held = frames if args.frames is None else args.frames.split(",")
    # The response pattern stays the one the run gave each frame -- built by
    # leave_one_out_patterns from the others -- so only the parameters come
    # from here; see the module docstring.
    nights = {name: NightCalibration.from_run(args.record or args.run_dir / "record.h5", None,
                                              exclude=[name])
              for name in held}
    orders = (sorted({o for _, o in rows}) if args.orders is None
              else [int(v) for v in args.orders.split(",")])

    def order_for(context, observation, name, extracted, inject_sigma_kms=None):
        order = igrins_spectral_order(
            extracted, source_flux_model_grid=context["source"],
            saturation_floor=driver.ORDER["saturation_floor"],
            throughput_floor=driver.ORDER["throughput_floor"],
            continuum_percentile=driver.ORDER["continuum_percentile"],
            mask_hydrogen_kms=driver.ORDER["mask_hydrogen_kms"] if stellar is None else None)
        cached = np.load(args.run_dir / f"{name}_{extracted.name}.npz")
        if "response_pattern" in cached.files:
            response = native_axis(cached["response_pattern"], order.wavelength_vacuum_nm)
            scale = 1.0 + np.clip(response, -0.8, 5.0)
            order = SpectralOrder(
                order.wavelength_vacuum_nm, np.asarray(order.flux) / scale,
                np.asarray(order.uncertainty) / scale, mask=order.mask,
                zenith_angle_deg=order.zenith_angle_deg,
                source_flux_model_grid=order.source_flux_model_grid)
        if injected is not None:
            lines = injected_lines(injected, order.wavelength_vacuum_nm,
                                   args.inject_velocity_kms, inject_sigma_kms)
            # The uncertainty scales with the flux, as a flat field would; a
            # real star's lines would also lower the photon noise, but that is
            # second order in a test of bias.
            order = SpectralOrder(
                order.wavelength_vacuum_nm, np.asarray(order.flux) * lines,
                np.asarray(order.uncertainty) * lines, mask=order.mask,
                zenith_angle_deg=order.zenith_angle_deg,
                source_flux_model_grid=order.source_flux_model_grid)
        return order, cached

    def bounds_for(model, parameters, free):
        pinned = lambda value: (float(value), float(value))  # noqa: E731
        bounds = {s: ((-2.0, 2.0) if s in free else pinned(parameters.log_column_scales[s]))
                  for s in model.species}
        bounds["velocity_kms"] = ((-8.0, 8.0) if "velocity_kms" in free
                                  else pinned(parameters.velocity_kms))
        bounds["wavelength_stretch"] = (0.0, 0.0)
        bounds["lsf_sigma_kms"] = ((1.0, 6.0) if "lsf_sigma_kms" in free
                                   else pinned(parameters.lsf_sigma_kms))
        bound = driver.ORDER["continuum_bound"]
        for index in range(degree + 1):
            bounds[f"continuum_{index}"] = (-2.0, 2.0) if index == 0 else (-bound, bound)
        bounds["log_jitter"] = (np.log(1e-5), np.log(0.5))
        bounds["stellar_velocity_kms"] = ((-60.0, 60.0) if stellar is not None
                                          else pinned(parameters.stellar_velocity_kms))
        return bounds

    def measure(context, order, cached, parameters, star_model):
        """Residual and correction operator, on the full fit's own pixels."""

        exact = context["model"].precompute_opacity(parameters)
        axis = np.argsort(1.0e7 / np.asarray(order.wavelength_vacuum_nm))
        model_flux = np.asarray(exact.predict(order, parameters))[axis]
        star = np.asarray(star_model.predict(order, parameters))[axis]
        observed = np.asarray(order.flux)[axis]
        reliable = cached["reliable"].astype(bool)
        mask = cached["mask"].astype(bool)
        sigma = np.asarray(order.uncertainty)[axis]
        pixel_sigma = float(np.median(sigma[mask]))
        residual = observed - model_flux
        effective = model_flux / star
        reference = cached["model_flux"] / cached["stellar_only"]
        difference = (effective - reference)[reliable]
        return {
            "residual_rms_over_noise": float(
                np.sqrt(np.mean(residual[reliable] ** 2)) / pixel_sigma),
            # The correction operator against the full fit's, in continuum
            # units, beside the noise in the same units -- what a science user
            # would inherit.
            "transmission_rms_difference": float(np.sqrt(np.mean(difference**2))),
            "transmission_p99_difference": float(np.percentile(np.abs(difference), 99)),
            "noise_continuum_units": float(np.median((sigma / star)[reliable])),
        }

    def fit(context, objective, order, parameters, free, star_model, cached):
        began = time.time()
        result = fit_order(context["fit_model"], order, parameters,
                           bounds_for(context["model"], parameters, free),
                           objective=objective, covariance=False)
        out = measure(context, order, cached, result.parameters, star_model)
        out.update({
            "velocity_kms": float(result.parameters.velocity_kms),
            "lsf_sigma_kms": float(result.parameters.lsf_sigma_kms),
            "log_column_H2O": float(result.parameters.log_column_scales.get("H2O", 0.0)),
            **{f"log_column_{s}": float(result.parameters.log_column_scales.get(s, 0.0))
               for s in DRY},
            "stellar_velocity_kms": float(result.parameters.stellar_velocity_kms),
            "converged": bool(result.success), "seconds": round(time.time() - began, 2),
        })
        return result.parameters, out

    results = []
    retained = {}
    started = time.time()
    for number in orders:
        first = observations[frames[0]]
        try:
            context = driver.build_order_context(first, number, context_args, args.data_root, profile,
                                                 stellar)
        except (RuntimeError, ValueError) as exc:
            print(f"order {number}: skipped ({exc})")
            continue
        model = context["model"]
        star_model = TelluricModel(
            profile, context["grid"],
            ArrayOpacityBackend({s: np.zeros((len(profile.temperature_k), context["grid"].size))
                                 for s in model.species}),
            accuracy_mode="fast", max_lsf_sigma_kms=driver.PHYSICS["max_lsf_sigma_kms"],
            pixel_integration=driver.PHYSICS["pixel_integration"])
        objective = None
        water_free = "H2O" in context["free_species"]
        for name in held:
            if (name, number) not in rows:
                continue
            reference = rows[(name, number)]
            cal = calibration(nights[name], number, mjd[name])
            if cal is None:
                continue
            observation = observations[name]
            extracted = observation.order(number)
            order, cached = order_for(context, observation, name, extracted,
                                      inject_sigma_kms=cal["lsf_sigma_kms"])
            if objective is None:
                objective = OrderObjective(context["fit_model"], order, degree + 1)
            else:
                objective = objective.rebind(order)
            usable = np.asarray(order.flux)[np.asarray(order.mask)]
            parameters = TelluricParameters(
                log_column_scales={s: float(cal["columns"].get(s, 0.0)) for s in model.species},
                velocity_kms=cal["velocity_kms"], wavelength_stretch=0.0,
                lsf_sigma_kms=cal["lsf_sigma_kms"],
                continuum_coeffs=np.concatenate([[float(np.log(np.median(usable)))],
                                                 np.zeros(degree)]),
                log_jitter=float(np.log(np.median(
                    np.asarray(order.uncertainty)[np.asarray(order.mask)]))),
                stellar_velocity_kms=float(reference["stellar_velocity_kms"]))

            entry = {
                "frame": name, "order": number,
                "hours": round(24.0 * (mjd[name] - mjd[frames[0]]), 3),
                "water_free": water_free,
                "free_species": context["free_species"],
                "calibration": cal,
                "full": {
                    "residual_rms_over_noise": float(reference["residual_rms_over_noise"]),
                    "velocity_kms": float(reference["velocity_kms"]),
                    "lsf_sigma_kms": float(reference["lsf_sigma_kms"]),
                    "log_column_H2O": float(reference["log_column_H2O"]),
                    "median_transmission": float(reference["median_transmission"]),
                },
                "levels": {},
            }
            frees = {
                "L0": set(),
                "L1": {"velocity_kms"},
                "L2": {"velocity_kms"} | ({"H2O"} if water_free else set()),
                "L3": {"velocity_kms", "lsf_sigma_kms"} | ({"H2O"} if water_free else set()),
            }
            current = parameters
            fitted = {}
            for level in LEVELS:
                current, entry["levels"][level] = fit(
                    context, objective, order, current, frees[level], star_model, cached)
                fitted[level] = current
            entry["dry_free"] = [s for s in DRY if s in context["free_species"]]
            if args.dry_shift:
                # From L2, not L3: the LSF stays the calibration's.
                _, entry["levels"]["L4"] = fit(
                    context, objective, order, fitted["L2"],
                    frees["L2"] | set(entry["dry_free"]), star_model, cached)
            results.append(entry)
            retained[(name, number)] = (order, cached, parameters)
            L = entry["levels"]
            print(f"  {name}  order {number:3d}  "
                  f"full {entry['full']['residual_rms_over_noise']:5.2f}  "
                  + "  ".join(f"{k} {L[k]['residual_rms_over_noise']:5.2f}" for k in LEVELS)
                  + f"  dT(L2) {L['L2']['transmission_rms_difference']:.4f}", flush=True)

        # The shared levels need every order of a frame before any of them, so
        # they run as a second pass over this order once the frame-level
        # numbers exist -- which they do not yet. Keep the context instead of
        # rebuilding it: see below.
        retained[("context", number)] = (context, star_model, objective)

    def frame_shifts(level):
        """Median over a frame's orders of how far ``level`` moved each from its calibration."""

        shifts = {}
        for name in held:
            mine = [r for r in results if r["frame"] == name and level in r["levels"]]
            if not mine:
                continue
            dv = [r["levels"][level]["velocity_kms"] - r["calibration"]["velocity_kms"]
                  for r in mine]
            dw = [r["levels"][level]["log_column_H2O"] - r["calibration"]["columns"]["H2O"]
                  for r in mine if r["water_free"]]
            shifts[name] = {"velocity_kms": float(np.median(dv)),
                            "log_column_H2O": float(np.median(dw)) if dw else 0.0,
                            "orders": len(mine), "water_orders": len(dw)}
            for species in DRY:
                d = [r["levels"][level][f"log_column_{species}"]
                     - r["calibration"]["columns"][species]
                     for r in mine if species in r.get("dry_free", ())]
                shifts[name][f"log_column_{species}"] = float(np.median(d)) if d else 0.0
                shifts[name][f"{species}_orders"] = len(d)
        return shifts

    def shifted(parameters, shift, water_free, dry=()):
        columns = dict(parameters.log_column_scales)
        if water_free:
            columns["H2O"] = columns["H2O"] + shift["log_column_H2O"]
        for species in dry:
            columns[species] = columns[species] + shift[f"log_column_{species}"]
        return parameters._replace(
            velocity_kms=parameters.velocity_kms + shift["velocity_kms"],
            log_column_scales=columns)

    shared = frame_shifts("L2")
    fitted_s2 = {}
    for entry in results:
        name, number = entry["frame"], entry["order"]
        context, star_model, objective = retained[("context", number)]
        order, cached, parameters = retained[(name, number)]
        objective = objective.rebind(order)
        velocity_only = dict(shared[name], log_column_H2O=0.0)
        _, entry["levels"]["S1"] = fit(context, objective, order,
                                       shifted(parameters, velocity_only, False), set(),
                                       star_model, cached)
        fitted_s2[(name, number)], entry["levels"]["S2"] = fit(
            context, objective, order, shifted(parameters, shared[name], entry["water_free"]),
            set(), star_model, cached)

    dry_shared = None
    if args.dry_shift:
        dry_shared = frame_shifts("L4")
        for entry in results:
            name, number = entry["frame"], entry["order"]
            context, star_model, objective = retained[("context", number)]
            order, cached, parameters = retained[(name, number)]
            objective = objective.rebind(order)
            _, entry["levels"]["S3"] = fit(
                context, objective, order,
                shifted(parameters, dry_shared[name], entry["water_free"], entry["dry_free"]),
                set(), star_model, cached)
            tied = dict(dry_shared[name], log_column_CH4=dry_shared[name]["log_column_CO2"])
            _, entry["levels"]["S3t"] = fit(
                context, objective, order,
                shifted(parameters, tied, entry["water_free"], entry["dry_free"]),
                set(), star_model, cached)

    clipped_shared = None
    if args.clip_sigma is not None:
        clipped_orders = {}
        width = 2 * args.clip_pixels + 1
        for entry in results:
            name, number = entry["frame"], entry["order"]
            context, star_model, objective = retained[("context", number)]
            order, cached, parameters = retained[(name, number)]
            best = fitted_s2[(name, number)]
            prediction = np.asarray(context["fit_model"].predict(order, best))
            sigma = np.sqrt(np.asarray(order.uncertainty) ** 2
                            + np.exp(2.0 * float(best.log_jitter)))
            mask = np.asarray(order.mask)
            low = mask & ((np.asarray(order.flux) - prediction) / sigma < -args.clip_sigma)
            reach = np.convolve(low.astype(float), np.ones(width), mode="same") > 0
            clipped = SpectralOrder(
                order.wavelength_vacuum_nm, order.flux, order.uncertainty,
                mask=mask & ~reach, zenith_angle_deg=order.zenith_angle_deg,
                source_flux_model_grid=order.source_flux_model_grid)
            entry["clipped_fraction"] = float(np.count_nonzero(mask & reach)
                                              / max(np.count_nonzero(mask), 1))
            clipped_orders[(name, number)] = clipped
            objective = objective.rebind(clipped)
            # From S2's answer, not the calibration: the point is to re-measure
            # the frame on cleaner pixels, starting where it already stands.
            free = {"velocity_kms"} | ({"H2O"} if entry["water_free"] else set())
            _, entry["levels"]["L2c"] = fit(context, objective, clipped, best, free,
                                            star_model, cached)
        clipped_shared = frame_shifts("L2c")
        for entry in results:
            name, number = entry["frame"], entry["order"]
            context, star_model, objective = retained[("context", number)]
            order, cached, parameters = retained[(name, number)]
            clipped = clipped_orders[(name, number)]
            objective = objective.rebind(clipped)
            _, entry["levels"]["S2c"] = fit(
                context, objective, clipped,
                shifted(parameters, clipped_shared[name], entry["water_free"]),
                set(), star_model, cached)

    report = {
        "description": "Leave-one-standard-out transfer of telluric parameters to a held-out "
                       "IGRINS frame; see scripts/validate_igrins_transfer.py",
        "run_dir": str(args.run_dir), "profile": str(profile_path), "stellar": stellar_path,
        "injected": None if injected is None else {
            "source": str(args.inject), "velocity_kms": args.inject_velocity_kms},
        "frames": {f: {"object": observations[f].object_name, "mjd": mjd[f],
                       "airmass": float(1.0 / np.cos(np.radians(
                           observations[f].zenith_angle_deg)))} for f in frames},
        "shared_shifts": shared,
        "clipped_shifts": clipped_shared,
        "dry_shifts": dry_shared,
        "clip": None if args.clip_sigma is None else {
            "sigma": args.clip_sigma, "pixels": args.clip_pixels},
        "rows": results,
        "seconds": round(time.time() - started, 1),
    }
    write_report(args.output, report)
    print(json.dumps(report["summary"]["all"], indent=1))
    print(f"wrote {args.output} in {report['seconds']:.0f} s")


if __name__ == "__main__":
    main()
