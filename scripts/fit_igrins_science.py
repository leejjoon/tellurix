#!/usr/bin/env python
"""Correct IGRINS science frames with a night's telluric calibration.

A science target has no standard of its own. Its telluric model comes from the
night's A0V standards (``build_igrins_calibration.py``), carried to its airmass
and time, and only what the standards cannot know is fitted on the frame
itself -- because the target's own lines are what a refit would absorb. What to
fit was measured by holding standards out (``docs/igrins_transfer.md``):

* per order, from the calibration: the dry columns, the LSF, the velocity zero
  point, water interpolated in time, and the instrument response pattern;
* per **frame**, fitted: one velocity shift and one water scale, each the median
  over orders of a per-order fit. Per-order water is *not* kept -- with a
  line-rich target's lines in the data it lets each order's water chase that
  order's stellar lines, and ends worse than the frame's single scale;
* per order, fitted: the continuum, which carries the target's SED and the blaze.

``--clip-sigma`` measures the frame's shifts a second time without the pixels
the first fit leaves far *below* the model -- a target's unmodelled lines. It
brought the water bias from an injected K giant back to the clean level in H
and did nothing in K, where the CO blends and a weak-line forest never cross
the threshold.

When both bands of an exposure are fitted, their water shifts are compared.
Clean, they agree to about 0.02 in log column; a line-rich target biased them in
opposite directions, about 3% apart. A large disagreement is reported, not
resolved: it says the stellar lines are in the water scale.

    uv run python scripts/fit_igrins_science.py \\
        --spec data/igrins/20181220_0059/SDC?_20181220_0059.spec.fits \\
        --calibration data/calibration/igrins_dct2018_h.h5 data/calibration/igrins_dct2018_k.h5

Products, per frame and order, in ``--output-dir``: ``<frame>_<order>.npz`` with
``corrected`` = ``observed / effective_transmission``, where the effective
transmission is ``model_flux / stellar_only`` -- the *convolved* operator, not
the unconvolved ``transmission`` saved beside it. The record and one summary
per frame are written alongside.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Clean H and K agree on the frame's water shift to 0.018-0.021 rms; a K giant's
# lines pushed them 0.033 apart. Twice the clean scatter separates the two.
BAND_DISAGREEMENT = 0.04


def load_driver(root: Path):
    """The standards driver's order rule and context builder, so both agree."""

    spec = importlib.util.spec_from_file_location(
        "fit_igrins_standard", root / "scripts/fit_igrins_standard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exposure(observation) -> str:
    """Date and frame number, which the H and K files of one exposure share."""
    return observation.path.name.split(".")[0][5:]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, nargs="+", required=True,
                        help="science frames, H and/or K, of the calibration's night")
    parser.add_argument("--calibration", type=Path, nargs="+", required=True,
                        help="one NightCalibration per band")
    parser.add_argument("--orders", default=None,
                        help="comma-separated physical echelle orders; default every order "
                             "both the frame and the calibration have")
    parser.add_argument("--profile", type=Path, default=None,
                        help="default: the profile the calibration's standards were fitted with")
    parser.add_argument("--data-root", type=Path, default=root,
                        help="where the line files (data/lblrtm) are")
    parser.add_argument("--stellar", default="flat",
                        help="'flat', or a stellar npz for the target -- which turns a science "
                             "frame into the A0V case and removes the line-rich floor")
    parser.add_argument("--vsini-kms", type=float, default=0.0)
    parser.add_argument("--stellar-velocity-kms", type=float, default=0.0,
                        help="the target's starting velocity when --stellar is a model")
    parser.add_argument("--clip-sigma", type=float, default=None)
    parser.add_argument("--clip-pixels", type=int, default=2)
    parser.add_argument("--min-transmission", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=root / "data/corrected/igrins/science")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--compilation-cache", default=str(root / ".jax-cache"))
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()

    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.compilation_cache:
        import jax
        jax.config.update("jax_compilation_cache_dir", str(Path(args.compilation_cache)))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    driver = load_driver(root)
    from tellurix import (
        ArrayOpacityBackend, NightCalibration, OrderObjective, SpectralOrder, StellarSpectrum,
        TelluricModel, TelluricParameters, chebyshev_continuum, fit_order,
        igrins_spectral_order, load_atmosphere_csv, read_igrins_observation,
    )

    calibrations = {}
    for path in args.calibration:
        calibration = NightCalibration.load(path)
        calibrations[calibration.band] = (calibration, path)
    stellar = None if args.stellar == "flat" else StellarSpectrum.from_npz(args.stellar)
    observations = [read_igrins_observation(path) for path in args.spec]
    bands = defaultdict(list)
    for observation in observations:
        if observation.band not in calibrations:
            raise SystemExit(f"{observation.path.name}: no {observation.band} calibration given")
        bands[observation.band].append(observation)

    started = time.time()
    frame_reports = {}
    record_rows = defaultdict(list)
    for band, frames in sorted(bands.items()):
        calibration, calibration_path = calibrations[band]
        config = calibration.source["config"]
        for key in driver.ORDER:
            if key in config:
                driver.ORDER[key] = config[key]
        profile_path = args.profile or Path(calibration.source["inputs"]["profile"])
        if not profile_path.exists():
            profile_path = root / "data/profiles" / profile_path.name
        profile = load_atmosphere_csv(profile_path)
        context_args = argparse.Namespace(
            resolving_power=config["resolving_power"],
            samples_per_resolution=config["samples_per_resolution"],
            margin_cm1=config["margin_cm1"], grid_margin_cm1=config["grid_margin_cm1"],
            vsini_kms=args.vsini_kms, precompute_opacity=True, self_broadening="linear",
            min_optical_depth=config["min_optical_depth"], stellar=args.stellar)
        degree = int(config["continuum_degree"])
        numbers = (sorted(set(calibration.orders) & set().union(*(o.orders for o in frames)))
                   if args.orders is None else [int(v) for v in args.orders.split(",")])
        print(f"{band}: {len(frames)} frame(s), orders {numbers[0]}-{numbers[-1]}, "
              f"calibration {calibration_path.name} "
              f"({len({f for o in calibration.orders.values() for f in o.frames})} standards)")

        def bounds_for(model, parameters, free):
            pinned = lambda value: (float(value), float(value))  # noqa: E731
            bounds = {s: ((-2.0, 2.0) if s in free else pinned(parameters.log_column_scales[s]))
                      for s in model.species}
            bounds["velocity_kms"] = ((-8.0, 8.0) if "velocity_kms" in free
                                      else pinned(parameters.velocity_kms))
            bounds["wavelength_stretch"] = (0.0, 0.0)
            bounds["lsf_sigma_kms"] = pinned(parameters.lsf_sigma_kms)
            bound = driver.ORDER["continuum_bound"]
            for index in range(degree + 1):
                bounds[f"continuum_{index}"] = (-2.0, 2.0) if index == 0 else (-bound, bound)
            bounds["log_jitter"] = (np.log(1e-5), np.log(0.5))
            v0 = args.stellar_velocity_kms
            bounds["stellar_velocity_kms"] = ((v0 - 50.0, v0 + 50.0) if stellar is not None
                                              else pinned(parameters.stellar_velocity_kms))
            return bounds

        def fit(context, objective, order, parameters, free):
            result = fit_order(context["fit_model"], order, parameters,
                               bounds_for(context["model"], parameters, free),
                               objective=objective, covariance=False)
            return result

        work = {}
        contexts = {}
        for number in numbers:
            ocal = calibration.order(number)
            holders = [o for o in frames if number in o.orders]
            if not holders:
                continue
            try:
                context = driver.build_order_context(holders[0], number, context_args,
                                                     args.data_root, profile, stellar)
            except (RuntimeError, ValueError) as exc:
                print(f"  {band}{number}: skipped ({exc})")
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
            for observation in holders:
                extracted = observation.order(number)
                try:
                    order = igrins_spectral_order(
                        extracted, source_flux_model_grid=context["source"],
                        saturation_floor=driver.ORDER["saturation_floor"],
                        throughput_floor=driver.ORDER["throughput_floor"],
                        continuum_percentile=driver.ORDER["continuum_percentile"],
                        # The target is not an A0V: there is no hydrogen series
                        # to mask, and masking its own lines is --clip-sigma's job.
                        mask_hydrogen_kms=None)
                except ValueError as exc:
                    print(f"  {band}{number} {exposure(observation)}: skipped ({exc})")
                    continue
                pattern = ocal.pattern_on(order.wavelength_vacuum_nm)
                scale = 1.0 + np.clip(pattern, -0.8, 5.0)
                order = SpectralOrder(
                    order.wavelength_vacuum_nm, np.asarray(order.flux) / scale,
                    np.asarray(order.uncertainty) / scale, mask=order.mask,
                    zenith_angle_deg=order.zenith_angle_deg,
                    source_flux_model_grid=order.source_flux_model_grid)
                if objective is None:
                    objective = OrderObjective(context["fit_model"], order, degree + 1)
                else:
                    try:
                        objective = objective.rebind(order)
                    except ValueError:
                        objective = OrderObjective(context["fit_model"], order, degree + 1)
                columns = ocal.log_columns_at(observation.mjd)
                usable = np.asarray(order.flux)[np.asarray(order.mask)]
                start = TelluricParameters(
                    log_column_scales={s: float(columns.get(s, 0.0)) for s in model.species},
                    velocity_kms=ocal.velocity_kms, wavelength_stretch=0.0,
                    lsf_sigma_kms=ocal.lsf_sigma_kms,
                    continuum_coeffs=np.concatenate([[float(np.log(np.median(usable)))],
                                                     np.zeros(degree)]),
                    log_jitter=float(np.log(np.median(
                        np.asarray(order.uncertainty)[np.asarray(order.mask)]))),
                    stellar_velocity_kms=args.stellar_velocity_kms)
                # Continuum first, then the shifts, from the calibration: the
                # sequence the transfer test measured.
                first = fit(context, objective, order, start, set()).parameters
                free = {"velocity_kms"} | ({"H2O"} if water_free else set())
                per_order = fit(context, objective, order, first, free).parameters
                work[(exposure(observation), number)] = {
                    "observation": observation, "order": order, "start": start,
                    "per_order": per_order, "water_free": water_free, "pattern": pattern,
                    "objective": objective, "extracted": extracted,
                }
            contexts[number] = (context, star_model)
            print(f"  {band}{number}: {len(holders)} frame(s) fitted per order", flush=True)

        def frame_shifts(level):
            shifts = {}
            for key in {k[0] for k in work}:
                mine = [w for (e, _), w in work.items() if e == key and level in w]
                dv = [float(w[level].velocity_kms - w["start"].velocity_kms) for w in mine]
                dw = [float(w[level].log_column_scales["H2O"] - w["start"].log_column_scales["H2O"])
                      for w in mine if w["water_free"]]
                shifts[key] = {"velocity_kms": float(np.median(dv)) if dv else 0.0,
                               "log_column_H2O": float(np.median(dw)) if dw else 0.0,
                               "orders": len(mine), "water_orders": len(dw)}
            return shifts

        def shifted(parameters, shift, water_free):
            columns = dict(parameters.log_column_scales)
            if water_free:
                columns["H2O"] = columns["H2O"] + shift["log_column_H2O"]
            return parameters._replace(
                velocity_kms=parameters.velocity_kms + shift["velocity_kms"],
                log_column_scales=columns)

        def frame_level(level, source_level, order_key="order"):
            shifts = frame_shifts(source_level)
            for (key, number), w in work.items():
                context, _ = contexts[number]
                objective = w["objective"].rebind(w[order_key])
                w[level] = fit(context, objective, w[order_key],
                               shifted(w["start"], shifts[key], w["water_free"]), set()).parameters
            return shifts

        shifts = frame_level("frame", "per_order")
        clipped_shifts = None
        if args.clip_sigma is not None:
            width = 2 * args.clip_pixels + 1
            for (key, number), w in work.items():
                context, _ = contexts[number]
                order, best = w["order"], w["frame"]
                prediction = np.asarray(context["fit_model"].predict(order, best))
                sigma = np.sqrt(np.asarray(order.uncertainty) ** 2
                                + np.exp(2.0 * float(best.log_jitter)))
                mask = np.asarray(order.mask)
                low = mask & ((np.asarray(order.flux) - prediction) / sigma < -args.clip_sigma)
                reach = np.convolve(low.astype(float), np.ones(width), mode="same") > 0
                w["clipped"] = SpectralOrder(
                    order.wavelength_vacuum_nm, order.flux, order.uncertainty,
                    mask=mask & ~reach, zenith_angle_deg=order.zenith_angle_deg,
                    source_flux_model_grid=order.source_flux_model_grid)
                w["clipped_fraction"] = float(np.count_nonzero(mask & reach)
                                              / max(np.count_nonzero(mask), 1))
                objective = w["objective"].rebind(w["clipped"])
                free = {"velocity_kms"} | ({"H2O"} if w["water_free"] else set())
                w["per_order_clipped"] = fit(context, objective, w["clipped"], best, free).parameters
            clipped_shifts = frame_level("frame_clipped", "per_order_clipped", "clipped")
        final_level = "frame_clipped" if args.clip_sigma is not None else "frame"

        # Products, from the final frame-level parameters, on all the frame's
        # pixels -- clipping chooses the shifts, not what gets corrected.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rows_by_frame = defaultdict(list)
        for (key, number), w in sorted(work.items()):
            context, star_model = contexts[number]
            observation, order, parameters = w["observation"], w["order"], w[final_level]
            exact = context["model"].precompute_opacity(parameters)
            axis = np.argsort(1.0e7 / np.asarray(order.wavelength_vacuum_nm))
            nu = (1.0e7 / np.asarray(order.wavelength_vacuum_nm))[axis]
            model_flux = np.asarray(exact.predict(order, parameters))[axis]
            star = np.asarray(star_model.predict(order, parameters))[axis]
            effective = model_flux / star
            observed = np.asarray(order.flux)[axis]
            sigma = np.asarray(order.uncertainty)[axis]
            mask = np.asarray(order.mask)[axis]
            transmission = np.interp(nu, np.asarray(context["model"].wavenumber_cm1),
                                     np.asarray(exact.transmission(parameters,
                                                                   order.zenith_angle_deg)))
            continuum = np.asarray(chebyshev_continuum(
                parameters.continuum_coeffs,
                np.linspace(-1.0, 1.0, len(order.wavelength_vacuum_nm))))[axis]
            reliable = mask & (transmission >= args.min_transmission)
            residual = observed - model_flux
            pixel_sigma = float(np.median(sigma[mask]))
            name = w["extracted"].name
            stem = observation.path.name.split(".")[0]
            np.savez_compressed(
                args.output_dir / f"{stem}_{name}.npz",
                wavenumber_cm1=nu, observed=np.where(mask, observed, np.nan),
                uncertainty=sigma, mask=mask, reliable=reliable,
                model_flux=model_flux, stellar_only=star,
                effective_transmission=effective, transmission=transmission,
                corrected=observed / np.maximum(effective, 1e-6),
                corrected_uncertainty=sigma / np.maximum(effective, 1e-6),
                continuum=continuum, response_pattern=w["pattern"][axis])
            row = {
                "order": number, "name": name,
                "reliable": int(reliable.sum()), "pixels": int(mask.size),
                "median_transmission": float(np.median(transmission[mask])),
                "residual_rms_over_noise": float(
                    np.sqrt(np.mean(residual[reliable] ** 2)) / pixel_sigma)
                if reliable.any() else float("nan"),
                "velocity_kms": float(parameters.velocity_kms),
                "log_column_H2O": float(parameters.log_column_scales.get("H2O", 0.0)),
                "log_column_H2O_interpolated": float(w["start"].log_column_scales.get("H2O", 0.0)),
                "log_column_H2O_per_order": float(w["per_order"].log_column_scales.get("H2O", 0.0)),
                "water_free": w["water_free"],
                "clipped_fraction": w.get("clipped_fraction"),
            }
            rows_by_frame[key].append(row)
            record_rows[band].append({
                "frame": stem, "order": name, "page_sha256": observation.sha256["spec"],
                "log_column_scales": {s: float(v) for s, v in parameters.log_column_scales.items()},
                "continuum_coeffs": np.asarray(parameters.continuum_coeffs, dtype=float),
                "v1": float(nu.min()), "v2": float(nu.max()), "pixels": row["pixels"],
                "grid_points": int(context["grid"].size), "reliable": row["reliable"],
                "velocity_kms": row["velocity_kms"],
                "stellar_velocity_kms": float(parameters.stellar_velocity_kms),
                "lsf_sigma_kms": float(parameters.lsf_sigma_kms),
                "log_jitter": float(parameters.log_jitter), "pixel_sigma": pixel_sigma,
                "residual_rms": float(np.sqrt(np.mean(residual[reliable] ** 2)))
                if reliable.any() else float("nan"),
                "residual_rms_over_noise": row["residual_rms_over_noise"],
                "median_transmission": row["median_transmission"],
                "negligible_telluric": not context["free_species"],
                "free_species": "+".join(context["free_species"]),
                "band": band, "order_number": number,
                "order_source": observation.order_source,
                "airmass": float(1.0 / np.cos(np.radians(observation.zenith_angle_deg))),
                "zenith_angle_deg": float(observation.zenith_angle_deg),
                "mjd": observation.mjd, "object": observation.object_name,
                "water_shift": (clipped_shifts or shifts)[key]["log_column_H2O"],
                "velocity_shift": (clipped_shifts or shifts)[key]["velocity_kms"],
            })

        for key, rows in rows_by_frame.items():
            observation = next(w["observation"] for (k, _), w in work.items() if k == key)
            frame_reports[(key, band)] = {
                "observation": {
                    "path": str(observation.path), "object": observation.object_name,
                    "object_type": observation.object_type, "band": band,
                    "date_obs": observation.date_obs, "mjd": observation.mjd,
                    "airmass": float(1.0 / np.cos(np.radians(observation.zenith_angle_deg))),
                    "orders": list(observation.orders), "order_source": observation.order_source,
                    "sha256": dict(observation.sha256),
                },
                "calibration": {"path": str(calibration_path), "source": calibration.source["record"],
                                "standards": sorted({f for o in calibration.orders.values()
                                                     for f in o.frames}),
                                "standards_mjd": [float(m) for m in sorted({
                                    float(t) for o in calibration.orders.values()
                                    for t in o.water_mjd})]},
                "shifts": {"frame": shifts[key],
                           "frame_clipped": None if clipped_shifts is None else clipped_shifts[key],
                           "final": final_level},
                "settings": {"stellar": args.stellar, "vsini_kms": args.vsini_kms,
                             "clip_sigma": args.clip_sigma, "clip_pixels": args.clip_pixels,
                             "min_transmission": args.min_transmission},
                "results": sorted(rows, key=lambda r: r["order"]),
            }
            final = (clipped_shifts or shifts)[key]
            residual = np.nanmedian([r["residual_rms_over_noise"] for r in rows])
            print(f"  {observation.path.name.split('.')[0]} {observation.object_name[:14]:14s} "
                  f"water {final['log_column_H2O']:+.3f} ({final['water_orders']} orders), "
                  f"velocity {final['velocity_kms']:+.3f} km/s, median residual {residual:.2f}")

    # One exposure seen in both bands measures its water twice, independently.
    agreement = {}
    for key in sorted({k for k, _ in frame_reports}):
        if (key, "H") in frame_reports and (key, "K") in frame_reports:
            pick = lambda b: frame_reports[(key, b)]["shifts"]  # noqa: E731
            water = {b: (pick(b)["frame_clipped"] or pick(b)["frame"])["log_column_H2O"]
                     for b in "HK"}
            difference = water["H"] - water["K"]
            agreement[key] = {"H": water["H"], "K": water["K"], "difference": difference,
                              "flag": bool(abs(difference) > BAND_DISAGREEMENT)}
            for band in "HK":
                frame_reports[(key, band)]["band_agreement"] = agreement[key]
            note = "  <-- the bands disagree: stellar lines are in the water scale" \
                if agreement[key]["flag"] else ""
            print(f"  {key}: water shift H {water['H']:+.3f}, K {water['K']:+.3f}, "
                  f"difference {difference:+.3f}{note}")

    for (key, band), report in frame_reports.items():
        stem = Path(report["observation"]["path"]).name.split(".")[0]
        (args.output_dir / f"{stem}_science_summary.json").write_text(json.dumps(report, indent=2))

    from tellurix import write_record

    for band, rows in record_rows.items():
        species = sorted({s for r in rows for s in r["log_column_scales"]})
        names = species + ["velocity_kms", "stellar_velocity_kms", "wavelength_stretch",
                           "lsf_sigma_kms"] + [f"continuum_{i}" for i in range(degree + 1)] + [
                               "log_jitter"]
        calibration, calibration_path = calibrations[band]
        record_path = args.record or args.output_dir / f"record_{band}.h5"
        write_record(
            record_path if args.record is None or len(record_rows) == 1
            else record_path.with_name(f"{record_path.stem}_{band}{record_path.suffix}"),
            run={"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "driver": Path(__file__).name},
            config={**{k: v for k, v in calibration.source["config"].items()},
                    "clip_sigma": args.clip_sigma if args.clip_sigma is not None else "",
                    "clip_pixels": args.clip_pixels, "stellar": args.stellar,
                    "vsini_kms": args.vsini_kms},
            physics=dict(driver.PHYSICS),
            inputs={"calibration": str(calibration_path),
                    "calibration_record": calibration.source["record"],
                    "frames": [r["frame"] for r in rows]},
            parameter_names=names, species=species,
            pages=sorted(rows, key=lambda r: (r["frame"], r["order_number"])),
            continuum_degree=degree, key_fields=("frame", "order"),
            extra_columns=(("band", "S256"), ("order_number", "i4"), ("order_source", "S64"),
                           ("airmass", "f8"), ("zenith_angle_deg", "f8"), ("mjd", "f8"),
                           ("object", "S256"), ("water_shift", "f8"), ("velocity_shift", "f8")))
    print(f"\n{sum(len(r) for r in record_rows.values())} order-frames in "
          f"{time.time() - started:.0f} s -> {args.output_dir}")


if __name__ == "__main__":
    main()
