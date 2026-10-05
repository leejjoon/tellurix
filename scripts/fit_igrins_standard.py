#!/usr/bin/env python
"""Fit the telluric absorption in IGRINS A0V standards, order by order.

The fit is ``tellurix_igrins.standard``; this drives a night of frames through
it, reading data from this checkout, and writes the summaries and the record.

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
        --spec data/igrins/20180402_0104/SDCH_20180402_0104.spec.fits --orders 100

Give several ``--spec`` paths to fit a whole night at once, which is far cheaper
per frame than running them one at a time -- see below. ``--orders`` takes a
comma-separated list of *physical* echelle orders -- the WAT header's beam
numbers, H 98-125 and K 71-96 -- never row positions, which differ between
reductions; leave it out for all of them.

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

from tellurix.download import DataPaths
from tellurix_igrins.standard import (
    ORDER_RULE, PHYSICS, StandardFitSettings, build_order_context, fit_one, run_provenance,
    stages_for,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True, nargs="+",
                        help="one or more PLP .spec.fits files; giving a whole night at once "
                             "shares the grid, the opacity and the compilations across frames")
    parser.add_argument("--orders", default=None,
                        help="comma-separated physical echelle orders (H 98-125, K 71-96), "
                             "as the WAT header names them; default every order present")
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
    parser.add_argument("--record", type=Path, default=None,
                        help="the run's HDF5 record, which is its product; the .npz arrays "
                             "are a regenerable cache. Defaults beside the summaries.")
    parser.add_argument("--summary-suffix", default="",
                        help="appended to the summary filename. Sharding one band across "
                             "devices by order needs this: every shard writes a summary for "
                             "the same frames, and without a distinct name they overwrite "
                             "each other and each keeps only its own orders.")
    parser.add_argument("--precompute-opacity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-broadening", choices=("linear", "frozen"), default="linear")
    parser.add_argument("--hydrogen-mask-kms", type=float, default=None,
                        help="override the order rule's mask_hydrogen_kms. At the default 600 the "
                             "Brackett line is still 6-8%% deep in an A0V model, so the flat "
                             "source leaks stellar absorption into the continuum.")
    parser.add_argument("--throughput-floor", type=float, default=None,
                        help="override the order rule's throughput_floor; the cut on the smoothed "
                             "blaze")
    parser.add_argument("--fixed-pattern", action=argparse.BooleanOptionalAction, default=True,
                        help="measure each order's repeatable instrument response from the "
                             "night's OTHER frames and divide it out, then refit. Worth about "
                             "1.7x on the residual and most of that at the order edges.")
    parser.add_argument("--pattern-smooth-pixels", type=int, default=51,
                        help="boxcar applied to the pattern. Without it the correction also "
                             "removes our own telluric model error, which every frame shares; "
                             "0 disables the guard and invalidates the residual as a test.")
    parser.add_argument("--fixed-pattern-min-frames", type=int, default=5,
                        help="below this many frames the pattern is too noisy to be a calibration")
    parser.add_argument("--blaze", type=Path, default=None,
                        help="a FlatBlaze (build_igrins_flat_blaze.py) to divide out before "
                             "fitting, so the continuum only models the lamp-to-star colour; "
                             "pair it with a low --continuum-degree")
    parser.add_argument("--response-pattern", type=Path, default=None,
                        help="a MasterPattern (build_igrins_master_pattern.py) to divide out "
                             "instead of the night's own leave-one-out pattern -- for a night "
                             "with fewer than --fixed-pattern-min-frames standards")
    parser.add_argument("--line-coupling", action="store_true",
                        help="apply AER's first-order line coupling (CO2 in both bands), as "
                             "LBLRTM does; off reproduces earlier runs")
    parser.add_argument("--no-covariance", action="store_true",
                        help="skip the formal errors, and with them the 8.2 s Hessian compile")
    parser.add_argument("--compilation-cache", default=str(root / ".jax-cache"),
                        help="pass an empty string to disable")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--fix-columns-from", type=Path, default=None,
                        help="a run record of the same night and band: hold --fix-species at "
                             "each order's median over that run's frames instead of fitting "
                             "them (a diagnostic for what their per-frame freedom absorbs)")
    parser.add_argument("--fix-species", default="CO2,CH4")
    args = parser.parse_args()

    rule = dict(ORDER_RULE)
    if args.hydrogen_mask_kms is not None:
        rule["mask_hydrogen_kms"] = args.hydrogen_mask_kms
    if args.throughput_floor is not None:
        rule["throughput_floor"] = args.throughput_floor
    fixed_columns = {}
    if args.fix_columns_from is not None:
        from tellurix.record import read_record

        pages = read_record(args.fix_columns_from).pages
        for species in (s.strip().upper() for s in args.fix_species.split(",")):
            for number in sorted(set(int(n) for n in pages["order_number"])):
                values = pages[f"log_column_{species}"][pages["order_number"] == number]
                fixed_columns.setdefault(number, {})[species] = float(np.median(values))
    settings = StandardFitSettings(
        fixed_columns=fixed_columns,
        stellar=args.stellar, vsini_kms=args.vsini_kms, resolving_power=args.resolving_power,
        samples_per_resolution=args.samples_per_resolution, margin_cm1=args.margin_cm1,
        grid_margin_cm1=args.grid_margin_cm1, continuum_degree=args.continuum_degree,
        min_optical_depth=args.min_optical_depth, min_transmission=args.min_transmission,
        precompute_opacity=args.precompute_opacity, self_broadening=args.self_broadening,
        covariance=not args.no_covariance, line_coupling=args.line_coupling, order_rule=rule)
    paths = DataPaths.bootstrapped(root)
    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Path("") is Path("."), which is truthy -- check the string, not the path,
    # or an empty argument silently writes a cache into the repository root.
    if args.compilation_cache:
        import jax
        jax.config.update("jax_compilation_cache_dir", str(Path(args.compilation_cache)))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    from tellurix import StellarSpectrum, load_atmosphere_csv
    from tellurix_igrins import leave_one_out_patterns, read_igrins_observation

    observations = [read_igrins_observation(path) for path in args.spec]
    flat_blaze = None
    if args.blaze is not None:
        from tellurix_igrins import FlatBlaze

        flat_blaze = FlatBlaze.load(args.blaze)
        if {o.band for o in observations} != {flat_blaze.band}:
            raise SystemExit(f"{args.blaze} is a {flat_blaze.band} blaze")
    master = None
    if args.response_pattern is not None:
        from tellurix_igrins import MasterPattern

        master = MasterPattern.load(args.response_pattern)
        if {o.band for o in observations} != {master.band}:
            raise SystemExit(f"{args.response_pattern} is a {master.band} pattern")
    profile = load_atmosphere_csv(root / args.profile)
    stellar = None if args.stellar == "flat" else StellarSpectrum.from_npz(args.stellar)

    # Physical echelle orders, never row positions: the number of orders a
    # band ships differs between reductions, so row N of one frame need not be
    # row N of the next. An order a frame lacks is skipped for that frame.
    numbers = (sorted(set().union(*(o.orders for o in observations))) if args.orders is None
               else [int(value) for value in args.orders.split(",")])

    print(f"{len(observations)} frame(s), orders {numbers[0]}-{numbers[-1]} ({len(numbers)}), "
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

    for number in numbers:
        context, objective = None, None
        first_pass = []
        for observation in observations:
            if context is None:
                try:
                    context = build_order_context(observation, number, settings, paths,
                                                  profile, stellar)
                    setup_total += context["setup_seconds"]
                except (RuntimeError, ValueError) as exc:
                    # The window belongs to the order, so a failure here is
                    # usually shared; try the next frame anyway, because a bad
                    # extraction in one frame is not a property of the order.
                    failures[id(observation)].append({"order": number, "error": str(exc)})
                    print(f"  order {number:3d}  {observation.path.name[5:18]}  skipped: {exc}")
                    continue
            began = time.time()
            try:
                row, objective = fit_one(context, observation, settings, objective,
                                         write_arrays=not args.fixed_pattern,
                                         blaze=flat_blaze, output_dir=args.output_dir)
            except (RuntimeError, ValueError) as exc:
                failures[id(observation)].append({"order": number, "error": str(exc)})
                print(f"  order {number:3d}  {observation.path.name[5:18]}  skipped: {exc}")
                continue
            row["seconds"] = round(time.time() - began, 1)
            first_pass.append((observation, row))

        # Second pass. The night's other frames measure this order's fixed
        # instrument response, which is the dominant residual at the red edge;
        # one pass cannot see it, because a single frame cannot tell a
        # repeatable response error from its own noise.
        corrected = None
        use_master = master is not None and first_pass
        if use_master or (args.fixed_pattern
                          and len(first_pass) >= args.fixed_pattern_min_frames):
            if use_master:
                # A night too thin to measure its own response borrows the
                # instrument's, measured on other nights. No leave-one-out is
                # needed: none of these frames went into it.
                patterns = [master.pattern_on(number, observation.order(number).wavelength_vacuum_nm)
                            for observation, _ in first_pass]
            else:
                patterns = leave_one_out_patterns(
                    [r["_fractional_residual"] for _, r in first_pass],
                    smooth_pixels=args.pattern_smooth_pixels)
            corrected = []
            for (observation, previous), pattern in zip(first_pass, patterns):
                began = time.time()
                try:
                    row, objective = fit_one(context, observation, settings, objective,
                                             response=pattern, blaze=flat_blaze,
                                             output_dir=args.output_dir)
                except (RuntimeError, ValueError) as exc:
                    failures[id(observation)].append({"order": number, "error": str(exc)})
                    continue
                row["seconds"] = round(previous["seconds"] + time.time() - began, 1)
                row["residual_rms_over_noise_uncorrected"] = previous["residual_rms_over_noise"]
                row["_record"]["response_pattern_rms"] = row["response_pattern_rms"]
                corrected.append((observation, row))
        elif args.fixed_pattern:
            print(f"  order {number:3d}  fewer than {args.fixed_pattern_min_frames} frames; "
                  "no fixed-pattern correction")

        for observation, row in (corrected if corrected is not None else first_pass):
            results[id(observation)].append(row)
            tag = "reused" if row["reused_compilation"] else "built "
            was = (f"  was {row['residual_rms_over_noise_uncorrected']:5.2f}"
                   if "residual_rms_over_noise_uncorrected" in row else "")
            print(f"  order {number:3d}  {observation.path.name[5:18]}  "
                  f"T={row['median_transmission']:.3f}  rms/sig={row['residual_rms_over_noise']:6.2f}  "
                  f"R={row['resolving_power_fitted']:6.0f}  v={row['velocity_kms']:+5.2f}  "
                  f"{tag} {row['seconds']:5.1f}s{was}")

    elapsed = time.time() - started
    for observation in observations:
        rows = [{k: v for k, v in row.items() if not k.startswith("_")}
                for row in sorted(results[id(observation)], key=lambda r: r["order"])]
        stem = observation.path.name.split(".")[0]
        summary = {
            "observation": {
                "path": str(observation.path), "object": observation.object_name,
                "object_type": observation.object_type, "band": observation.band,
                "telescope": observation.telescope, "date_obs": observation.date_obs,
                "mjd": observation.mjd, "exposure_time_s": observation.exposure_time_s,
                "zenith_angle_deg": observation.zenith_angle_deg,
                "orders": list(observation.orders), "order_source": observation.order_source,
                "zenith_source": observation.zenith_source,
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
                "line_coupling": args.line_coupling,
                "fixed_pattern": args.fixed_pattern,
                "fixed_pattern_min_frames": args.fixed_pattern_min_frames,
                "response_pattern": None if master is None else str(args.response_pattern),
                "blaze": None if args.blaze is None else str(args.blaze),
                "pattern_smooth_pixels": args.pattern_smooth_pixels,
                "frames_in_run": len(observations),
                "fixed_columns": {str(n): dict(v) for n, v in settings.fixed_columns.items()},
                "fixed_columns_from": (None if args.fix_columns_from is None
                                       else str(args.fix_columns_from)),
            },
            "physics": {**PHYSICS, "line_coupling": args.line_coupling,
                       "stages": list(stages_for(args.stellar))},
            "order_rule": dict(settings.order_rule),
            "results": rows, "failures": failures[id(observation)],
        }
        path = args.output_dir / f"{stem}{args.summary_suffix}_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        if rows:
            ratio = np.array([r["residual_rms_over_noise"] for r in rows])
            print(f"\n{stem}: {len(rows)} orders, median rms/sigma {np.median(ratio):.2f}, "
                  f"range {ratio.min():.2f}-{ratio.max():.2f}")

    record_rows = [r["_record"] for rows in results.values() for r in rows]
    if record_rows:
        from tellurix import write_record

        all_rows = [r for rows in results.values() for r in rows]
        # Which molecules have lines depends on the window, so an order's
        # parameter vector is not the run's. Build the union and remap every
        # row's sigma and correlation into it, or the record would silently
        # line up one order's CH4 against another's CO2.
        species = sorted({s for r in all_rows for s in r["_species"]})
        template = max(all_rows, key=lambda r: len(r["_parameter_names"]))
        trailing = [n for n in template["_parameter_names"] if n not in template["_species"]]
        run_names = species + trailing
        position = {name: i for i, name in enumerate(run_names)}
        for row in all_rows:
            names = list(row["_parameter_names"])
            index = np.array([position[n] for n in names])
            sigma = np.zeros(len(run_names))
            sigma[index] = np.asarray(row["_record"]["sigma"], dtype=float)
            correlation = np.zeros((len(run_names), len(run_names)))
            correlation[np.ix_(index, index)] = np.asarray(
                row["_record"]["correlation"], dtype=float)
            row["_record"]["sigma"] = sigma
            row["_record"]["correlation"] = correlation
        run, inputs = run_provenance(Path(__file__), paths, settings, observations,
                                     root / args.profile)
        record_path = args.record or (
            args.output_dir / f"record{args.summary_suffix}.h5")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        write_record(
            record_path,
            run=run,
            config={"resolving_power": args.resolving_power,
                    "samples_per_resolution": args.samples_per_resolution,
                    "margin_cm1": args.margin_cm1,
                    "grid_margin_cm1": args.grid_margin_cm1,
                    "continuum_degree": args.continuum_degree,
                    "min_optical_depth": args.min_optical_depth,
                    "min_transmission": args.min_transmission,
                    "vsini_kms": args.vsini_kms, "stellar": args.stellar,
                    "blaze": "" if args.blaze is None else str(args.blaze),
                    "line_coupling": args.line_coupling,
                    "fixed_columns_from": ("" if args.fix_columns_from is None
                                           else str(args.fix_columns_from)),
                    **settings.order_rule},
            physics={**PHYSICS, "line_coupling": args.line_coupling,
                       "stages": list(stages_for(args.stellar))},
            inputs=inputs,
            parameter_names=run_names,
            species=species,
            pages=sorted(record_rows, key=lambda r: (r["frame"], r["order"])),
            continuum_degree=args.continuum_degree,
            # An atlas row is a page-epoch; here it is one echelle order of one
            # exposure. Shards of one band write separate records that
            # merge_records combines.
            key_fields=("frame", "order"),
            extra_columns=(("band", "S256"), ("order_number", "i4"),
                           ("order_source", "S64"),
                           ("airmass", "f8"), ("zenith_angle_deg", "f8"),
                           ("mjd", "f8"), ("telescope", "S256"), ("object", "S256"),
                           ("date_obs", "S256"),
                           ("surface_temperature_k", "f8"),
                           ("surface_pressure_hpa", "f8"),
                           ("surface_humidity_percent", "f8"),
                           ("response_pattern_rms", "f8"),
                           ("residual_z_rms", "f8"),
                           ("reused_compilation", "?")),
        )
        print(f"wrote {record_path} ({record_path.stat().st_size / 1e6:.2f} MB)")

    fitted = sum(len(r) for r in results.values())
    reused = sum(1 for rows in results.values() for r in rows if r["reused_compilation"])
    print(f"\n{fitted} order-frames in {elapsed:.0f} s ({elapsed / max(fitted, 1):.1f} s each); "
          f"{reused} reused a compilation; {setup_total:.0f} s of that was per-order setup")


if __name__ == "__main__":
    main()
