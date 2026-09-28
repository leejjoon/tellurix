#!/usr/bin/env python
"""Rank every absorber in a window, so the species list is computed not chosen.

A hand-written species list is what hid OCS and O3 in the 4.9 um solar window,
worth a factor of 2.5 in residual. Nine other explanations were eliminated
first -- velocity, LSF, column scale, solar model, instrument profile, continuum
degree, vertical profile, layer count, line physics -- and every one of those
tests was sound and beside the point, because they all assumed the species list
was right. See docs/solar_fts_residual.md.

For every molecule AER ships, this selects the lines in the window and computes
the **peak vertical optical depth from the profile's own column**. That last
part is the one to get right: ranking ozone by its surface mixing ratio
understates it 17x, because its column is stratospheric. The profile carries
the column, so use it -- which means the profile must carry the species at all
(`make_site_profile.py --afgl-model`).

    UV_CACHE_DIR=.uv-cache uv run python scripts/scan_window_species.py \\
        --v1 2030 --v2 2060 --profile data/profiles/kitt_peak_1990_afgl.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from tellurix import (
    AER_MOLECULE_IDS,
    AERLineDatabase,
    ExoJAXOpacityBackend,
    constant_velocity_grid,
    line_optical_depth_bound,
    ScanIdentity,
    load_atmosphere_csv,
    load_scan,
    save_scan,
    select_significant_lines,
    trim_wavenumber_grid,
)
from tellurix.nso import MEASURED_FWHM_CM1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1", type=float, required=True)
    parser.add_argument("--v2", type=float, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--zenith-angle-deg", type=float, default=0.0,
                        help="0 ranks the vertical column; a slant path scales every "
                             "species alike and so does not change the ranking")
    parser.add_argument("--fwhm-cm1", type=float, default=MEASURED_FWHM_CM1)
    parser.add_argument("--samples-per-resolution", type=float, default=2.0,
                        help="coarser than a fit needs: this ranks, it does not fit")
    parser.add_argument("--margin-cm1", type=float, default=25.0)
    parser.add_argument("--threshold", type=float, default=1.0e-3,
                        help="peak optical depth below which a species is not worth "
                             "fitting. 1e-3 is about a tenth of the NSO FTS noise.")
    parser.add_argument("--line-budget", type=float, default=1.0e-5,
                        help="optical depth discarded by dropping weak lines; must stay "
                             "well below --threshold so it cannot move a species across it")
    parser.add_argument("--layer-chunk-size", type=int, default=0,
                        help="layers evaluated at once; lower it if a line-rich species "
                             "exhausts the device")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/scans"),
                        help="where scans are kept. A scan depends only on the window and "
                             "the profile, not on any spectrum, so one entry serves every "
                             "file, every refit and every re-analysis -- which is what makes "
                             "a 6-minute evaluation affordable at atlas scale.")
    parser.add_argument("--refresh", action="store_true",
                        help="recompute even if a matching entry exists")
    parser.add_argument("--output", type=Path, default=None,
                        help="an extra copy, beside the cache entry")
    args = parser.parse_args()
    if not 0.0 < args.v1 < args.v2:
        parser.error("require 0 < v1 < v2")

    root = Path(__file__).resolve().parents[1]
    profile_path = args.profile if args.profile.is_absolute() else root / args.profile
    profile = load_atmosphere_csv(profile_path)

    identity = ScanIdentity.for_profile(
        profile_path, (args.v1, args.v2),
        threshold=args.threshold, line_budget=args.line_budget,
        margin_cm1=args.margin_cm1, fwhm_cm1=args.fwhm_cm1,
        samples_per_resolution=args.samples_per_resolution)
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else root / args.cache_dir

    def deliver(report: dict) -> None:
        if args.output:
            out = args.output if args.output.is_absolute() else root / args.output
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {out}")

    cached = None if args.refresh else load_scan(cache_dir, identity)
    if cached is not None:
        print(f"cached {identity.path(cache_dir)}")
        print("fit:      " + ", ".join(cached["species"]))
        deliver(cached)
        return

    centre = 0.5 * (args.v1 + args.v2)
    grid = constant_velocity_grid(
        1.0e7 / args.v2, 1.0e7 / args.v1,
        resolving_power=centre / args.fwhm_cm1,
        samples_per_resolution=args.samples_per_resolution,
        margin_cm1=args.margin_cm1)
    grid = trim_wavenumber_grid(grid, args.v1, args.v2, 5.0)
    column = np.asarray(profile.air_column_cm2)
    secant = 1.0 / np.cos(np.radians(args.zenith_angle_deg))

    line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    found, absent, missing, reject_by_bound = [], [], [], []
    for name, molecule_id in sorted(AER_MOLECULE_IDS.items(), key=lambda kv: kv[1]):
        directory = next((d for d in line_root.glob(f"{molecule_id:02d}_*") if d.is_dir()), None)
        if directory is None:
            continue
        if name not in profile.vmr:
            missing.append(name)
            continue
        try:
            database = AERLineDatabase(directory / directory.name, name,
                                       (args.v1, args.v2), margin_cm1=args.margin_cm1)
        except ValueError as exc:
            if str(exc).startswith(f"no {name} lines found"):
                absent.append(name)
                continue
            raise
        # Reject analytically before touching the kernel. The sum of the
        # per-line bounds is an upper bound on the peak optical depth anywhere
        # -- it assumes every line peaks at the same wavenumber, which they do
        # not -- so a species whose bound is below the threshold provably
        # cannot reach it. This is numpy, not XLA: it costs nothing next to a
        # compile, and it removes most of the periodic table from every window.
        bound = float(np.sum(line_optical_depth_bound(database, profile, name)))
        if bound < args.threshold:
            reject_by_bound.append({"species": name, "molecule": molecule_id,
                                    "lines": int(database.nu_lines.size),
                                    "optical_depth_bound": bound})
            print(f"  {name:8s} {database.nu_lines.size:7d} lines  "
                  f"bound {bound:.3e}  -- below threshold, not evaluated", flush=True)
            continue

        # A ranking does not need every line, and some species carry tens of
        # thousands: O3 has 20,143 here, whose dense line-by-grid intermediates
        # ask for 31 GB. select_significant_lines bounds the optical-depth error
        # by its budget, which is two orders below the threshold being tested,
        # so it cannot change which side of the cut a species falls on.
        full = int(database.nu_lines.size)
        database = select_significant_lines(database, profile, name,
                                            optical_depth_budget=args.line_budget)
        opacity = ExoJAXOpacityBackend.prepare(
            {name: database}, grid, methods="direct_sparse",
            temperature_range_k=(float(np.min(profile.temperature_k)),
                                 float(np.max(profile.temperature_k))),
            maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
            vectorize_layers=True, mixed_precision=True, pressure_shift=True,
            layer_chunk_size=args.layer_chunk_size or None)
        # Never call this eagerly. An uncompiled kernel dispatches operation by
        # operation and materialises the dense line-by-grid offset matrix, which
        # asks for 16 GB on a 20,000-line species like O3; jit fuses it away.
        # The backend takes the self-broadening partial pressure per species,
        # keyed like the cross sections it returns.
        pressure = np.asarray(profile.pressure_layer_bar)
        evaluate = jax.jit(lambda t, p_, s: opacity.cross_sections(t, p_, {name: s})[name])
        cross_section = np.asarray(evaluate(
            np.asarray(profile.temperature_k), pressure,
            pressure * np.asarray(profile.vmr[name])))
        # Optical depth is the cross section times the species column, summed
        # over layers -- the profile's own column, which is the whole point.
        tau = np.sum(cross_section * (np.asarray(profile.vmr[name]) * column)[:, None], axis=0)
        found.append({"species": name, "molecule": molecule_id,
                      "lines": full, "lines_kept": int(database.nu_lines.size),
                      "peak_optical_depth": float(np.max(tau) * secant),
                      "median_optical_depth": float(np.median(tau) * secant)})
        print(f"  {name:8s} {full:7d} lines ({database.nu_lines.size:6d} kept)  "
              f"peak tau {found[-1]['peak_optical_depth']:.3e}", flush=True)

    found.sort(key=lambda row: -row["peak_optical_depth"])
    keep = [row for row in found if row["peak_optical_depth"] >= args.threshold]
    reject = [row for row in found if row["peak_optical_depth"] < args.threshold]
    reject_by_bound.sort(key=lambda row: -row["optical_depth_bound"])
    report = {
        "window_cm1": [args.v1, args.v2],
        "profile": str(args.profile),
        "zenith_angle_deg": args.zenith_angle_deg,
        "threshold_peak_optical_depth": args.threshold,
        "species": [row["species"] for row in keep],
        "fit": keep,
        "rejected": reject,
        "rejected_by_bound": reject_by_bound,
        "no_lines_in_window": absent,
        "not_in_profile": missing,
        "evaluated": len(found),
        "rejected_without_evaluating": len(reject_by_bound),
        "headroom": (None if not reject else
                     args.threshold / reject[0]["peak_optical_depth"]),
    }
    print()
    print(f"evaluated {len(found)} species; {len(reject_by_bound)} rejected on their bound")
    print("fit:      " + ", ".join(report["species"]))
    if reject:
        print(f"strongest rejected: {reject[0]['species']} at "
              f"{reject[0]['peak_optical_depth']:.2e}, "
              f"{report['headroom']:.0f}x below the cut")
    if missing:
        print(f"NOT IN THE PROFILE and so unrankable: {', '.join(missing)}")
    print(f"cached {save_scan(cache_dir, identity, report)}")
    deliver(report)


if __name__ == "__main__":
    main()
