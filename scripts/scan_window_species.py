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

from tellurix import ScanIdentity, load_atmosphere_csv, load_scan, save_scan, scan_window
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

    report = scan_window(
        profile, root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule",
        identity, zenith_angle_deg=args.zenith_angle_deg,
        layer_chunk_size=args.layer_chunk_size,
        progress=lambda line: print(line, flush=True))
    report["profile"] = str(args.profile)
    reject = report["rejected"]
    missing = report["not_in_profile"]
    print()
    print(f"evaluated {report['evaluated']} species; "
          f"{report['rejected_without_evaluating']} rejected on their bound")
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
