#!/usr/bin/env python
"""Check a night's header zenith distances against geometry; write overrides where they fail.

The slant path is the zenith distance, and the header's pointing cards are
not always right. McDonald 2015-12 writes -1 in every pointing card of half its
frames and, in others, a ZDSTART 5-24 deg away from where the target actually
was (HR 1061 on 2015-12-03: 18.9 deg in the header, 43.1 by geometry). The
catalog's RA/Dec cannot stand in: it is empty for most of those rows and, where
filled, sometimes wrong (chi Tau at 70.47 deg on one frame, 65.90 on the rest).
Its ``NAME`` column is a SIMBAD identifier, so the target is resolved by name
(network, SIMBAD) and the zenith distance computed over DATE-OBS..DATE-END at
the site. On DCT 2016/2018, McDonald 2017 and Gemini South 2021 that agrees
with the header to 0.2 deg.

    uv run python scripts/igrins_pointing.py --spec data/igrins/20151203_*/SDCH_*.spec.fits
    uv run python scripts/igrins_pointing.py --spec ... --write disagreeing
    uv run python scripts/igrins_pointing.py --spec ... --sequence --write all

``--sequence`` writes the zenith distance of the whole combined sequence
instead of one exposure's: a PLP spectrum sums every exposure listed in the
catalog's ``FILES`` (4-10 on most nights, up to 35 minutes), while the header
describes the first. Left uncorrected this moved the dry-gas columns of a frame
by up to 6% (GitHub #1). The sequence starts at DATE-OBS and lasts
n x (EXPTIME + overhead); the catalog's JD is the sequence midpoint, so the
overhead comes from it where that is plausible, and from the archive's measured
cadence otherwise.

``--write`` puts ``pointing.json`` beside a frame -- for every frame, or only
those whose header is unusable or off by more than ``--tolerance-deg`` -- and
``read_igrins_observation`` then uses it for both bands of that exposure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True, nargs="+")
    parser.add_argument("--catalog", type=Path, default=Path("data/igrins/reduced_log.csv"))
    parser.add_argument("--write", choices=("none", "disagreeing", "all"), default="none")
    parser.add_argument("--tolerance-deg", type=float, default=1.0)
    parser.add_argument("--sequence", action="store_true",
                        help="write the whole combined sequence's zenith distance (implies "
                             "comparing against geometry over the sequence, not one exposure)")
    args = parser.parse_args()

    from astropy.coordinates import SkyCoord
    from astropy.io import fits

    from astropy.time import Time

    from tellurix_igrins.igrins import (EXPOSURE_OVERHEAD_S, POINTING_SIDECAR,
                                        geometric_zenith_angle_deg, sequence_zenith_angle_deg,
                                        site_for, zenith_angle_deg)

    rows = {(r["CIVIL"], int(r["FILENUMBER"])): r for r in csv.DictReader(args.catalog.open())}
    resolved = {}
    for path in args.spec:
        header = dict(fits.getheader(path))
        _, night, number = path.name.split(".")[0].split("_")[:3]
        row = rows.get((night, int(number)))
        if row is None:
            raise SystemExit(f"{path.name}: no catalog row for {night} {number}")
        name = " ".join((row.get("NAME") or row["OBJNAME_super"]).split())
        if name not in resolved:
            resolved[name] = SkyCoord.from_name(name)
        target = resolved[name]
        geometry = geometric_zenith_angle_deg(target.ra.deg, target.dec.deg, str(header["DATE-OBS"]),
                                              str(header["DATE-END"]), site_for(header))
        sequence = None
        if args.sequence:
            exposures = len(row["FILES"].split())
            exposure_s = float(row["EXPTIME"])
            # JD is the sequence midpoint on every night checked but DCT
            # 2018-12-20, whose JD sits a constant 241 s past one exposure
            # (single AB pairs there, so the sequence barely matters).
            half_s = (Time(float(row["JD"]), format="jd") - Time(str(header["DATE-OBS"]))).sec
            implied = 2.0 * half_s / exposures - exposure_s
            overhead = implied if 10.0 <= implied <= 120.0 else EXPOSURE_OVERHEAD_S
            duration = exposures * (exposure_s + overhead)
            sequence = {"exposures": exposures, "exposure_time_s": exposure_s,
                        "overhead_s": round(overhead, 1),
                        "overhead_from": "catalog JD" if overhead == implied else "default",
                        "duration_s": round(duration, 1),
                        "zenith_angle_deg": sequence_zenith_angle_deg(
                            target.ra.deg, target.dec.deg, str(header["DATE-OBS"]), duration,
                            site_for(header))}
        # The header value as the reader would take it, ignoring any sidecar.
        try:
            from_header = zenith_angle_deg(header)
        except ValueError:
            from_header = None
        off = None if from_header is None else from_header - geometry
        failing = off is None or abs(off) > args.tolerance_deg
        print(f"{path.name:32s} {name:14s} geometry {geometry:6.2f}  header "
              f"{'unusable' if from_header is None else f'{from_header:6.2f} ({off:+.2f})'}"
              f"{'  <-' if failing else ''}"
              + (f"  sequence of {sequence['exposures']} over {sequence['duration_s'] / 60:.1f} min "
                 f"{sequence['zenith_angle_deg']:6.2f} (airmass "
                 f"{1 / np.cos(np.radians(sequence['zenith_angle_deg'])):.3f} against "
                 f"{1 / np.cos(np.radians(geometry)):.3f})" if sequence else ""))
        if args.write == "all" or (args.write == "disagreeing" and failing):
            sidecar = path.parent / POINTING_SIDECAR
            sidecar.write_text(json.dumps({
                "zenith_angle_deg": sequence["zenith_angle_deg"] if sequence else geometry,
                "source": "sequence" if sequence else "geometry",
                "geometry_zenith_angle_deg": geometry, "sequence": sequence, "target": name,
                "ra_deg": target.ra.deg, "dec_deg": target.dec.deg, "resolver": "SIMBAD by name",
                "date_obs": str(header["DATE-OBS"]), "date_end": str(header["DATE-END"]),
                "header_zenith_angle_deg": from_header, "computed_from": path.name,
            }, indent=1) + "\n")


if __name__ == "__main__":
    main()
