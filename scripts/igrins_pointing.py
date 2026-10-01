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

``--write`` puts ``pointing.json`` beside a frame -- for every frame, or only
those whose header is unusable or off by more than ``--tolerance-deg`` -- and
``read_igrins_observation`` then uses it for both bands of that exposure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True, nargs="+")
    parser.add_argument("--catalog", type=Path, default=Path("data/igrins/reduced_log.csv"))
    parser.add_argument("--write", choices=("none", "disagreeing", "all"), default="none")
    parser.add_argument("--tolerance-deg", type=float, default=1.0)
    args = parser.parse_args()

    from astropy.coordinates import SkyCoord
    from astropy.io import fits

    from tellurix.igrins import (POINTING_SIDECAR, geometric_zenith_angle_deg, site_for,
                                 zenith_angle_deg)

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
        # The header value as the reader would take it, ignoring any sidecar.
        try:
            from_header = zenith_angle_deg(header)
        except ValueError:
            from_header = None
        off = None if from_header is None else from_header - geometry
        failing = off is None or abs(off) > args.tolerance_deg
        print(f"{path.name:32s} {name:14s} geometry {geometry:6.2f}  header "
              f"{'unusable' if from_header is None else f'{from_header:6.2f} ({off:+.2f})'}"
              f"{'  <-' if failing else ''}")
        if args.write == "all" or (args.write == "disagreeing" and failing):
            sidecar = path.parent / POINTING_SIDECAR
            sidecar.write_text(json.dumps({
                "zenith_angle_deg": geometry, "source": "geometry", "target": name,
                "ra_deg": target.ra.deg, "dec_deg": target.dec.deg, "resolver": "SIMBAD by name",
                "date_obs": str(header["DATE-OBS"]), "date_end": str(header["DATE-END"]),
                "header_zenith_angle_deg": from_header, "computed_from": path.name,
            }, indent=1) + "\n")


if __name__ == "__main__":
    main()
