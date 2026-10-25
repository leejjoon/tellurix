#!/usr/bin/env python
"""Describe each scan of the Kurucz et al. 1984 flux atlas, for `tellurix.CoaddedTelluricModel`.

The atlas splices eight McMath FTS scans of 1980-81 (Table 1, reprinted in Kurucz's
2005 re-reduction as `kurucz2005/fluxscans.dat`; `tellurix_fts.flux_atlas.FLUXATL_SCANS`).
Each scan co-adds 16-36 interferograms over about two hours of one day, so its
telluric lines are a mean over a changing air mass: 1.16 to 1.45 in scan 113. One
component per scan, on the ERA5 profile at the hour nearest the scan's middle, with
nodes evenly spaced in time from its start to its stop, each at the Sun's zenith
angle then. Table 1's times are local (MST): with UTC = MST + 7 h the Sun's air mass
at each start and stop matches the table's to 0.01.

The nodes carry no velocity. Each scan was recorded in the observatory's frame and
corrected afterwards by one factor (Table 1's "FTS frequency correction"), so the
telluric lines are not smeared, only moved together, which the fitted velocity takes.

    for t in 19810324T21 19810325T19 19810325T21 19810511T19; do
      uv run python scripts/era5_site_profile.py --site "Kitt Peak" --epoch 1983 \\
          --time ${t:0:4}-${t:4:2}-${t:6:2}T${t:9:2}:00 --output data/profiles/fluxatl1981/kitt_peak_${t}00.csv
    done
    uv run python scripts/make_fluxatl_coadd.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", default="data/profiles/fluxatl1981")
    parser.add_argument("--node-minutes", type=float, default=20.0)
    args = parser.parse_args()

    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_sun
    from astropy.time import Time

    from tellurix_fts.flux_atlas import FLUXATL_SCANS
    from tellurix_fts.nso import KITT_PEAK_ALTITUDE_KM, KITT_PEAK_LATITUDE_DEG, KITT_PEAK_LONGITUDE_DEG

    site = EarthLocation(lat=KITT_PEAK_LATITUDE_DEG * u.deg, lon=KITT_PEAK_LONGITUDE_DEG * u.deg,
                         height=KITT_PEAK_ALTITUDE_KM * u.km)
    for number, scan in FLUXATL_SCANS.items():
        midnight = Time(f"{scan['date']}T00:00:00", scale="utc")
        start, stop = (seconds / 3600.0 + 7.0 for seconds in scan["mst_seconds"])
        count = max(1, int(round((stop - start) * 60.0 / args.node_minutes)))
        nodes = []
        for k in range(count):
            hour = start + (k + 0.5) * (stop - start) / count
            time = midnight + hour * u.hour
            zenith = 90.0 - float(get_sun(time).transform_to(AltAz(obstime=time, location=site)).alt.deg)
            nodes.append({"utc_hour": round(hour, 3), "weight": 1.0, "zenith_angle_deg": round(zenith, 4),
                          "airmass": round(1.0 / np.cos(np.radians(zenith)), 4), "velocity_kms": 0.0})
        era5_hour = int(np.floor(0.5 * (start + stop) + 0.5))
        profile = f"{args.profiles}/kitt_peak_{scan['date'].replace('-', '')}T{era5_hour:02d}00.csv"
        output = Path(args.profiles) / f"coadd_scan{number}.json"
        output.write_text(json.dumps({
            "about": " ".join(__doc__.split("\n\n")[1:3]).replace("\n", " "),
            "source": f"Kurucz, Furenlid, Brault & Testerman 1984, NSO Atlas No. 1, Table 1, scan {number}",
            "components": [{"label": f"scan{number}", "date": scan["date"],
                            "utc_hours": [round(start, 3), round(stop, 3)], "profile": profile,
                            "nodes": nodes}]}, indent=1) + "\n", encoding="utf-8")
        print(f"scan {number}: {scan['date']} {start:.2f}-{stop:.2f} UTC, {count} nodes, X "
              f"{nodes[0]['airmass']:.3f}-{nodes[-1]['airmass']:.3f} (table {scan['airmass'][0]}-"
              f"{scan['airmass'][1]}), ERA5 {era5_hour:02d}:00 -> {output}")


if __name__ == "__main__":
    main()
