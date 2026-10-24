#!/usr/bin/env python
"""Describe how the IAG solar flux atlas was co-added, for `tellurix.CoaddedTelluricModel`.

Reiners et al. 2016 (A&A 587, A65; Table 2 and Sect. 3) co-add 1190 VIS scans from
nine days of 2014, in packages of 10 scans taking ~15 min, each package shifted by
the Sun-observer radial velocity (JPL Horizons) into the solar frame before
averaging. So each package's telluric lines sit at -v_r in the atlas, and v_r moves
by 0.6 km/s between April and July and by +-0.2 km/s within a day.

The paper gives the scans per day but not when they were taken. Each day's
packages are put back to back, centred on the Sun's transit at Göttingen, and that
stretch is cut into hour-long nodes (at least one), each weighted by its share of
the day's scans, with its own air mass and v_r at its midpoint. Each day gets the
ERA5 profile at 11 UTC (transit is at ~11:20 UTC on these dates):

    for d in 20140307 20140310 20140416 20140417 20140420 20140711 20140717 20140722 20140723; do
      uv run python scripts/era5_site_profile.py --site Goettingen \\
          --time ${d:0:4}-${d:4:2}-${d:6:2}T11:00 --output data/profiles/iag2014/goettingen_${d}T1100.csv
    done
    uv run python scripts/make_iag_coadd.py --output data/profiles/iag2014/coadd_vis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Reiners et al. 2016, Table 2: the VIS setting's days and scans.
VIS_SCANS = {"2014-03-07": 220, "2014-03-10": 170, "2014-04-16": 50, "2014-04-17": 280,
             "2014-04-20": 190, "2014-07-11": 100, "2014-07-17": 60, "2014-07-22": 50,
             "2014-07-23": 70}
SCANS_PER_PACKAGE = 10
MINUTES_PER_PACKAGE = 15.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", default="data/profiles/iag2014")
    parser.add_argument("--node-hours", type=float, default=1.0)
    parser.add_argument("--centre-offset-hours", type=float, default=0.0,
                        help="centre each day's packages this far from transit (negative: morning)")
    args = parser.parse_args()

    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body_barycentric_posvel, get_sun
    from astropy.time import Time

    from tellurix_fts.baker import IAG_ALTITUDE_KM, IAG_LATITUDE_DEG, IAG_LONGITUDE_DEG

    site = EarthLocation(lat=IAG_LATITUDE_DEG * u.deg, lon=IAG_LONGITUDE_DEG * u.deg,
                         height=IAG_ALTITUDE_KM * u.km)

    def geometry(time):
        """The Sun's radial velocity from the site (km/s, receding positive) and its zenith angle."""
        sun, sun_v = get_body_barycentric_posvel("sun", time)
        earth, earth_v = get_body_barycentric_posvel("earth", time)
        place, place_v = site.get_gcrs_posvel(time)
        r = (sun - earth - place).xyz.to(u.km).value
        v = (sun_v - earth_v - place_v).xyz.to(u.km / u.s).value
        altitude = get_sun(time).transform_to(AltAz(obstime=time, location=site)).alt.deg
        return float(np.dot(r, v) / np.linalg.norm(r)), float(90.0 - altitude)

    components = []
    for day, scans in VIS_SCANS.items():
        hours = np.arange(0.0, 24.0, 1.0 / 60.0)
        midnight = Time(f"{day}T00:00:00", scale="utc")
        altitude = get_sun(midnight + hours * u.hour).transform_to(
            AltAz(obstime=midnight + hours * u.hour, location=site)).alt.deg
        transit = float(hours[int(np.argmax(altitude))])
        duration = scans / SCANS_PER_PACKAGE * MINUTES_PER_PACKAGE / 60.0
        count = max(1, int(round(duration / args.node_hours)))
        nodes = []
        for k in range(count):
            hour = transit + args.centre_offset_hours - 0.5 * duration + (k + 0.5) * duration / count
            radial, zenith = geometry(midnight + hour * u.hour)
            nodes.append({"utc_hour": round(hour, 3), "weight": scans / count,
                          "zenith_angle_deg": round(zenith, 4), "airmass": round(1.0 / np.cos(np.radians(zenith)), 4),
                          "sun_radial_velocity_kms": round(radial, 4), "velocity_kms": round(-radial, 4)})
        components.append({"label": day, "scans": scans, "transit_utc_hour": round(transit, 3),
                           "centre_offset_hours": args.centre_offset_hours,
                           "profile": f"{args.profiles}/goettingen_{day.replace('-', '')}T1100.csv",
                           "nodes": nodes})
        print(f"{day}: {scans} scans, {duration:.1f} h about {transit:.2f} UTC, {count} nodes, "
              f"X {nodes[0]['airmass']:.2f}-{nodes[-1]['airmass']:.2f}, telluric v "
              + ", ".join(f"{n['velocity_kms']:+.2f}" for n in nodes))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "about": __doc__.split("\n\n")[1].replace("\n", " ") + " " + __doc__.split("\n\n")[2].replace("\n", " "),
        "source": "Reiners et al. 2016, A&A 587, A65, Table 2 (VIS)", "components": components},
        indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
