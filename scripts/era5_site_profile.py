#!/usr/bin/env python
"""Build a layer profile from ERA5 instead of a lapse rate.

`make_site_profile.py` assumes a constant 6.5 K/km lapse rate and an exponential
water profile with a 2 km scale height. The fit gives each species one free
scale factor on its whole column, which absorbs an error in the *total* but not
in the vertical *distribution* -- and line strength depends on the pressure and
temperature at which the absorption happens. That is the shape of the
unexplained systematic in `docs/igrins_a0v.md`: well-mixed columns wandering a
few percent per unit airmass in inconsistent directions between nights.

This reads the real profile from ERA5 and writes the same CSV, with the same
layer edges above the site, so a fit against the two differs in T(z) and q(z)
and in nothing else.

    uv run python scripts/era5_site_profile.py \\
        --spec data/igrins/20181220_*/SDCH_*.spec.fits \\
        --output data/profiles/dct_2018_era5.csv

Three ways to say where and when. ``--spec`` reads IGRINS frames, ``--fts``
reads a raw NSO FTS spectrum, and ``--latitude/--longitude/--altitude-km/--time``
takes it directly. Only the first can supply a station pressure from a header;
the other two anchor on ERA5's own geopotential, which is what
``station_pressure_from_era5`` exists for.

    uv run python scripts/era5_site_profile.py --epoch 1990 \\
        --fts .../telluric_near_ir/ftsspec_901218_5.txt \\
        --output data/profiles/kitt_peak_19901218_1800.csv

Fetching and layering are ``tellurix.era5``, which says where the data comes
from and why the grid cell's orography does not matter; this is its command
line. The column is anchored at the *station pressure from the frame's own
header* when there is one.
"""

from __future__ import annotations

import argparse
import functools
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from tellurix.era5 import build_era5_profile, fetch_column, station_pressure_from_era5

# Where each site actually is. ERA5 needs a latitude and longitude; the reader's
# SITES table carries only the altitude, because that is all the header
# normalisation needs.
SITE_COORDINATES = {
    "McDonald Observatory": (30.6717, -104.0225),
    "Lowell Discovery Telescope": (34.7444, -111.4223),
    "Gemini South": (-30.2408, -70.7367),
}


def _named_sites():
    """Sites addressable by name with no instrument header to read.

    Kitt Peak is here because the NSO solar spectra date and time themselves
    but carry no weather card at all, so every number but the position has to
    come from ERA5.
    """

    from tellurix_fts.nso import (
        KITT_PEAK_ALTITUDE_KM, KITT_PEAK_LATITUDE_DEG, KITT_PEAK_LONGITUDE_DEG,
    )

    return {"Kitt Peak": (KITT_PEAK_LATITUDE_DEG, KITT_PEAK_LONGITUDE_DEG, KITT_PEAK_ALTITUDE_KM)}


def main() -> None:
    from tellurix.site_profile import (
        AFGL_MODELS, DEFAULT_EDGES_KM, EPOCH_DRY_VMR, LAYERINGS, afgl_dry_vmr, write_profile_csv,
    )

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, nargs="+",
                        help="the night's PLP .spec.fits files; the site, the time and the "
                             "station pressure all come from their headers")
    parser.add_argument("--fts", type=Path, nargs="+",
                        help="raw NSO ftsspec_*.txt spectra; the site is Kitt Peak and the "
                             "time is the midpoint of the exposure")
    parser.add_argument("--site", default=None, help="a name from _named_sites()")
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--altitude-km", type=float, default=None)
    parser.add_argument("--time", default=None, help="ISO 8601 UTC, e.g. 1990-12-18T18:00")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", default="2020", choices=sorted(EPOCH_DRY_VMR))
    parser.add_argument("--afgl-model", default=None, choices=AFGL_MODELS,
                        help="carry every AFGL molecule with its vertical structure. ERA5 "
                             "supplies T(z) and q(z), which is what it measures; the other "
                             "41 molecules come from the climatology, because a species-scan "
                             "can only rank what the profile knows about and a stratospheric "
                             "gas cannot be described by one surface number.")
    parser.add_argument("--layering", choices=LAYERINGS, default="weighted",
                        help="weighted: LBLRTM's air-weighted layer averages; centre: "
                             "sample each layer's centre, as profiles before 2026-10 were")
    parser.add_argument("--anchor", choices=("header", "era5", "auto"), default="auto",
                        help="where the station pressure comes from. 'auto' prefers the "
                             "header and falls back to ERA5, which is what every Gemini "
                             "South frame from 2020 needs.")
    args = parser.parse_args()

    chosen = [name for name, value in
              (("--spec", args.spec), ("--fts", args.fts), ("--time", args.time)) if value]
    if len(chosen) != 1:
        raise SystemExit("give exactly one of --spec, --fts or --time; got " + (", ".join(chosen) or "none"))

    pressures: list[float] = []
    if args.spec:
        from tellurix_igrins import read_igrins_observation

        sites, times = set(), []
        for path in args.spec:
            observation = read_igrins_observation(path)
            surface = observation.surface
            sites.add((surface["site"], surface["altitude_km"]))
            if surface["pressure_hpa"] is not None:
                pressures.append(surface["pressure_hpa"])
            times.append(datetime.strptime(observation.date_obs[:19], "%Y-%m-%dT%H:%M:%S")
                         .replace(tzinfo=timezone.utc))
        if len(sites) != 1:
            raise SystemExit(f"these frames are from more than one site: {sorted(sites)}")
        site, altitude = sites.pop()
        if site not in SITE_COORDINATES:
            raise SystemExit(f"no coordinates for {site}; add them to SITE_COORDINATES")
        latitude, longitude = SITE_COORDINATES[site]
        span = f"{len(args.spec)} frames spanning {min(times):%H:%M}-{max(times):%H:%M} UT"
    elif args.fts:
        from tellurix_fts import read_fts_spectrum

        site = args.site or "Kitt Peak"
        named = _named_sites()
        if site not in named:
            raise SystemExit(f"no coordinates for {site}; add them to _named_sites()")
        latitude, longitude, altitude = named[site]
        times = []
        for path in args.fts:
            # The exposure midpoint, not the start: these scans run 40-80
            # minutes and the air mass moves a long way inside one of them.
            spectrum = read_fts_spectrum(path)
            if spectrum.observed_utc_mid is None:
                raise SystemExit(f"{path} has no usable time in its header")
            times.append(spectrum.observed_utc_mid)
        span = (f"{len(args.fts)} spectra centred "
                f"{min(times):%H:%M}-{max(times):%H:%M} UT")
    else:
        if args.latitude is None or args.longitude is None or args.altitude_km is None:
            named = _named_sites()
            if args.site not in named:
                raise SystemExit("--time needs --latitude/--longitude/--altitude-km, or a "
                                 f"--site from {sorted(named)}")
            latitude, longitude, altitude = named[args.site]
        else:
            latitude, longitude, altitude = args.latitude, args.longitude, args.altitude_km
        site = args.site or f"({latitude:+.4f}, {longitude:+.4f})"
        stamp = datetime.fromisoformat(args.time)
        times = [stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)]
        span = f"the time given, {times[0]:%H:%M} UT"

    middle = min(times) + (max(times) - min(times)) / 2
    when = (middle + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)

    print(f"{site} ({latitude:+.4f}, {longitude:+.4f}) at {altitude:.3f} km")
    print(f"  {span}; ERA5 at {when:%Y-%m-%d %H:00} UT")
    column = fetch_column(latitude, longitude, when)

    from_era5 = station_pressure_from_era5(column, altitude)
    use_header = args.anchor == "header" or (args.anchor == "auto" and pressures)
    if args.anchor == "header" and not pressures:
        raise SystemExit("no frame carries a station pressure; use --anchor era5")
    surface_pressure = float(np.median(pressures)) if use_header else from_era5
    source = "the headers" if use_header else "ERA5's own geopotential"
    print(f"  anchored at {surface_pressure:.1f} hPa from {source}"
          + (f" (ERA5 would say {from_era5:.1f})" if pressures else ""))
    print(f"  ERA5 cell ({column['grid'][0]:+.2f}, {column['grid'][1]:.2f}), "
          f"its orography {column['grid_surface_m']:.0f} m against the telescope's "
          f"{altitude*1000:.0f} m")

    # A function of altitude, so build_era5_profile can sample the AFGL gases at the
    # layer centres or average them across each layer, as --layering asks.
    dry_vmr = (EPOCH_DRY_VMR[args.epoch] if args.afgl_model is None
               else functools.partial(afgl_dry_vmr, args.afgl_model, epoch=args.epoch))
    profile = build_era5_profile(column, surface_pressure, altitude,
                            dry_vmr, DEFAULT_EDGES_KM, layering=args.layering)
    water = profile.pop("_precipitable_water_mm")
    lapse = profile.pop("_lapse_k_km")
    print(f"  ERA5 gives {water:.2f} mm of precipitable water and a {lapse:.2f} K/km "
          f"lapse over the first 3 km (the analytic profile assumes 6.5)")

    rows = write_profile_csv(args.output, profile, [
        f"{site}, {args.epoch} trace-gas abundances, generated by scripts/era5_site_profile.py."
        + ("" if args.afgl_model is None else f" AFGL {args.afgl_model} trace set."),
        f"ERA5 at {when:%Y-%m-%dT%H:00}Z, cell ({column['grid'][0]:+.3f}, {column['grid'][1]:.3f}), "
        f"ARCO-ERA5 37-level.",
        f"anchored at {surface_pressure:.1f} hPa / {altitude:.3f} km from {source}; "
        f"{water:.2f} mm vertical water.",
        f"{args.layering} layers."
        + (" pressure_bar is each layer's air-weighted mean pressure." if args.layering == "weighted" else ""),
        "VMR columns are moist-air volume mixing ratios, dimensionless.",
    ])
    print(f"wrote {args.output} with {rows} layers")


if __name__ == "__main__":
    main()
