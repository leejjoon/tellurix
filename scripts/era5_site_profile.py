#!/usr/bin/env python
"""Build a layer profile for one IGRINS night from ERA5 instead of a lapse rate.

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

    uv run --with aiohttp python scripts/era5_site_profile.py \\
        --spec data/igrins/20181220_*/SDCH_*.spec.fits \\
        --output data/profiles/dct_2018_era5.csv

Data comes from ARCO-ERA5 on Google Cloud, which is public and needs no
credentials, unlike the Copernicus CDS API. One hourly timestep of one variable
is a single global chunk of about 50 MB, so a profile costs three of those.

**The grid cell's orography is not the observatory.** At 0.25 degrees the cell
containing DCT has a surface elevation of 1844 m against the telescope's 2360 m,
and Cerro Pachon and Mt Locke are worse. That does not matter here, because the
column is anchored at the *station pressure from the frame's own header* and
integrated upward: ERA5 supplies T and q as functions of pressure, which is a
real atmospheric column regardless of where the model thinks the ground is. The
levels ERA5 reports below the telescope are extrapolated and are discarded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ARCO_ERA5 = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
             "full_37-1h-0p25deg-chunk-1.zarr-v3")
EPOCH = datetime(1900, 1, 1, tzinfo=timezone.utc)
_G0 = 9.80665
_BOLTZMANN_ERG_K = 1.380649e-16
_AVOGADRO = 6.02214076e23
_DRY_MOLAR_MASS_G_MOL = 28.9647
_WATER_MOLAR_MASS_G_MOL = 18.01528
_EARTH_RADIUS_KM = 6371.0

# Where each site actually is. ERA5 needs a latitude and longitude; the reader's
# SITES table carries only the altitude, because that is all the header
# normalisation needs.
SITE_COORDINATES = {
    "McDonald Observatory": (30.6717, -104.0225),
    "Lowell Discovery Telescope": (34.7444, -111.4223),
    "Gemini South": (-30.2408, -70.7367),
}


def _gravity(altitude_km):
    return _G0 * (_EARTH_RADIUS_KM / (_EARTH_RADIUS_KM + altitude_km)) ** 2


def specific_humidity_to_vmr(q):
    """Moist-air volume mixing ratio from specific humidity.

    q is a mass fraction of moist air; the profile stores number fractions.
    """

    q = np.clip(np.asarray(q, dtype=float), 0.0, 0.999)
    ratio = (_DRY_MOLAR_MASS_G_MOL / _WATER_MOLAR_MASS_G_MOL) * q / (1.0 - q)
    return ratio / (1.0 + ratio)


def fetch_column(latitude, longitude, when):
    """ERA5 temperature, specific humidity and geopotential height over a point."""

    import fsspec
    import zarr

    store = zarr.open_consolidated(fsspec.get_mapper(ARCO_ERA5), mode="r")
    latitudes = store["latitude"][:]
    longitudes = store["longitude"][:]
    index_lat = int(np.abs(latitudes - latitude).argmin())
    index_lon = int(np.abs(longitudes - (longitude % 360.0)).argmin())
    index_time = int((when - EPOCH).total_seconds() // 3600)
    if not 0 <= index_time < store["time"].shape[0]:
        raise SystemExit(f"{when:%Y-%m-%d %H:%M} is outside the ERA5 archive")
    return {
        "level_hpa": store["level"][:].astype(float),
        "temperature_k": store["temperature"][index_time, :, index_lat, index_lon].astype(float),
        "specific_humidity": store["specific_humidity"][index_time, :, index_lat, index_lon].astype(float),
        "height_m": store["geopotential"][index_time, :, index_lat, index_lon].astype(float) / _G0,
        "grid": (float(latitudes[index_lat]), float(longitudes[index_lon])),
        "grid_surface_m": float(store["geopotential_at_surface"][index_time, index_lat, index_lon]) / _G0,
        "time": when,
    }


def station_pressure_from_era5(column, site_altitude_km):
    """The pressure at the telescope, read off ERA5's own geopotential.

    The header's BARPRESS is the better number when it exists, but every Gemini
    South frame from 2020 on drops it -- 31% of the archive -- and without an
    anchor no profile can be built at all. Interpolating ERA5's height-pressure
    relation to the site altitude supplies one. Checked against the three nights
    that do carry the card, it agrees to 0.4-1.7 hPa, or 0.2%.
    """

    order = np.argsort(column["height_m"])
    height = column["height_m"][order]
    pressure = column["level_hpa"][order]
    site_m = site_altitude_km * 1000.0
    if not height[0] <= site_m <= height[-1]:
        raise SystemExit(f"the site at {site_m:.0f} m is outside ERA5's height range")
    return float(np.exp(np.interp(site_m, height, np.log(pressure))))


def build_profile(column, surface_pressure_hpa, site_altitude_km, dry_vmr, edges_km):
    """Layers above the site, with ERA5's temperature and water."""

    edges = np.asarray(edges_km, dtype=float)
    site_m = site_altitude_km * 1000.0
    above = column["height_m"] > site_m
    if above.sum() < 8:
        raise SystemExit("ERA5 reports too few levels above the site")
    # ERA5 runs top-down; sort by height so interpolation is monotonic.
    order = np.argsort(column["height_m"])
    height_km = (column["height_m"][order] - site_m) / 1000.0
    temperature_profile = column["temperature_k"][order]
    water_profile = specific_humidity_to_vmr(column["specific_humidity"][order])

    fine = np.linspace(0.0, edges[-1], 40001)
    fine_temperature = np.interp(fine, height_km, temperature_profile)
    gravity_cm_s2 = _gravity(site_altitude_km + fine) * 100.0
    scale = _DRY_MOLAR_MASS_G_MOL * gravity_cm_s2 / (_AVOGADRO * _BOLTZMANN_ERG_K)
    integrand = scale / fine_temperature
    log_pressure = np.log(surface_pressure_hpa) - np.concatenate(
        [[0.0], np.cumsum(np.diff(fine) * 1.0e5 * 0.5 * (integrand[1:] + integrand[:-1]))]
    )
    edge_pressure_hpa = np.exp(np.interp(edges, fine, log_pressure))

    center = 0.5 * (edges[:-1] + edges[1:])
    temperature = np.interp(center, height_km, temperature_profile)
    # Interpolate water in the log, which is how it varies, and never below the
    # stratospheric value ERA5 itself reports at the top.
    water_vmr = np.exp(np.interp(center, height_km, np.log(np.maximum(water_profile, 1.0e-12))))
    thickness_cm = np.diff(edges) * 1.0e5
    center_pressure_hpa = np.exp(np.interp(center, fine, log_pressure))
    number_density = center_pressure_hpa * 1000.0 / (_BOLTZMANN_ERG_K * temperature)

    vmr = {"H2O": water_vmr}
    for species, dry in dry_vmr.items():
        vmr[species] = dry * (1.0 - water_vmr)

    water_column = float(np.sum(water_vmr * number_density * thickness_cm))
    precipitable_water_mm = water_column * _WATER_MOLAR_MASS_G_MOL / _AVOGADRO * 10.0

    order_out = np.argsort(center)[::-1]
    return {
        "pressure_top_bar": edge_pressure_hpa[1:][order_out] / 1000.0,
        "pressure_bottom_bar": edge_pressure_hpa[:-1][order_out] / 1000.0,
        "temperature_k": temperature[order_out],
        "altitude_km": (site_altitude_km + center)[order_out],
        "mean_molecular_weight_g_mol": np.full(len(order_out), _DRY_MOLAR_MASS_G_MOL),
        "gravity_m_s2": _gravity(site_altitude_km + center)[order_out],
        **{name: values[order_out] for name, values in vmr.items()},
        "_precipitable_water_mm": precipitable_water_mm,
        "_lapse_k_km": float((temperature_profile[np.argmin(np.abs(height_km - 0.1))]
                              - temperature_profile[np.argmin(np.abs(height_km - 3.0))]) / 2.9),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(root / "scripts"))
    from make_site_profile import DEFAULT_EDGES_KM, EPOCH_DRY_VMR

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True, nargs="+",
                        help="the night's PLP .spec.fits files; the site, the time and the "
                             "station pressure all come from their headers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", default="2020", choices=sorted(EPOCH_DRY_VMR))
    parser.add_argument("--anchor", choices=("header", "era5", "auto"), default="auto",
                        help="where the station pressure comes from. 'auto' prefers the "
                             "header and falls back to ERA5, which is what every Gemini "
                             "South frame from 2020 needs.")
    args = parser.parse_args()

    from tellurix import read_igrins_observation

    sites, pressures, times = set(), [], []
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
    middle = min(times) + (max(times) - min(times)) / 2
    when = (middle + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)

    print(f"{site} ({latitude:+.4f}, {longitude:+.4f}) at {altitude:.3f} km")
    print(f"  {len(args.spec)} frames spanning {min(times):%H:%M}-{max(times):%H:%M} UT; "
          f"ERA5 at {when:%Y-%m-%d %H:00} UT")
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

    profile = build_profile(column, surface_pressure, altitude,
                            EPOCH_DRY_VMR[args.epoch], DEFAULT_EDGES_KM)
    water = profile.pop("_precipitable_water_mm")
    lapse = profile.pop("_lapse_k_km")
    print(f"  ERA5 gives {water:.2f} mm of precipitable water and a {lapse:.2f} K/km "
          f"lapse over the first 3 km (the analytic profile assumes 6.5)")

    names = list(profile)
    lines = [
        f"# {site}, {args.epoch} trace-gas abundances, generated by scripts/era5_site_profile.py.",
        f"# ERA5 at {when:%Y-%m-%dT%H:00}Z, cell ({column['grid'][0]:+.3f}, {column['grid'][1]:.3f}), "
        f"ARCO-ERA5 37-level.",
        f"# anchored at {surface_pressure:.1f} hPa / {altitude:.3f} km from {source}; "
        f"{water:.2f} mm vertical water.",
        "# VMR columns are moist-air volume mixing ratios, dimensionless.",
        ",".join(names),
    ]
    for index in range(len(profile["temperature_k"])):
        values = []
        for name in names:
            value = profile[name][index]
            values.append(f"{value:.6f}" if name in ("temperature_k", "altitude_km",
                                                     "mean_molecular_weight_g_mol",
                                                     "gravity_m_s2") else f"{value:.8e}")
        lines.append(",".join(values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output} with {len(profile['temperature_k'])} layers")


if __name__ == "__main__":
    main()
