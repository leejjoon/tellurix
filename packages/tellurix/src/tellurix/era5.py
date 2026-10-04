"""ERA5 temperature and water over a site, and the layers built from them.

ERA5 is level data, as LBLRTM's input is, so a profile built from it can be
layered either way ``site_profile.LAYERINGS`` offers. Data comes from
ARCO-ERA5 on Google Cloud, which is public and needs no credentials, unlike the
Copernicus CDS API. One hourly timestep of one variable is a single global chunk
of about 50 MB, so a column costs three of those. Fetching needs the ``era5``
extra (zarr, fsspec, aiohttp); building layers from a column does not.

**The grid cell's orography is not the observatory.** At 0.25 degrees the cell
containing DCT has a surface elevation of 1844 m against the telescope's 2360 m,
and Cerro Pachon and Mt Locke are worse. That does not matter, because the
column is anchored at a station pressure and integrated upward: ERA5 supplies T
and q as functions of pressure, which is a real atmospheric column regardless of
where the model thinks the ground is. The levels ERA5 reports below the
telescope are extrapolated and are discarded.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .site_profile import _gravity, weighted_layers

ARCO_ERA5 = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
             "full_37-1h-0p25deg-chunk-1.zarr-v3")
EPOCH = datetime(1900, 1, 1, tzinfo=timezone.utc)
_G0 = 9.80665
_BOLTZMANN_ERG_K = 1.380649e-16
_AVOGADRO = 6.02214076e23
_DRY_MOLAR_MASS_G_MOL = 28.9647
_WATER_MOLAR_MASS_G_MOL = 18.01528


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
        raise ValueError(f"{when:%Y-%m-%d %H:%M} is outside the ERA5 archive")
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
        raise ValueError(f"the site at {site_m:.0f} m is outside ERA5's height range")
    return float(np.exp(np.interp(site_m, height, np.log(pressure))))


def build_era5_profile(column, surface_pressure_hpa, site_altitude_km, dry_vmr, edges_km,
                  layering="weighted"):
    """Layers above the site, with ERA5's temperature and water.

    ERA5 is level data, as LBLRTM's input is: temperature is interpolated
    linearly in height between levels and water in its log, and ``layering``
    either averages the resulting atmosphere over each layer
    (``tellurix.site_profile.weighted_layers``) or samples it at the centre,
    as every profile before 2026-10 did.
    """

    edges = np.asarray(edges_km, dtype=float)
    site_m = site_altitude_km * 1000.0
    above = column["height_m"] > site_m
    if above.sum() < 8:
        raise ValueError("ERA5 reports too few levels above the site")
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
    lapse = float((temperature_profile[np.argmin(np.abs(height_km - 0.1))]
                   - temperature_profile[np.argmin(np.abs(height_km - 3.0))]) / 2.9)

    if layering == "weighted":
        fine_pressure = np.exp(log_pressure)
        fine_water = np.exp(np.interp(fine, height_km, np.log(np.maximum(water_profile, 1.0e-12))))
        profile = weighted_layers(fine, fine_pressure, fine_temperature, fine_water,
                                  dry_vmr, site_altitude_km, edges)
        density = fine_pressure * 1000.0 / (_BOLTZMANN_ERG_K * fine_temperature) * fine_water
        water_column = float(np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(fine))) * 1.0e5
        profile["_precipitable_water_mm"] = water_column * _WATER_MOLAR_MASS_G_MOL / _AVOGADRO * 10.0
        profile["_lapse_k_km"] = lapse
        return profile
    if layering != "centre":
        raise ValueError(f"unknown layering {layering!r}")

    center = 0.5 * (edges[:-1] + edges[1:])
    temperature = np.interp(center, height_km, temperature_profile)
    # Interpolate water in the log, which is how it varies, and never below the
    # stratospheric value ERA5 itself reports at the top.
    water_vmr = np.exp(np.interp(center, height_km, np.log(np.maximum(water_profile, 1.0e-12))))
    thickness_cm = np.diff(edges) * 1.0e5
    center_pressure_hpa = np.exp(np.interp(center, fine, log_pressure))
    number_density = center_pressure_hpa * 1000.0 / (_BOLTZMANN_ERG_K * temperature)

    dry_at_centres = (dry_vmr(site_altitude_km + center) if callable(dry_vmr) else dry_vmr)
    vmr = {"H2O": water_vmr}
    for species, dry in dry_at_centres.items():
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
        "_lapse_k_km": lapse,
    }
