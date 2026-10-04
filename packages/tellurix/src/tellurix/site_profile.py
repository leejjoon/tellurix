"""A hydrostatic layer profile for an observing site and epoch.

The packaged ``example_midlatitude.csv`` is illustrative and carries present-day
trace-gas abundances. Retrievals that lean on a well-mixed species to break the
airmass degeneracy need the abundances of the epoch the data were taken in: CO2
rose from 357 ppm in 1993-94 to about 420 ppm, and that 17.6% error would map
one for one into the retrieved airmass.

The AFGL standard atmospheres ship with the package (``afgl/``), as written by
``scripts/extract_afgl_profiles.py``; they carry all 47 HITRAN molecules with
their vertical structure, including the stratospheric ones a single mixing ratio
cannot represent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence, Union

import numpy as np

_BOLTZMANN_ERG_K = 1.380649e-16
_AVOGADRO = 6.02214076e23
_EARTH_RADIUS_KM = 6371.0
_DRY_MOLAR_MASS_G_MOL = 28.9647
_WATER_MOLAR_MASS_G_MOL = 18.01528
_SEA_LEVEL_GRAVITY = 9.80665

AFGL_DIRECTORY = Path(__file__).resolve().parent / "afgl"

# Gases whose abundance has trended, as dry-air volume mixing ratios at the
# surface. Everything else comes from the AFGL standard atmosphere. The 1993-94
# column is the epoch of the Hinkle, Wallace & Livingston Arcturus atlas;
# 1990-12 is the epoch of the NSO solar spectra. NOAA GML global annual means.
EPOCH_SURFACE_VMR = {
    # For niratl and ftsspec_830626_{2,3}, 1983 June. CO2 is NOAA GML's global
    # annual mean (342.53 ppm). Its global CH4 starts in 1984 (1644.84 ppb) and
    # is taken back one year at the mid-1980s growth of ~13 ppb/yr; N2O is the
    # 1990 value below taken back seven years at ~0.7 ppb/yr. Every column is
    # fitted, so these seed the fit and the scan's ranking and nothing else.
    "1983": {"CO2": 342.5e-6, "CH4": 1.632e-6, "N2O": 0.3035e-6},
    "1990": {"CO2": 354.4e-6, "CH4": 1.714e-6, "N2O": 0.3085e-6},
    "1994": {"CO2": 357.0e-6, "CH4": 1.72e-6, "N2O": 0.310e-6},
    "2020": {"CO2": 414.0e-6, "CH4": 1.87e-6, "N2O": 0.333e-6},
}
# The surface values above plus the two that have not trended, for a profile
# that carries only the well-mixed species.
EPOCH_DRY_VMR = {
    epoch: {**values, "CO": 0.10e-6, "O2": 0.2095}
    for epoch, values in EPOCH_SURFACE_VMR.items()
}
AFGL_MODELS = ("tropical", "midlatitude_summer", "midlatitude_winter",
               "subarctic_summer", "subarctic_winter", "us_standard_1976")

# Height above the site, in km. Fine near the ground where the water sits, and
# with four layers above 0.25 bar where the well-mixed line cores form.
DEFAULT_EDGES_KM = (0.0, 0.3, 0.7, 1.2, 1.8, 2.6, 3.6, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0)

# How a layer's state is formed from the atmosphere it spans. "weighted" is
# LBLRTM's Curtis-Godson averaging (lblatm.f90, ALAYER and FPACK): pressure and
# temperature weighted by air density, every gas's amount integrated. "centre"
# samples the layer centre and is how every profile before 2026-10 was built.
# Against a 150-layer reference on two ERA5 nights, after the column scales are
# fitted, the 12 default layers are off by 0.04-0.06% (weighted) and 0.05-0.20%
# (centre) at the 99th percentile -- both under IGRINS noise.
LAYERINGS = ("weighted", "centre")

# A dry-air VMR is a number, a per-layer array, or a function of altitude above
# sea level in km -- the last is what lets a weighted layer average a gas with
# vertical structure rather than sample it.
DryVMR = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]

# Columns written with fixed decimals; everything else is a mixing ratio and
# written in exponent form.
_FIXED_COLUMNS = ("temperature_k", "altitude_km", "gravity_m_s2", "mean_molecular_weight_g_mol")


def load_afgl(model: str, directory: Path | None = None) -> dict:
    """The AFGL standard atmosphere, as arrays over its own 50-level grid."""

    path = Path(directory or AFGL_DIRECTORY) / f"{model}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/extract_afgl_profiles.py")
    rows = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    names = rows[0].split(",")
    values = np.asarray([[float(v) for v in row.split(",")] for row in rows[1:]])
    return {name: values[:, index] for index, name in enumerate(names)}


def afgl_dry_vmr(model: str, altitude_km: np.ndarray, epoch: str,
                 directory: Path | None = None) -> dict:
    """Every AFGL molecule on the requested altitudes, as dry-air fractions.

    Interpolated in the log, because these span fourteen orders of magnitude
    and ozone changes by a factor of 260 between the surface and 38 km. Water
    is excluded: it comes from the site's own precipitable-water argument or
    from ERA5, never from a climatology.
    """

    table = load_afgl(model, directory)
    grid = table["altitude_km"]
    surface = EPOCH_SURFACE_VMR[epoch]
    out = {}
    for name, ppmv in table.items():
        if name in ("altitude_km", "H2O"):
            continue
        dry = np.exp(np.interp(altitude_km, grid, np.log(np.maximum(ppmv, 1e-30)))) * 1.0e-6
        # Scale the trended gases to the epoch by their surface ratio, which
        # preserves the AFGL vertical shape.
        if name in surface:
            dry = dry * (surface[name] / (ppmv[0] * 1.0e-6))
        out[name] = dry
    return out


def _gravity(altitude_km: np.ndarray) -> np.ndarray:
    return _SEA_LEVEL_GRAVITY * (_EARTH_RADIUS_KM / (_EARTH_RADIUS_KM + altitude_km)) ** 2


def _dry_profile(dry_vmr, altitude_km: np.ndarray) -> dict:
    """Every dry VMR evaluated at the given altitudes above sea level."""

    if callable(dry_vmr):
        return dict(dry_vmr(altitude_km))
    return {name: value(altitude_km) if callable(value) else value
            for name, value in dry_vmr.items()}


def weighted_layers(
    height_km: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    water_vmr: np.ndarray,
    dry_vmr,
    site_altitude_km: float,
    edges_km,
) -> dict:
    """Layers of a finely sampled atmosphere by LBLRTM's Curtis-Godson averaging.

    ``height_km`` is height above the site, increasing from ``edges_km[0]`` to at
    least ``edges_km[-1]``, sampled finely enough that the trapezoid rule is
    exact to the precision wanted; ``pressure_hpa``, ``temperature_k`` and the
    moist-air ``water_vmr`` are on it. ``dry_vmr`` maps species to dry-air VMRs
    as numbers, per-layer arrays, or functions of altitude above sea level, or is
    one function returning that mapping.

    Pressure and temperature are weighted by air density and each gas's VMR is
    its integrated amount over the integrated air, as LBLRTM's ALAYER and FPACK
    form them. The amounts then agree with the hydrostatic column the model
    computes from the edges, which centre sampling does not: there a layer's
    water is its centre value, 9% below its mean for a 3 km layer at a 2 km
    scale height. Returns per-layer arrays top to bottom, as the builders do,
    including the air-weighted pressure as ``pressure_bar``.
    """

    height = np.asarray(height_km, dtype=float)
    edges = np.asarray(edges_km, dtype=float)
    if height[0] > edges[0] or height[-1] < edges[-1] or np.any(np.diff(height) <= 0.0):
        raise ValueError("the fine grid must increase and span the layer edges")
    # Number density up to a constant, which cancels in every ratio below.
    density = np.asarray(pressure_hpa, dtype=float) / np.asarray(temperature_k, dtype=float)

    def per_layer(values):
        cumulative = np.concatenate(
            [[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(height))])
        return np.diff(np.interp(edges, height, cumulative))

    air = per_layer(density)
    pressure = per_layer(density * pressure_hpa) / air
    temperature = per_layer(density * temperature_k) / air
    water = per_layer(density * water_vmr) / air

    nlayer = len(edges) - 1
    vmr = {"H2O": water}
    for species, dry in _dry_profile(dry_vmr, site_altitude_km + height).items():
        dry = np.asarray(dry, dtype=float)
        if dry.shape == (nlayer,):
            # A per-layer value is constant within its layer, so only the water
            # dilution varies across it.
            vmr[species] = dry * (1.0 - water)
        else:
            vmr[species] = per_layer(density * np.broadcast_to(dry, height.shape)
                                     * (1.0 - water_vmr)) / air

    edge_pressure_hpa = np.exp(np.interp(edges, height, np.log(pressure_hpa)))
    center = 0.5 * (edges[:-1] + edges[1:])
    order = np.argsort(center)[::-1]
    return {
        "pressure_top_bar": edge_pressure_hpa[1:][order] / 1000.0,
        "pressure_bottom_bar": edge_pressure_hpa[:-1][order] / 1000.0,
        "pressure_bar": pressure[order] / 1000.0,
        "temperature_k": temperature[order],
        "altitude_km": (site_altitude_km + center)[order],
        "mean_molecular_weight_g_mol": np.full(nlayer, _DRY_MOLAR_MASS_G_MOL),
        "gravity_m_s2": _gravity(site_altitude_km + center)[order],
        **{name: values[order] for name, values in vmr.items()},
    }


def build_site_profile(
    surface_pressure_hpa: float,
    surface_temperature_k: float,
    site_altitude_km: float,
    lapse_rate_k_km: float,
    tropopause_temperature_k: float,
    water_scale_height_km: float,
    stratospheric_water_vmr: float,
    precipitable_water_mm: float,
    dry_vmr,
    edges_km: tuple = DEFAULT_EDGES_KM,
    layering: str = "weighted",
) -> dict:
    """Return per-layer arrays, ordered top to bottom as the model expects.

    ``dry_vmr`` takes what :func:`weighted_layers` takes. ``layering`` is one of
    :data:`LAYERINGS`; ``"centre"`` reproduces profiles built before weighted
    layers existed.
    """

    if layering not in LAYERINGS:
        raise ValueError(f"layering must be one of {LAYERINGS}")
    edges = np.asarray(edges_km, dtype=float)
    if edges.ndim != 1 or np.any(np.diff(edges) <= 0.0) or edges[0] != 0.0:
        raise ValueError("layer edges must start at the site and increase")

    # Integrate hydrostatic equilibrium on a fine sub-grid so the layer edges
    # are not sensitive to how coarsely the layers themselves are chosen.
    fine = np.linspace(0.0, edges[-1], 40001)
    fine_temperature = np.maximum(surface_temperature_k - lapse_rate_k_km * fine, tropopause_temperature_k)
    # CGS throughout: g/mol times cm/s2 over the gas constant in erg/(mol K).
    gravity_cm_s2 = _gravity(site_altitude_km + fine) * 100.0
    scale = _DRY_MOLAR_MASS_G_MOL * gravity_cm_s2 / (_AVOGADRO * _BOLTZMANN_ERG_K)
    integrand = scale / fine_temperature
    log_pressure = np.log(surface_pressure_hpa) - np.concatenate(
        [[0.0], np.cumsum(np.diff(fine) * 1.0e5 * 0.5 * (integrand[1:] + integrand[:-1]))]
    )
    edge_pressure_hpa = np.exp(np.interp(edges, fine, log_pressure))
    target_column = precipitable_water_mm * 0.1 / _WATER_MOLAR_MASS_G_MOL * _AVOGADRO

    if layering == "weighted":
        # The same water as below -- an exponential in number density on a
        # stratospheric mixing-ratio floor -- integrated rather than sampled.
        fine_density = np.exp(log_pressure) * 1000.0 / (_BOLTZMANN_ERG_K * fine_temperature)
        shape = np.exp(-fine / water_scale_height_km)

        def integral(values):
            return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(fine))) * 1.0e5

        remaining = target_column - stratospheric_water_vmr * integral(fine_density)
        if remaining <= 0.0:
            raise ValueError("requested precipitable water is below the stratospheric floor")
        fine_water = stratospheric_water_vmr + remaining / integral(shape) * shape / fine_density
        if np.any(fine_water >= 1.0):
            raise ValueError("water vapour mixing ratio reached unity")
        profile = weighted_layers(fine, np.exp(log_pressure), fine_temperature, fine_water,
                                  dry_vmr, site_altitude_km, edges)
        profile["_precipitable_water_mm"] = precipitable_water_mm
        return profile

    center = 0.5 * (edges[:-1] + edges[1:])
    temperature = np.maximum(surface_temperature_k - lapse_rate_k_km * center, tropopause_temperature_k)
    thickness_cm = np.diff(edges) * 1.0e5
    center_pressure_hpa = np.exp(np.interp(center, fine, log_pressure))
    number_density = center_pressure_hpa * 1000.0 / (_BOLTZMANN_ERG_K * temperature)

    # An exponential water profile on a stratospheric floor, renormalized so the
    # vertical column equals the requested precipitable water.
    shape = np.exp(-center / water_scale_height_km)
    floor_column = stratospheric_water_vmr * number_density * thickness_cm
    scaled = shape * thickness_cm
    remaining = target_column - float(np.sum(floor_column))
    if remaining <= 0.0:
        raise ValueError("requested precipitable water is below the stratospheric floor")
    water_column = scaled * remaining / float(np.sum(scaled)) + floor_column
    water_vmr = water_column / (number_density * thickness_cm)
    if np.any(water_vmr >= 1.0):
        raise ValueError("water vapour mixing ratio reached unity")

    # The public profile stores fractions of moist air, so the well-mixed
    # species are their dry-air values diluted by the local water.
    vmr = {"H2O": water_vmr}
    for species, dry in _dry_profile(dry_vmr, site_altitude_km + center).items():
        vmr[species] = dry * (1.0 - water_vmr)

    order = np.argsort(center)[::-1]
    return {
        "pressure_top_bar": edge_pressure_hpa[1:][order] / 1000.0,
        "pressure_bottom_bar": edge_pressure_hpa[:-1][order] / 1000.0,
        "temperature_k": temperature[order],
        "altitude_km": (site_altitude_km + center)[order],
        "mean_molecular_weight_g_mol": np.full(len(order), _DRY_MOLAR_MASS_G_MOL),
        "gravity_m_s2": _gravity(site_altitude_km + center)[order],
        **{name: values[order] for name, values in vmr.items()},
        "_precipitable_water_mm": precipitable_water_mm,
    }


def write_profile_csv(path: Path, profile: Mapping[str, np.ndarray],
                      comments: Sequence[str]) -> int:
    """Write per-layer arrays as the CSV ``load_atmosphere_csv`` reads.

    Keys starting with an underscore are diagnostics, not columns, and are
    skipped. Returns the number of layers written.
    """

    columns = [name for name in profile if not name.startswith("_")]
    rows = len(profile["temperature_k"])
    lines = [f"# {comment}" for comment in comments] + [",".join(columns)]
    for index in range(rows):
        lines.append(",".join(
            f"{profile[name][index]:.6f}" if name in _FIXED_COLUMNS
            else f"{profile[name][index]:.8e}" for name in columns))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
