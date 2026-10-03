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
from typing import Mapping, Sequence

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


def build_site_profile(
    surface_pressure_hpa: float,
    surface_temperature_k: float,
    site_altitude_km: float,
    lapse_rate_k_km: float,
    tropopause_temperature_k: float,
    water_scale_height_km: float,
    stratospheric_water_vmr: float,
    precipitable_water_mm: float,
    dry_vmr: dict,
    edges_km: tuple = DEFAULT_EDGES_KM,
) -> dict:
    """Return per-layer arrays, ordered top to bottom as the model expects."""

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

    center = 0.5 * (edges[:-1] + edges[1:])
    temperature = np.maximum(surface_temperature_k - lapse_rate_k_km * center, tropopause_temperature_k)
    thickness_cm = np.diff(edges) * 1.0e5
    center_pressure_hpa = np.exp(np.interp(center, fine, log_pressure))
    number_density = center_pressure_hpa * 1000.0 / (_BOLTZMANN_ERG_K * temperature)

    # An exponential water profile on a stratospheric floor, renormalized so the
    # vertical column equals the requested precipitable water.
    shape = np.exp(-center / water_scale_height_km)
    floor_column = stratospheric_water_vmr * number_density * thickness_cm
    target_column = precipitable_water_mm * 0.1 / _WATER_MOLAR_MASS_G_MOL * _AVOGADRO
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
    for species, dry in dry_vmr.items():
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
