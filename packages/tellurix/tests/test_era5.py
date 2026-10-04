"""Layers from an ERA5-shaped column, with no network."""

import numpy as np
import pytest

from tellurix import AtmosphereProfile
from tellurix.era5 import build_era5_profile, specific_humidity_to_vmr, station_pressure_from_era5
from tellurix.site_profile import DEFAULT_EDGES_KM, EPOCH_DRY_VMR

AVOGADRO = 6.02214076e23


def _column():
    """An isothermal-ish column on ERA5's 37 levels, top down as ERA5 gives it."""

    level_hpa = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
                          300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
                          875, 900, 925, 950, 975, 1000], dtype=float)
    height_m = 7400.0 * np.log(1013.25 / level_hpa)
    temperature = np.maximum(288.0 - 6.5e-3 * height_m, 216.65)
    # Water falling off with a 2 km scale height onto a stratospheric floor.
    specific_humidity = 6.0e-3 * np.exp(-height_m / 2000.0) + 3.0e-6
    return {"level_hpa": level_hpa, "temperature_k": temperature,
            "specific_humidity": specific_humidity, "height_m": height_m}


def _profile(layers):
    edges = np.concatenate([layers["pressure_top_bar"][:1], layers["pressure_bottom_bar"]])
    return AtmosphereProfile(edges, layers["temperature_k"], layers["altitude_km"],
                             {k: layers[k] for k in ("H2O", "CO2")},
                             layers["mean_molecular_weight_g_mol"], layers["gravity_m_s2"],
                             mean_pressure_bar=layers.get("pressure_bar"))


def test_specific_humidity_is_a_mass_fraction_of_moist_air():
    # 1 g/kg of water: (q/18.015) / (q/18.015 + (1-q)/28.965) mol/mol.
    assert specific_humidity_to_vmr(1.0e-3) == pytest.approx(1.6068e-3, rel=1e-4)


def test_the_station_pressure_is_read_off_the_height_pressure_relation():
    column = _column()
    expected = 1013.25 * np.exp(-2360.0 / 7400.0)
    assert station_pressure_from_era5(column, 2.360) == pytest.approx(expected, rel=1e-3)


def test_weighted_layers_carry_the_water_the_column_holds():
    """The diagnostic and the profile agree, because both integrate the column."""

    column = _column()
    layers = build_era5_profile(column, 768.5, 2.360, EPOCH_DRY_VMR["2020"], DEFAULT_EDGES_KM)
    profile = _profile(layers)
    water_mm = float(np.sum(profile.air_column_cm2 * profile.vmr["H2O"])) * 18.01528 / AVOGADRO * 10.0

    assert water_mm == pytest.approx(layers["_precipitable_water_mm"], rel=1e-4)
    np.testing.assert_array_equal(profile.pressure_layer_bar, layers["pressure_bar"])


def test_centre_layering_writes_no_layer_pressure_and_more_water_in_thick_layers():
    column = _column()
    centre = build_era5_profile(column, 768.5, 2.360, EPOCH_DRY_VMR["2020"], DEFAULT_EDGES_KM,
                                layering="centre")
    weighted = build_era5_profile(column, 768.5, 2.360, EPOCH_DRY_VMR["2020"], DEFAULT_EDGES_KM)

    assert "pressure_bar" not in centre
    np.testing.assert_allclose(centre["pressure_top_bar"], weighted["pressure_top_bar"])
    # A layer's centre value of a gas falling off exponentially is below its
    # mean, by most in the thickest layers.
    assert centre["H2O"][4] < weighted["H2O"][4]


def test_an_unknown_layering_is_refused():
    with pytest.raises(ValueError, match="layering"):
        build_era5_profile(_column(), 768.5, 2.360, EPOCH_DRY_VMR["2020"], DEFAULT_EDGES_KM,
                           layering="lbl")
