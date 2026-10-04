from pathlib import Path

import numpy as np
import pytest

from tellurix import AtmosphereProfile, SpectralOrder, load_atmosphere_csv, load_mipas_profile

DATA = Path(__file__).resolve().parent / "data"


def test_profile_rejects_pressure_in_wrong_order():
    with pytest.raises(ValueError, match="increase"):
        AtmosphereProfile(
            pressure_edges_bar=[1.0, 0.1],
            temperature_k=[280.0],
            altitude_km=[2.0],
            vmr={"H2O": [0.001]},
        )


def test_order_defaults_mask_and_source():
    order = SpectralOrder([1000.0, 1000.1, 1000.2], [1.0] * 3, [0.1] * 3)
    np.testing.assert_array_equal(order.mask, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(order.source_flux, np.ones(3))


def test_example_profile_loads_all_target_species():
    profile = load_atmosphere_csv(str(DATA / "profiles/example_midlatitude.csv"))
    assert len(profile.temperature_k) == 6
    assert set(profile.vmr) == {"H2O", "CO2", "CH4", "O2", "CO", "N2O"}
    assert np.all(profile.air_column_cm2 > 0.0)


def test_mipas_profile_is_converted_from_levels_to_top_down_layers(tmp_path):
    mipas = tmp_path / "profile.atm"
    mipas.write_text(
        """3 ! levels
*HGT [km]
0 2 10
*PRE [hPa]
1000 800 250
*TEM [K]
290 275 230
*H2O [ppmv]
10000 5000 100
*END
"""
    )
    profile = load_mipas_profile(mipas, observatory_altitude_km=2.0, species=("H2O",))
    np.testing.assert_allclose(profile.pressure_edges_bar, [0.25, 0.8])
    np.testing.assert_allclose(profile.temperature_k, [252.5])
    np.testing.assert_allclose(profile.vmr["H2O"], [0.00255])


def test_a_profile_without_a_mean_pressure_keeps_each_term_on_its_own():
    """Lines on the geometric mean, continua on the arithmetic, as before."""

    profile = AtmosphereProfile([0.1, 0.4], [250.0], [5.0], {"H2O": [1.0e-3]})

    assert profile.pressure_layer_bar == pytest.approx([0.2])
    assert profile.continuum_pressure_bar == pytest.approx([0.25])


def test_a_mean_pressure_is_used_by_lines_and_continua_alike():
    profile = AtmosphereProfile([0.1, 0.4], [250.0], [5.0], {"H2O": [1.0e-3]},
                                mean_pressure_bar=[0.24])

    assert profile.pressure_layer_bar == pytest.approx([0.24])
    assert profile.continuum_pressure_bar == pytest.approx([0.24])


@pytest.mark.parametrize("mean_pressure", ([0.1], [0.5], [0.2, 0.3], [np.nan]))
def test_a_mean_pressure_outside_its_layer_is_rejected(mean_pressure):
    with pytest.raises(ValueError, match="mean pressure"):
        AtmosphereProfile([0.1, 0.4], [250.0], [5.0], {"H2O": [1.0e-3]},
                          mean_pressure_bar=mean_pressure)
