"""The window settings and the solar source lookup, without fitting anything."""

import dataclasses
import json
from pathlib import Path

import pytest

import numpy as np

from tellurix_fts.window import STAGES, WindowSettings, build_bounds, solar_line_mask, solar_source_for


def test_settings_are_frozen_and_default_like_the_command_line():
    settings = WindowSettings(spectrum=Path("s.txt"), profile=Path("p.csv"))

    assert settings.stages == "continuum,velocity,columns"
    assert settings.accuracy_mode == "mt_ckd"
    assert settings.pin == ()
    # HITRAN's H2O pressure shifts unless AER's are asked for.
    assert settings.h2o_pressure_shift == "hitran"
    # AER's widths and every pixel fitting the telluric parameters, as before
    # either option existed.
    assert settings.pressure_width_scale == 1.0
    assert settings.telluric_from_clean_pixels is False
    # And Voigt lines: LBLRTM has no speed dependence.
    assert settings.speed_dependence == 0.0
    # And AER's strengths: no line-strength table.
    assert settings.line_strength_table is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.v1 = 1.0


def test_the_source_is_the_band_that_covers_the_window(tmp_path):
    for name, (lo, hi) in {"red": (1600.0, 1700.0), "nir": (1650.0, 2100.0)}.items():
        (tmp_path / f"solar_payne_zero_{name}.json").write_text(
            json.dumps({"wavelength_nm": [lo, hi]}))

    # 6000-6030 cm-1 is 1658.4-1666.7 nm: inside both, and the first sorted wins.
    assert solar_source_for(6000.0, 6030.0, tmp_path).name == "solar_payne_zero_nir.npz"
    with pytest.raises(ValueError, match="no solar source covers"):
        solar_source_for(1876.0, 1906.0, tmp_path)


def test_a_pinned_parameter_stays_pinned_in_every_stage():
    initial = dataclasses.make_dataclass("P", [
        "log_column_scales", "velocity_kms", "lsf_sigma_kms", "continuum_coeffs", "log_jitter",
        "stellar_velocity_kms"])({"H2O": 0.3}, 0.0, 0.5, [0.0, 0.0], -4.0, 0.0)
    for stage in STAGES:
        bounds = build_bounds(("H2O",), 1, STAGES[stage], initial, include_stellar=True,
                              pinned=("H2O",))
        assert bounds["H2O"] == (0.3, 0.3)
        assert bounds["wavelength_stretch"] == (0.0, 0.0)


def test_the_solar_line_mask_measures_depth_below_the_models_own_top():
    # Descending in wavenumber, as fit_window's arrays are.
    nu = np.linspace(9300.0, 9290.0, 2001)
    solar = 1.0 - 0.3 * np.exp(-0.5 * ((nu - 9295.0) / 0.05) ** 2)
    # A continuum 3% above the solar model's top, as deep in a water band: by
    # the continuum every pixel would look like a solar line.
    arrays = {"wavenumber_cm1": nu, "stellar_only_pixels": 0.97 * solar,
              "continuum": np.ones_like(nu)}
    mask = solar_line_mask(arrays, depth=0.01)
    assert mask[np.argmin(np.abs(nu - 9295.0))]
    assert not mask[np.abs(nu - 9295.0) > 0.5].any()
    assert 0.0 < mask.mean() < 0.05


def test_the_closing_envelope_lets_a_broad_wing_through_and_keeps_narrow_lines():
    nu = np.linspace(11710.0, 11680.0, 6001)
    # A Ca II-like wing: 30% deep at 11704, a few cm-1 wide, and a narrow line on it.
    wing = 1.0 - 0.3 / (1.0 + ((nu - 11704.0) / 3.0) ** 2)
    solar = wing * (1.0 - 0.2 * np.exp(-0.5 * ((nu - 11695.0) / 0.05) ** 2))
    arrays = {"wavenumber_cm1": nu, "stellar_only_pixels": solar, "continuum": np.ones_like(nu)}
    flank = (np.abs(nu - 11698.0) < 1.0)
    assert solar_line_mask(arrays, depth=0.01)[flank].mean() > 0.9
    closing = solar_line_mask(arrays, depth=0.01, envelope="closing")
    assert not closing[flank].any()
    assert closing[np.argmin(np.abs(nu - 11695.0))]
    with pytest.raises(ValueError):
        solar_line_mask(arrays, envelope="hull")
