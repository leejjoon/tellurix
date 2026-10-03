"""The window settings and the solar source lookup, without fitting anything."""

import dataclasses
import json
from pathlib import Path

import pytest

from tellurix_fts.window import STAGES, WindowSettings, build_bounds, solar_source_for


def test_settings_are_frozen_and_default_like_the_command_line():
    settings = WindowSettings(spectrum=Path("s.txt"), profile=Path("p.csv"))

    assert settings.stages == "continuum,velocity,columns"
    assert settings.accuracy_mode == "mt_ckd"
    assert settings.pin == ()
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
