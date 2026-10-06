"""The standard-fit settings: a run's order rule is its own, not a shared global."""

from types import SimpleNamespace

import numpy as np
import pytest

from tellurix_igrins.standard import ORDER_RULE, StandardFitSettings, stage_bounds, stages_for


def test_a_partial_order_rule_keeps_the_defaults_for_the_rest():
    settings = StandardFitSettings(order_rule={"throughput_floor": 0.45})

    assert settings.order_rule["throughput_floor"] == 0.45
    assert settings.order_rule["mask_hydrogen_kms"] == ORDER_RULE["mask_hydrogen_kms"]
    # Overriding one run's rule must not reach the default every other run reads,
    # which is what mutating the driver's module-level dict used to do.
    assert ORDER_RULE["throughput_floor"] == 0.25


def test_the_rule_cannot_be_changed_after_the_fact():
    settings = StandardFitSettings()
    with pytest.raises(TypeError):
        settings.order_rule["throughput_floor"] = 0.5
    with pytest.raises(TypeError):
        ORDER_RULE["throughput_floor"] = 0.5


def test_an_unknown_rule_key_is_refused():
    with pytest.raises(ValueError, match="throughput_flor"):
        StandardFitSettings(order_rule={"throughput_flor": 0.45})


def test_a_stellar_model_adds_the_stellar_stage():
    assert stages_for("flat") == ("continuum", "velocity", "columns")
    assert stages_for("a0v.npz")[-1] == "stellar"


def test_the_continuum_bound_comes_from_the_rule():
    parameters = SimpleNamespace(
        velocity_kms=0.0, lsf_sigma_kms=2.8, continuum_coeffs=np.zeros(4), log_jitter=-5.0,
        stellar_velocity_kms=0.0)
    bounds = stage_bounds("columns", ("H2O",), ("H2O",), parameters, 3, fit_stellar=False,
                          continuum_bound=1.5)

    assert bounds["continuum_0"] == (-2.0, 2.0)
    assert bounds["continuum_3"] == (-1.5, 1.5)
    assert bounds["H2O"] == (-2.0, 2.0)
    assert bounds["stellar_velocity_kms"] == (0.0, 0.0)


def test_a_fixed_column_is_pinned_in_every_stage_and_the_rest_stay_free():
    settings = StandardFitSettings(fixed_columns={"89": {"co2": -0.02}})
    fixed = settings.fixed_columns[89]
    assert dict(fixed) == {"CO2": -0.02}
    parameters = SimpleNamespace(
        velocity_kms=0.0, lsf_sigma_kms=2.8, continuum_coeffs=np.zeros(4), log_jitter=-5.0,
        stellar_velocity_kms=0.0)
    for stage in stages_for("a0v.npz"):
        bounds = stage_bounds(stage, ("H2O", "CO2"), ("H2O", "CO2"), parameters, 3,
                              fit_stellar=True, fixed_columns=fixed)
        assert bounds["CO2"] == (-0.02, -0.02)
    assert bounds["H2O"] == (-2.0, 2.0)
    with pytest.raises(TypeError):
        settings.fixed_columns[89]["CO2"] = 0.0


def test_a_whole_frame_summary_supersedes_the_shards_left_beside_it(tmp_path):
    from tellurix_igrins.standard import summary_paths

    for name in ("SDCK_1_0039.s0_summary.json", "SDCK_1_0039.s1_summary.json",
                 "SDCK_1_0039_summary.json", "SDCK_1_0055.s0_summary.json",
                 "SDCK_1_0055.s1_summary.json", "SDCK_1_0055_K80.npz"):
        (tmp_path / name).write_text("{}")
    names = [p.name for p in summary_paths(tmp_path)]
    # 0039 was refitted whole; 0055 only ever ran sharded and still reads.
    assert names == ["SDCK_1_0039_summary.json", "SDCK_1_0055.s0_summary.json",
                     "SDCK_1_0055.s1_summary.json"]
