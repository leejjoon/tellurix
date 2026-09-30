"""A night's calibration carries its standards' telluric model to other frames."""

import numpy as np
import pytest

from tellurix import MasterPattern, NightCalibration, OrderCalibration, write_record

NU = np.linspace(6000.0, 6080.0, 400)
TRUE_PATTERN = 0.02 * np.sin(np.linspace(0.0, 3.0, NU.size))


def _night(tmp_path, frames=5, with_cache=True):
    """A record of ``frames`` standards over one H order, and its npz cache."""

    rng = np.random.default_rng(1)
    pages = []
    for i in range(frames):
        stem = f"SDCH_20990101_{i:04d}"
        pages.append({
            "frame": stem, "order": "H109", "band": "H", "order_number": 109,
            "order_source": "wat", "mjd": 60000.0 + 0.1 * i,
            "log_column_scales": {"H2O": -0.5 + 0.05 * i, "CH4": 0.01 * i, "CO2": 0.0},
            "continuum_coeffs": np.zeros(3), "velocity_kms": 0.1 * i,
            "lsf_sigma_kms": 2.8 + 0.01 * i,
        })
        if with_cache:
            model = np.ones_like(NU)
            observed = (1.0 + TRUE_PATTERN) * model * (1.0 + 1e-4 * rng.standard_normal(NU.size))
            # What the run saved: the data after its own leave-one-out division.
            applied = TRUE_PATTERN * 0.9
            np.savez(tmp_path / f"{stem}_H109.npz", wavenumber_cm1=NU,
                     observed=observed / (1.0 + applied), model_flux=model,
                     continuum=np.ones_like(NU), mask=np.ones(NU.size, bool),
                     response_pattern=applied)
    record = write_record(
        tmp_path / "record.h5", run={}, config={"resolving_power": 45000.0},
        physics={}, inputs={"profile": "x.csv"},
        parameter_names=["CH4", "CO2", "H2O"], species=["CH4", "CO2", "H2O"], pages=pages,
        continuum_degree=2, key_fields=("frame", "order"),
        extra_columns=(("band", "S256"), ("order_number", "i4"), ("order_source", "S64"),
                       ("mjd", "f8")))
    return record


def test_medians_and_the_water_series(tmp_path):
    calibration = NightCalibration.from_run(_night(tmp_path), None)
    order = calibration.order(109)
    assert order.name == "H109"
    assert order.columns == pytest.approx({"CH4": 0.02, "CO2": 0.0})
    assert order.velocity_kms == pytest.approx(0.2)
    assert order.lsf_sigma_kms == pytest.approx(2.82)
    # Linear in time between standards, flat outside them.
    assert order.water_at(60000.05) == pytest.approx(-0.475)
    assert order.water_at(59999.0) == pytest.approx(-0.5)
    assert order.water_at(60001.0) == pytest.approx(-0.3)
    assert order.log_columns_at(60000.1)["H2O"] == pytest.approx(-0.45)
    assert order.pattern is None


def test_the_pattern_is_the_response_the_standards_share(tmp_path):
    calibration = NightCalibration.from_run(_night(tmp_path), tmp_path, smooth_pixels=0)
    pattern = calibration.order(109).pattern_on(1.0e7 / NU)
    assert np.max(np.abs(pattern - TRUE_PATTERN)) < 1e-3
    # Nothing is invented where nothing was measured.
    assert calibration.order(109).pattern_on(np.array([1.0e7 / 7000.0]))[0] == 0.0


def test_excluding_a_standard_leaves_it_out(tmp_path):
    record = _night(tmp_path)
    calibration = NightCalibration.from_run(record, None, exclude=["SDCH_20990101_0004"])
    order = calibration.order(109)
    assert "SDCH_20990101_0004" not in order.frames
    assert order.water_at(60001.0) == pytest.approx(-0.35)


def test_too_few_standards_give_no_calibration(tmp_path):
    calibration = NightCalibration.from_run(_night(tmp_path, frames=2), None)
    assert not calibration.orders
    with pytest.raises(KeyError, match="no calibration for H109"):
        calibration.order(109)


def test_save_and_load_round_trip(tmp_path):
    calibration = NightCalibration.from_run(_night(tmp_path), tmp_path)
    loaded = NightCalibration.load(calibration.save(tmp_path / "night.h5"))
    a, b = calibration.order(109), loaded.order(109)
    assert loaded.band == "H" and b.columns == a.columns and b.frames == a.frames
    assert np.array_equal(a.water_mjd, b.water_mjd)
    # Stored in float32; see night._store.
    assert np.allclose(a.pattern, b.pattern, rtol=0, atol=1e-7)
    assert loaded.source["record_sha256"] == calibration.source["record_sha256"]


def test_water_is_not_a_fixed_column():
    with pytest.raises(ValueError, match="time series"):
        OrderCalibration(band="H", number=109, columns={"H2O": 0.0}, velocity_kms=0.0,
                         lsf_sigma_kms=2.8, water_mjd=np.array([1.0]),
                         water_log_column=np.array([0.0]))


def test_a_row_named_record_is_refused(tmp_path):
    record = write_record(
        tmp_path / "old.h5", run={}, config={}, physics={}, inputs={},
        parameter_names=["H2O"], species=["H2O"],
        pages=[{"frame": "f", "order": "H05", "log_column_scales": {"H2O": 0.0},
                "continuum_coeffs": np.zeros(1), "band": "H", "order_index": 5}],
        continuum_degree=0, key_fields=("frame", "order"),
        extra_columns=(("band", "S256"), ("order_index", "i4")))
    with pytest.raises(ValueError, match="migrate_igrins_order_names"):
        NightCalibration.from_run(record, None)


# --- the master pattern ---


def _calibration_with(pattern, band="H", number=109):
    order = OrderCalibration(band=band, number=number, columns={"CO2": 0.0}, velocity_kms=0.0,
                             lsf_sigma_kms=2.8, water_mjd=np.array([1.0]),
                             water_log_column=np.array([0.0]),
                             pattern_wavenumber_cm1=NU, pattern=pattern)
    return NightCalibration(band=band, orders={number: order}, source={"record": "x"})


def test_the_master_is_the_median_night():
    nights = [_calibration_with(TRUE_PATTERN * k) for k in (0.8, 1.0, 1.5)]
    grid, pattern = MasterPattern.from_calibrations(nights).orders[109]
    assert np.allclose(pattern, TRUE_PATTERN)


def test_an_unmeasured_night_does_not_drag_the_master_to_zero():
    """A calibration writes exactly zero where too few standards measured a pixel."""

    blank = TRUE_PATTERN.copy()
    blank[:100] = 0.0
    master = MasterPattern.from_calibrations(
        [_calibration_with(TRUE_PATTERN), _calibration_with(blank)])
    assert np.allclose(master.orders[109][1][:100], TRUE_PATTERN[:100])


def test_the_master_round_trips_and_serves_a_thin_night(tmp_path):
    master = MasterPattern.from_calibrations(
        [_calibration_with(TRUE_PATTERN), _calibration_with(TRUE_PATTERN)])
    loaded = MasterPattern.load(master.save(tmp_path / "master.h5"))
    assert np.allclose(loaded.orders[109][1], master.orders[109][1], rtol=0, atol=1e-7)
    # A calibration built with it takes its pattern and never reads the cache.
    night = NightCalibration.from_run(_night(tmp_path, with_cache=False), tmp_path / "absent",
                                      master=loaded)
    assert np.allclose(night.order(109).pattern_on(1.0e7 / NU), TRUE_PATTERN, rtol=0, atol=1e-6)
    assert night.source["pattern"] == "master"


def test_one_band_per_master():
    with pytest.raises(ValueError, match="one band"):
        MasterPattern.from_calibrations([_calibration_with(TRUE_PATTERN, "H"),
                                         _calibration_with(TRUE_PATTERN, "K", 80)])
