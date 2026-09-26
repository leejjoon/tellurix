import datetime as dt

import numpy as np
import pytest

from tellurix import (
    fts_spectral_order,
    photatl_spectral_order,
    read_fts_spectrum,
    read_photatl_page,
    uniform_wavenumber_grid,
    zenith_angle_deg_for_airmass,
)


FTS_FIXTURE = "tests/data/nso_ftsspec_sample.txt"
PHOTATL_FIXTURE = "tests/data/nso_photatl_sample"
SAMPLING_CM1 = 0.0094771


def test_reads_the_observing_header():
    spectrum = read_fts_spectrum(FTS_FIXTURE)

    assert spectrum.source_name == "CENT DISK E AUX"
    assert spectrum.airmass_start == pytest.approx(5.88)
    assert spectrum.airmass_stop == pytest.approx(3.58)
    assert spectrum.airmass_mean == pytest.approx(4.73)
    # The transform length is not the number of spectral points, and nothing
    # may validate one against the other.
    assert spectrum.transform_samples == 831488
    assert spectrum.flux.size == 400


def test_parses_a_clock_field_with_an_internal_space():
    # The components are right-justified, so a single-digit second arrives as
    # "15:18: 9.0". Matching the field as one token drops exactly these lines,
    # and silently -- which is what this guards.
    spectrum = read_fts_spectrum(FTS_FIXTURE)

    assert spectrum.observed_utc_start == dt.datetime(1990, 12, 18, 15, 18, 9, tzinfo=dt.timezone.utc)
    assert spectrum.observed_utc_stop == dt.datetime(1990, 12, 18, 15, 57, 43, tzinfo=dt.timezone.utc)
    # A 40-minute scan: the midpoint is what an ERA5 lookup should use.
    assert spectrum.observed_utc_mid == dt.datetime(1990, 12, 18, 15, 37, 56, tzinfo=dt.timezone.utc)


def test_refuses_a_date_its_own_julian_day_contradicts(tmp_path):
    broken = tmp_path / "ftsspec_broken.txt"
    broken.write_text(
        open(FTS_FIXTURE).read().replace("julian day number=  2448244",
                                         "julian day number=  2440000")
    )

    with pytest.raises(ValueError, match="Julian day"):
        read_fts_spectrum(broken)


def test_unknown_air_mass_is_none_rather_than_a_guess(tmp_path):
    # ftsspec_830626_3 writes "?.??" here, and its free-text comment says
    # "EAST 3. AIRMASSES" -- prose that must not be parsed as a measurement.
    unknown = tmp_path / "ftsspec_unknown.txt"
    unknown.write_text(
        open(FTS_FIXTURE).read().replace("airmass           5.88             3.58",
                                         "airmass           ?.??             ?.??")
    )
    spectrum = read_fts_spectrum(unknown)

    assert spectrum.airmass_start is None and spectrum.airmass_mean is None
    with pytest.raises(ValueError, match="no air mass"):
        fts_spectral_order(spectrum.select(5000.5, 5003.0))


def test_reconstructs_the_uniform_grid_the_file_quantizes():
    stored = read_fts_spectrum(FTS_FIXTURE, reconstruct_grid=False)
    grid = read_fts_spectrum(FTS_FIXTURE)

    # The file writes four decimals, so its own spacings are not uniform.
    assert np.ptp(np.diff(stored.wavenumber_vacuum_cm1)) > 1e-5
    assert np.ptp(np.diff(grid.wavenumber_vacuum_cm1)) < 1e-12
    np.testing.assert_allclose(grid.spacing_cm1, SAMPLING_CM1, rtol=1e-6)
    # Fitting every sample averages the quantization down by sqrt(N), so the
    # grid is better than any stored value it was built from.
    assert grid.grid_residual_cm1 == pytest.approx(5e-5, abs=1e-5)


def test_uniform_grid_residual_reports_real_departures():
    values = np.arange(64) * 0.01
    grid, residual = uniform_wavenumber_grid(values)
    assert residual == pytest.approx(0.0, abs=1e-12)

    values[32] += 0.005
    _, residual = uniform_wavenumber_grid(values)
    assert residual > 1e-3


def test_air_mass_round_trips_through_the_zenith_angle():
    # The model divides by cos z, so the header's own air mass -- which already
    # carries refraction and curvature -- must come back out exactly.
    for airmass in (1.0, 1.8, 2.17, 4.73, 5.88, 6.66):
        zenith = zenith_angle_deg_for_airmass(airmass)
        assert 1.0 / np.cos(np.radians(zenith)) == pytest.approx(airmass, rel=1e-12)
        assert 0.0 <= zenith < 90.0

    with pytest.raises(ValueError):
        zenith_angle_deg_for_airmass(0.5)


def test_order_uses_the_header_air_mass_and_ascends_in_wavelength():
    spectrum = read_fts_spectrum(FTS_FIXTURE)
    order = fts_spectral_order(spectrum.select(5000.5, 5003.0))

    assert np.all(np.diff(order.wavelength_vacuum_nm) > 0.0)
    assert order.zenith_angle_deg == pytest.approx(zenith_angle_deg_for_airmass(4.73))
    # The saturated core is masked as a region, not pixel by pixel.
    assert 0.0 < order.mask.mean() < 1.0
    assert np.all(order.flux[order.mask] > 0.0)

    # Passing zero folds the air mass into the column scales instead, which is
    # the Arcturus configuration and the diagnostic half of the slant-path test.
    assert fts_spectral_order(spectrum.select(5000.5, 5003.0),
                              zenith_angle_deg=0.0).zenith_angle_deg == 0.0


def test_refuses_a_window_that_passes_no_light():
    # A window with no signal has a continuum made of noise, and every pixel
    # clears a fraction of it, so the relative saturation floor alone would
    # happily fit it. Measured on the real data, live windows reach 188-2702
    # sigma and dead ones 5-11.
    from tellurix.nso import _saturation_mask

    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1e-3, 512)

    with pytest.raises(ValueError, match="sigma above zero"):
        _saturation_mask(noise, 0.02, 5, 30.0)


def test_photatl_drops_the_row_that_holds_dg():
    page = read_photatl_page(PHOTATL_FIXTURE)

    # The README: dg "replaces the last point in the total", so that row's
    # fourth column is a threshold and not a measurement.
    assert page.dg == pytest.approx(0.6)
    assert page.total.size == 200
    assert page.total[-1] != pytest.approx(0.6)
    # Looser than the ftsspec check on purpose: single precision over a
    # 200-row fixture averages down by sqrt(200), where a real 3060-row page
    # recovers the documented spacing to the eighth decimal.
    np.testing.assert_allclose(page.spacing_cm1, SAMPLING_CM1, rtol=1e-5)
    # photatl stores single precision, so its jitter is far coarser than the
    # ftsspec files' four decimals.
    assert page.grid_residual_cm1 > 1e-5


def test_photatl_marks_where_the_solar_column_was_filled_in():
    page = read_photatl_page(PHOTATL_FIXTURE)

    np.testing.assert_array_equal(page.interpolated, page.atmospheric < page.dg)
    assert page.interpolated.any()


def test_photatl_refuses_every_column_but_the_observed_one():
    page = read_photatl_page(PHOTATL_FIXTURE)

    order = photatl_spectral_order(page)
    assert order.zenith_angle_deg == 0.0
    assert np.all(np.diff(order.wavelength_vacuum_nm) > 0.0)

    # 'solar' is interpolated wherever the sky is opaque and 'atmospheric' is
    # the estimate this work replaces; fitting either is a category error.
    for column in ("solar", "atmospheric", "ratioed"):
        with pytest.raises(ValueError, match="must be 'total'"):
            photatl_spectral_order(page, column=column)


def test_selecting_a_range_keeps_the_provenance():
    spectrum = read_fts_spectrum(FTS_FIXTURE)
    window = spectrum.select(5001.0, 5002.0)

    assert window.sha256 == spectrum.sha256
    assert window.airmass_mean == spectrum.airmass_mean
    assert window.flux.size < spectrum.flux.size
    with pytest.raises(ValueError):
        spectrum.select(5002.0, 5001.0)
