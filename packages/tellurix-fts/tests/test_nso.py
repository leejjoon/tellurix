from pathlib import Path
import datetime as dt

import numpy as np
import pytest

from tellurix_fts import (
    fts_spectral_order, photatl_as_fts_spectrum, photatl_spectral_order, read_fts_spectrum,
    read_niratl_page, read_photatl_page, read_solar_spectrum, uniform_wavenumber_grid,
    zenith_angle_deg_for_airmass,
)

DATA = Path(__file__).resolve().parent / "data"


FTS_FIXTURE = str(DATA / "nso_ftsspec_sample.txt")
PHOTATL_FIXTURE = str(DATA / "nso_photatl_sample")
# 200 rows of ph08900 around its first -1.0 fill in the solar column.
NIRATL_FIXTURE = str(DATA / "nso_niratl_sample")
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
    from tellurix_fts.nso import _saturation_mask

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


def test_a_page_through_the_fts_path_is_the_same_order():
    page = read_photatl_page(PHOTATL_FIXTURE)
    spectrum = photatl_as_fts_spectrum(page)

    # The batch fit reads pages through the FTS path; it must build exactly the
    # order the photatl path would, from the observed column and nothing else.
    via_fts = fts_spectral_order(spectrum, zenith_angle_deg=60.0)
    direct = photatl_spectral_order(page, zenith_angle_deg=60.0)
    np.testing.assert_array_equal(via_fts.flux, direct.flux)
    np.testing.assert_array_equal(via_fts.mask, direct.mask)
    np.testing.assert_array_equal(via_fts.wavelength_vacuum_nm, direct.wavelength_vacuum_nm)
    assert spectrum.sha256 == page.sha256

    # A page records no air mass, and that must fail loudly rather than
    # default to anything.
    assert spectrum.airmass_mean is None
    with pytest.raises(ValueError, match="no air mass"):
        fts_spectral_order(spectrum)


def test_read_solar_spectrum_dispatches_on_the_page_name(tmp_path):
    page_path = tmp_path / "wn1850"
    page_path.write_bytes(open(PHOTATL_FIXTURE, "rb").read())

    np.testing.assert_array_equal(read_solar_spectrum(page_path).flux,
                                  read_photatl_page(page_path).total)
    assert read_solar_spectrum(FTS_FIXTURE).airmass_mean == pytest.approx(4.73)


def test_selecting_a_range_keeps_the_provenance():
    spectrum = read_fts_spectrum(FTS_FIXTURE)
    window = spectrum.select(5001.0, 5002.0)

    assert window.sha256 == spectrum.sha256
    assert window.airmass_mean == spectrum.airmass_mean
    assert window.flux.size < spectrum.flux.size
    with pytest.raises(ValueError):
        spectrum.select(5002.0, 5001.0)


def test_a_window_is_normalized_before_it_is_fitted():
    """The failure this prevents: the fitted continuum's constant term is a log
    flux against a bound of +-2, and these spectra are not normalized -- the raw
    flux falls to 0.017 at the blue edge of the transform, whose log is -4.07.
    Fourteen of 224 windows per file were fitted with the continuum pinned at
    exp(-2); 9046-9076 went from 293 times the noise to 2.0 once normalized."""
    import numpy as np

    from tellurix_fts import fts_continuum_level

    faint = np.full(512, 0.017)
    faint[::7] = 0.010
    assert fts_continuum_level(faint) == pytest.approx(0.017, rel=1e-6)

    # The level is a scale, so scaling the data scales it by the same factor:
    # that is what makes dividing by it exactly degenerate with the continuum.
    assert fts_continuum_level(faint * 58.8) == pytest.approx(
        fts_continuum_level(faint) * 58.8, rel=1e-9)


def test_a_continuum_level_needs_positive_finite_pixels():
    import numpy as np

    from tellurix_fts import fts_continuum_level

    with pytest.raises(ValueError, match="too few finite pixels"):
        fts_continuum_level(np.full(4, np.nan))
    with pytest.raises(ValueError, match="no positive continuum level"):
        fts_continuum_level(np.full(64, -1.0))
    with pytest.raises(ValueError, match="percentile"):
        fts_continuum_level(np.ones(64), percentile=0.0)


def test_niratl_keeps_the_header_as_continuum_levels():
    page = read_niratl_page(NIRATL_FIXTURE)

    # The first line is three continuum levels, not a sample; reading it as
    # one would put a spurious point at the start of the grid.
    assert len(page.continuum_levels) == 3
    assert page.observed.size == 200
    # 4096 samples over 30 cm-1. The printed steps are 7 or 8 float32 quanta
    # (0.006836 and 0.007812 at 8900 cm-1), and their median is the larger --
    # the true step is only recovered by the grid reconstruction.
    np.testing.assert_allclose(page.spacing_cm1, 30.0 / 4095, rtol=2e-3)


def test_niratl_fill_is_in_the_components_never_in_the_observed_column():
    page = read_niratl_page(NIRATL_FIXTURE)

    assert page.filled.any()
    assert np.all(page.observed[page.filled] != -1.0)


def test_a_niratl_page_goes_through_the_fts_path_with_no_air_mass(tmp_path):
    page_path = tmp_path / "ph08900"
    page_path.write_bytes(open(NIRATL_FIXTURE, "rb").read())

    spectrum = read_solar_spectrum(page_path)
    np.testing.assert_array_equal(spectrum.flux, read_niratl_page(page_path).observed)
    # The README says 1.0 and the O2 A-band says 1.06-1.10; neither may become
    # a silent default.
    assert spectrum.airmass_mean is None
    with pytest.raises(ValueError, match="no air mass"):
        fts_spectral_order(spectrum)
    order = fts_spectral_order(spectrum, zenith_angle_deg=0.0)
    assert np.all(np.diff(order.wavelength_vacuum_nm) > 0.0)
