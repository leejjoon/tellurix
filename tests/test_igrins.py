"""The IGRINS reader must survive nine years of header drift across three sites."""

import numpy as np
import pytest

from tellurix import (
    continuum_level,
    hydrogen_series_um,
    igrins_spectral_order,
    read_igrins_observation,
    site_for,
    stellar_line_mask,
    surface_conditions,
    zenith_angle_deg,
)

FIXTURE = "tests/data/igrins/SDCH_test_0001.spec.fits"

# The three conventions actually observed in the archive, each with the value a
# real frame carried. Nothing in the file says which is in force.
MCDONALD = {"TELESCOP": "McDonald Observatory", "AIRTEMP": 78.0, "BARPRESS": 23.6,
            "HUMIDITY": 56.0, "ZDSTART": 30.0, "ZDEND": 31.0}
DCT = {"TELESCOP": "Discovery Channel", "AIRTEMP": 10.6, "BARPRESS": 1025.0,
       "HUMIDITY": 22.0, "ZDSTART": 20.0, "ZDEND": 20.4}
GEMINI = {"TELESCOP": "Gemini South", "AIRTEMP": 11.0, "BARPRESS": 730.0,
          "HUMIDITY": 36.1, "ZDSTART": 15.9, "ZDEND": 15.7}


@pytest.fixture(scope="module")
def observation():
    return read_igrins_observation(FIXTURE)


def test_a_real_plp_product_round_trips(observation):
    assert observation.band == "H"
    assert observation.object_type == "STD"
    assert observation.telescope == "Gemini South"
    assert observation.orders == 2
    assert set(observation.sha256) == {"spec", "variance", "flattened"}
    assert all(len(digest) == 64 for digest in observation.sha256.values())
    # The flattened product is read for comparison only, never for the fit.
    assert observation.telluric_model is not None
    assert observation.plp_continuum is not None


def test_an_order_is_ascending_and_finite(observation):
    for index in range(observation.orders):
        order = observation.order(index)
        wavelength = order.wavelength_vacuum_nm
        assert np.all(np.isfinite(wavelength))
        assert np.all(np.diff(wavelength) > 0.0), "SpectralOrder requires ascending wavelength"
        low, high = order.wavenumber_range_cm1
        assert low < high
        assert 1.40e3 < wavelength[0] < 1.85e3, "the H band, in nanometres"


def test_the_uncertainty_is_the_measured_variance(observation):
    """The atlas had to estimate its noise; this does not, and must not."""

    order = observation.order(0)
    fitted = igrins_spectral_order(order, normalize=False, throughput_floor=0.0,
                                   mask_hydrogen_kms=None)
    mask = np.asarray(fitted.mask)
    np.testing.assert_allclose(
        np.asarray(fitted.uncertainty)[mask], np.sqrt(np.asarray(order.variance))[mask], rtol=1e-12
    )


def test_masked_pixels_keep_a_finite_but_worthless_uncertainty(observation):
    """SpectralOrder demands a positive sigma everywhere, including where it is junk."""

    fitted = igrins_spectral_order(observation.order(0))
    sigma = np.asarray(fitted.uncertainty)
    mask = np.asarray(fitted.mask)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0.0)
    assert np.all(np.isfinite(np.asarray(fitted.flux)))
    assert sigma[~mask].min() > 1.0e3 * sigma[mask].max()


def test_normalizing_puts_the_continuum_near_one(observation):
    """The fitted continuum's constant term is a log flux, so the scale matters."""

    order = observation.order(0)
    level = continuum_level(order)
    assert level > 1.0e3, "PLP counts run to tens of thousands"

    raw = igrins_spectral_order(order, normalize=False)
    scaled = igrins_spectral_order(order, normalize=True)
    mask = np.asarray(scaled.mask)
    np.testing.assert_array_equal(mask, np.asarray(raw.mask))
    np.testing.assert_allclose(
        np.asarray(scaled.flux)[mask], np.asarray(raw.flux)[mask] / level, rtol=1e-12
    )
    assert 0.1 < np.median(np.asarray(scaled.flux)[mask]) < 2.0


def test_the_throughput_cut_removes_the_blaze_roll_off(observation):
    """It is asymmetric: the roll-off is at the start of an order, not both ends."""

    order = observation.order(0)
    without = np.asarray(igrins_spectral_order(order, throughput_floor=0.0).mask)
    with_cut = np.asarray(igrins_spectral_order(order, throughput_floor=0.25).mask)
    assert with_cut.sum() < without.sum()
    assert np.all(without[with_cut]), "the cut may only remove pixels"

    dropped = np.flatnonzero(without & ~with_cut)
    assert dropped.size > 100
    # Far more of what goes is at the blue start of the order than the red end.
    start = np.count_nonzero(dropped < without.size // 2)
    assert start > 3 * (dropped.size - start)


def test_the_plp_mask_is_not_used(observation):
    """It flags over half the band by its own flattening criterion, not ours."""

    fitted = igrins_spectral_order(observation.order(0), mask_hydrogen_kms=None)
    assert np.mean(np.asarray(fitted.mask)) > 0.4


# --- the hydrogen series, which is the whole of an A0V spectrum here ---


def test_the_brackett_series_lands_where_it_should():
    brackett = hydrogen_series_um(4, (1.40, 2.55))
    # Br-gamma, the n = 7 -> 4 line, is the one everyone quotes.
    assert np.min(np.abs(brackett - 2.1661)) < 5.0e-4
    # Br-delta and Br-epsilon.
    assert np.min(np.abs(brackett - 1.9451)) < 5.0e-4
    assert np.min(np.abs(brackett - 1.8179)) < 5.0e-4
    # The series converges on its limit and never crosses it.
    assert brackett.min() > 1.4584
    assert np.all(np.diff(brackett) > 0.0)


def test_the_series_sum_has_to_be_bounded():
    with pytest.raises(ValueError, match="max_upper_level"):
        hydrogen_series_um(4, (1.4, 2.5), max_upper_level=4)
    near = hydrogen_series_um(4, (1.40, 2.55), max_upper_level=12)
    far = hydrogen_series_um(4, (1.40, 2.55), max_upper_level=40)
    assert far.size > near.size
    assert far.min() < near.min(), "more terms crowd closer to the limit"


def test_the_mask_removes_a_brackett_line_and_leaves_a_clean_order(observation):
    """Order 11 of this frame carries Br12 at 1.6403 um; order 2 carries none."""

    with_line = stellar_line_mask(observation.order(1).wavelength_vacuum_nm)
    clean = stellar_line_mask(observation.order(0).wavelength_vacuum_nm)
    assert np.count_nonzero(~with_line) > 200
    assert np.all(clean)


def test_a_wider_mask_removes_more():
    wavelength = np.linspace(1630.0, 1655.0, 2048)
    narrow = np.count_nonzero(~stellar_line_mask(wavelength, 200.0))
    wide = np.count_nonzero(~stellar_line_mask(wavelength, 800.0))
    assert 0 < narrow < wide
    with pytest.raises(ValueError, match="half width"):
        stellar_line_mask(wavelength, 0.0)


# --- the three weather conventions ---


@pytest.mark.parametrize(
    "header,temperature_k",
    [(MCDONALD, 298.7), (DCT, 283.75), (GEMINI, 284.15)],
)
def test_each_site_normalizes_to_one_unit_system(header, temperature_k):
    conditions = surface_conditions(header)
    assert conditions["temperature_k"] == pytest.approx(temperature_k, abs=0.1)
    # Whatever the file said, the answer is a station pressure consistent with
    # the site altitude -- which is the only thing that makes the three
    # comparable at all.
    site = site_for(header)
    expected = 1013.25 * (1.0 - 0.0065 * site.altitude_km * 1000.0 / 288.15) ** 5.25588
    assert conditions["pressure_hpa"] == pytest.approx(expected, rel=0.05)
    assert conditions["relative_humidity_percent"] == header["HUMIDITY"]


def test_a_changed_convention_fails_loudly():
    """A DCT frame that switched to station pressure must not pass silently."""

    with pytest.raises(ValueError, match="convention has changed"):
        surface_conditions({**DCT, "BARPRESS": 763.0})


def test_an_unknown_telescope_is_refused():
    with pytest.raises(ValueError, match="unknown IGRINS telescope"):
        surface_conditions({**GEMINI, "TELESCOP": "Subaru"})


def test_missing_weather_cards_come_back_as_none():
    """Gemini 2021 and 2023 keep HUMIDITY and drop the rest; 2015 McDonald drops all four."""

    partial = surface_conditions({"TELESCOP": "Gemini South", "HUMIDITY": 16.1})
    assert partial["relative_humidity_percent"] == 16.1
    assert partial["temperature_k"] is None and partial["pressure_hpa"] is None

    blank = surface_conditions({"TELESCOP": "McDonald Observatory", "AIRTEMP": "",
                                "BARPRESS": "", "HUMIDITY": ""})
    assert all(blank[key] is None for key in
               ("temperature_k", "pressure_hpa", "relative_humidity_percent"))


# --- the slant path, which the atlas never had ---


def test_the_zenith_angle_prefers_the_measured_distance():
    assert zenith_angle_deg(GEMINI) == pytest.approx(15.8)


def test_the_airmass_is_the_fallback():
    header = {"TELESCOP": "Gemini South", "AMSTART": 2.0, "AMEND": 2.0}
    assert zenith_angle_deg(header) == pytest.approx(np.degrees(np.arccos(0.5)))


def test_a_sentinel_airmass_is_not_a_measurement():
    """reduced_log.csv writes -1 for a missing airmass; a header can too."""

    with pytest.raises(ValueError, match="neither a usable zenith distance nor an airmass"):
        zenith_angle_deg({"TELESCOP": "Gemini South", "AMSTART": -1.0, "AMEND": -1.0})


def test_the_order_carries_the_slant_path(observation):
    order = observation.order(0)
    fitted = igrins_spectral_order(order)
    assert fitted.zenith_angle_deg == pytest.approx(observation.zenith_angle_deg)
    assert 0.0 <= fitted.zenith_angle_deg < 90.0


# --- refusals ---


def test_an_order_outside_the_band_is_refused(observation):
    with pytest.raises(ValueError, match="outside"):
        observation.order(observation.orders)


@pytest.mark.parametrize("floor", [-0.1, 1.0, 2.0])
def test_an_impossible_floor_is_refused(observation, floor):
    with pytest.raises(ValueError, match="floor must be in"):
        igrins_spectral_order(observation.order(0), saturation_floor=floor)


def test_masking_everything_is_an_error_not_an_empty_fit(observation):
    with pytest.raises(ValueError, match="keeps too few pixels"):
        igrins_spectral_order(observation.order(1), mask_hydrogen_kms=50_000.0)


def test_a_missing_variance_file_is_refused(tmp_path):
    import shutil

    spec = tmp_path / "SDCH_test_0001.spec.fits"
    shutil.copy(FIXTURE, spec)
    with pytest.raises(FileNotFoundError):
        read_igrins_observation(spec)


# --- what the real archive actually puts in TELESCOP ---

# The reduced McDonald headers name the telescope, not the observatory, and the
# spelling changed: 2014 says '2.7-m Harlen J. Smith' with OBSERVAT 'McDonald',
# 2017 says 'Harlen J. Smith' with OBSERVAT 'McDonald Observatory'. An earlier
# version of SITES keyed on a 'McDonald' prefix of TELESCOP and matched neither.
MCDONALD_2017 = {"TELESCOP": "Harlen J. Smith", "OBSERVAT": "McDonald Observatory",
                 "AIRTEMP": 65.0, "BARPRESS": 23.6, "HUMIDITY": 49.0, "ZDSTART": 37.79,
                 "ZDEND": 37.93}
MCDONALD_2014 = {"TELESCOP": "2.7-m Harlen J. Smith", "OBSERVAT": "McDonald",
                 "AIRTEMP": "", "BARPRESS": "", "HUMIDITY": "", "ZDSTART": "",
                 "AMSTART": 1.0110, "AMEND": 1.0115}


@pytest.mark.parametrize("header", [MCDONALD_2017, MCDONALD_2014])
def test_mcdonald_is_recognised_however_the_header_spells_it(header):
    assert site_for(header).name == "McDonald Observatory"


def test_the_mcdonald_convention_on_a_real_header():
    """65 degF and 23.6 inHg, which is 291.5 K and 799 hPa at 2077 m."""
    conditions = surface_conditions(MCDONALD_2017)
    assert conditions["temperature_k"] == pytest.approx(291.48, abs=0.05)
    assert conditions["pressure_hpa"] == pytest.approx(799.2, abs=1.0)
    assert conditions["relative_humidity_percent"] == 49.0


def test_the_observatory_card_is_used_when_the_telescope_card_is_unknown():
    assert site_for({"TELESCOP": "something new", "OBSERVAT": "Lowell Observatory"}).name == (
        "Lowell Discovery Telescope")


def test_a_bare_gemini_card_is_refused_because_it_cannot_tell_north_from_south():
    """IGRINS-2 is at Gemini North, 1900 m higher up a different mountain."""
    with pytest.raises(ValueError, match="unknown IGRINS telescope"):
        site_for({"TELESCOP": "Gemini", "OBSERVAT": "Gemini Observatory"})
    assert site_for({"TELESCOP": "Gemini South", "OBSERVAT": "Gemini Observatory"}).name == (
        "Gemini South")


def test_blank_numeric_cards_do_not_break_the_reader():
    """The 2014 schema leaves several numeric cards as empty strings."""
    from tellurix.igrins import _number

    assert np.isnan(_number(""))
    assert np.isnan(_number(None))
    assert _number("", default=0.0) == 0.0
    assert _number("1.25") == pytest.approx(1.25)
    # And the zenith angle still comes out, from the airmass fallback.
    assert zenith_angle_deg(MCDONALD_2014) == pytest.approx(
        np.degrees(np.arccos(1.0 / 1.01125)), abs=1e-6)
    blank = surface_conditions(MCDONALD_2014)
    assert all(blank[key] is None for key in
               ("temperature_k", "pressure_hpa", "relative_humidity_percent"))


# --- the fixed instrument-response pattern ---


def test_a_frames_pattern_never_uses_that_frame():
    """The whole validity of the correction rests on this."""
    from tellurix import leave_one_out_patterns

    common = np.array([0.01, -0.02, 0.03, 0.00])
    stack = np.array([common + 0.001 * i for i in range(6)])
    # Corrupt one frame beyond recognition; every *other* frame's pattern must
    # be untouched by it, and its own must come from the rest.
    stack[2] = 99.0
    # smoothing off: this checks the leave-one-out median, not the boxcar
    patterns = leave_one_out_patterns(stack, smooth_pixels=0)
    for index, pattern in enumerate(patterns):
        assert np.max(np.abs(pattern)) < 1.0, f"frame {index} absorbed the outlier"
    # The outlier's own correction is the clean median of the others.
    np.testing.assert_allclose(patterns[2], np.median(np.delete(stack, 2, axis=0), axis=0))


def test_the_pattern_recovers_an_injected_response():
    from tellurix import leave_one_out_patterns

    rng = np.random.default_rng(0)
    truth = np.array([0.05, -0.03, 0.10, -0.08, 0.00])
    stack = truth + rng.normal(0.0, 0.01, (12, truth.size))
    for pattern in leave_one_out_patterns(stack, smooth_pixels=0):
        np.testing.assert_allclose(pattern, truth, atol=0.01)


def test_too_few_frames_leave_a_pixel_uncorrected():
    """A pattern from one or two frames is that frame's noise, not a calibration."""
    from tellurix import leave_one_out_patterns

    stack = np.full((4, 3), 0.2)
    stack[:, 1] = np.nan          # nothing measures this pixel
    stack[1:, 2] = np.nan         # only one frame measures this one
    patterns = leave_one_out_patterns(stack, minimum_frames=3, smooth_pixels=0)
    for pattern in patterns:
        assert pattern[0] == pytest.approx(0.2)
        assert pattern[1] == 0.0, "an unmeasured pixel must be left alone"
        assert pattern[2] == 0.0, "one frame is not enough to call something a pattern"
    assert np.all(np.isfinite(np.asarray(patterns)))


def test_dividing_by_the_response_is_the_same_as_scaling_the_model():
    """Which is why the correction needs no change to the forward model."""
    rng = np.random.default_rng(1)
    flux, model, sigma = rng.normal(10, 1, 50), rng.normal(10, 1, 50), np.full(50, 0.2)
    response = rng.normal(0.0, 0.05, 50)
    scale = 1.0 + response
    scaled_model = np.sum(((flux - scale * model) / sigma) ** 2)
    scaled_data = np.sum(((flux / scale - model) / (sigma / scale)) ** 2)
    assert scaled_data == pytest.approx(scaled_model, rel=1e-12)


def test_the_pattern_keeps_the_broad_response_and_drops_line_scale_structure():
    """Leave-one-out stops the pattern eating one frame's noise. Smoothing stops
    it eating the telluric model error that every frame shares."""
    from tellurix import leave_one_out_patterns

    rng = np.random.default_rng(0)
    x = np.arange(400)
    broad = 0.03 * np.sin(x / 120.0)      # instrument: hundreds of pixels
    fine = 0.02 * np.sin(x / 1.5)         # line list: a few pixels
    stack = broad + fine + rng.normal(0.0, 0.002, (9, 400))

    smoothed = leave_one_out_patterns(stack)[0]
    raw = leave_one_out_patterns(stack, smooth_pixels=0)[0]
    assert np.std(smoothed - broad) < 0.15 * np.std(fine), "line-scale structure leaked through"
    assert np.std(raw - broad) > 0.8 * np.std(fine), "the unsmoothed pattern should carry it"
    np.testing.assert_allclose(smoothed, broad, atol=0.005)


def test_an_even_smoothing_window_is_refused():
    from tellurix import leave_one_out_patterns

    with pytest.raises(ValueError, match="odd"):
        leave_one_out_patterns(np.zeros((5, 50)), smooth_pixels=50)


def test_smoothing_never_invents_a_correction_where_nothing_was_measured():
    from tellurix import leave_one_out_patterns

    stack = np.full((6, 60), 0.05)
    stack[:, 20:30] = np.nan          # a gap no frame measured
    pattern = leave_one_out_patterns(stack, smooth_pixels=11)[0]
    assert np.all(pattern[20:30] == 0.0), "a boxcar must not bleed into an unmeasured gap"
    assert pattern[5] == pytest.approx(0.05, abs=1e-9)


# --- seeding the water column from the dewpoint ---


def test_the_magnus_formula_matches_known_values():
    from tellurix import saturation_vapour_pressure_hpa as es

    # Standard table values, good to a few tenths of a percent.
    assert es(0.0) == pytest.approx(6.112, rel=1e-3)
    assert es(20.0) == pytest.approx(23.39, rel=5e-3)
    assert es(-10.0) == pytest.approx(2.86, rel=1e-2)
    assert es(30.0) > es(20.0) > es(10.0) > es(0.0) > es(-10.0)


def test_the_water_column_comes_from_the_dewpoint_not_the_humidity():
    """The dewpoint *is* the vapour pressure; humidity needs the temperature too."""
    from tellurix import precipitable_water_mm, saturation_vapour_pressure_hpa

    surface = {"temperature_k": 280.0, "dewpoint_c": -7.4,
               "relative_humidity_percent": 34.0}
    water = precipitable_water_mm(surface)
    expected = 100.0 * saturation_vapour_pressure_hpa(-7.4) * 1400.0 / (461.5 * 280.0)
    assert water == pytest.approx(expected, rel=1e-9)
    # Changing the humidity must not move it while a dewpoint is present.
    assert precipitable_water_mm({**surface, "relative_humidity_percent": 90.0}) == \
        pytest.approx(water, rel=1e-12)


def test_the_humidity_is_the_fallback_when_there_is_no_dewpoint():
    from tellurix import precipitable_water_mm

    with_dew = precipitable_water_mm({"temperature_k": 280.0, "dewpoint_c": -7.4})
    # 34% at 280 K is very nearly a dewpoint of -7.4 C, so the two should agree.
    without = precipitable_water_mm({"temperature_k": 280.0, "dewpoint_c": None,
                                     "relative_humidity_percent": 34.0})
    assert without == pytest.approx(with_dew, rel=0.05)


def test_no_water_information_returns_none_rather_than_a_guess():
    from tellurix import precipitable_water_mm

    assert precipitable_water_mm({"temperature_k": 280.0, "dewpoint_c": None,
                                  "relative_humidity_percent": None}) is None
    assert precipitable_water_mm({"temperature_k": None, "dewpoint_c": -5.0}) is None
    assert precipitable_water_mm({"temperature_k": 280.0, "dewpoint_c": None,
                                  "relative_humidity_percent": 0.0}) is None


def test_the_dewpoint_is_converted_from_fahrenheit_at_mcdonald():
    """A real McDonald header: 42 degF is 5.6 C, not 42 C."""
    conditions = surface_conditions({**MCDONALD_2017, "DEWPOINT": 42.0})
    assert conditions["dewpoint_c"] == pytest.approx((42.0 - 32.0) * 5.0 / 9.0, abs=1e-9)
    assert surface_conditions({**GEMINI, "DEWPOINT": -3.8})["dewpoint_c"] == pytest.approx(-3.8)


def test_the_estimate_lands_within_a_factor_of_two_on_the_fitted_nights():
    """The seed only has to be close enough for the self-broadening expansion."""
    from tellurix import precipitable_water_mm

    # Median surface conditions and the fitted column, from the three nights.
    for dewpoint, temperature, fitted in ((-7.4, 281.15, 2.36), (5.9, 290.09, 9.99),
                                          (-2.2, 275.85, 6.89)):
        estimate = precipitable_water_mm(
            {"temperature_k": temperature, "dewpoint_c": dewpoint})
        assert 0.5 < estimate / fitted < 2.0, f"{estimate:.2f} against {fitted:.2f}"
