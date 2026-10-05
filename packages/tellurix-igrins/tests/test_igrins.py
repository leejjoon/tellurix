"""The IGRINS reader must survive nine years of header drift across three sites."""

from pathlib import Path

import numpy as np
import pytest

from tellurix_igrins import (
    IGRINS_ORDER_CENTRES_UM, WatSpec, continuum_level, format_wat2_cards, hydrogen_series_um,
    identify_orders, parse_wat_specs, igrins_spectral_order, read_igrins_observation, site_for,
    stellar_line_mask, surface_conditions, zenith_angle_deg,
)

DATA = Path(__file__).resolve().parent / "data"

FIXTURE = str(DATA / "igrins/SDCH_test_0001.spec.fits")

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
    # Cut from a 28-order frame: its rows are physical orders 100 and 109.
    assert observation.orders == (100, 109)
    assert observation.order_source == "wat"
    assert set(observation.sha256) == {"spec", "variance", "flattened"}
    assert all(len(digest) == 64 for digest in observation.sha256.values())
    # The flattened product is read for comparison only, never for the fit.
    assert observation.telluric_model is not None
    assert observation.plp_continuum is not None


def test_an_order_is_ascending_and_finite(observation):
    for number in observation.orders:
        order = observation.order(number)
        wavelength = order.wavelength_vacuum_nm
        assert np.all(np.isfinite(wavelength))
        assert np.all(np.diff(wavelength) > 0.0), "SpectralOrder requires ascending wavelength"
        low, high = order.wavenumber_range_cm1
        assert low < high
        assert 1.40e3 < wavelength[0] < 1.85e3, "the H band, in nanometres"


def test_the_uncertainty_is_the_measured_variance(observation):
    """The atlas had to estimate its noise; this does not, and must not."""

    order = observation.order(100)
    fitted = igrins_spectral_order(order, normalize=False, throughput_floor=0.0,
                                   mask_hydrogen_kms=None)
    mask = np.asarray(fitted.mask)
    np.testing.assert_allclose(
        np.asarray(fitted.uncertainty)[mask], np.sqrt(np.asarray(order.variance))[mask], rtol=1e-12
    )


def test_masked_pixels_keep_a_finite_but_worthless_uncertainty(observation):
    """SpectralOrder demands a positive sigma everywhere, including where it is junk."""

    fitted = igrins_spectral_order(observation.order(100))
    sigma = np.asarray(fitted.uncertainty)
    mask = np.asarray(fitted.mask)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0.0)
    assert np.all(np.isfinite(np.asarray(fitted.flux)))
    assert sigma[~mask].min() > 1.0e3 * sigma[mask].max()


def test_normalizing_puts_the_continuum_near_one(observation):
    """The fitted continuum's constant term is a log flux, so the scale matters."""

    order = observation.order(100)
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

    order = observation.order(100)
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

    fitted = igrins_spectral_order(observation.order(100), mask_hydrogen_kms=None)
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
    """Order H109 carries Br12 at 1.6403 um; order H100 carries none."""

    with_line = stellar_line_mask(observation.order(109).wavelength_vacuum_nm)
    clean = stellar_line_mask(observation.order(100).wavelength_vacuum_nm)
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


def test_mcdonald_2015_reports_metric_and_the_frame_says_so():
    """Same telescope, other units: 2015-12-03 is Celsius and station hPa.

    Pressure alone rules out inHg; the dewpoint and humidity agree only in
    Celsius. The 2017 frame, read the same way, still comes out Fahrenheit.
    """

    metric = {"TELESCOP": "Harlen J. Smith", "OBSERVAT": "McDonald Observatory",
              "AIRTEMP": 5.6, "BARPRESS": 799.5, "HUMIDITY": 34.0, "DEWPOINT": -8.7}
    conditions = surface_conditions(metric)
    assert conditions["temperature_k"] == pytest.approx(278.75)
    assert conditions["pressure_hpa"] == pytest.approx(799.5)
    assert conditions["dewpoint_c"] == pytest.approx(-8.7)
    imperial = surface_conditions({**metric, "AIRTEMP": 66.0, "BARPRESS": 23.4,
                                   "HUMIDITY": 37.0, "DEWPOINT": 38.6})
    assert imperial["temperature_k"] == pytest.approx(292.04, abs=0.01)
    # a humidity neither unit reproduces is refused, not guessed
    with pytest.raises(ValueError, match="convention has changed"):
        surface_conditions({**metric, "HUMIDITY": 80.0})


def test_geometry_reproduces_a_header_zenith_distance():
    """DCT 2018-12-20 frame 0045, HD 31069 (SIMBAD): ZDSTART/ZDEND 49.30/49.08."""

    from tellurix_igrins.igrins import SITES, geometric_zenith_angle_deg

    angle = geometric_zenith_angle_deg(73.71351218, 44.06086201, "2018-12-21T02:05:43.189",
                                       "2018-12-21T02:06:54.301", SITES["DCT"])
    assert angle == pytest.approx(0.5 * (49.30 + 49.08), abs=0.15)


def test_a_pointing_sidecar_overrides_the_header(tmp_path):
    import json
    import shutil

    for suffix in ("spec.fits", "variance.fits"):
        shutil.copy(FIXTURE.replace("spec.fits", suffix), tmp_path)
    spec = tmp_path / "SDCH_test_0001.spec.fits"
    assert read_igrins_observation(spec).zenith_source == "header"
    (tmp_path / "pointing.json").write_text(json.dumps({"zenith_angle_deg": 42.5}))
    observation = read_igrins_observation(spec)
    assert observation.zenith_angle_deg == 42.5
    assert observation.zenith_source == "geometry"
    assert observation.order(observation.orders[0]).meta["zenith_source"] == "geometry"
    (tmp_path / "pointing.json").write_text(
        json.dumps({"zenith_angle_deg": 44.0, "source": "sequence"}))
    assert read_igrins_observation(spec).zenith_source == "sequence"


def test_a_sequence_takes_the_mean_airmass_of_a_setting_star():
    """McDonald 2017-04-20 frame 0039, k Tau setting at airmass 3: eight 90 s
    exposures over 16 minutes, of which the header describes the first."""

    from tellurix_igrins.igrins import SITES, geometric_zenith_angle_deg, sequence_zenith_angle_deg

    start = "2017-04-21T03:21:53.464"
    instant = sequence_zenith_angle_deg(74.53913355, 25.05040692, start, 0.0, SITES["McDonald"])
    assert instant == pytest.approx(geometric_zenith_angle_deg(
        74.53913355, 25.05040692, start, start, SITES["McDonald"]), abs=1e-6)
    whole = sequence_zenith_angle_deg(74.53913355, 25.05040692, start, 969.1, SITES["McDonald"])
    airmass = 1.0 / np.cos(np.radians([instant, whole]))
    # 9% more air than at the first exposure's start -- the size of the dry-gas
    # excess this frame showed before the correction.
    assert 1.07 < airmass[1] / airmass[0] < 1.11
    with pytest.raises(ValueError):
        sequence_zenith_angle_deg(74.53913355, 25.05040692, start, -1.0, SITES["McDonald"])


def test_the_all_minus_one_sentinel_is_missing_weather():
    """2015-12-01 McDonald writes -1 in every weather card."""

    blank = surface_conditions({"TELESCOP": "Harlen J. Smith", "AIRTEMP": -1.0,
                                "BARPRESS": -1.0, "HUMIDITY": -1.0, "DEWPOINT": -1.0})
    assert all(blank[key] is None for key in
               ("temperature_k", "pressure_hpa", "relative_humidity_percent", "dewpoint_c"))


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


# The H and K files of one DCT exposure, GJ 281 on 2018-12-20: the K file's end
# cards describe another frame -- it ends before it starts -- and averaging its
# ZDEND put the K slant path 5.8% short of the H file's.
GJ281_H = {"ZDSTART": 45.85, "ZDEND": 45.59, "DATE-OBS": "2018-12-21T06:47:50.045",
           "DATE-END": "2018-12-21T06:49:20.051", "EXPTIME": 60.0, "NCOMBINE": 8}
GJ281_K = {**GJ281_H, "ZDEND": 39.18, "DATE-END": "2018-12-21T06:45:53.214"}


def test_a_self_consistent_exposure_averages_its_ends():
    assert zenith_angle_deg(GJ281_H) == pytest.approx(0.5 * (45.85 + 45.59))


def test_end_cards_from_another_frame_are_not_averaged():
    assert zenith_angle_deg(GJ281_K) == pytest.approx(45.85)


def test_a_zenith_distance_faster_than_the_sky_is_refused():
    """Gemini South 2021-03-16 frame 162, K: 0.71 deg in 90 s; the sky allows 0.38."""

    header = {"ZDSTART": 40.5388, "ZDEND": 41.2459, "DATE-OBS": "2021-03-17T08:04:08.002",
              "DATE-END": "2021-03-17T08:05:37.959"}
    assert zenith_angle_deg(header) == pytest.approx(40.5388)
    # Without DATE-END, the exposure time bounds it instead.
    loose = {"ZDSTART": 40.0, "ZDEND": 40.4, "EXPTIME": 60.0}
    assert zenith_angle_deg(loose) == pytest.approx(40.2)
    tight = {"ZDSTART": 40.0, "ZDEND": 45.0, "EXPTIME": 60.0}
    assert zenith_angle_deg(tight) == pytest.approx(40.0)


def test_the_airmass_is_the_fallback():
    header = {"TELESCOP": "Gemini South", "AMSTART": 2.0, "AMEND": 2.0}
    assert zenith_angle_deg(header) == pytest.approx(np.degrees(np.arccos(0.5)))


def test_a_sentinel_airmass_is_not_a_measurement():
    """reduced_log.csv writes -1 for a missing airmass; a header can too."""

    with pytest.raises(ValueError, match="neither a usable zenith distance nor an airmass"):
        zenith_angle_deg({"TELESCOP": "Gemini South", "AMSTART": -1.0, "AMEND": -1.0})


def test_the_order_carries_the_slant_path(observation):
    order = observation.order(100)
    fitted = igrins_spectral_order(order)
    assert fitted.zenith_angle_deg == pytest.approx(observation.zenith_angle_deg)
    assert 0.0 <= fitted.zenith_angle_deg < 90.0


# --- refusals ---


def test_an_order_the_file_does_not_hold_is_refused(observation):
    # H98 exists on the instrument but not in this cut-down file; 0 is a row
    # index, which is exactly the name this reader must not accept.
    for number in (98, 0, 1):
        with pytest.raises(ValueError, match="has no H order"):
            observation.order(number)


@pytest.mark.parametrize("floor", [-0.1, 1.0, 2.0])
def test_an_impossible_floor_is_refused(observation, floor):
    with pytest.raises(ValueError, match="floor must be in"):
        igrins_spectral_order(observation.order(100), saturation_floor=floor)


def test_masking_everything_is_an_error_not_an_empty_fit(observation):
    with pytest.raises(ValueError, match="keeps too few pixels"):
        igrins_spectral_order(observation.order(109), mask_hydrogen_kms=50_000.0)


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
    from tellurix_igrins.igrins import _number

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
    from tellurix_igrins import leave_one_out_patterns

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
    from tellurix_igrins import leave_one_out_patterns

    rng = np.random.default_rng(0)
    truth = np.array([0.05, -0.03, 0.10, -0.08, 0.00])
    stack = truth + rng.normal(0.0, 0.01, (12, truth.size))
    for pattern in leave_one_out_patterns(stack, smooth_pixels=0):
        np.testing.assert_allclose(pattern, truth, atol=0.01)


def test_too_few_frames_leave_a_pixel_uncorrected():
    """A pattern from one or two frames is that frame's noise, not a calibration."""
    from tellurix_igrins import leave_one_out_patterns

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
    from tellurix_igrins import leave_one_out_patterns

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
    from tellurix_igrins import leave_one_out_patterns

    with pytest.raises(ValueError, match="odd"):
        leave_one_out_patterns(np.zeros((5, 50)), smooth_pixels=50)


def test_smoothing_never_invents_a_correction_where_nothing_was_measured():
    from tellurix_igrins import leave_one_out_patterns

    stack = np.full((6, 60), 0.05)
    stack[:, 20:30] = np.nan          # a gap no frame measured
    pattern = leave_one_out_patterns(stack, smooth_pixels=11)[0]
    assert np.all(pattern[20:30] == 0.0), "a boxcar must not bleed into an unmeasured gap"
    assert pattern[5] == pytest.approx(0.05, abs=1e-9)


# --- seeding the water column from the dewpoint ---


def test_the_magnus_formula_matches_known_values():
    from tellurix_igrins import saturation_vapour_pressure_hpa as es

    # Standard table values, good to a few tenths of a percent.
    assert es(0.0) == pytest.approx(6.112, rel=1e-3)
    assert es(20.0) == pytest.approx(23.39, rel=5e-3)
    assert es(-10.0) == pytest.approx(2.86, rel=1e-2)
    assert es(30.0) > es(20.0) > es(10.0) > es(0.0) > es(-10.0)


def test_the_water_column_comes_from_the_dewpoint_not_the_humidity():
    """The dewpoint *is* the vapour pressure; humidity needs the temperature too."""
    from tellurix_igrins import precipitable_water_mm, saturation_vapour_pressure_hpa

    surface = {"temperature_k": 280.0, "dewpoint_c": -7.4,
               "relative_humidity_percent": 34.0}
    water = precipitable_water_mm(surface)
    expected = 100.0 * saturation_vapour_pressure_hpa(-7.4) * 1400.0 / (461.5 * 280.0)
    assert water == pytest.approx(expected, rel=1e-9)
    # Changing the humidity must not move it while a dewpoint is present.
    assert precipitable_water_mm({**surface, "relative_humidity_percent": 90.0}) == \
        pytest.approx(water, rel=1e-12)


def test_the_humidity_is_the_fallback_when_there_is_no_dewpoint():
    from tellurix_igrins import precipitable_water_mm

    with_dew = precipitable_water_mm({"temperature_k": 280.0, "dewpoint_c": -7.4})
    # 34% at 280 K is very nearly a dewpoint of -7.4 C, so the two should agree.
    without = precipitable_water_mm({"temperature_k": 280.0, "dewpoint_c": None,
                                     "relative_humidity_percent": 34.0})
    assert without == pytest.approx(with_dew, rel=0.05)


def test_no_water_information_returns_none_rather_than_a_guess():
    from tellurix_igrins import precipitable_water_mm

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
    from tellurix_igrins import precipitable_water_mm

    # Median surface conditions and the fitted column, from the three nights.
    for dewpoint, temperature, fitted in ((-7.4, 281.15, 2.36), (5.9, 290.09, 9.99),
                                          (-2.2, 275.85, 6.89)):
        estimate = precipitable_water_mm(
            {"temperature_k": temperature, "dewpoint_c": dewpoint})
        assert 0.5 < estimate / fitted < 2.0, f"{estimate:.2f} against {fitted:.2f}"


# --- physical order numbers ---
#
# Row position means nothing across IGRINS files -- K ships 24, 25 or 26 orders,
# a wavelength-ascending file reverses them, a custom extraction drops some --
# so orders are named by the WAT beam, matched to rows by wavelength.


def _fixture_arrays():
    from astropy.io import fits

    with fits.open(FIXTURE) as handle:
        return dict(handle[0].header), np.asarray(handle[1].data, dtype=float)


def test_wat_cards_round_trip():
    header, _ = _fixture_arrays()
    specs = parse_wat_specs(header)
    assert [s.order for s in specs] == [100, 109]
    rebuilt = parse_wat_specs(dict(format_wat2_cards(specs)))
    assert [(s.order, s.w1_angstrom, s.dw_angstrom, s.nw) for s in rebuilt] == \
        [(s.order, s.w1_angstrom, s.dw_angstrom, s.nw) for s in specs]
    # Every card fits the 68-character slices IRAF writes.
    assert all(len(value) <= 68 for _, value in format_wat2_cards(specs))


def test_a_card_split_on_a_space_survives_the_stripped_blank():
    """A FITS reader strips a slice's trailing blank; it may be a separator."""

    header, _ = _fixture_arrays()
    specs = parse_wat_specs(header)
    cards = format_wat2_cards(specs)
    stripped = {key: value.rstrip() for key, value in cards}
    assert [s.text for s in parse_wat_specs(stripped)] == [s.text for s in specs]


def test_reversed_rows_keep_their_physical_orders():
    """What the PLP's invert_order writes: rows and WAT entries both reversed."""

    header, wavelength = _fixture_arrays()
    specs = parse_wat_specs(header)
    flipped = {k: v for k, v in header.items() if not k.startswith("WAT2_")}
    flipped.update(format_wat2_cards(specs[::-1]))
    assert identify_orders("H", wavelength[::-1], flipped) == ((109, 100), "wat")


def test_wat_entries_are_matched_by_wavelength_not_position():
    header, wavelength = _fixture_arrays()
    specs = parse_wat_specs(header)
    # Entries listed in the opposite order to the rows they describe.
    swapped = {k: v for k, v in header.items() if not k.startswith("WAT2_")}
    swapped.update(format_wat2_cards(specs[::-1]))
    assert identify_orders("H", wavelength, swapped) == ((100, 109), "wat")


def test_without_wat_the_wavelength_names_the_order():
    header, wavelength = _fixture_arrays()
    bare = {k: v for k, v in header.items() if not k.startswith("WAT2_")}
    assert identify_orders("H", wavelength, bare) == ((100, 109), "wavelength")


def test_wat_describing_other_rows_is_caught():
    """The fixture's own history: a full frame's cards kept on two of its rows."""

    header, wavelength = _fixture_arrays()
    specs = parse_wat_specs(header)
    wrong = [WatSpec(aperture=s.aperture, order=s.order - 2,
                     w1_angstrom=s.w1_angstrom + 356.0, dw_angstrom=s.dw_angstrom,
                     nw=s.nw, text=s.text) for s in specs]
    moved = []
    for spec in wrong:
        fields = spec.text.split()
        fields[1], fields[3] = str(spec.order), repr(spec.w1_angstrom)
        moved.append(spec.__class__(**{**spec.__dict__, "text": " ".join(fields)}))
    lied = {k: v for k, v in header.items() if not k.startswith("WAT2_")}
    lied.update(format_wat2_cards(moved))
    with pytest.warns(UserWarning, match="do not describe these rows"):
        assert identify_orders("H", wavelength, lied) == \
            ((100, 109), "wavelength; wat inconsistent")


def test_an_empty_row_has_no_order():
    """The PLP fills an order it did not extract with NaN."""

    _, wavelength = _fixture_arrays()
    padded = np.vstack([wavelength, np.full((1, wavelength.shape[1]), np.nan)])
    assert identify_orders("H", padded) == ((100, 109, None), "wavelength")


def test_a_row_no_known_order_matches_is_refused():
    _, wavelength = _fixture_arrays()
    with pytest.raises(ValueError, match="cannot name the physical order"):
        identify_orders("H", wavelength + 0.008)


def test_the_centre_table_is_ordered_like_an_echelle():
    """Higher order, shorter wavelength, and m * lambda roughly constant."""

    for band, table in IGRINS_ORDER_CENTRES_UM.items():
        numbers = sorted(table)
        assert numbers == list(range(numbers[0], numbers[-1] + 1))
        assert np.all(np.diff([table[m] for m in numbers]) < 0)
        product = np.array([m * table[m] for m in numbers])
        assert np.ptp(product) / np.median(product) < 0.01
