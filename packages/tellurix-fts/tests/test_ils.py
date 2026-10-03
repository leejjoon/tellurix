import numpy as np
import pytest

from tellurix_fts import (
    BOXCAR_FWHM_CONSTANT, describe_truncation, interferogram_envelope, measure_mopd,
)
from tellurix_fts.ils import running_median


def _truncated_spectrum(mopd_cm, spacing_cm1=0.0094771, samples=4096, seed=0):
    """A spectrum whose interferogram is flat out to ``mopd_cm`` and then zero.

    Built in the interferogram domain so the answer is known exactly: this is
    what the measurement has to recover.
    """

    rng = np.random.default_rng(seed)
    wavenumber = 5000.0 + np.arange(samples) * spacing_cm1
    path = np.fft.rfftfreq(samples, d=spacing_cm1)
    interferogram = rng.normal(0.0, 1.0, path.size) * (path <= mopd_cm)
    flux = np.fft.irfft(interferogram, n=samples)
    return wavenumber, 1.0 + 0.05 * flux / np.max(np.abs(flux))


@pytest.mark.parametrize("mopd_cm", [8.0, 15.0, 34.4])
def test_recovers_a_known_truncation(mopd_cm):
    wavenumber, flux = _truncated_spectrum(mopd_cm)

    result = measure_mopd(*interferogram_envelope(wavenumber, flux))

    # The FFT bin is the resolution of the answer, so that is the tolerance.
    assert result["mopd_cm"] == pytest.approx(
        mopd_cm, abs=2.0 * result["path_difference_resolution_cm"]
    )


def test_zero_filling_does_not_move_the_cut():
    # The premise behind "the absolute scale is not obtainable this way" is
    # that an FTS product is zero-filled by an unknown factor. Appending zeros
    # to an interferogram does not move where it was truncated, which is why
    # the measurement survives it.
    wavenumber, flux = _truncated_spectrum(15.0, samples=4096)
    padded_wavenumber = 5000.0 + np.arange(8192) * (0.0094771 / 2.0)
    padded = np.interp(padded_wavenumber, wavenumber, flux, left=1.0, right=1.0)

    plain = measure_mopd(*interferogram_envelope(wavenumber, flux))["mopd_cm"]
    oversampled = measure_mopd(*interferogram_envelope(padded_wavenumber, padded))["mopd_cm"]

    assert oversampled == pytest.approx(plain, rel=0.05)


def test_describes_what_the_truncation_implies():
    wavenumber, flux = _truncated_spectrum(34.4)

    described = describe_truncation(wavenumber, flux)

    fwhm = BOXCAR_FWHM_CONSTANT / (2.0 * described["mopd_cm"])
    assert described["ils_fwhm_cm1"] == pytest.approx(fwhm)
    assert described["resolving_power_at_center"] == pytest.approx(
        described["wavenumber_center_cm1"] / fwhm
    )
    assert described["wavenumber_center_cm1"] == pytest.approx(
        0.5 * (wavenumber[0] + wavenumber[-1])
    )


def test_refuses_a_spectrum_with_no_truncation_to_find():
    # A spectrum whose interferogram fills the band -- white noise -- still
    # yields a confident-looking MOPD, because the steepest drop of a flat
    # envelope is somewhere. What gives it away is that the envelope does not
    # step: 0.005 decades here against a minimum of 1.14 over 1,240 real
    # measurements of three atlases.
    rng = np.random.default_rng(1)
    samples = 4096
    wavenumber = 5000.0 + np.arange(samples) * 0.0094771
    flux = 1.0 + 0.01 * rng.normal(0.0, 1.0, samples)

    described = describe_truncation(wavenumber, flux)

    assert described["envelope_step_decades"] < 0.1
    assert described["mopd_measurable"] is False


def test_running_median_requires_an_odd_width():
    values = np.arange(16, dtype=float)
    np.testing.assert_array_equal(running_median(values, 1), values)
    assert running_median(values, 3).shape == values.shape
    with pytest.raises(ValueError, match="odd"):
        running_median(values, 4)
