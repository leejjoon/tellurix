import numpy as np
import pytest

from tellurix_fts.flatness import scatter, truncate_to_sinc


def test_truncate_to_sinc_gives_the_requested_width():
    step = 0.0018833
    nu = 7000.0 + step * np.arange(8001)
    line = np.zeros(nu.size)
    line[4000] = 1.0
    profile = truncate_to_sinc(nu, line, 0.01753)
    half = profile > 0.5 * profile.max()
    lo, hi = np.flatnonzero(half)[[0, -1]]
    # Interpolate the half-maximum crossings on both sides.
    left = nu[lo - 1] + step * (0.5 * profile.max() - profile[lo - 1]) / (profile[lo] - profile[lo - 1])
    right = nu[hi] + step * (profile[hi] - 0.5 * profile.max()) / (profile[hi] - profile[hi + 1])
    assert right - left == pytest.approx(0.01753, rel=0.01)
    assert profile.sum() == pytest.approx(1.0, rel=1e-3)


def test_truncate_to_sinc_leaves_a_smooth_spectrum_alone():
    nu = np.linspace(7000.0, 7030.0, 15001)
    smooth = 1.0 + 0.1 * np.sin(nu / 3.0)
    assert np.allclose(truncate_to_sinc(nu, smooth, 0.01753), smooth, atol=1e-6)


def test_truncate_to_sinc_refuses_an_uneven_grid():
    with pytest.raises(ValueError):
        truncate_to_sinc(np.array([0.0, 1.0, 3.0, 4.0]), np.ones(4), 0.5)


def test_scatter_is_zero_for_a_flat_spectrum():
    nu = np.arange(500.0)
    values = np.ones(500)
    reference = np.arange(500) % 2 == 0
    assert scatter(nu, values, reference, ~reference) == pytest.approx(0.0, abs=1e-12)
