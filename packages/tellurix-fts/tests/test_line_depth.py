"""The per-feature ratio and offset, on one synthetic line."""

import numpy as np
import pytest

from tellurix_fts.line_depth import SPEED_OF_LIGHT_MS, feature_measurements

CENTRE = 13000.0


def _line(scale=1.0, shift_cm1=0.0):
    nu = CENTRE - 1.0 + 0.015 * np.arange(134)
    tau = 1.5 * scale * np.exp(-0.5 * ((nu - CENTRE - shift_cm1) / 0.02) ** 2)
    return nu, np.exp(-tau)


def _measure(scale, shift_cm1):
    nu, model = _line()
    _, observed = _line(scale, shift_cm1)
    ones = np.ones_like(nu)
    return feature_measurements(nu, observed, model, ones, ones.astype(bool), np.array([CENTRE]))[0]


def test_a_strength_error_is_a_ratio_and_not_a_shift():
    ratio, peak, offset = _measure(1.2, 0.0)
    assert ratio == pytest.approx(1.2, rel=1e-3)
    assert peak == pytest.approx(1.5, rel=0.05)  # the nearest pixel
    assert abs(offset) < 0.01


def test_a_shift_is_recovered_beside_a_strength_error():
    # 0.001 cm-1 is 23 m/s at 13000 cm-1, the size of the offsets measured.
    _, _, offset = _measure(1.2, 0.001)
    assert offset == pytest.approx(0.001 / CENTRE * SPEED_OF_LIGHT_MS, rel=0.02)


def test_a_feature_the_model_does_not_describe_gives_no_offset():
    # A line the model lacks altogether: the fit would chase it out of the window.
    nu, model = _line()
    observed = model * np.exp(-2.0 * np.exp(-0.5 * ((nu - CENTRE - 0.06) / 0.02) ** 2))
    ones = np.ones_like(nu)
    found = feature_measurements(nu, observed, model, ones, ones.astype(bool), np.array([CENTRE]))
    assert np.isnan(found[0][2]) or abs(found[0][2]) < 0.02 / CENTRE * SPEED_OF_LIGHT_MS
