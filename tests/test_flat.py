"""The lamp flat's blaze: traced, collapsed, cleaned of the lamp's own lines, named."""

import numpy as np
import pytest

from tellurix import FlatBlaze
from tellurix.flat import collapse, name_bands, smooth_blaze, trace_bands

NX = NY = 512


def _blaze(x, centre=300.0, width=180.0):
    """A blaze with a steep red roll-off, like an IGRINS order's."""
    rise = np.exp(-0.5 * ((x - centre) / (0.6 * width)) ** 2)
    return rise * (1 / (1 + np.exp((x - 460.0) / 6.0)))


def _flat():
    """Four gently curved bands, 60 rows tall, 110 apart."""
    x = np.arange(NX)
    image = np.full((NY, NX), 1.0)
    centres = []
    for k, y0 in enumerate((80, 190, 300, 410)):
        yc = y0 + 1e-5 * (x - 256) ** 2
        centres.append(yc)
        for col in range(NX):
            lo, hi = int(yc[col]) - 30, int(yc[col]) + 30
            image[lo:hi, col] += 1000.0 * (1 + 0.1 * k) * _blaze(x)[col]
    return image, centres


def test_bands_are_traced_and_collapsed():
    image, truth = _flat()
    traced = trace_bands(image, step=4)
    assert len(traced) == 4
    x = np.arange(NX)
    for coefficients, centre in zip(traced, truth):
        lit = _blaze(x) > 0.3
        assert np.max(np.abs(np.polyval(coefficients, x)[lit] - centre[lit])) < 1.5
    one = collapse(image - 1.0, traced[0])
    peak = np.nanargmax(one)
    assert abs(peak - np.argmax(_blaze(x))) < 3


def test_the_lamps_own_lines_are_not_part_of_the_blaze():
    x = np.arange(NX, dtype=float)
    blaze = 1000 * _blaze(x) + 1.0
    lamp = blaze.copy()
    for centre in (150, 220, 330, 400):          # narrow telluric dips in the lamp
        lamp *= 1 - 0.3 * np.exp(-0.5 * ((x - centre) / 1.5) ** 2)
    smooth = smooth_blaze(lamp)
    body = (x > 140) & (x < 440)
    assert np.max(np.abs(smooth[body] / blaze[body] - 1)) < 0.01
    # ... while the steep red roll-off, which is the whole point, is followed.
    edge = (x > 445) & (x < 470)
    assert np.max(np.abs(smooth[edge] - blaze[edge])) / blaze.max() < 0.03


def test_bands_are_named_in_either_direction():
    spectra = {100: (10, 400), 101: (20, 410), 102: (30, 420), 103: (40, 430)}
    upward = [(11, 401), (21, 409), (29, 421), (41, 431)]
    names, mismatch = name_bands(upward, spectra)
    assert names == {0: 100, 1: 101, 2: 102, 3: 103} and mismatch < 3
    names, _ = name_bands(upward[::-1], spectra)
    assert names == {0: 103, 1: 102, 2: 101, 3: 100}


def test_the_blaze_round_trips_and_is_read_by_pixel(tmp_path):
    x = np.arange(NX, dtype=float)
    blaze = FlatBlaze(band="H", orders={109: 500 * _blaze(x) + 1.0}, source={"test": True})
    loaded = FlatBlaze.load(blaze.save(tmp_path / "blaze.h5"))
    pixel = np.array([100, 300, 450])
    assert np.allclose(loaded.blaze_on(109, pixel), blaze.blaze_on(109, pixel), rtol=1e-6)
    assert loaded.blaze_on(109, np.array([int(np.argmax(_blaze(x)))]))[0] == pytest.approx(1.0)
    assert loaded.blaze_on(110, pixel) is None


def test_the_blend_is_narrow_at_the_ends_and_wide_inside():
    from tellurix.flat import edge_blended_blaze

    x = np.arange(NX, dtype=float)
    blaze = 1000 * _blaze(x) + 1.0
    # lamp structure on ~15-pixel scales, which must not reach the star
    lamp = blaze * (1 + 0.004 * np.sin(2 * np.pi * x / 15.0))
    blended = edge_blended_blaze(lamp, edge_window=31, interior_window=151)
    narrow = smooth_blaze(lamp, window=31)
    wide = smooth_blaze(lamp, window=151)
    lit = np.flatnonzero(narrow > 0.05 * np.nanmax(narrow))
    inside = (x > lit.min() + 220) & (x < lit.max() - 220)
    assert np.allclose(blended[inside], wide[inside])
    red_end = (x > lit.max() - 60) & (x <= lit.max())
    assert np.allclose(blended[red_end], narrow[red_end])
