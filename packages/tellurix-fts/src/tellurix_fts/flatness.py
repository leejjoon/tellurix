"""How flat a corrected solar spectrum is where only the atmosphere absorbs.

The score of `scripts/score_telluric_correction.py`, shared with the comparisons that
score two atlases on one set of pixels: at pixels where the model transmission is below
a ceiling and the solar model is flat, the rms of the corrected spectrum about a running
mean of nearby pixels with neither solar nor telluric absorption.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BINS = {"telluric": 0.97, "deep": 0.7}
EDGE_PIXELS = 50
MIN_TRANSMISSION = 0.15
MEAN_PIXELS = 51
# The FWHM of an unapodized FTS sinc is this over twice the maximum OPD.
SINC_FWHM_TIMES_TWO_MOPD = 1.2067


def corrected(path: Path):
    """Wavenumber (ascending), corrected/continuum, solar model/continuum, T, mask."""

    with np.load(path) as data:
        nu = np.asarray(data["wavenumber_cm1"])
        order = np.argsort(nu)
        take = lambda key: np.asarray(data[key])[order]
        flat = take("flux") / (take("model_flux") / take("stellar_only_pixels")) / take("continuum")
        return (nu[order], flat, take("stellar_only_pixels") / take("continuum"),
                take("transmission_pixels"), take("mask").astype(bool))


def scatter(nu, values, reference, scored) -> float:
    # A normalized running mean, so a reference set shorter than the kernel or
    # a page end does not pull the baseline toward zero.
    kernel = np.ones(min(MEAN_PIXELS, int(reference.sum())))
    smooth = (np.convolve(values[reference], kernel, mode="same")
              / np.convolve(np.ones(int(reference.sum())), kernel, mode="same"))
    base = np.interp(nu, nu[reference], smooth)
    return float(100.0 * np.sqrt(np.mean((values[scored] - base[scored]) ** 2)))


def photatl_solar(pages, nu):
    """photatl's solar column on ``nu``, and where photatl interpolated it."""

    solar = np.full(nu.size, np.nan)
    filled = np.zeros(nu.size, bool)
    for page in pages:
        w = page.wavenumber_vacuum_cm1
        inside = (nu >= w.min()) & (nu <= w.max())
        if inside.any():
            order = np.argsort(w)
            solar[inside] = np.interp(nu[inside], w[order], page.solar[order])
            filled[inside] |= np.interp(nu[inside], w[order],
                                        (page.atmospheric < page.dg)[order].astype(float)) > 0
    return solar, filled


def truncate_to_sinc(nu, values, fwhm_cm1: float):
    """An FTS spectrum on an even grid, as an unapodized FTS of sinc FWHM ``fwhm_cm1`` saw it.

    Zeroes the spectrum's Fourier transform beyond the maximum OPD of that sinc. Exact for
    a spectrum measured with a shorter unapodized sinc (a longer boxcar in OPD), whose own
    transform is 1 inside the cut. The line through the two end values is taken out and the
    rest extended as an odd function (a sine series), so value and slope stay continuous
    across the transform's wrap; a plain mirror leaves a kink that rings 3e-4 deep half a
    wavenumber in. A linear term passes a symmetric kernel unchanged.
    """

    nu = np.asarray(nu, dtype=float)
    values = np.asarray(values, dtype=float)
    step = nu[1] - nu[0]
    if not np.allclose(np.diff(nu), step, rtol=1e-6):
        raise ValueError("truncate_to_sinc needs an evenly spaced grid")
    ramp = values[0] + (values[-1] - values[0]) * np.arange(values.size) / (values.size - 1)
    rest = values - ramp
    extended = np.concatenate([rest, -rest[-2:0:-1]])
    opd_cm = np.fft.rfftfreq(extended.size, d=abs(step))
    mopd_cm = SINC_FWHM_TIMES_TWO_MOPD / (2.0 * fwhm_cm1)
    out = np.fft.irfft(np.fft.rfft(extended) * (opd_cm <= mopd_cm), n=extended.size)
    return out[:values.size] + ramp
