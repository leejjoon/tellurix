"""Measuring an FTS instrument profile from the spectrum itself.

An FTS spectrum is the Fourier transform of an interferogram truncated at the
maximum optical path difference, so the transform of a spectrum *is* that
interferogram: its envelope is flat out to the MOPD and then cuts. Both the
MOPD and the apodization are therefore measurable, and neither has to be
assumed -- which matters, because the atlases that use this either document no
resolution at all or document one their own data contradicts.

Zero-filling does not defeat it. An FTS product is commonly zero-filled by a
factor its sampling does not reveal, but appending zeros to an interferogram
does not move where it was truncated, so the cut is still the real one.

This lives in the package rather than in a script because every atlas here is
an FTS product and all of them need it: Arcturus
(``scripts/measure_atlas_ils.py``) and the NSO solar atlases
(``scripts/measure_solar_ils.py``).
"""

from __future__ import annotations

import numpy as np


# Full width at half maximum of sinc(x) = sin(x)/x, in units of 1/(2 L).
BOXCAR_FWHM_CONSTANT = 1.20671

# How far the envelope must fall across the cut before the cut is believed.
# A spectrum with no truncation inside its sampling -- noise -- still yields a
# confident-looking MOPD, because the steepest drop of a flat envelope is
# somewhere. What separates the two is the *step*: measured over 1,240 real
# page and window measurements spanning photatl, the ftsspec spectra and the
# Arcturus atlas, the smallest step is 1.14 decades, while white noise gives
# 0.005. Anything below this threshold is not a truncation.
MINIMUM_ENVELOPE_STEP_DECADES = 1.0


def running_median(values: np.ndarray, width: int) -> np.ndarray:
    """Median filter with edge padding. ``width`` must be odd."""

    if width % 2 != 1:
        raise ValueError("running median width must be odd")
    if width <= 1:
        return np.asarray(values, dtype=float)
    half = width // 2
    padded = np.pad(np.asarray(values, dtype=float), half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=-1)


def interferogram_envelope(
    wavenumber_cm1: np.ndarray, flux: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return optical path difference in cm and the interferogram envelope."""

    spacing = float(np.median(np.diff(wavenumber_cm1)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("wavenumbers must be increasing and evenly spaced")
    # The mean is a delta at zero path difference and the window suppresses the
    # leakage that would otherwise fill the region beyond the cut.
    windowed = (flux - np.mean(flux)) * np.hanning(len(flux))
    envelope = np.abs(np.fft.rfft(windowed))
    return np.fft.rfftfreq(len(flux), d=spacing), envelope


def measure_mopd(
    path_difference_cm: np.ndarray, envelope: np.ndarray, smoothing: int = 5
) -> dict:
    """Locate the truncation of the interferogram and describe its sharpness.

    The envelope does not fall to zero beyond the cut: an atlas stores a finite
    number of significant digits, and that quantization leaves a floor. So the
    cut is found as the steepest sustained drop, not as the last sample above a
    noise threshold -- a threshold detector locks onto the floor and
    overestimates the MOPD by several centimetres.
    """

    peak = float(np.max(envelope))
    if peak <= 0.0:
        raise ValueError("degenerate interferogram envelope")
    logarithmic = np.log10(np.maximum(running_median(envelope, smoothing), 1e-30) / peak)
    nyquist_cm = float(path_difference_cm[-1])
    spacing_cm = float(path_difference_cm[1])

    span = max(int(round(0.3 / spacing_cm)), 3)
    searchable = np.flatnonzero(
        (path_difference_cm > 2.0) & (path_difference_cm < nyquist_cm - 1.0)
    )
    searchable = searchable[searchable + span < len(logarithmic)]
    if len(searchable) < 3:
        raise ValueError("too few samples to locate a truncation")
    drops = logarithmic[searchable] - logarithmic[searchable + span]
    start = int(searchable[int(np.argmax(drops))])
    drop_decades = float(drops[int(np.argmax(drops))])

    in_band = float(
        np.median(logarithmic[(path_difference_cm > 2.0) & (path_difference_cm < path_difference_cm[start])])
    )
    beyond = (path_difference_cm > path_difference_cm[start] + 0.5) & (
        path_difference_cm < 0.95 * nyquist_cm
    )
    tail = float(np.median(logarithmic[beyond])) if np.any(beyond) else in_band - 3.0
    threshold = 0.5 * (in_band + tail)

    within = np.flatnonzero(
        (logarithmic > threshold) & (path_difference_cm < path_difference_cm[start] + 0.5)
    )
    mopd_cm = float(path_difference_cm[within[-1]]) if len(within) else float("nan")

    # An unapodized cut falls within a couple of samples. Norton-Beer tapers
    # over roughly L/3, so the transition width separates the two.
    taper = np.flatnonzero(
        (logarithmic < in_band - 0.5)
        & (logarithmic > tail + 0.5)
        & (path_difference_cm > path_difference_cm[start] - 0.5)
        & (path_difference_cm < path_difference_cm[start] + 1.0)
    )
    transition_cm = (
        float(path_difference_cm[taper[-1]] - path_difference_cm[taper[0]])
        if len(taper) > 1
        else spacing_cm
    )

    return {
        "mopd_cm": mopd_cm,
        "drop_decades": drop_decades,
        "in_band_log10": in_band,
        "tail_log10": tail,
        "transition_cm": transition_cm,
        "path_difference_resolution_cm": spacing_cm,
        "nyquist_path_difference_cm": nyquist_cm,
    }


def describe_truncation(wavenumber_cm1: np.ndarray, flux: np.ndarray, smoothing: int = 5) -> dict:
    """Measure the cut and derive what it implies for the instrument profile."""

    wavenumber_cm1 = np.asarray(wavenumber_cm1, dtype=float)
    result = measure_mopd(*interferogram_envelope(wavenumber_cm1, flux), smoothing=smoothing)
    center = float(0.5 * (wavenumber_cm1[0] + wavenumber_cm1[-1]))
    result.update(
        {
            "wavenumber_min_cm1": float(wavenumber_cm1[0]),
            "wavenumber_max_cm1": float(wavenumber_cm1[-1]),
            "wavenumber_center_cm1": center,
            "samples": int(wavenumber_cm1.size),
            "spacing_cm1": float(np.median(np.diff(wavenumber_cm1))),
        }
    )
    step = result["in_band_log10"] - result["tail_log10"]
    result["envelope_step_decades"] = float(step)
    mopd = result["mopd_cm"]
    if np.isfinite(mopd) and mopd > 0.0 and step >= MINIMUM_ENVELOPE_STEP_DECADES:
        fwhm = BOXCAR_FWHM_CONSTANT / (2.0 * mopd)
        result["ils_fwhm_cm1"] = float(fwhm)
        result["resolving_power_at_center"] = float(center / fwhm)
        result["samples_per_fwhm"] = float(fwhm / result["spacing_cm1"])
        # Beyond this the MOPD is longer than the sampling can express and the
        # measurement only recovers the decimation, not the interferogram.
        result["mopd_measurable"] = bool(mopd < 0.95 * result["nyquist_path_difference_cm"])
    else:
        result["mopd_measurable"] = False
    return result
