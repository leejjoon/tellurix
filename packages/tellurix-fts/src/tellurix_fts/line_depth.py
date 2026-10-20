"""Per telluric feature, observed against model: optical depth ratio and position offset.

The measurement behind docs/line_intensity_corrections.md. A feature is a cluster
of lines with a centre (``scripts/measure_line_depth_ratios.py`` builds them from
AER); in one fitted window its pixels are those within ``CORE_CM1`` of the centre
that are in the fit mask, solar-clean, not saturated (model above 0.02) and
absorbing (model optical depth above 0.1), optical depths taken against the
solar model. Shared by the solar fits' measurement and Baker et al. 2020's
telluric spectra, so the two are one definition.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

CORE_CM1 = 0.05
# A line's position is in its flanks, which the core cut drops.
SHIFT_CM1 = 0.1
# 0.02 cm-1 is ~400 m/s: twenty times the line-list offsets being measured.
MAX_SHIFT_CM1 = 0.02
MIN_PIXELS = 3
SPEED_OF_LIGHT_MS = 299792458.0


def feature_measurements(nu, observed, model, solar, mask, centres, *, clean=None,
                         solar_clean: float = 0.97, edge_pixels: int = 10):
    """``{feature index: (tau ratio, peak model tau, offset m/s)}``.

    ``observed``, ``model`` and ``solar`` are over the continuum and on ascending
    ``nu``. ``clean`` overrides ``solar > solar_clean`` as the solar-clean cut.
    The ratio is sum(tau_observed) / sum(tau_model) over the feature's pixels. The
    offset is :func:`shift_and_scale`'s over the unsaturated pixels within
    ``SHIFT_CM1``, as a velocity: positive puts the observed line at higher
    wavenumber. A centroid over a window fixed on the model's centre is pulled
    toward that centre, and read offsets 3-5x too small.
    """

    nu = np.asarray(nu, dtype=float)
    transmitted_model = np.asarray(model) / np.asarray(solar)
    transmitted_observed = np.asarray(observed) / np.asarray(solar)
    tau_model = -np.log(np.clip(transmitted_model, 1e-6, None))
    tau_observed = -np.log(np.clip(transmitted_observed, 1e-6, None))
    if clean is None:
        clean = np.asarray(solar) > solar_clean
    usable = np.asarray(mask, bool) & clean & (np.asarray(model) > 0.02)
    good = usable & (tau_model > 0.1)
    # The lines are 1-2 pixels wide (sigma): a finite-difference slope reads the
    # shift ~18% high, so the model is shifted along a cubic spline instead.
    spline = CubicSpline(nu, transmitted_model)
    out = {}
    inside = (centres > nu[edge_pixels]) & (centres < nu[-edge_pixels])
    for i in np.where(inside)[0]:
        core = (np.abs(nu - centres[i]) < CORE_CM1) & good
        if core.sum() < MIN_PIXELS:
            continue
        near = (np.abs(nu - centres[i]) < SHIFT_CM1) & usable
        offset = shift_and_scale(spline, nu[near], transmitted_observed[near])[0]
        out[int(i)] = (float(tau_observed[core].sum() / tau_model[core].sum()),
                       float(tau_model[core].max()),
                       float(offset / centres[i] * SPEED_OF_LIGHT_MS))
    return out


def shift_and_scale(spline, nu, observed, iterations: int = 5):
    """Least-squares (shift cm-1, scale) taking the model onto ``observed``.

    The model is ``spline(nu - shift) ** (1 + scale)``: shifted, and its optical
    depth scaled. The scale is free so that a strength error, even about the line
    but cut unevenly by the mask and the neighbours, is not read as a shift.
    """

    shift, scale = 0.0, 0.0
    for _ in range(iterations):
        model = np.clip(spline(nu - shift), 1e-6, 1.0)
        predicted = model ** (1.0 + scale)
        jacobian = np.stack([-(1.0 + scale) * predicted / model * spline(nu - shift, 1),
                             predicted * np.log(model)], axis=1)
        try:
            step = np.linalg.lstsq(jacobian, observed - predicted, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        shift, scale = shift + step[0], scale + step[1]
        # A feature this far off is a blend the model has wrong or a masked core,
        # not a position: give up rather than follow it.
        if not (np.isfinite(shift) and abs(shift) < MAX_SHIFT_CM1 and -0.9 < scale < 3.0):
            return np.nan, np.nan
    return shift, scale
