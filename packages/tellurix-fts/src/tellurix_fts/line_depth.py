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

# The feature catalogue (docs/line_depth_ratios.json): AER H2O and O2 lines above
# FEATURE_LINE_MIN_S, linked into a cluster when within FEATURE_LINK_CM1 of the next,
# kept when the strongest reaches FEATURE_MIN_S. A feature's index is its place in
# that list, which every report keys by, so the rule must not drift between them.
FEATURE_MOLECULES = ("01_H2O", "07_O2")
FEATURE_RANGE_CM1 = (8800.0, 17600.0)
FEATURE_LINK_CM1 = 0.08
FEATURE_MIN_S = 3e-25
FEATURE_LINE_MIN_S = 3e-27

CORE_CM1 = 0.05
# A line's position is in its flanks, which the core cut drops.
SHIFT_CM1 = 0.1
# 0.02 cm-1 is ~400 m/s: twenty times the line-list offsets being measured.
MAX_SHIFT_CM1 = 0.02
MIN_PIXELS = 3
SPEED_OF_LIGHT_MS = 299792458.0


def read_feature_lines(line_root) -> dict:
    """The catalogue's lines from AER's per-molecule files, sorted by wavenumber.

    Arrays ``nu``, ``strength``, ``elower``, ``molecule``, ``isotope``. Read
    record by record, not through :class:`tellurix.AERLineDatabase`, because the
    catalogue predates it and its order (wavenumber, then strength) is part of the
    feature indices.
    """

    from pathlib import Path

    lines = []
    for name in FEATURE_MOLECULES:
        for record in open(Path(line_root) / name / name):
            try:
                nu = float(record[3:15]); strength = float(record[15:25].replace("D", "E"))
                lower = float(record[45:55])
            except ValueError:
                continue
            if FEATURE_RANGE_CM1[0] < nu < FEATURE_RANGE_CM1[1] and strength > FEATURE_LINE_MIN_S:
                lines.append((nu, strength, lower, int(record[:2]), int(record[2])))
    nu, strength, lower, molecule, isotope = np.array(sorted(lines)).T
    return dict(nu=nu, strength=strength, elower=lower, molecule=molecule.astype(int), isotope=isotope.astype(int))


def cluster_features(lines: dict) -> list[dict]:
    """The catalogue's features, each with the indices of its member lines (``members``)."""

    nu, strength = lines["nu"], lines["strength"]
    out = []
    for index in np.split(np.arange(nu.size), np.where(np.diff(nu) > FEATURE_LINK_CM1)[0] + 1):
        weight = strength[index]
        if weight.max() < FEATURE_MIN_S:
            continue
        strongest = index[np.argmax(weight)]
        out.append(dict(nu=round(float(np.sum(nu[index] * weight) / weight.sum()), 4), lines=int(index.size),
                        s_max=float(weight.max()), nu_strongest=float(nu[strongest]),
                        e_lower_strongest=float(lines["elower"][strongest]),
                        molecule=int(lines["molecule"][strongest]), members=index))
    return out


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
