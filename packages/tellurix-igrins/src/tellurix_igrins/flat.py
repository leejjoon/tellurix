"""The lamp flat's blaze, per physical IGRINS order, from a night's calibration bundle.

RRISA ships every night's PLP calibrations beside its spectra (the catalog's
``CAL_URL``): 2-D lamp-on and lamp-off flats, among other things. What they are
good for here is the **blaze**. The fitted continuum is a degree-9 Chebyshev in
log flux, and the blaze rolls off more steeply at the order ends than that can
bend; what it leaves behind is most of the "instrument response pattern" the
standards fits divide out. Measured on DCT 2018-12-20, the lamp's own residual
from the same degree-9 continuum reproduces that pattern at r = +0.74 to +0.89,
red-edge rise and fall included. So the blaze is measured from the lamp and
divided out, and the continuum is left only the lamp-to-star colour.

Nothing here reads the PLP's order traces, which the bundle does not include.
Orders are traced in the lamp flat itself -- bright bands ~61 rows tall, ~100
apart, running along the columns -- collapsed along the slit, and named by
physical order by matching each band's lit column range to the extracted
spectra's.

The lamp's light crosses air, so its spectrum carries telluric lines of its own
(obvious in H120). :func:`smooth_blaze` rejects them: they are narrow and only
ever dips, while the blaze is smooth and only ever bends on tens of pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from tellurix.record import file_sha256, text

FORMAT_VERSION = 1
# How deep the lamp's own telluric lines may go -- the 1st percentile of the
# lamp against its blaze -- before the blaze is not trusted. Noise alone reaches
# about -3% in H; the water bands take the lamp to -19% (H120-121) and -59%
# (K93), and there the smoother cannot tell an absorption band from the blaze.
MAXIMUM_LAMP_ABSORPTION = 0.065


def trace_bands(
    lamp: np.ndarray,
    threshold: float = 0.25,
    minimum_height: int = 40,
    step: int = 8,
    maximum_jump: float = 6.0,
    degree: int = 3,
) -> list[np.ndarray]:
    """Centre-line polynomials, row against column, of every full order in a flat.

    Bands are found column by column above ``threshold`` of the column's bright
    level, followed outward from the detector's middle column, and fitted with
    a polynomial. A band cut by the detector edge -- shorter than
    ``minimum_height`` rows -- is not an order the PLP extracted and is skipped.
    Returned bottom to top.
    """

    lamp = np.asarray(lamp, dtype=float)
    ny, nx = lamp.shape
    columns = np.arange(4, nx - 4, step)

    def centres(x):
        profile = np.nanmedian(lamp[:, max(0, x - 3):x + 4], axis=1)
        bright = profile > threshold * np.nanpercentile(profile, 99)
        edges = np.flatnonzero(np.diff(bright.astype(int)))
        starts, ends = list(edges[::2] + 1), list(edges[1::2] + 1)
        if bright[0]:
            starts = [0] + starts
        if len(ends) < len(starts):
            ends = ends + [ny]
        found = []
        for a, b in zip(starts, ends):
            if b - a >= minimum_height:
                weight = profile[a:b]
                found.append(float(np.sum(np.arange(a, b) * weight) / np.sum(weight)))
        return found

    middle = int(np.argmin(np.abs(columns - nx // 2)))
    reference = centres(columns[middle])
    tracks = [{columns[middle]: y} for y in reference]
    for direction in (1, -1):
        last = list(reference)
        j = middle + direction
        while 0 <= j < len(columns):
            found = np.asarray(centres(columns[j]))
            for i, y in enumerate(last):
                if found.size:
                    k = int(np.argmin(np.abs(found - y)))
                    if abs(found[k] - y) < maximum_jump:
                        tracks[i][columns[j]] = float(found[k])
                        last[i] = float(found[k])
            j += direction
    fits = []
    for track in tracks:
        if len(track) < 40:
            continue
        xs = np.array(sorted(track))
        fits.append(np.polyfit(xs, [track[x] for x in xs], degree))
    return sorted(fits, key=lambda c: np.polyval(c, nx // 2))


def collapse(lamp: np.ndarray, centre: np.ndarray, half_width: int = 28) -> np.ndarray:
    """One band summed along the slit, per detector column; NaN off the detector."""

    lamp = np.asarray(lamp, dtype=float)
    ny, nx = lamp.shape
    rows = np.rint(np.polyval(centre, np.arange(nx))).astype(int)
    out = np.full(nx, np.nan)
    for x in range(nx):
        lo, hi = rows[x] - half_width, rows[x] + half_width + 1
        if lo >= 0 and hi <= ny:
            out[x] = np.nansum(lamp[lo:hi, x])
    return out


def smooth_blaze(
    lamp1d: np.ndarray, window: int = 31, clip_sigma: float = 2.5, iterations: int = 6,
    minimum_depth: float = 0.005, max_width: int = 10,
) -> np.ndarray:
    """The lamp's blaze without its narrow telluric dips.

    A Savitzky-Golay filter follows the steep order-end roll-off that a boxcar
    would round off; points more than ``clip_sigma`` robust sigmas *below* the
    running fit are the lamp's absorption lines and are replaced by the fit and
    smoothed again. Only dips are rejected: nothing in a lamp is in emission.
    """

    from scipy.signal import savgol_filter

    y = np.asarray(lamp1d, dtype=float)
    good = np.isfinite(y) & (y > 0)
    out = np.full(y.shape, np.nan)
    if np.count_nonzero(good) < window:
        return out
    index = np.flatnonzero(good)
    values = y[good].copy()
    keep = np.ones(values.size, bool)
    for _ in range(iterations):
        filled = np.where(keep, values, np.interp(index, index[keep], values[keep]))
        fit = savgol_filter(filled, window, 3)
        ratio = values / np.maximum(fit, 1e-12) - 1.0
        sigma = 1.4826 * np.median(np.abs(ratio[keep] - np.median(ratio[keep])))
        # A floor, so a clean lamp does not clip the filter's own ripple and
        # oscillate: a lamp line worth removing is at least half a percent deep.
        new = ratio > -max(clip_sigma * sigma, minimum_depth)
        # Only *narrow* dips are lamp lines. The order-end roll-off also sits
        # below a lagging filter, for tens of pixels, and is the very thing
        # being measured: a flagged run longer than max_width is kept.
        flagged = np.flatnonzero(~new)
        if flagged.size:
            runs = np.split(flagged, np.flatnonzero(np.diff(flagged) > 1) + 1)
            for run in runs:
                if run.size > max_width:
                    new[run] = True
        if np.array_equal(new, keep):
            break
        keep = new
    filled = np.where(keep, values, np.interp(index, index[keep], values[keep]))
    out[good] = savgol_filter(filled, window, 3)
    return out


def edge_blended_blaze(lamp1d: np.ndarray, edge_window: int = 31, interior_window: int = 151,
                       edge_reach: int = 100, blend: int = 100) -> np.ndarray:
    """The narrow filter at the order ends, the wide one inside, blended between.

    One window cannot serve both. The order ends roll off over tens of pixels
    and need a ~31-pixel window to follow; the interior carries the lamp's own
    0.2-0.4% structure on ~15-pixel scales, which a 31-pixel window keeps and
    so divides into the star -- measured on DCT 2018, it raised the interior
    residual on 16-pixel scales from 1.82 to 2.28 times white noise in H. The
    narrow fit is used within ``edge_reach`` pixels of either end of the lit
    range, the wide one beyond ``edge_reach + blend``, linearly between.
    """

    narrow = smooth_blaze(lamp1d, window=edge_window)
    wide = smooth_blaze(lamp1d, window=interior_window)
    lit = np.flatnonzero(np.isfinite(narrow) & (narrow > 0.05 * np.nanmax(narrow)))
    if lit.size == 0:
        return narrow
    x = np.arange(np.asarray(lamp1d).size)
    distance = np.minimum(np.abs(x - lit.min()), np.abs(x - lit.max()))
    weight = np.clip((distance - edge_reach) / blend, 0.0, 1.0)      # 0 at the ends, 1 inside
    out = np.where(np.isfinite(wide), (1 - weight) * narrow + weight * wide, narrow)
    return np.where(np.isfinite(narrow), out, np.nan)


def lamp_spectra(flat_on: str | Path, flat_off: str | Path, spectrum: str | Path,
                 half_width: int = 28) -> tuple[dict[int, np.ndarray], float, int]:
    """The lamp's unsmoothed 1-D spectrum per physical order, on detector columns.

    Returns the spectra, the median column mismatch of the naming, and how many
    bands were traced.
    """

    from astropy.io import fits

    from .igrins import read_igrins_observation

    lamp = (np.asarray(fits.getdata(flat_on, 0), dtype=float)
            - np.asarray(fits.getdata(flat_off, 0), dtype=float))
    centres = trace_bands(lamp)
    raw = [collapse(lamp, c, half_width) for c in centres]
    extents = []
    for values in raw:
        lit = np.flatnonzero(np.isfinite(values) & (values > 0.05 * np.nanmax(values)))
        extents.append((int(lit.min()), int(lit.max())))
    observation = read_igrins_observation(spectrum)
    spectrum_extents = {}
    for row, number in enumerate(observation.order_numbers):
        if number is None:
            continue
        flux = observation.flux[row]
        lit = np.flatnonzero(np.isfinite(flux) & (flux > 0.05 * np.nanmax(flux)))
        if lit.size:
            spectrum_extents[number] = (int(lit.min()), int(lit.max()))
    names, mismatch = name_bands(extents, spectrum_extents)
    return {number: raw[k] for k, number in names.items()}, mismatch, len(centres)


def name_bands(extents: list[tuple[int, int]], spectrum_extents: Mapping[int, tuple[int, int]]):
    """Physical order of each band, bottom to top, and the median column mismatch.

    Each band's lit column range is matched to the extracted orders', trying
    both directions up the detector and small offsets; the arrangement whose
    ranges agree best wins. On DCT 2018-12-20 they agree to 9 columns (H) and
    5 (K) against order-to-order differences of tens.
    """

    numbers = sorted(spectrum_extents)
    best = None
    for arrangement in (numbers, numbers[::-1]):
        for offset in range(-4, 5):
            pairs = [(k, arrangement[k + offset]) for k in range(len(extents))
                     if 0 <= k + offset < len(arrangement)]
            # Most bands must be named: an offset that pairs almost none would
            # win on the median of nothing.
            if len(pairs) < max(2, int(0.75 * min(len(extents), len(numbers)))):
                continue
            cost = float(np.median([abs(extents[k][0] - spectrum_extents[m][0])
                                    + abs(extents[k][1] - spectrum_extents[m][1])
                                    for k, m in pairs]))
            if best is None or cost < best[0]:
                best = (cost, dict(pairs))
    if best is None:
        raise ValueError("could not match the flat's bands to the extracted orders")
    return best[1], best[0]


@dataclass(frozen=True)
class FlatBlaze:
    """One band's lamp blaze, per physical order, on detector columns."""

    band: str
    orders: Mapping[int, np.ndarray]
    source: Mapping[str, object] = field(default_factory=dict)
    # Per order, the 1st percentile of lamp / blaze - 1: how deep the lamp's
    # own telluric absorption goes.
    lamp_absorption: Mapping[int, float] = field(default_factory=dict)

    def usable(self, number: int) -> bool:
        return (int(number) in self.orders
                and -self.lamp_absorption.get(int(number), 0.0) <= MAXIMUM_LAMP_ABSORPTION)

    def blaze_on(self, number: int, pixel: np.ndarray) -> np.ndarray | None:
        """The order's blaze at these detector columns, peak 1; None if unusable.

        None when the order was not traced, or when the lamp's own telluric
        absorption there is too deep to separate from the blaze.
        """

        if not self.usable(number):
            return None
        blaze = self.orders[int(number)]
        values = blaze[np.asarray(pixel, dtype=int)]
        return values / np.nanmax(blaze)

    @classmethod
    def from_flats(cls, flat_on: str | Path, flat_off: str | Path, spectrum: str | Path,
                   half_width: int = 28, window: int = 31,
                   interior_window: int | None = 151) -> "FlatBlaze":
        """Trace, collapse, smooth and name, using one extracted spectrum of the night."""

        from .igrins import read_igrins_observation

        raw, mismatch, traced = lamp_spectra(flat_on, flat_off, spectrum, half_width)
        if interior_window:
            orders = {number: edge_blended_blaze(values, window, interior_window)
                      for number, values in raw.items()}
        else:
            orders = {number: smooth_blaze(values, window=window) for number, values in raw.items()}
        absorption = {}
        for number, values in raw.items():
            smooth = orders[number]
            lit = np.isfinite(smooth) & (smooth > 0.25 * np.nanmax(smooth))
            absorption[number] = float(np.percentile(values[lit] / smooth[lit] - 1.0, 1))
        source = {"flat_on": str(flat_on), "flat_on_sha256": file_sha256(flat_on),
                  "flat_off": str(flat_off), "flat_off_sha256": file_sha256(flat_off),
                  "named_with": str(spectrum), "column_mismatch": mismatch,
                  "bands_traced": traced, "half_width": half_width, "window": window,
                  "interior_window": interior_window or 0}
        return cls(band=read_igrins_observation(spectrum).band, orders=orders, source=source,
                   lamp_absorption=absorption)

    def save(self, path: str | Path) -> Path:
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.attrs["format_version"] = FORMAT_VERSION
            handle.attrs["kind"] = "flat_blaze"
            handle.attrs["band"] = self.band
            handle.attrs["source"] = json.dumps(dict(self.source))
            for number, blaze in sorted(self.orders.items()):
                dataset = handle.create_dataset(f"{self.band}{number}",
                                                data=np.asarray(blaze, dtype=np.float32),
                                                compression="gzip", shuffle=True)
                dataset.attrs["number"] = number
                dataset.attrs["lamp_absorption"] = self.lamp_absorption.get(number, 0.0)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FlatBlaze":
        import h5py

        with h5py.File(Path(path), "r") as handle:
            if text(handle.attrs.get("kind", "")) != "flat_blaze":
                raise ValueError(f"{path} is not a flat blaze")
            orders = {int(d.attrs["number"]): d[:].astype(float) for d in handle.values()}
            absorption = {int(d.attrs["number"]): float(d.attrs.get("lamp_absorption", 0.0))
                          for d in handle.values()}
            return cls(band=text(handle.attrs["band"]), orders=orders,
                       source=json.loads(text(handle.attrs["source"])),
                       lamp_absorption=absorption)
