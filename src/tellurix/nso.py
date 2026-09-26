"""Readers for the NSO Kitt Peak FTS solar atlases.

Two formats, one instrument. ``ftsspec_*`` are raw McMath-Pierce FTS solar
spectra carrying an observing header; ``photatl`` is the Livingston & Wallace
(1991) atlas built from spectra of the same programme. References:

    Livingston, W., & Wallace, L. 1991, NSO Technical Report 91-001
    Wallace, L., Livingston, W., Hinkle, K., & Bernath, P. 1996, ApJS, 106, 165

The data is not fetched by this package; see ``docs/solar_fit_plan.md`` for the
retrieval route and ``docs/solar_atlases.md`` for the survey. Any publication
using it must carry the acknowledgement in :data:`NSO_ACKNOWLEDGEMENT`.

Loaders for the same files exist in the sibling ``differentiable_stellar_-
spectroscopy`` project (``dss/data/atlases.py``). These are deliberately
separate and self-contained, for the reasons ``atlas.py`` gives: that project
is not installable, and its ``mask`` marks *bad* pixels where
:class:`~tellurix.types.SpectralOrder` marks usable ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from .atlas import robust_noise
from .types import SpectralOrder


NSO_ACKNOWLEDGEMENT = "NSO/Kitt Peak FTS data used here were produced by NSF/NOAO."
PHOTATL_REFERENCE = (
    "Livingston, W., & Wallace, L. 1991, An Atlas of the Solar Spectrum in the Infrared "
    "from 1850 to 9000 cm-1, NSO Technical Report 91-001"
)
FTS_SERIES_REFERENCE = (
    "Wallace, L., Livingston, W., Hinkle, K., & Bernath, P. 1996, ApJS, 106, 165"
)

# McMath-Pierce, Kitt Peak. The altitude is what make_site_profile.py already
# defaults to, and the position is what an ERA5 profile needs.
KITT_PEAK_LATITUDE_DEG = 31.9583
KITT_PEAK_LONGITUDE_DEG = -111.5967
KITT_PEAK_ALTITUDE_KM = 2.096

# Mountain Standard Time is UT-7 year round; Arizona does not observe DST.
_MST_OFFSET_HOURS = 7

# The sampling both formats share, measured by least squares over every page of
# photatl (0.00947710) and over ftsspec_901218_5 (0.00947709), and documented
# by the sibling project at 0.0094771. That they agree to the eighth decimal is
# the strongest evidence that photatl is built from spectra like these.
SAMPLING_CM1 = 0.0094771

# The sinc FWHM measured from the interferogram cut; see docs/solar_fit_plan.md.
# Constant across the atlas because these spectra hold the path difference
# fixed, not the resolving power.
MEASURED_FWHM_CM1 = 0.01727


def zenith_angle_deg_for_airmass(airmass: float) -> float:
    """Zenith angle whose secant is ``airmass``.

    :meth:`TelluricModel.transmission` divides the vertical optical depth by
    ``cos z``, so feeding back ``arccos(1/X)`` reproduces the header's own air
    mass exactly. That is the point: the observatory's X already carries
    refraction and Earth curvature, and a plane-parallel ``sec z`` computed
    from a true zenith angle would not. Round-tripping through this function
    keeps the model on the measured quantity rather than on an approximation
    to it, which matters at X = 5.88 where the two differ by about 1%.
    """

    if not np.isfinite(airmass) or airmass < 1.0:
        raise ValueError("air mass must be finite and at least 1")
    zenith = math.degrees(math.acos(1.0 / airmass))
    if not 0.0 <= zenith < 90.0:
        raise ValueError(f"air mass {airmass} is outside what SpectralOrder accepts")
    return zenith


def uniform_wavenumber_grid(wavenumber_cm1: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares uniform grid through stored wavenumbers, and the residual.

    **Both formats store a uniform grid at reduced precision and neither says
    so.** ``photatl`` writes single-precision wavenumbers: the spacings in a
    page take two values one float32 ulp apart (measured ratio 1.01-1.15 at
    1850, 5000 and 8975 cm-1), and the jitter about a uniform grid grows with
    wavenumber -- 0.4% of the 0.01727 cm-1 resolution element at 1850 cm-1,
    1.5% at 5000, **2.8% at 9000**. The ftsspec files store four decimals,
    which is a flat 0.3% everywhere.

    Fitting all N samples averages the quantization down by sqrt(N), so the
    reconstructed grid is better than any stored sample. Use it. Taking the
    spacing from the two endpoints instead would inherit their quantization in
    full, and using the stored values leaves a jitter that aliases straight
    into a fitted velocity.

    Returns the grid and the maximum absolute residual of the stored values
    about it, which is a useful check that a file really is evenly sampled.
    """

    values = np.asarray(wavenumber_cm1, dtype=float)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("need at least eight samples to fit a grid")
    index = np.arange(values.size, dtype=float)
    design = np.vstack([index, np.ones_like(index)]).T
    (spacing, offset), *_ = np.linalg.lstsq(design, values, rcond=None)
    if spacing <= 0.0:
        raise ValueError("wavenumbers must increase")
    grid = offset + spacing * index
    return grid, float(np.max(np.abs(values - grid)))


# --------------------------------------------------------------- raw FTS ----

_HEADER_PATTERNS = {
    "source_name": re.compile(r"^source name:(.*)$"),
    "comment": re.compile(r"^comment:(.*)$"),
    "date_mst": re.compile(r"^date \(MST\):\s*(\S+)"),
    "julian_day": re.compile(r"^julian day number=\s*(\d+)"),
}
_PAIR_PATTERNS = {
    "airmass": re.compile(r"^airmass\s+(\S+)\s+(\S+)\s*$"),
}
# The clock fields are right-justified per component, so a single-digit second
# arrives as "15:18: 9.0" -- with a space inside the token. Matching \S+ drops
# exactly those lines, and silently, which is why the fields are matched
# component by component instead.
_CLOCK = r"(\?+|\d+)\s*:\s*(\?+|\d+)\s*:\s*([?.]+|[\d.]+)"
_UNIVERSAL_TIME = re.compile(rf"^universal time\s+{_CLOCK}\s+{_CLOCK}\s*$")
_SAMPLES = re.compile(r"^number of samples=\s*(\d+)\s+point number of center=\s*(\d+)")
_RESOLUTION = re.compile(r"^resolution=\s*([0-9.]+)\s*cm-1")
_DATA_LINE = re.compile(r"^\s*[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s*$")


def _optional_float(text: str) -> float | None:
    """Parse a header number that may be a row of question marks.

    ``ftsspec_830626_3`` writes ``?.??`` for air mass and ``?: ?: ?.?`` for
    sidereal time and hour angle. Returning None keeps that honest; its
    free-text comment says "EAST 3. AIRMASSES", which is prose and is not
    parsed here.
    """

    try:
        return float(text)
    except ValueError:
        return None


def _clock_to_hours(text: str) -> float | None:
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (float(part) for part in parts)
    except ValueError:
        return None
    return hours + minutes / 60.0 + seconds / 3600.0


@dataclass(frozen=True)
class FTSSpectrum:
    """One raw McMath-Pierce FTS solar spectrum, ascending in wavenumber."""

    path: Path
    sha256: str
    source_name: str
    comment: str
    date_mst: str
    julian_day: int
    universal_time_start: str
    universal_time_stop: str
    airmass_start: float | None
    airmass_stop: float | None
    stated_resolution_cm1: float
    transform_samples: int
    point_of_center: int
    wavenumber_vacuum_cm1: np.ndarray
    flux: np.ndarray
    grid_residual_cm1: float
    observed_utc: _dt.datetime | None = field(default=None)

    def __post_init__(self) -> None:
        nu = np.asarray(self.wavenumber_vacuum_cm1, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        if nu.ndim != 1 or nu.size < 8:
            raise ValueError("an FTS spectrum needs at least eight samples")
        if flux.shape != nu.shape:
            raise ValueError("flux must match the wavenumber grid")
        if np.any(~np.isfinite(nu)) or np.any(np.diff(nu) <= 0.0):
            raise ValueError("wavenumbers must be finite and strictly increasing")
        object.__setattr__(self, "wavenumber_vacuum_cm1", nu)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "path", Path(self.path))

    @property
    def spacing_cm1(self) -> float:
        return float(np.median(np.diff(self.wavenumber_vacuum_cm1)))

    @property
    def airmass_mean(self) -> float | None:
        """Mean of the start and stop air mass.

        **This is not the air mass of the observation when the scan is long.**
        The FTS co-adds over the whole exposure, so what was recorded is
        ``<exp(-tau X)>`` over time, not ``exp(-tau <X>)``. For
        ``ftsspec_901218_5`` (2.17 -> 1.80) the difference peaks at 0.0008 in
        transmission, about one sigma of that file's noise; for
        ``ftsspec_901218_4`` (5.88 -> 3.58) it reaches 0.0053, which is 5-10
        sigma. See docs/solar_fit_plan.md.
        """

        if self.airmass_start is None or self.airmass_stop is None:
            return None
        return 0.5 * (self.airmass_start + self.airmass_stop)

    def select(self, wavenumber_min_cm1: float, wavenumber_max_cm1: float) -> "FTSSpectrum":
        if not wavenumber_min_cm1 < wavenumber_max_cm1:
            raise ValueError("the selected range must be increasing")
        keep = (self.wavenumber_vacuum_cm1 >= wavenumber_min_cm1) & (
            self.wavenumber_vacuum_cm1 <= wavenumber_max_cm1
        )
        if np.count_nonzero(keep) < 8:
            raise ValueError("the selected range leaves too few samples")
        return FTSSpectrum(
            path=self.path, sha256=self.sha256, source_name=self.source_name,
            comment=self.comment, date_mst=self.date_mst, julian_day=self.julian_day,
            universal_time_start=self.universal_time_start,
            universal_time_stop=self.universal_time_stop,
            airmass_start=self.airmass_start, airmass_stop=self.airmass_stop,
            stated_resolution_cm1=self.stated_resolution_cm1,
            transform_samples=self.transform_samples, point_of_center=self.point_of_center,
            wavenumber_vacuum_cm1=self.wavenumber_vacuum_cm1[keep],
            flux=self.flux[keep], grid_residual_cm1=self.grid_residual_cm1,
            observed_utc=self.observed_utc,
        )


def read_fts_spectrum(path: str | Path, *, reconstruct_grid: bool = True) -> FTSSpectrum:
    """Read one ``ftsspec_*.txt`` file: free-form header, then two columns.

    The header is parsed by field rather than by line number, because the files
    differ in trailing whitespace and in whether a field is filled in at all.

    ``transform_samples`` is the header's ``number of samples``, which is the
    transform length and **not** the number of spectral points stored: the two
    agree for the 1990 files (831,488) and do not for the 1983 pair, where the
    header says 745,472 and 1,048,576 against 540,672 rows on disk. Nothing is
    validated against it for that reason.

    ``stated_resolution_cm1`` is recorded and **must not be used as the
    instrument profile**. The header of ``ftsspec_901218_4`` says 0.053 cm-1,
    which is a 9.43 cm path difference read as ``1/(2L)`` or 18.87 cm read as
    ``1/L``; the interferogram cut of that same file is at 34.4 cm, giving
    0.0175 cm-1. Measure the cut; see ``scripts/measure_atlas_ils.py``.
    """

    path = Path(path)
    content = path.read_bytes()
    text = content.decode("ascii", errors="replace")

    header: dict = {}
    pairs: dict = {}
    resolution = float("nan")
    samples = point_of_center = 0
    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        if _DATA_LINE.match(line):
            first, second = line.split()
            rows.append((float(first), float(second)))
            continue
        if not rows:
            for name, pattern in _HEADER_PATTERNS.items():
                found = pattern.match(line)
                if found:
                    header[name] = found.group(1).strip()
            for name, pattern in _PAIR_PATTERNS.items():
                found = pattern.match(line)
                if found:
                    pairs[name] = (found.group(1), found.group(2))
            found = _UNIVERSAL_TIME.match(line)
            if found:
                parts = found.groups()
                pairs["universal_time"] = (":".join(parts[:3]), ":".join(parts[3:]))
            found = _SAMPLES.match(line)
            if found:
                samples, point_of_center = int(found.group(1)), int(found.group(2))
            found = _RESOLUTION.match(line)
            if found:
                resolution = float(found.group(1))
    if len(rows) < 8:
        raise ValueError(f"no usable data records found in {path}")
    for required in ("source_name", "date_mst", "julian_day"):
        if required not in header:
            raise ValueError(f"{path} is missing the '{required}' header field")

    values = np.asarray(rows, dtype=float)
    stored, flux = values[:, 0], values[:, 1]
    grid, residual = uniform_wavenumber_grid(stored)
    wavenumber = grid if reconstruct_grid else stored

    universal_time = pairs.get("universal_time", ("", ""))
    airmass = pairs.get("airmass", ("", ""))
    observed = _observed_utc(header["date_mst"], universal_time[0])
    if observed is not None:
        _check_julian_day(observed, int(header["julian_day"]), path)

    return FTSSpectrum(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        source_name=header["source_name"],
        comment=header.get("comment", ""),
        date_mst=header["date_mst"],
        julian_day=int(header["julian_day"]),
        universal_time_start=universal_time[0],
        universal_time_stop=universal_time[1],
        airmass_start=_optional_float(airmass[0]),
        airmass_stop=_optional_float(airmass[1]),
        stated_resolution_cm1=resolution,
        transform_samples=samples,
        point_of_center=point_of_center,
        wavenumber_vacuum_cm1=wavenumber,
        flux=flux,
        grid_residual_cm1=residual,
        observed_utc=observed,
    )


def _observed_utc(date_mst: str, universal_time: str) -> _dt.datetime | None:
    """Combine the MST calendar date with the UT clock time.

    The header dates the run in MST but times it in UT, and the two fall on
    different calendar days whenever UT is before 07:00 -- which is every night
    observation, though not the daytime solar runs this reads. Getting it wrong
    moves an ERA5 profile by a day.
    """

    hours = _clock_to_hours(universal_time)
    if hours is None:
        return None
    try:
        month, day, year = (int(part) for part in date_mst.split("/"))
    except ValueError:
        return None
    # Two-digit years: this instrument ran from the 1970s, so there is no
    # ambiguity to resolve below 70.
    year += 1900 if year >= 70 else 2000
    stamp = _dt.datetime(year, month, day, tzinfo=_dt.timezone.utc) + _dt.timedelta(hours=hours)
    if hours < _MST_OFFSET_HOURS:
        stamp += _dt.timedelta(days=1)
    return stamp


def _check_julian_day(observed: _dt.datetime, julian_day: int, path: Path) -> None:
    """Refuse a date the header's own Julian day contradicts.

    The archive is not self-describing and a misparsed date would silently
    fetch the wrong day's atmosphere. This is the same guard
    ``igrins.surface_conditions`` applies to the weather cards.
    """

    unix = observed.timestamp()
    derived = unix / 86400.0 + 2440587.5
    if abs(derived - julian_day) > 1.0:
        raise ValueError(
            f"{path}: parsed {observed.isoformat()} (JD {derived:.2f}) but the header "
            f"says Julian day {julian_day}"
        )


# ---------------------------------------------------------------- photatl ----


@dataclass(frozen=True)
class PhotatlPage:
    """One ``wnXXXX`` page of Livingston & Wallace (1991), ascending in wavenumber.

    ``total`` is the observed spectrum and is the only column that should be
    fitted. ``solar`` is **linearly interpolated wherever ``atmospheric < dg``**
    -- the README says so, and it is measurable: in those pixels the solar
    column's median second difference is 1e-16 against 9e-04 in transparent
    pixels, while ``total`` and ``atmospheric`` carry *more* structure there
    than outside, as real absorption does.
    """

    path: Path
    sha256: str
    dg: float
    wavenumber_vacuum_cm1: np.ndarray
    solar: np.ndarray
    atmospheric: np.ndarray
    total: np.ndarray
    grid_residual_cm1: float

    def __post_init__(self) -> None:
        nu = np.asarray(self.wavenumber_vacuum_cm1, dtype=float)
        if nu.ndim != 1 or nu.size < 8:
            raise ValueError("an atlas page needs at least eight samples")
        if np.any(~np.isfinite(nu)) or np.any(np.diff(nu) <= 0.0):
            raise ValueError("atlas wavenumbers must be finite and strictly increasing")
        for name in ("solar", "atmospheric", "total"):
            column = np.asarray(getattr(self, name), dtype=float)
            if column.shape != nu.shape:
                raise ValueError(f"the {name} column must match the wavenumber grid")
            object.__setattr__(self, name, column)
        object.__setattr__(self, "wavenumber_vacuum_cm1", nu)
        object.__setattr__(self, "path", Path(self.path))

    @property
    def spacing_cm1(self) -> float:
        return float(np.median(np.diff(self.wavenumber_vacuum_cm1)))

    @property
    def interpolated(self) -> np.ndarray:
        """True where the *solar* column was filled in rather than measured."""

        return self.atmospheric < self.dg

    def select(self, wavenumber_min_cm1: float, wavenumber_max_cm1: float) -> "PhotatlPage":
        """Restrict to a wavenumber range.

        Pages overlap their neighbours by 2 cm-1 and each carries its own
        predetermined intensity scale, so a fit should stay inside one page
        rather than blend across a seam.
        """

        if not wavenumber_min_cm1 < wavenumber_max_cm1:
            raise ValueError("the selected range must be increasing")
        keep = (self.wavenumber_vacuum_cm1 >= wavenumber_min_cm1) & (
            self.wavenumber_vacuum_cm1 <= wavenumber_max_cm1
        )
        if np.count_nonzero(keep) < 8:
            raise ValueError("the selected range leaves too few samples")
        return PhotatlPage(
            self.path, self.sha256, self.dg, self.wavenumber_vacuum_cm1[keep],
            self.solar[keep], self.atmospheric[keep], self.total[keep],
            self.grid_residual_cm1,
        )


def read_photatl_page(path: str | Path, *, reconstruct_grid: bool = True) -> PhotatlPage:
    """Read one ``wnXXXX`` page: four columns, 3061 rows, the last one special.

    **The final row is dropped.** The README says the parameter ``dg``
    "replaces the last point in the ``total``", so that row's fourth column is
    a threshold and not a measurement -- on ``wn5000`` it reads 0.6 where the
    atmospheric column is 0.988. Its wavenumber is on grid and its solar and
    atmospheric values look real, but ``total`` is what gets fitted, so the row
    goes and ``dg`` is kept as a field.
    """

    path = Path(path)
    content = path.read_bytes()
    values = np.loadtxt(path)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] < 9:
        raise ValueError(f"{path} is not a four-column photatl page")
    dg = float(values[-1, 3])
    values = values[:-1]
    grid, residual = uniform_wavenumber_grid(values[:, 0])
    return PhotatlPage(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        dg=dg,
        wavenumber_vacuum_cm1=grid if reconstruct_grid else values[:, 0],
        solar=values[:, 1],
        atmospheric=values[:, 2],
        total=values[:, 3],
        grid_residual_cm1=residual,
    )


# ------------------------------------------------------------ to an order ----


def _running_median(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(windows, axis=-1)


def _saturation_mask(
    flux: np.ndarray, saturation_floor: float, smooth_pixels: int, minimum_continuum_snr: float
) -> np.ndarray:
    """Usable where the smoothed flux clears a fraction of the local continuum.

    Masking individual non-positive pixels would bias a saturated core upward:
    the true flux there is zero and the noise scatters either side, so keeping
    only the positive excursions keeps only half the distribution. Deciding on
    a *smoothed* flux masks the core as a region instead.

    **The relative threshold needs an absolute companion**, because a window
    that passes no light at all has a "continuum" made of noise and every pixel
    clears a fraction of it. Measured on ``ftsspec_901218_5``, the continuum
    over the window's own noise separates the two cases by a wide margin:

    ======================  =====  ==========================================
    window (cm-1)           SNR    what it is
    ======================  =====  ==========================================
    6000-6030                2702  clean, peak response
    2200-2230                 188  real signal, heavily absorbed
    3700-3730                  11  the 2.7 um hole
    2340-2370                   6  the CO2 band core
    1700-1730                   5  below the filter cut
    ======================  =====  ==========================================

    So an opaque window is refused here rather than fitted to noise, which is
    the lesson ``minimum_reliable`` records for IGRINS: skip the order, do not
    write a NaN row.
    """

    if not 0.0 < saturation_floor < 1.0:
        raise ValueError("saturation_floor must lie in (0, 1)")
    finite = np.isfinite(flux)
    if not np.any(finite):
        raise ValueError("no finite pixels in this window")
    # The response envelope is flat across a window this narrow, so a high
    # percentile of the window is a serviceable continuum estimate.
    continuum = float(np.percentile(flux[finite], 99.0))
    if not np.isfinite(continuum) or continuum <= 0.0:
        raise ValueError("this window has no positive continuum; it is opaque or off the filter")
    noise = robust_noise(flux[finite])
    if continuum < minimum_continuum_snr * noise:
        raise ValueError(
            f"window continuum {continuum:.5g} is only {continuum / noise:.1f} sigma above "
            f"zero (need {minimum_continuum_snr:g}); it is opaque or off the filter"
        )
    return finite & (_running_median(np.where(finite, flux, 0.0), smooth_pixels)
                     >= saturation_floor * continuum)


def _to_order(
    wavenumber_cm1: np.ndarray,
    flux: np.ndarray,
    *,
    uncertainty,
    saturation_floor: float,
    smooth_pixels: int,
    minimum_continuum_snr: float,
    zenith_angle_deg: float,
    source_flux_model_grid,
) -> SpectralOrder:
    order = np.argsort(1.0e7 / wavenumber_cm1)
    wavelength_nm = (1.0e7 / wavenumber_cm1)[order]
    values = np.asarray(flux, dtype=float)[order]

    mask = _saturation_mask(values, saturation_floor, smooth_pixels, minimum_continuum_snr)
    if not np.any(mask):
        raise ValueError("every pixel was masked")
    if uncertainty is None:
        sigma = np.full(values.size, robust_noise(values[mask]))
    else:
        sigma = np.broadcast_to(np.asarray(uncertainty, dtype=float), values.shape).copy()
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("uncertainties must be finite and positive")

    # SpectralOrder validates the flux array itself, so masked-out pixels still
    # need a finite positive placeholder.
    return SpectralOrder(
        wavelength_vacuum_nm=wavelength_nm,
        flux=np.where(mask, values, 1.0),
        uncertainty=sigma,
        mask=mask,
        zenith_angle_deg=zenith_angle_deg,
        source_flux_model_grid=source_flux_model_grid,
    )


def fts_spectral_order(
    spectrum: FTSSpectrum,
    *,
    uncertainty: float | np.ndarray | None = None,
    saturation_floor: float = 0.02,
    smooth_pixels: int = 5,
    minimum_continuum_snr: float = 30.0,
    zenith_angle_deg: float | None = None,
    source_flux_model_grid: np.ndarray | None = None,
) -> SpectralOrder:
    """Convert a window of a raw FTS spectrum into a fittable order.

    ``zenith_angle_deg`` defaults to the header's mean air mass through
    :func:`zenith_angle_deg_for_airmass`. Pass 0.0 to fold the air mass into
    the fitted column scales instead, which is the Arcturus configuration and
    the diagnostic half of the slant-path test.

    The atlas ascends in wavenumber and :class:`SpectralOrder` requires
    ascending vacuum wavelength, so the columns are reversed here. Getting that
    wrong mirrors the spectrum and still looks plausible.
    """

    if zenith_angle_deg is None:
        airmass = spectrum.airmass_mean
        if airmass is None:
            raise ValueError(
                f"{spectrum.path.name} has no air mass in its header; pass zenith_angle_deg"
            )
        zenith_angle_deg = zenith_angle_deg_for_airmass(airmass)
    return _to_order(
        spectrum.wavenumber_vacuum_cm1, spectrum.flux, uncertainty=uncertainty,
        saturation_floor=saturation_floor, smooth_pixels=smooth_pixels,
        minimum_continuum_snr=minimum_continuum_snr, zenith_angle_deg=zenith_angle_deg,
        source_flux_model_grid=source_flux_model_grid,
    )


def photatl_spectral_order(
    page: PhotatlPage,
    *,
    column: str = "total",
    uncertainty: float | np.ndarray | None = None,
    saturation_floor: float = 0.02,
    smooth_pixels: int = 5,
    minimum_continuum_snr: float = 30.0,
    zenith_angle_deg: float = 0.0,
    source_flux_model_grid: np.ndarray | None = None,
) -> SpectralOrder:
    """Convert a page into a fittable order.

    ``column`` must be ``"total"``, the observed spectrum. ``"solar"`` is
    refused: it is linearly interpolated over 23.5% of the atlas on average
    (30.7% over 2.5-4.0 um), so fitting it would fit fabricated pixels.
    ``"atmospheric"`` is the authors' own telluric estimate at atlas
    resolution, which is what this work replaces rather than reproduces.

    ``zenith_angle_deg`` defaults to 0: the atlas records no air mass, so it is
    absorbed into the fitted column scales, and a fitted scale is then
    ``true column x air mass / profile column``. Two pages cannot separate the
    two factors. That is the Arcturus caveat, inherited.
    """

    if column != "total":
        raise ValueError("column must be 'total'; see the docstring on 'solar' and 'atmospheric'")
    return _to_order(
        page.wavenumber_vacuum_cm1, page.total, uncertainty=uncertainty,
        saturation_floor=saturation_floor, smooth_pixels=smooth_pixels,
        minimum_continuum_snr=minimum_continuum_snr, zenith_angle_deg=zenith_angle_deg,
        source_flux_model_grid=source_flux_model_grid,
    )
