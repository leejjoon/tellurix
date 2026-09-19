"""Reader for IGRINS reduced spectra as distributed by RRISA.

RRISA (the Raw and Reduced IGRINS Spectral Archive) publishes IGRINS PLP v3
output for every night the instrument observed between 2014 and 2023, at three
telescopes. Reference:

    Kaplan, K. F., et al. 2024, RRISA, https://igrinscontact.github.io/

Two things make an A0V telluric standard from this archive a better test of the
forward model than the Arcturus atlas ``atlas.py`` reads. The reduction ships a
real per-pixel ``variance``, so the uncertainty is measured rather than
estimated from second differences. And the header carries the standard's own
zenith distance, so the slant path is *known* -- on the atlas it was pinned at
zero and every airmass effect was absorbed into the fitted column scales.

The archive spans three sites and nine years of header drift, so the weather
cards arrive in three mutually incompatible unit conventions with nothing in
the file to say which is in force. :func:`surface_conditions` resolves that from
``TELESCOP`` and then checks the answer against the site's altitude, because a
silent convention change is exactly the failure this module has to survive.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from .record import file_sha256
from .types import SpectralOrder


RRISA_REFERENCE = "Kaplan, K. F., et al. 2024, RRISA, https://igrinscontact.github.io/"

# Rydberg constant for hydrogen, 1/m, which puts the series in *vacuum* -- the
# frame the PLP wavelength solution and SpectralOrder both use.
_RYDBERG_HYDROGEN_PER_M = 1.09677583e7

# Standard-atmosphere constants, used only to turn a sea-level-reduced pressure
# back into a station pressure and to sanity-check the result.
_SEA_LEVEL_PRESSURE_HPA = 1013.25
_SEA_LEVEL_TEMPERATURE_K = 288.15
_STANDARD_LAPSE_K_PER_M = 0.0065
_BAROMETRIC_EXPONENT = 5.25588
_HECTOPASCAL_PER_INCH_HG = 33.863886

# How far the normalized station pressure may sit from the hydrostatic
# expectation for the site before the reading is refused. The three conventions
# observed land within 1.5%; 8% admits real weather and a little instrument
# slop while still rejecting a sea-level value read as a station one, which is
# off by 35% at DCT.
_PRESSURE_TOLERANCE = 0.08


@dataclass(frozen=True)
class Site:
    """An IGRINS host telescope and the unit conventions its headers use."""

    name: str
    altitude_km: float
    temperature_unit: str  # "C" or "F"
    pressure_unit: str  # "hPa" or "inHg"
    pressure_reference: str  # "station" or "sea_level"
    # Matched case-insensitively as substrings of TELESCOP and then OBSERVAT.
    # Neither card is stable: one McDonald frame says TELESCOP='Harlen J.
    # Smith' and OBSERVAT='McDonald Observatory', an older one says
    # '2.7-m Harlen J. Smith' and plain 'McDonald'.
    aliases: tuple[str, ...] = ()


# Keyed on the TELESCOP card. The conventions are measured, not documented:
# McDonald 2017-08-12 reports 78.0/23.6, which is Fahrenheit and inches of
# mercury (23.6 inHg = 799 hPa, right for Mt Locke); DCT reports 1025 hPa,
# which is impossible at 2360 m and is therefore reduced to sea level; Gemini
# South reports 730 hPa, which is the true station pressure at Cerro Pachon.
SITES: Mapping[str, Site] = {
    "McDonald": Site("McDonald Observatory", 2.077, "F", "inHg", "station",
                     ("mcdonald", "harlen j. smith", "otto struve")),
    "DCT": Site("Lowell Discovery Telescope", 2.360, "C", "hPa", "sea_level",
                ("discovery channel", "lowell", "dct")),
    # Deliberately not aliased to a bare "gemini observatory": that card cannot
    # tell South from North, and IGRINS-2 is at Gemini North on a different
    # mountain. An unrecognised Gemini frame should raise, not be placed on
    # Cerro Pachon by default.
    "Gemini South": Site("Gemini South", 2.722, "C", "hPa", "station",
                         ("gemini south", "cerro pachon")),
}


def hydrogen_series_um(
    lower_level: int,
    wavelength_range_um: tuple[float, float] = (1.40, 2.55),
    max_upper_level: int = 25,
) -> np.ndarray:
    """Vacuum wavelengths of one hydrogen series inside a range, in microns.

    An A0V has no metal lines worth the name across H and K; the hydrogen
    recombination series is the whole of its spectrum there, which is why these
    are computed rather than tabulated.

    ``max_upper_level`` bounds the sum because the series does not end -- it
    converges. Near the limit the lines are separated by far less than their
    own width and merge into one depression, so enumerating them past about 25
    adds terms that are individually negligible and collectively already
    covered by their neighbours' masking windows.
    """

    if lower_level < 1:
        raise ValueError("the lower level must be a positive principal quantum number")
    if max_upper_level <= lower_level:
        raise ValueError("max_upper_level must exceed the lower level")
    low, high = wavelength_range_um
    if not 0.0 < low < high:
        raise ValueError("the wavelength range must be positive and increasing")
    upper = np.arange(lower_level + 1, max_upper_level + 1)
    inverse = _RYDBERG_HYDROGEN_PER_M * (1.0 / lower_level**2 - 1.0 / upper**2)
    wavelength = 1.0e6 / inverse
    return np.sort(wavelength[(wavelength >= low) & (wavelength <= high)])


def stellar_line_mask(
    wavelength_vacuum_nm: np.ndarray,
    half_width_kms: float = 600.0,
    series: tuple[int, ...] = (4, 5),
) -> np.ndarray:
    """``True`` where a pixel is clear of the hydrogen series.

    The default half width is wide because these lines are Stark broadened to
    several hundred km/s in an A0V, far wider than any rotation the star adds.
    Masking them lets a telluric fit use a flat source and carry no stellar
    model error at all, which is the cleanest measurement of the atmosphere
    this package can make.
    """

    if half_width_kms <= 0.0:
        raise ValueError("the mask half width must be positive")
    wavelength_um = np.asarray(wavelength_vacuum_nm, dtype=float) / 1000.0
    # Widen by well over the mask width so a line just outside the order still
    # masks the pixels its wing reaches.
    span = (wavelength_um.min() * 0.99, wavelength_um.max() * 1.01)
    lines = np.concatenate([hydrogen_series_um(level, span) for level in series])
    if lines.size == 0:
        return np.ones(wavelength_um.shape, dtype=bool)
    velocity = 299_792.458 * (wavelength_um[:, None] - lines[None, :]) / lines[None, :]
    return np.all(np.abs(velocity) > half_width_kms, axis=1)


def _number(value, default=np.nan) -> float:
    """A header value as a float, tolerating the blanks the old schema leaves.

    The 2014 McDonald headers carry several numeric cards as empty strings,
    which ``float()`` refuses; there is nothing wrong with the frame.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _card(header: Mapping[str, object], key: str) -> float | None:
    """A numeric header card, or ``None`` when it is absent or blank.

    Recent Gemini frames keep ``HUMIDITY`` but drop ``AIRTEMP``, ``BARPRESS``
    and ``DEWPOINT``, and 2015 McDonald frames carry none of the four, so every
    weather card has to be treated as optional.
    """

    value = header.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def site_for(header: Mapping[str, object]) -> Site:
    """The :class:`Site` a frame was taken at, from ``TELESCOP`` or ``OBSERVAT``.

    Both cards are tried because neither is stable across the archive, and
    neither is the observatory's formal name: McDonald frames identify the
    telescope ('Harlen J. Smith', '2.7-m Harlen J. Smith') while DCT frames
    identify the programme ('Discovery Channel'). Matching is on a per-site
    alias list rather than on a prefix, so a new spelling fails loudly instead
    of being silently attached to whichever site sorts first.
    """

    cards = [str(header.get(key, "")).strip() for key in ("TELESCOP", "OBSERVAT")]
    for card in cards:
        lowered = card.lower()
        if not lowered:
            continue
        for site in SITES.values():
            if any(alias in lowered for alias in site.aliases):
                return site
    named = " / ".join(c for c in cards if c) or "nothing"
    raise ValueError(
        f"unknown IGRINS telescope: the header says {named}. Add an alias to SITES "
        "with the site's altitude and the unit conventions its headers use."
    )


def _station_pressure_ratio(altitude_km: float) -> float:
    """Station over sea-level pressure in the standard atmosphere."""

    lapse = _STANDARD_LAPSE_K_PER_M * altitude_km * 1000.0 / _SEA_LEVEL_TEMPERATURE_K
    return float((1.0 - lapse) ** _BAROMETRIC_EXPONENT)


def surface_conditions(header: Mapping[str, object]) -> dict:
    """Surface temperature, station pressure and humidity in one unit system.

    Returns Kelvin, hectopascals **at the telescope**, and percent. Any of the
    three may be ``None``: the cards are unreliably populated across the
    archive, and a caller that needs a number should fall back to a reanalysis
    rather than to a default that looks like a measurement.

    The pressure is checked against the hydrostatic expectation for the site,
    so a future header that changes convention fails here instead of quietly
    placing the observatory at sea level.
    """

    site = site_for(header)
    temperature = _card(header, "AIRTEMP")
    if temperature is not None:
        if site.temperature_unit == "F":
            temperature = (temperature - 32.0) * 5.0 / 9.0
        temperature += 273.15

    pressure = _card(header, "BARPRESS")
    if pressure is not None:
        if site.pressure_unit == "inHg":
            pressure *= _HECTOPASCAL_PER_INCH_HG
        ratio = _station_pressure_ratio(site.altitude_km)
        if site.pressure_reference == "sea_level":
            pressure *= ratio
        expected = _SEA_LEVEL_PRESSURE_HPA * ratio
        if abs(pressure / expected - 1.0) > _PRESSURE_TOLERANCE:
            raise ValueError(
                f"{site.name} reports BARPRESS={header.get('BARPRESS')!r}, which normalizes "
                f"to {pressure:.1f} hPa against {expected:.1f} hPa expected at "
                f"{site.altitude_km:.3f} km -- the header convention has changed"
            )

    humidity = _card(header, "HUMIDITY")
    return {
        "site": site.name,
        "altitude_km": site.altitude_km,
        "temperature_k": temperature,
        "pressure_hpa": pressure,
        "relative_humidity_percent": humidity,
    }


def zenith_angle_deg(header: Mapping[str, object]) -> float:
    """The mean zenith distance of an exposure, in degrees.

    ``ZDSTART``/``ZDEND`` are the telescope's own measurement and are present
    throughout the archive; the airmass cards are the fallback. Averaging the
    two ends is good to better than a degree for IGRINS exposure times, and the
    residual curvature of sec(z) over one exposure is far below the accuracy of
    the atmosphere profile.
    """

    start, end = _card(header, "ZDSTART"), _card(header, "ZDEND")
    if start is not None and end is not None:
        angle = 0.5 * (start + end)
    else:
        airmass = [_card(header, key) for key in ("AMSTART", "AMEND")]
        usable = [value for value in airmass if value is not None and value >= 1.0]
        if not usable:
            raise ValueError("the header carries neither a usable zenith distance nor an airmass")
        angle = float(np.degrees(np.arccos(1.0 / np.mean(usable))))
    if not 0.0 <= angle < 90.0:
        raise ValueError(f"the zenith distance is outside [0, 90): {angle:.3f} deg")
    return float(angle)


@dataclass(frozen=True)
class IGRINSOrder:
    """One echelle order, ascending in vacuum wavelength."""

    index: int
    band: str
    wavelength_vacuum_nm: np.ndarray
    flux: np.ndarray
    variance: np.ndarray
    zenith_angle_deg: float
    telluric_model: np.ndarray | None = None
    plp_continuum: np.ndarray | None = None
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength_vacuum_nm, dtype=float)
        if wavelength.ndim != 1 or wavelength.size < 8:
            raise ValueError("an order needs at least eight pixels")
        if np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0.0):
            raise ValueError("order wavelengths must be finite and strictly increasing")
        for name in ("flux", "variance", "telluric_model", "plp_continuum"):
            values = getattr(self, name)
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            if values.shape != wavelength.shape:
                raise ValueError(f"{name} must match the order's wavelength grid")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "wavelength_vacuum_nm", wavelength)

    @property
    def wavenumber_range_cm1(self) -> tuple[float, float]:
        return (
            float(1.0e7 / self.wavelength_vacuum_nm[-1]),
            float(1.0e7 / self.wavelength_vacuum_nm[0]),
        )

    @property
    def name(self) -> str:
        return f"{self.band}{self.index:02d}"


@dataclass(frozen=True)
class IGRINSObservation:
    """One band of one IGRINS exposure, as the PLP extracted it."""

    path: Path
    band: str
    object_name: str
    object_type: str
    telescope: str
    date_obs: str
    mjd: float
    exposure_time_s: float
    zenith_angle_deg: float
    wavelength_vacuum_nm: np.ndarray
    flux: np.ndarray
    variance: np.ndarray
    sha256: Mapping[str, str]
    surface: Mapping[str, object]
    telluric_model: np.ndarray | None = None
    plp_continuum: np.ndarray | None = None

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength_vacuum_nm, dtype=float)
        if wavelength.ndim != 2:
            raise ValueError("an IGRINS band is an (order, pixel) array")
        for name in ("flux", "variance", "telluric_model", "plp_continuum"):
            values = getattr(self, name)
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            if values.shape != wavelength.shape:
                raise ValueError(f"{name} must match the wavelength array")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "wavelength_vacuum_nm", wavelength)
        object.__setattr__(self, "path", Path(self.path))

    @property
    def orders(self) -> int:
        return int(self.wavelength_vacuum_nm.shape[0])

    def order(self, index: int) -> IGRINSOrder:
        """One order, with its non-finite edges trimmed away.

        The PLP leaves the first and last pixels of an order undefined where
        the extraction aperture runs off the detector. Those cannot be masked
        and kept, because :class:`SpectralOrder` requires a strictly increasing
        wavelength and a positive uncertainty everywhere; interior gaps are
        masked instead.
        """

        if not 0 <= index < self.orders:
            raise ValueError(f"order {index} is outside 0-{self.orders - 1}")
        wavelength = self.wavelength_vacuum_nm[index]
        flux = self.flux[index]
        variance = self.variance[index]
        usable = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(variance)
        if np.count_nonzero(usable) < 8:
            raise ValueError(f"order {index} has too few usable pixels")
        first, last = int(np.argmax(usable)), int(usable.size - np.argmax(usable[::-1]))
        cut = slice(first, last)
        # The wavelength solution descends with order index in half the bands;
        # SpectralOrder wants ascending wavelength either way.
        rising = wavelength[cut][-1] > wavelength[cut][0]
        step = slice(None) if rising else slice(None, None, -1)

        def take(values):
            return None if values is None else np.asarray(values[index][cut][step], dtype=float)

        return IGRINSOrder(
            index=index,
            band=self.band,
            wavelength_vacuum_nm=wavelength[cut][step],
            flux=flux[cut][step],
            variance=variance[cut][step],
            zenith_angle_deg=self.zenith_angle_deg,
            telluric_model=take(self.telluric_model),
            plp_continuum=take(self.plp_continuum),
            meta={
                "object": self.object_name,
                "telescope": self.telescope,
                "date_obs": self.date_obs,
                "mjd": self.mjd,
                "band": self.band,
                "surface": dict(self.surface),
            },
        )


def _companion(spec_path: Path, suffix: str) -> Path:
    """The sibling PLP product of one exposure, e.g. ``.variance.fits``."""

    return spec_path.with_name(f"{spec_path.name.split('.')[0]}.{suffix}")


def read_igrins_observation(
    spec_path: str | Path,
    *,
    variance_path: str | Path | None = None,
    flattened_path: str | Path | None = None,
) -> IGRINSObservation:
    """Read one band of one exposure from the PLP products RRISA ships.

    ``spec.fits`` carries the extracted counts in its primary HDU and the
    wavelength solution, in microns, in the first extension. The variance and
    the ``spec_flattened`` products are found beside it unless given.

    The flattened product is read for its ``MODEL_TELTRANS`` and
    ``FITTED_CONTINUUM`` only. Neither enters the fit: they are the PLP's own
    telluric model and its own continuum, and the point of fitting the raw
    counts is that both stay outside the forward model and available as an
    independent comparison.
    """

    from astropy.io import fits

    spec_path = Path(spec_path)
    variance_path = Path(variance_path) if variance_path else _companion(spec_path, "variance.fits")
    flattened_path = (
        Path(flattened_path) if flattened_path else _companion(spec_path, "spec_flattened.fits")
    )

    with fits.open(spec_path) as handle:
        header = dict(handle[0].header)
        flux = np.asarray(handle[0].data, dtype=float)
        wavelength_um = np.asarray(handle[1].data, dtype=float)
    if flux.ndim != 2 or wavelength_um.shape != flux.shape:
        raise ValueError(f"{spec_path} is not a PLP (order, pixel) spectrum with a wavelength HDU")

    with fits.open(variance_path) as handle:
        variance = np.asarray(handle[0].data, dtype=float)
    if variance.shape != flux.shape:
        raise ValueError(f"{variance_path} does not match the shape of {spec_path}")

    telluric_model = plp_continuum = None
    digests = {"spec": file_sha256(spec_path), "variance": file_sha256(variance_path)}
    if flattened_path.exists():
        with fits.open(flattened_path) as handle:
            if "MODEL_TELTRANS" in handle:
                telluric_model = np.asarray(handle["MODEL_TELTRANS"].data, dtype=float)
            if "FITTED_CONTINUUM" in handle:
                plp_continuum = np.asarray(handle["FITTED_CONTINUUM"].data, dtype=float)
        digests["flattened"] = file_sha256(flattened_path)

    return IGRINSObservation(
        path=spec_path,
        band=str(header.get("BAND", spec_path.name[3:4])).strip(),
        object_name=str(header.get("OBJECT", "")).strip(),
        object_type=str(header.get("OBJTYPE", "")).strip(),
        telescope=str(header.get("TELESCOP", "")).strip(),
        date_obs=str(header.get("DATE-OBS", "")).strip(),
        mjd=_number(header.get("MJD-OBS")),
        exposure_time_s=_number(header.get("EXPTIME")),
        zenith_angle_deg=zenith_angle_deg(header),
        wavelength_vacuum_nm=wavelength_um * 1000.0,
        flux=flux,
        variance=variance,
        sha256=digests,
        surface=surface_conditions(header),
        telluric_model=telluric_model,
        plp_continuum=plp_continuum,
    )


def smoothed_blaze(flux: np.ndarray, window: int = 51) -> np.ndarray:
    """The order's throughput profile, with the telluric lines averaged out.

    A boxcar this wide spans several resolution elements, so individual
    absorption lines do not survive it while the blaze, which varies over the
    whole order, does. Non-finite pixels are skipped rather than treated as
    zero, which would dig a hole around every bad pixel.
    """

    if window < 3 or window % 2 == 0:
        raise ValueError("the smoothing window must be odd and at least three pixels")
    values = np.asarray(flux, dtype=float)
    finite = np.isfinite(values)
    kernel = np.ones(window) / window
    total = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    weight = np.convolve(finite.astype(float), kernel, mode="same")
    return total / np.maximum(weight, 1.0e-9)


def leave_one_out_patterns(fractional, minimum_frames=3, smooth_pixels=51):
    """One instrument-response pattern per frame, from every *other* frame.

    The red edge of an IGRINS order carries a residual that repeats frame to
    frame: measured over ten standards of five different stars spanning airmass
    1.07 to 2.50, 56-76% of each frame's residual variance is common, it is
    strongest in orders with no telluric absorption at all, and its amplitude
    does not grow with airmass. That is the instrument, not the atmosphere.

    Deriving a frame's correction from the other frames only is what keeps this
    a calibration rather than a way of fitting the noise: a pattern taken from
    the frame it corrects would absorb genuine residual and flatter the telluric
    model. Every number this pipeline reports is therefore leave-one-out.

    ``smooth_pixels`` is the other half of that guard, and it is not optional.
    Leave-one-out stops the pattern absorbing one frame's *noise*; it does
    nothing about a systematic every frame shares -- and our own telluric model
    error is exactly that, since every frame looks through the same sky with the
    same line list. Measured on this night, the unsmoothed pattern's
    high-frequency component correlates with the absorption depth at r = +0.44
    and with the transmission gradient at +0.39, and its amplitude scales with
    how much absorption an order has: 0.013 of the continuum where the median
    transmission is 0.67, 0.003 where it is 0.999, which is the noise floor.
    That component is the line list, not the instrument, and removing it would
    make the telluric model look better than it is.

    A boxcar of 51 pixels -- about fifteen resolution elements -- keeps the
    broad response error, which runs over hundreds of pixels at the red edge of
    an order, and leaves telluric-scale structure in the residual where it
    belongs. It drops the pattern's correlation with absorption depth from
    +0.44 to +0.15. ``smooth_pixels=0`` disables the guard and must not be used
    when the residual is being quoted as a test of the telluric model.
    """

    stack = np.asarray(fractional, dtype=float)
    patterns = []
    for index in range(stack.shape[0]):
        others = np.delete(stack, index, axis=0)
        with warnings.catch_warnings():
            # A pixel no frame measured is an all-NaN slice; that is the
            # ordinary case at an order edge, and it is handled below.
            warnings.simplefilter("ignore", RuntimeWarning)
            pattern = np.nanmedian(others, axis=0)
        enough = np.sum(np.isfinite(others), axis=0) >= minimum_frames
        pattern = np.where(enough & np.isfinite(pattern), np.nan_to_num(pattern), 0.0)
        if smooth_pixels and smooth_pixels > 1:
            if smooth_pixels % 2 == 0:
                raise ValueError("smooth_pixels must be odd so the boxcar is centred")
            kernel = np.ones(smooth_pixels) / smooth_pixels
            weight = np.convolve(enough.astype(float), kernel, mode="same")
            total = np.convolve(np.where(enough, pattern, 0.0), kernel, mode="same")
            # Where nothing was measured the correction stays exactly zero
            # rather than bleeding in from a neighbour.
            pattern = np.where(enough, total / np.maximum(weight, 1e-9), 0.0)
        patterns.append(pattern)
    return patterns


def run_provenance(root: Path, args, observations, profile_path: Path) -> tuple[dict, dict]:
    """What produced this run, and the identity of everything that went into it.

    Paths alone are not provenance: a line list can be replaced under the same
    name. The hashes are what let a rebuild say whether it is looking at the
    same inputs.
    """

    import jax
    import jax_telluric
    from jax_telluric import file_sha256

    driver = Path(__file__).resolve()
    inputs = {
        "profile": str(profile_path), "profile_sha256": file_sha256(profile_path),
        "stellar": args.stellar,
        "frames": [str(o.path) for o in observations],
        "frame_sha256": [o.sha256["spec"] for o in observations],
    }
    if args.stellar != "flat" and Path(args.stellar).exists():
        inputs["stellar_sha256"] = file_sha256(args.stellar)
    mt_ckd = root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc"
    if mt_ckd.exists():
        inputs["mt_ckd"] = str(mt_ckd)
        inputs["mt_ckd_sha256"] = file_sha256(mt_ckd)
    line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    inputs["aer_line_root"] = str(line_root)
    for species, molecule_id in sorted(MOLECULE_IDS.items()):
        stem = f"{molecule_id:02d}_{species}"
        path = line_root / stem / stem
        if path.exists():
            inputs[f"aer_{species}_sha256"] = file_sha256(path)
    run = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "driver": driver.name, "driver_sha256": file_sha256(driver),
        "jax_telluric": getattr(jax_telluric, "__version__", ""),
        "jax": jax.__version__,
    }
    return run, inputs


def continuum_level(order: IGRINSOrder, percentile: float = 95.0) -> float:
    """A robust flux scale for one order, in the extraction's own counts.

    The PLP's counts run to tens of thousands and vary by an order of magnitude
    between the centre of a band and its edge. A high percentile of the finite
    flux sits on the blaze peak, above the telluric absorption that fills the
    lower part of the distribution, and is stable against the few bad pixels a
    maximum would catch.
    """

    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    flux = np.asarray(order.flux, dtype=float)
    finite = np.isfinite(flux)
    if np.count_nonzero(finite) < 8:
        raise ValueError(f"order {order.name} has too few finite pixels")
    level = float(np.percentile(flux[finite], percentile))
    if not np.isfinite(level) or level <= 0.0:
        raise ValueError(f"order {order.name} has no positive continuum level")
    return level


def igrins_spectral_order(
    order: IGRINSOrder,
    *,
    source_flux_model_grid: np.ndarray | None = None,
    saturation_floor: float = 0.02,
    throughput_floor: float = 0.25,
    mask_hydrogen_kms: float | None = 600.0,
    continuum_percentile: float = 95.0,
    normalize: bool = True,
) -> SpectralOrder:
    """Turn one extracted order into a fittable :class:`SpectralOrder`.

    ``mask`` is ``True`` where a pixel is usable. Pixels are dropped where the
    flux or variance is not finite, where the variance is not positive, where
    the flux falls below ``saturation_floor`` of the order's own continuum
    level -- a line core carrying no information, and the place sky-subtraction
    residuals show up -- and, when ``mask_hydrogen_kms`` is set, inside the
    hydrogen series.

    ``throughput_floor`` is a separate cut, on the *smoothed* flux, and it is
    the one that matters most. An IGRINS order's blaze does not roll off
    symmetrically: measured on this frame it falls below a quarter of its peak
    across the first 350-700 pixels while reaching almost to the last, so a
    fixed edge trim is the wrong shape. That roll-off region is not merely
    noisy -- with the PLP's own telluric model divided out, including it makes
    a weighted Chebyshev fit of the blaze worse at every degree, which says the
    extraction itself misbehaves there. Cutting at a quarter of peak costs
    about a quarter of the order and is worth it.

    That last cut is what lets a first pass run against a flat source: an A0V
    is featureless once the Brackett and Pfund lines are gone, so the fit
    carries no stellar model error at all. Pass ``None`` once a real A0V model
    is supplied through ``source_flux_model_grid``.

    Unlike the Arcturus atlas, the uncertainty here is measured. Masked pixels
    still need a finite positive value because :class:`SpectralOrder` requires
    one everywhere, so they are given a large one rather than a plausible one.

    ``normalize`` divides the flux and its uncertainty by
    :func:`continuum_level`, and matters more than it looks. The fitted
    continuum's constant term is a *log* flux, so raw PLP counts put it near 10
    and its bound has to span the counts scale; the optimizer rescales its
    variables onto that bound, and a first step of a fraction of a 60-unit
    range drives the continuum to zero. Normalizing keeps that coefficient near
    zero, where a bound of a couple of units is enough. Recover the counts with
    :func:`continuum_level` -- the fit itself is scale free.
    """

    if not 0.0 <= saturation_floor < 1.0:
        raise ValueError("saturation_floor must be in [0, 1)")
    if not 0.0 <= throughput_floor < 1.0:
        raise ValueError("throughput_floor must be in [0, 1)")

    flux = np.asarray(order.flux, dtype=float)
    variance = np.asarray(order.variance, dtype=float)
    finite = np.isfinite(flux) & np.isfinite(variance) & (variance > 0.0)
    if np.count_nonzero(finite) < 8:
        raise ValueError(f"order {order.name} has too few finite pixels")

    level = continuum_level(order, continuum_percentile)
    scale = level if normalize else 1.0
    flux = flux / scale
    variance = variance / scale**2

    mask = finite & (flux > saturation_floor * level / scale)
    if throughput_floor > 0.0:
        mask &= smoothed_blaze(np.where(finite, flux, np.nan)) > throughput_floor * level / scale
    if mask_hydrogen_kms is not None:
        mask &= stellar_line_mask(order.wavelength_vacuum_nm, mask_hydrogen_kms)
    if np.count_nonzero(mask) < 8:
        raise ValueError(f"order {order.name} keeps too few pixels after masking")

    sigma = np.sqrt(np.where(finite, variance, 1.0))
    # Large, not plausible: if anything ever ignores the mask, the pixel should
    # carry no weight rather than a weight that looks reasonable.
    sigma = np.where(mask, sigma, 1.0e6 * float(np.max(sigma[mask])))

    return SpectralOrder(
        wavelength_vacuum_nm=order.wavelength_vacuum_nm,
        flux=np.where(finite, flux, 0.0),
        uncertainty=sigma,
        mask=mask,
        zenith_angle_deg=order.zenith_angle_deg,
        source_flux_model_grid=source_flux_model_grid,
    )
