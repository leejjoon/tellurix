"""The telluric spectra of Baker, Blake & Reiners 2020 (ApJS 247, 24).

Their telluric-corrected IAG solar flux atlas came from fitting 141 Goettingen
spectra jointly in 11 groups (Zenodo 10.5281/zenodo.3598136), and they published
what each fit left once the Sun and the iodine cell were divided out:
``telluric_spectra_{0..10}.fits``. Each is one group: the primary header carries,
per spectrum ``NNNN``, its air mass ``AMS_NNNN``, Julian date ``JD_NNNN`` and H2O
optical depth ``TAU_NNNN``; the table carries the wavenumber ``v``, one
``telluric_NNNN`` per spectrum, the group's solar model ``stel_mod``, the median
residual ``res_med`` and ``iodine``.

``telluric_NNNN`` is data, not their telluric model: the normalised observation
over the iodine and the group's solar model, in the telluric frame. So it is the
atmosphere alone at R ~ 1e6, with the observation's noise (0.2-0.3% per pixel)
and whatever their solar model left under solar lines -- a test of the line list
that involves no solar model of ours.

106 distinct spectra appear, 35 of them in more than one group, divided there by
another group's solar model.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .flux_atlas import IAG_FWHM_CM1
from .nso import FTSSpectrum

# The Institut fuer Astrophysik, Goettingen Nordcampus, where the Vacuum Vertical
# Telescope feeds the FTS. A 0.1 km error in the altitude is 1% of the pressure,
# which the fitted columns take up.
IAG_LATITUDE_DEG = 51.5596
IAG_LONGITUDE_DEG = 9.9438
IAG_ALTITUDE_KM = 0.2
GROUPS = range(11)
_UNIX_EPOCH_JD = 2440587.5


@dataclass(frozen=True)
class BakerObservation:
    """One spectrum's header entries in one group's file."""

    group: int
    number: str
    airmass: float
    julian_day: float
    h2o_optical_depth: float

    @property
    def utc(self) -> _dt.datetime:
        return (_dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
                + _dt.timedelta(days=self.julian_day - _UNIX_EPOCH_JD))

    @property
    def era5_hour(self) -> _dt.datetime:
        """The ERA5 analysis nearest the spectrum, as era5_site_profile.py picks it."""
        return (self.utc + _dt.timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)


def group_file(directory: str | Path, group: int) -> Path:
    return Path(directory) / f"telluric_spectra_{group}.fits"


def baker_observations(directory: str | Path, groups=GROUPS) -> list[BakerObservation]:
    """Every spectrum of the given groups, from the headers alone."""

    from astropy.io import fits

    observations = []
    for group in groups:
        header = fits.getheader(group_file(directory, group), 0)
        for key in header:
            match = re.fullmatch(r"AMS_(\d+)", key)
            if match:
                number = match.group(1)
                observations.append(BakerObservation(
                    group, number, float(header[key]), float(header[f"JD_{number}"]),
                    float(header[f"TAU_{number}"])))
    return observations


def baker_wavenumber_grid(stored: np.ndarray) -> np.ndarray:
    """The exact grid under the file's float32 ``v`` column.

    float32 keeps 7 digits, so at 19990 cm-1 the column is rounded by up to
    1e-3 cm-1 -- 15 m/s, as large as the per-line position offsets the spectra
    are wanted for. The grid is linear in wavenumber; refitting the line
    recovers it, and the residual is checked to be the rounding and nothing else.
    """

    stored = np.asarray(stored, dtype=float)
    index = np.arange(stored.size)
    step, start = np.polyfit(index, stored, 1)
    grid = start + step * index
    worst = float(np.max(np.abs(grid - stored)))
    if worst > 1.0e-3:
        raise ValueError(f"v is not a linear grid: off by {worst:.2e} cm-1")
    return grid


def read_baker_table(directory: str | Path, group: int, columns) -> tuple[np.ndarray, dict]:
    """The exact grid and the named columns of one group's table, as float64."""

    from astropy.io import fits

    with fits.open(group_file(directory, group), memmap=True) as handle:
        table = handle[1].data
        grid = baker_wavenumber_grid(table["v"])
        return grid, {name: np.asarray(table[name], dtype=float) for name in columns}


def read_baker_telluric(directory: str | Path, observation: BakerObservation) -> FTSSpectrum:
    """One ``telluric_NNNN`` as an :class:`FTSSpectrum` at its air mass.

    The FTS's optics, not its scan length, set the resolution (Reiners et al.
    2016), so the width is the IAG atlas's.
    """

    path = group_file(directory, observation.group)
    nu, columns = read_baker_table(directory, observation.group, [f"telluric_{observation.number}"])
    utc = observation.utc
    return FTSSpectrum(
        path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_name=f"Baker 2020 group {observation.group} telluric_{observation.number}",
        comment="", date_mst="", julian_day=int(observation.julian_day),
        universal_time_start="", universal_time_stop="",
        airmass_start=observation.airmass, airmass_stop=observation.airmass,
        stated_resolution_cm1=IAG_FWHM_CM1, transform_samples=0, point_of_center=0,
        wavenumber_vacuum_cm1=nu, flux=columns[f"telluric_{observation.number}"],
        grid_residual_cm1=0.0, observed_utc_start=utc, observed_utc_stop=utc)
