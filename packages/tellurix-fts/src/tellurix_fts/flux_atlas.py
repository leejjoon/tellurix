"""Readers for the disc-integrated (flux) solar atlases.

The NSO raw spectra and the photatl/niratl pages are disc-centre intensity;
these two are the Sun as a star, which is the quantity Payne Zero synthesizes
(docs/solar_fit_plan.md, Phase 1). Both are plain text and record no header, so
what a fit needs from the header -- air mass, resolution -- is carried here.

- ``iag``: the IAG solar flux atlas, VIS setting (Reiners et al. 2016, A&A 587,
  A65; CDS J/A+A/587/A65), Göttingen 2014. Columns: vacuum wavenumber,
  normalized flux, uncertainty. It co-adds 1190 scans over nine days, so it has
  no one air mass.
- ``wallace2011``: Kitt Peak flux atlas #2 (Wallace, Hinkle, Livingston & Davis
  2011, ApJS 195, 6), ``sptr.regN``. Regions 1-3 carry vacuum wavenumber, the
  telluric-corrected flux, the transmission used for it, and the observed flux.
  The corrected column borrowed its transmission from the 1983-06-26 disc-centre
  pair, so a telluric fit reads the observed one. Regions 1 and 2 are each one
  McMath integrated-sun spectrum.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np

from .nso import FTSSpectrum

# Air mass is the paper's (Table 1 and text), to one decimal. The FWHM is the
# sinc's, measured from each region's own interferogram with
# `describe_truncation` in 100 cm-1 pieces: region 1 MOPD 26.1-26.8 cm over
# 10000-13400 cm-1, 0.0227 cm-1 (R 577,000 at 13050); region 2 18.3-18.5 cm over
# 13600-17400, 0.0328 cm-1 (R 441,000 at 14455). The paper's Table 1 gives
# 676,000 and 698,000, which neither interferogram supports.
WALLACE2011_REGIONS = {
    1: {"file": "sptr.reg1", "spectrum": "1989/10/13 #8", "airmass": 1.5, "fwhm_cm1": 0.02266,
        "atlas_range_cm1": (10800.0, 13500.0)},
    2: {"file": "sptr.reg2", "spectrum": "1989/10/13 #7", "airmass": 1.4, "fwhm_cm1": 0.03282,
        "atlas_range_cm1": (13500.0, 17500.0)},
}
# Reiners et al. (Table 1) put the optics limit at R ~ 1e6, 0.0145 cm-1
# at the B-band; the FTS was scanned finer than its optics resolve.
IAG_FWHM_CM1 = 0.0145


def _read_columns(atlas: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    if atlas == "iag":
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            data = np.loadtxt(handle, usecols=(0, 1))
    elif atlas == "wallace2011":
        data = np.loadtxt(path, usecols=(0, 3))
    else:
        raise ValueError(f"unknown flux atlas {atlas!r}")
    return data[:, 0], data[:, 1]


def read_flux_atlas(atlas: str, path: str | Path, keep: tuple[float, float], *,
                    cache_directory: str | Path | None = None, airmass: float | None = None,
                    fwhm_cm1: float = 0.0, source_name: str = "") -> FTSSpectrum:
    """One flux atlas file, cut to ``keep``, as an :class:`FTSSpectrum`.

    ``cache_directory`` keeps the cut as an npz beside nothing else, since the
    text files take seconds to parse and a run reads them once per window.
    """

    path = Path(path)
    cache = (None if cache_directory is None else
             Path(cache_directory) / f"{path.name.split('.')[0]}_{keep[0]:.0f}_{keep[1]:.0f}.npz")
    if cache is not None and cache.exists():
        stored = np.load(cache)
        nu, flux = stored["nu"], stored["flux"]
    else:
        nu, flux = _read_columns(atlas, path)
        inside = (nu > keep[0]) & (nu < keep[1])
        nu, flux = nu[inside], flux[inside]
        # Wallace's wavenumbers are written to 1e-4 cm-1, so a few neighbours tie.
        distinct = np.concatenate([[True], np.diff(nu) > 0])
        nu, flux = nu[distinct], flux[distinct]
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, nu=nu, flux=flux)
    return FTSSpectrum(
        path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_name=source_name,
        comment="", date_mst="", julian_day=0, universal_time_start="", universal_time_stop="",
        airmass_start=airmass, airmass_stop=airmass, stated_resolution_cm1=fwhm_cm1,
        transform_samples=0, point_of_center=0, wavenumber_vacuum_cm1=nu, flux=flux,
        grid_residual_cm1=0.0)
