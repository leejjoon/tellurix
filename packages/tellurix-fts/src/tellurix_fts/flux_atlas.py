"""Readers for the disc-integrated (flux) solar atlases.

The NSO raw spectra and the photatl/niratl pages are disc-centre intensity;
these two are the Sun as a star, which is the quantity Payne Zero synthesizes
(docs/solar_fit_plan.md, Phase 1). Both are plain text and record no header, so
what a fit needs from the header -- air mass, resolution -- is carried here.

- ``iag``: the IAG solar flux atlas (Reiners et al. 2016, A&A 587, A65; CDS
  J/A+A/587/A65), Göttingen 2014, VIS (``spvis``, 405-1065 nm) and NIR
  (``spnir``, 1000-2300 nm) settings. Columns: vacuum wavenumber, normalized
  flux, raw flux (the CDS ReadMe; there is no uncertainty). Each co-adds over
  1100 scans from six or nine days, so neither has one air mass.
- ``wallace2011``: Kitt Peak flux atlas #2 (Wallace, Hinkle, Livingston & Davis
  2011, ApJS 195, 6), ``sptr.regN``. Regions 1-3 carry vacuum wavenumber, the
  telluric-corrected flux, the transmission used for it, and the observed flux.
  The corrected column borrowed its transmission from the 1983-06-26 disc-centre
  pair, so a telluric fit reads the observed one. Regions 1-3 are each one
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
# 676,000 and 698,000, which neither interferogram supports. Region 3 is not in the
# paper's text: its air mass and time are the archive header's (891001R0.004,
# FTS_cdrom/FTS23: 11:41 MST, 37 min, air mass 1.24 -> 1.22), and its MOPD is
# 10.3-10.4 cm everywhere over 17400-22500, 0.0578 cm-1. The header's RESOLUTN
# times the sinc's 1.207 gives all three measured widths (0.0229, 0.0330, 0.0578),
# so the archive, not Table 1, is the one to trust. Region 3's authors' correction
# stops at 20040 cm-1: above it the corrected and transmission columns are zero.
WALLACE2011_REGIONS = {
    1: {"file": "sptr.reg1", "spectrum": "1989/10/13 #8", "airmass": 1.5, "fwhm_cm1": 0.02266,
        "atlas_range_cm1": (10800.0, 13500.0)},
    2: {"file": "sptr.reg2", "spectrum": "1989/10/13 #7", "airmass": 1.4, "fwhm_cm1": 0.03282,
        "atlas_range_cm1": (13500.0, 17500.0)},
    3: {"file": "sptr.reg3", "spectrum": "1989/10/01 #4", "airmass": 1.23, "fwhm_cm1": 0.0578,
        "atlas_range_cm1": (17500.0, 22500.0)},
}
# Reiners et al. (Table 1) put the optics limit at R ~ 1e6, 0.0145 cm-1
# at the B-band; the FTS was scanned finer than its optics resolve.
IAG_FWHM_CM1 = 0.0145
# The co-add has no air mass, and only its product with the column is measured.
# This one makes the A-band's O2 scale ~1 on the 2014-04-17 Göttingen profile
# (exp(0.55) = 1.73 at zenith 0, `fit_atlas_o2.py iag`), so the species cut,
# which is made along the path, sees about the optical depth the atlas holds.
IAG_EFFECTIVE_AIRMASS = 1.7
IAG_VIS = {"file": "spvis.dat.gz", "spectrum": "VIS setting, 2014, 1190 scans on nine days",
           "airmass": IAG_EFFECTIVE_AIRMASS, "fwhm_cm1": IAG_FWHM_CM1}
# The NIR setting was scanned at Bruker's 0.0044 cm-1 (Table 1), which is 0.9/MOPD:
# 204.5 cm. Measured from spnir itself with `describe_truncation` in 100 cm-1 pieces,
# the cut is at 198-210 cm wherever the envelope step is clean (4400-9800 cm-1; the
# rest find a band's own structure or the 266 cm sampling limit). Its sinc is
# 1.2067/(2 MOPD). The R ~ 1e6 optics limit is broader above ~3000 cm-1, and it
# grows with wavenumber, as the fitted Gaussian lsf_sigma_kms (constant in velocity)
# does. The air mass is only the species cut's, as for VIS: the weighted mean of
# data/profiles/iag2014/coadd_nir.json's nodes is 1.625 (VIS's is 1.545).
IAG_NIR_FWHM_CM1 = 0.00295
IAG_NIR = {"file": "spnir.dat.gz", "spectrum": "NIR setting, 2014, 1130 scans on six days",
           "airmass": 1.625, "fwhm_cm1": IAG_NIR_FWHM_CM1}


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
             # The whole name: sptr.reg1 and sptr.reg2 differ only after the dot.
             Path(cache_directory) / f"{path.name.removesuffix('.gz').removesuffix('.dat').replace('.', '_')}"
                                     f"_{keep[0]:.0f}_{keep[1]:.0f}.npz")
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


def read_wallace2011_correction(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The authors' telluric correction in a Wallace 2011 region: wavenumber, corrected, transmission.

    Their corrected column is the observed flux over the transmission where that
    transmission is above about 0.8, **zero** below it and ``NaN`` across the bands
    they did not correct (0.94 um, the A-band); both come back as ``NaN``, so only
    pixels they corrected compare. One field in region 1 overflowed its format
    (``*******``) and is read as ``NaN`` too.
    """

    with open(path) as handle:
        rows = [[float(field) if "*" not in field else np.nan for field in line.split()[:3]]
                for line in handle]
    nu, corrected, transmission = np.array(rows).T
    order = np.argsort(nu, kind="stable")
    nu, corrected, transmission = nu[order], corrected[order], transmission[order]
    distinct = np.concatenate([[True], np.diff(nu) > 0])
    corrected = np.where(corrected > 0, corrected, np.nan)
    return nu[distinct], corrected[distinct], transmission[distinct]
