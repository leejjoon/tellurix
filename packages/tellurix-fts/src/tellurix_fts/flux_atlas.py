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
- ``fluxatl``: the Kitt Peak solar flux atlas (Kurucz, Furenlid, Brault & Testerman
  1984, NSO Atlas No. 1), a directory of ``lmXXXX`` pages, 4 nm each plus 0.1 nm
  of overlap: air wavelength (nm), pseudo-residual flux, irradiance. It splices
  eight McMath FTS scans of 1980-81, each of one day and about two hours, and its
  telluric lines are left in. Kurucz's 2005 re-reduction of the same scans ships
  their Table 1 (``kurucz2005/fluxscans.dat``), which is where FLUXATL_SCANS comes
  from. Its per-scan files there (``flux1*.2005``, ``fluxspliced.2005``) still hold
  the tellurics; its corrected spectrum is ``irradrelwl.dat``
  (`read_kurucz_irradiance`), from Kurucz's ``sun/irradiance2005``.
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


# Table 1 of the atlas, as Kurucz's 2005 re-reduction reprints it, for the scans with
# telluric lines worth fitting (109 has few; 101-107 sit under ozone's Huggins and
# Chappuis bands, which AER does not carry). Limits are the splice points in vacuum
# nm. Times are local (MST) seconds: scan 111's start, 10:38, is 1.8 h before transit,
# as its hour angle (-6513 s) says, so the UTC is that plus 7 h. The resolving power
# is the sinc's; the FWHM a run uses is the scan's centre over it, since an FTS holds
# its path difference, not R, fixed across a scan.
FLUXATL_SCANS = {
    109: {"vacuum_nm": (472.82, 577.2), "date": "1981-03-24", "mst_seconds": (45035.48, 52805.08),
          "airmass": (1.15, 1.35), "resolving_power": 521360, "dispersion_cm1": 0.02381424},
    111: {"vacuum_nm": (577.2, 751.0), "date": "1981-03-25", "mst_seconds": (38297.73, 45284.97),
          "airmass": (1.30, 1.15), "resolving_power": 522900, "dispersion_cm1": 0.01815228},
    113: {"vacuum_nm": (751.0, 1000.0), "date": "1981-03-25", "mst_seconds": (47186.79, 54248.59),
          "airmass": (1.16, 1.45), "resolving_power": 522990, "dispersion_cm1": 0.01421564},
    115: {"vacuum_nm": (1000.0, 1300.0), "date": "1981-05-11", "mst_seconds": (37338.11, 45506.13),
          "airmass": (1.16, 1.03), "resolving_power": 522500, "dispersion_cm1": 0.01154261},
}
for _scan in FLUXATL_SCANS.values():
    _low_cm1, _high_cm1 = 1.0e7 / _scan["vacuum_nm"][1], 1.0e7 / _scan["vacuum_nm"][0]
    _scan["range_cm1"] = (_low_cm1, _high_cm1)
    _scan["fwhm_cm1"] = 0.5 * (_low_cm1 + _high_cm1) / _scan["resolving_power"]
    _scan["mean_airmass"] = 0.5 * sum(_scan["airmass"])


def edlen_refractive_index(wavenumber_vacuum_cm1):
    """Standard air's refractive index, Edlén 1966, at a vacuum wavenumber.

    The fluxatl pages were written in air with it (Kurucz et al. 1984); their
    steps come back to each scan's even wavenumber spacing in Table 1 through it.
    """

    sigma2 = (np.asarray(wavenumber_vacuum_cm1, dtype=float) * 1.0e-4) ** 2
    return 1.0 + 1.0e-8 * (8342.13 + 2406030.0 / (130.0 - sigma2) + 15997.0 / (38.9 - sigma2))


def air_to_vacuum_wavenumber(wavelength_air_nm):
    """Vacuum wavenumber (cm-1) of an air wavelength, inverting Edlén 1966 by iteration."""

    in_air = 1.0e7 / np.asarray(wavelength_air_nm, dtype=float)
    vacuum = in_air.copy()
    # n - 1 is ~3e-4 and varies slowly, so each pass gains about four decades.
    for _ in range(4):
        vacuum = in_air / edlen_refractive_index(vacuum)
    return vacuum


def _fluxatl_pages(directory: Path) -> list[Path]:
    return sorted(p for p in Path(directory).iterdir() if p.name.startswith("lm") and p.name[2:].isdigit())


def _sha256(path: Path) -> str:
    if not path.is_dir():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    # A page atlas: every page's name and bytes, in order.
    digest = hashlib.sha256()
    for page in _fluxatl_pages(path):
        digest.update(page.name.encode() + b"\0" + page.read_bytes())
    return digest.hexdigest()


def _read_columns(atlas: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    if atlas == "iag":
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            data = np.loadtxt(handle, usecols=(0, 1))
    elif atlas == "wallace2011":
        data = np.loadtxt(path, usecols=(0, 3))
    elif atlas == "fluxatl":
        # Pages overlap by 0.1 nm with the same samples; the first copy is kept.
        pages = [np.loadtxt(page, usecols=(0, 1)) for page in _fluxatl_pages(path)]
        data = np.concatenate(pages)
        order = np.argsort(data[:, 0], kind="stable")
        data = data[order]
        data = data[np.concatenate([[True], np.diff(data[:, 0]) > 0])]
        nu = air_to_vacuum_wavenumber(data[:, 0])
        order = np.argsort(nu)
        return nu[order], data[order, 1]
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
        path=path, sha256=_sha256(path), source_name=source_name,
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



KURUCZ_IRRADIANCE_STRAIGHT_SAMPLES = 8


def read_kurucz_irradiance(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kurucz's 2005 residual irradiance atlas: vacuum wavenumber, flux, hand-interpolated.

    ``irradrelwl.dat`` (Kurucz 2006, astro-ph/0605029) is the 1984 flux atlas's scans
    with a fitted telluric spectrum divided out, 299-1001 nm by 0.001 nm in vacuum
    wavelength, gravitational redshift kept. Where the division failed (saturated
    cores, wavelength mismatches) the abstract says the spectrum was replaced by hand
    with straight lines, and nothing flags them: the third array marks every sample
    in a run of at least KURUCZ_IRRADIANCE_STRAIGHT_SAMPLES whose second difference is
    within the 4-decimal rounding, so a score can leave them out. Flat continuum at
    exactly 1.0000 is caught too, which costs a comparison only pixels it did not
    need. Wavenumber ascending.
    """

    data = np.loadtxt(path, skiprows=5)
    wavelength_nm, flux = data[::-1, 0], data[::-1, 1]
    level = np.abs(np.diff(flux, 2)) <= 1.5e-4
    run = KURUCZ_IRRADIANCE_STRAIGHT_SAMPLES
    starts = np.nonzero(np.convolve(level.astype(int), np.ones(run, int), "valid") == run)[0]
    straight = np.zeros(flux.size, bool)
    for start in starts:
        straight[start:start + run + 2] = True
    return 1.0e7 / wavelength_nm, flux, straight
