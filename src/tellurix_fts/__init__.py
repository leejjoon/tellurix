"""Telluric fitting of Fourier-transform spectrometer atlases: Arcturus and the NSO solar atlases.

Readers for the Hinkle, Wallace & Livingston 1995 Arcturus atlas (``atlas``) and the
NSO Kitt Peak solar atlases (``nso``), and the FTS instrument profile measured from a
spectrum itself (``ils``). The forward model and the fitter are in ``tellurix``.
"""

# Imported first so that jax_enable_x64 is set before anything builds arrays.
import tellurix  # noqa: F401

from .atlas import (
    ArcturusPage,
    arcturus_spectral_order,
    epoch_velocity_kms,
    read_arcturus_page,
    robust_noise,
)
from .ils import (
    BOXCAR_FWHM_CONSTANT,
    MINIMUM_ENVELOPE_STEP_DECADES,
    describe_truncation,
    interferogram_envelope,
    measure_mopd,
)
from .nso import (
    FTSSpectrum,
    PhotatlPage,
    fts_continuum_level,
    fts_spectral_order,
    photatl_spectral_order,
    read_fts_spectrum,
    read_photatl_page,
    read_solar_spectrum,
    photatl_as_fts_spectrum,
    read_niratl_page,
    niratl_as_fts_spectrum,
    NiratlPage,
    uniform_wavenumber_grid,
    window_continuum_snr,
    zenith_angle_deg_for_airmass,
)

__all__ = [
    "arcturus_spectral_order",
    "ArcturusPage",
    "BOXCAR_FWHM_CONSTANT",
    "describe_truncation",
    "epoch_velocity_kms",
    "fts_continuum_level",
    "fts_spectral_order",
    "FTSSpectrum",
    "interferogram_envelope",
    "measure_mopd",
    "MINIMUM_ENVELOPE_STEP_DECADES",
    "niratl_as_fts_spectrum",
    "NiratlPage",
    "photatl_as_fts_spectrum",
    "photatl_spectral_order",
    "PhotatlPage",
    "read_arcturus_page",
    "read_fts_spectrum",
    "read_niratl_page",
    "read_photatl_page",
    "read_solar_spectrum",
    "robust_noise",
    "uniform_wavenumber_grid",
    "window_continuum_snr",
    "zenith_angle_deg_for_airmass",
]
