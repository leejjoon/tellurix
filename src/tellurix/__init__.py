"""Differentiable terrestrial telluric transmission."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .fit import FitResult, OrderObjective, fit_order
from .record import (
    Record,
    file_sha256,
    ils_fingerprint,
    merge_records,
    parameters_from_row,
    read_record,
    write_record,
)
from .corrections import LBLRTMOpticalDepthCorrection, build_lblrtm_correction
from .aer import (
    AERLineDatabase,
    line_optical_depth_bound,
    select_significant_lines,
)
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
    fts_spectral_order,
    photatl_spectral_order,
    read_fts_spectrum,
    read_photatl_page,
    uniform_wavenumber_grid,
    window_continuum_snr,
    zenith_angle_deg_for_airmass,
)
from .igrins import (
    IGRINSObservation,
    IGRINSOrder,
    Site,
    continuum_level,
    hydrogen_series_um,
    leave_one_out_patterns,
    precipitable_water_mm,
    saturation_vapour_pressure_hpa,
    igrins_spectral_order,
    read_igrins_observation,
    site_for,
    stellar_line_mask,
    surface_conditions,
    zenith_angle_deg,
)
from .exojax_backend import ExoJAXOpacityBackend
from .download import default_data_directory, download_aer_lines, download_mt_ckd
from .io import load_atmosphere_csv, load_mipas_profile
from .lblrtm import LBLRTMRunConfig, run_lblrtm, write_tape5
from .mt_ckd import MTCKDWaterContinuum
from .reference import (
    LBLRTMSpectrum,
    ValidationMetrics,
    compare_transmission,
    degrade_to_resolving_power,
    read_tape12_single_precision,
)
from .model import (
    ArrayOpacityBackend,
    chebyshev_continuum,
    LinearizedOpacityBackend,
    BoxcarFTSInstrumentProfile,
    ReferenceWaterContinuum,
    TelluricModel,
    constant_velocity_grid,
    trim_wavenumber_grid,
)
from .stellar import (
    StellarSpectrum,
    broaden_stellar_source,
    prepare_stellar_source,
    resample_stellar_continuum,
    resample_stellar_source,
)
from .types import AtmosphereProfile, SpectralOrder, TelluricParameters

__all__ = [
    "AERLineDatabase",
    "ArcturusPage",
    "ArrayOpacityBackend",
    "AtmosphereProfile",
    "BOXCAR_FWHM_CONSTANT",
    "BoxcarFTSInstrumentProfile",
    "ExoJAXOpacityBackend",
    "FTSSpectrum",
    "FitResult",
    "IGRINSObservation",
    "IGRINSOrder",
    "LBLRTMOpticalDepthCorrection",
    "LBLRTMRunConfig",
    "LBLRTMSpectrum",
    "LinearizedOpacityBackend",
    "MINIMUM_ENVELOPE_STEP_DECADES",
    "MTCKDWaterContinuum",
    "OrderObjective",
    "PhotatlPage",
    "Record",
    "ReferenceWaterContinuum",
    "Site",
    "SpectralOrder",
    "StellarSpectrum",
    "TelluricModel",
    "TelluricParameters",
    "ValidationMetrics",
    "arcturus_spectral_order",
    "broaden_stellar_source",
    "build_lblrtm_correction",
    "chebyshev_continuum",
    "compare_transmission",
    "constant_velocity_grid",
    "continuum_level",
    "default_data_directory",
    "degrade_to_resolving_power",
    "describe_truncation",
    "download_aer_lines",
    "download_mt_ckd",
    "epoch_velocity_kms",
    "file_sha256",
    "fit_order",
    "fts_spectral_order",
    "hydrogen_series_um",
    "igrins_spectral_order",
    "ils_fingerprint",
    "interferogram_envelope",
    "leave_one_out_patterns",
    "load_atmosphere_csv",
    "load_mipas_profile",
    "measure_mopd",
    "merge_records",
    "parameters_from_row",
    "photatl_spectral_order",
    "precipitable_water_mm",
    "prepare_stellar_source",
    "read_arcturus_page",
    "read_fts_spectrum",
    "read_igrins_observation",
    "read_photatl_page",
    "read_record",
    "read_tape12_single_precision",
    "resample_stellar_continuum",
    "resample_stellar_source",
    "robust_noise",
    "run_lblrtm",
    "saturation_vapour_pressure_hpa",
    "site_for",
    "stellar_line_mask",
    "surface_conditions",
    "trim_wavenumber_grid",
    "uniform_wavenumber_grid",
    "window_continuum_snr",
    "write_record",
    "write_tape5",
    "zenith_angle_deg",
    "zenith_angle_deg_for_airmass",
]
