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
from .igrins import (
    IGRINSObservation,
    IGRINSOrder,
    Site,
    continuum_level,
    hydrogen_series_um,
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
    igrins_wavenumber_grid,
    trim_wavenumber_grid,
)
from .stellar import (
    StellarSpectrum,
    broaden_stellar_source,
    prepare_stellar_source,
    resample_stellar_source,
)
from .types import AtmosphereProfile, SpectralOrder, TelluricParameters

__all__ = [
    "AERLineDatabase",
    "arcturus_spectral_order",
    "ArcturusPage",
    "ArrayOpacityBackend",
    "AtmosphereProfile",
    "BoxcarFTSInstrumentProfile",
    "broaden_stellar_source",
    "build_lblrtm_correction",
    "chebyshev_continuum",
    "compare_transmission",
    "continuum_level",
    "default_data_directory",
    "degrade_to_resolving_power",
    "download_aer_lines",
    "download_mt_ckd",
    "epoch_velocity_kms",
    "ExoJAXOpacityBackend",
    "file_sha256",
    "fit_order",
    "FitResult",
    "hydrogen_series_um",
    "igrins_spectral_order",
    "igrins_wavenumber_grid",
    "IGRINSObservation",
    "IGRINSOrder",
    "ils_fingerprint",
    "LBLRTMOpticalDepthCorrection",
    "LBLRTMRunConfig",
    "LBLRTMSpectrum",
    "LinearizedOpacityBackend",
    "load_atmosphere_csv",
    "load_mipas_profile",
    "merge_records",
    "MTCKDWaterContinuum",
    "OrderObjective",
    "parameters_from_row",
    "prepare_stellar_source",
    "read_arcturus_page",
    "read_igrins_observation",
    "read_record",
    "read_tape12_single_precision",
    "Record",
    "ReferenceWaterContinuum",
    "resample_stellar_source",
    "robust_noise",
    "run_lblrtm",
    "Site",
    "site_for",
    "SpectralOrder",
    "stellar_line_mask",
    "StellarSpectrum",
    "surface_conditions",
    "TelluricModel",
    "TelluricParameters",
    "trim_wavenumber_grid",
    "ValidationMetrics",
    "write_record",
    "write_tape5",
    "zenith_angle_deg",
]
