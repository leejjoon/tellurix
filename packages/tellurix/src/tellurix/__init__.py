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
    AER_MOLECULE_IDS,
    AERLineDatabase,
    line_optical_depth_bound,
    select_significant_lines,
)
from .o2_cia import ContinuumSum, O2CollisionInducedContinuum
from .exojax_backend import ExoJAXOpacityBackend
from .download import DataPaths, default_data_directory, download_aer_lines, download_mt_ckd
from .io import load_atmosphere_csv, load_mipas_profile
from .lblrtm import LBLRTMRunConfig, run_lblrtm, write_tape5
from .mt_ckd import MTCKDWaterContinuum
from .quality import species_at_upper_bound, write_upper_bound_flags
from .scan import (
    ScanIdentity,
    cached_scan,
    load_scan,
    read_scan,
    save_scan,
    scan_window,
    species_above,
)
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
from .site_profile import (
    LAYERINGS, afgl_dry_vmr, build_site_profile, load_afgl, weighted_layers, write_profile_csv,
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
    "AER_MOLECULE_IDS",
    "ArrayOpacityBackend",
    "AtmosphereProfile",
    "BoxcarFTSInstrumentProfile",
    "ExoJAXOpacityBackend",
    "FitResult",
    "LBLRTMOpticalDepthCorrection",
    "LBLRTMRunConfig",
    "LBLRTMSpectrum",
    "LinearizedOpacityBackend",
    "MTCKDWaterContinuum",
    "OrderObjective",
    "Record",
    "ReferenceWaterContinuum",
    "SpectralOrder",
    "StellarSpectrum",
    "TelluricModel",
    "TelluricParameters",
    "ValidationMetrics",
    "broaden_stellar_source",
    "build_lblrtm_correction",
    "chebyshev_continuum",
    "compare_transmission",
    "constant_velocity_grid",
    "default_data_directory",
    "degrade_to_resolving_power",
    "download_aer_lines",
    "download_mt_ckd",
    "file_sha256",
    "fit_order",
    "ils_fingerprint",
    "ScanIdentity",
    "cached_scan",
    "scan_window",
    "species_above",
    "load_scan",
    "read_scan",
    "save_scan",
    "load_atmosphere_csv",
    "load_mipas_profile",
    "merge_records",
    "parameters_from_row",
    "prepare_stellar_source",
    "ContinuumSum",
    "DataPaths",
    "afgl_dry_vmr",
    "build_site_profile",
    "weighted_layers",
    "LAYERINGS",
    "load_afgl",
    "species_at_upper_bound",
    "write_profile_csv",
    "write_upper_bound_flags",
    "O2CollisionInducedContinuum",
    "read_record",
    "read_tape12_single_precision",
    "resample_stellar_continuum",
    "resample_stellar_source",
    "run_lblrtm",
    "trim_wavenumber_grid",
    "write_record",
    "write_tape5",
]
