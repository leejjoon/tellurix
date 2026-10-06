"""Telluric fitting of IGRINS spectra.

The RRISA reduced-product reader (``igrins``), a night's calibration from its
standards (``night``) and the lamp-flat blaze (``flat``). The forward model and the
fitter are in ``tellurix``.
"""

# Imported first so that jax_enable_x64 is set before anything builds arrays.
import tellurix  # noqa: F401

from .flat import (
    FlatBlaze,
)
from .night import (
    MasterPattern,
    NightCalibration,
    OrderCalibration,
)
from .igrins import (
    IGRINS_ORDER_CENTRES_UM,
    IGRINSObservation,
    IGRINSOrder,
    Site,
    WatSpec,
    continuum_level,
    format_wat2_cards,
    hydrogen_series_um,
    identify_orders,
    leave_one_out_patterns,
    parse_wat_specs,
    smoothed_frame_median,
    precipitable_water_mm,
    saturation_vapour_pressure_hpa,
    igrins_spectral_order,
    read_igrins_observation,
    site_for,
    stellar_line_mask,
    surface_conditions,
    zenith_angle_deg,
)
from .standard import StandardFitSettings, build_order_context, fit_one, summary_paths

__all__ = [
    "continuum_level",
    "FlatBlaze",
    "format_wat2_cards",
    "hydrogen_series_um",
    "identify_orders",
    "IGRINS_ORDER_CENTRES_UM",
    "igrins_spectral_order",
    "IGRINSObservation",
    "IGRINSOrder",
    "leave_one_out_patterns",
    "MasterPattern",
    "NightCalibration",
    "OrderCalibration",
    "parse_wat_specs",
    "precipitable_water_mm",
    "read_igrins_observation",
    "saturation_vapour_pressure_hpa",
    "summary_paths",
    "Site",
    "site_for",
    "smoothed_frame_median",
    "stellar_line_mask",
    "surface_conditions",
    "WatSpec",
    "zenith_angle_deg",
    "StandardFitSettings",
    "build_order_context",
    "fit_one",
]
