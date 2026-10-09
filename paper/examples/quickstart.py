"""Sky transmission over 2.30-2.32 um, from a checkout with the line data bootstrapped."""

import jax
import numpy as np

import tellurix  # enables float64; import before ExoJAX
from tellurix import (
    AERLineDatabase, DataPaths, ExoJAXOpacityBackend, MTCKDWaterContinuum,
    TelluricModel, TelluricParameters, constant_velocity_grid, load_atmosphere_csv,
)
from tellurix.aer import AER_MOLECULE_IDS

data = DataPaths.bootstrapped(".")  # or DataPaths.downloaded() after tellurix-download-data all
profile = load_atmosphere_csv("data/profiles/gemini_2021_era5.csv")  # ERA5 over Gemini South
nu = constant_velocity_grid(2300.0, 2320.0, resolving_power=45_000)

species = ("H2O", "CH4", "CO")
lines = {
    s: AERLineDatabase(data.line_file(s, AER_MOLECULE_IDS[s]), s, (nu[0], nu[-1]))
    for s in species
}
opacity = ExoJAXOpacityBackend.prepare(
    lines, nu, methods="direct_sparse", vectorize_layers=True, pressure_shift=True
)
model = TelluricModel(
    profile, nu, opacity,
    accuracy_mode="mt_ckd", continuum=MTCKDWaterContinuum.from_netcdf(data.mt_ckd, nu),
)

parameters = TelluricParameters(
    log_column_scales={s: 0.0 for s in species},  # the profile's own columns
    velocity_kms=0.0, wavelength_stretch=0.0, lsf_sigma_kms=2.83,
    continuum_coeffs=np.zeros(1), log_jitter=-10.0,
)
# Compile once; an uncompiled call on a live opacity backend is very slow.
transmission = jax.jit(model.transmission)(parameters, 48.2)  # zenith angle 48.2 deg = air mass 1.5
print(f"{len(nu)} samples, minimum transmission {float(transmission.min()):.3f}")
