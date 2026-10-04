#!/usr/bin/env python
"""tellurix against LBLRTM 12.17 when both integrate the same atmosphere.

LBLRTM is given tellurix's own layers -- pressure, temperature and molecular
column of each, as IATM=0 layer input (``LBLRTMRunConfig.user_layers``) -- so
the comparison measures physics, not two different atmospheres. The layer
pressure is the air-weighted mean of the edges for both codes, lines and
continua alike. Reports the transmission error at R=45,000 before and after
fitting the H2O and CO2 column scales and a linear continuum, the way any fit
would, and the per-species line and continuum ratios behind it. Writes
``docs/lblrtm_identical_layers.json``.

    UV_CACHE_DIR=.uv-cache uv run python scripts/compare_lblrtm_layers.py

Needs ``bootstrap_lblrtm.sh``; four LBLRTM runs of a few minutes each.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import tellurix  # noqa: F401  (x64 before exojax)
import jax.numpy as jnp
from scipy.optimize import least_squares

from tellurix import (
    AER_MOLECULE_IDS, AERLineDatabase, DataPaths, ExoJAXOpacityBackend, LBLRTMRunConfig,
    MTCKDWaterContinuum, TelluricModel, TelluricParameters, constant_velocity_grid,
    load_atmosphere_csv, run_lblrtm,
)

WINDOW_CM1 = (5000.0, 5020.0)
RESOLVING_POWER = 45_000.0
SAMPLES_PER_RESOLUTION = 4.0
LINE_MARGIN_CM1 = 25.0
# LBLRTM lays its own grid inside the request and can fall short of the ends.
PAD_CM1 = 0.5


def convolve(transmission: np.ndarray) -> np.ndarray:
    sigma = SAMPLES_PER_RESOLUTION / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    offsets = np.arange(-int(6 * sigma) - 1, int(6 * sigma) + 2)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return np.convolve(transmission, kernel / kernel.sum(), "same")


def on_grid(spectrum, nu: np.ndarray) -> np.ndarray:
    transmission = np.clip(np.asarray(spectrum.transmission, float), np.finfo(np.float32).tiny, None)
    return np.interp(nu, np.asarray(spectrum.wavenumber_cm1), -np.log(transmission))


def errors(candidate: np.ndarray, target: np.ndarray, keep: np.ndarray) -> dict:
    error = np.abs(candidate - target)[keep]
    return {"median": float(np.median(error)), "p99": float(np.percentile(error, 99)),
            "max": float(error.max())}


def compare(tau_species: dict, extra: np.ndarray, target_tau: np.ndarray, nu: np.ndarray) -> dict:
    """Unfitted and fitted error of exp(-tau) against LBLRTM, monochromatic and at R=45,000."""

    inner = (nu >= WINDOW_CM1[0]) & (nu <= WINDOW_CM1[1])
    reference = np.exp(-target_tau)
    total = sum(tau_species.values()) + extra
    convolved_reference = convolve(reference)
    keep = inner & (convolved_reference > 0.05)
    result = {
        "unfitted_monochromatic": errors(np.exp(-total), reference, inner & (reference > 0.05)),
        "unfitted_R45000": errors(convolve(np.exp(-total)), convolved_reference, keep),
    }
    others = sum(t for s, t in tau_species.items() if s not in ("H2O", "CO2")) + extra
    x = (nu - 0.5 * sum(WINDOW_CM1)) / (0.5 * (WINDOW_CM1[1] - WINDOW_CM1[0]))

    def model(p):
        tau = np.exp(p[0]) * tau_species["H2O"] + np.exp(p[1]) * tau_species["CO2"] + others
        return convolve(np.exp(-tau)) * (1.0 + p[2] + p[3] * x)

    fit = least_squares(lambda p: (model(p) - convolved_reference)[keep], np.zeros(4))
    result["fitted_R45000"] = {**errors(model(fit.x), convolved_reference, keep),
                               "h2o_scale": float(np.exp(fit.x[0])),
                               "co2_scale": float(np.exp(fit.x[1]))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", type=Path, default=Path("data/profiles/example_midlatitude.csv"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/lblrtm/run_identical_layers"))
    parser.add_argument("--output", type=Path, default=Path("docs/lblrtm_identical_layers.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = DataPaths.bootstrapped(root)
    reference = root / "data/lblrtm"
    executable = reference / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
    tape3 = reference / "run_lnfl_igrins/TAPE3"
    source = load_atmosphere_csv(root / args.profile)
    # One pressure per layer, as LBLRTM has: the air-weighted mean.
    profile = dataclasses.replace(source, mean_pressure_bar=source.continuum_pressure_bar)
    nu = constant_velocity_grid(1.0e7 / WINDOW_CM1[1], 1.0e7 / WINDOW_CM1[0],
                                resolving_power=RESOLVING_POWER,
                                samples_per_resolution=SAMPLES_PER_RESOLUTION,
                                margin_cm1=LINE_MARGIN_CM1)

    def lblrtm(run_profile, continuum_flag, name):
        config = LBLRTMRunConfig(float(nu[0]) - PAD_CM1, float(nu[-1]) + PAD_CM1,
                                 continuum_flag=continuum_flag, user_layers=True,
                                 description=f"tellurix identical layers {name}")
        spectrum = run_lblrtm(root / args.run_dir / name, run_profile, config, executable, tape3,
                              paths.mt_ckd)
        print(f"LBLRTM {name} done", flush=True)
        return on_grid(spectrum, nu)

    lblrtm_tau = {
        "continua": lblrtm(profile, 1, "continua"),
        "lines": lblrtm(profile, 0, "lines"),
        **{f"lines_{s.lower()}": lblrtm(dataclasses.replace(profile, vmr={s: profile.vmr[s]}), 0,
                                         f"lines_{s.lower()}")
           for s in ("H2O", "CO2")},
    }

    databases = {}
    for species in profile.vmr:
        try:
            # Lines LBLRTM reaches for: everything within 25 cm-1 of the grid.
            databases[species] = AERLineDatabase(paths.line_file(species, AER_MOLECULE_IDS[species]),
                                                 species, (float(nu[0]), float(nu[-1])),
                                                 margin_cm1=LINE_MARGIN_CM1)
        except ValueError as exc:
            if "lines found" not in str(exc):
                raise
    backend = ExoJAXOpacityBackend.prepare(
        databases, nu, methods="direct_sparse",
        temperature_range_k=(float(profile.temperature_k.min()), float(profile.temperature_k.max())),
        maximum_pressure_bar=float(profile.pressure_layer_bar.max()), vectorize_layers=True,
        pressure_shift=True)
    continuum = MTCKDWaterContinuum.from_netcdf(paths.mt_ckd, nu)
    model = TelluricModel(profile, nu, backend, continuum=continuum)
    parameters = TelluricParameters({s: 0.0 for s in model.species}, 0.0, 0.0, 3.0,
                                    jnp.asarray([0.0]), np.log(1.0e-5))
    pieces = {k: -np.log(np.asarray(v)) for k, v in model.species_transmission(parameters).items()}
    mt_ckd = pieces.pop("continuum")

    inner = (nu >= WINDOW_CM1[0]) & (nu <= WINDOW_CM1[1])
    lblrtm_continuum = lblrtm_tau["continua"] - lblrtm_tau["lines"]
    species_lines = {}
    for s in ("H2O", "CO2"):
        theirs, ours = lblrtm_tau[f"lines_{s.lower()}"], pieces[s]
        species_lines[s] = {
            "integrated_ratio_tellurix_over_lblrtm": float(ours[inner].sum() / theirs[inner].sum()),
            "residual_rms": float(np.sqrt(np.mean((theirs - ours)[inner] ** 2))),
        }
    report = {
        "description": __doc__.split("\n\n")[1].replace("\n", " "),
        "generated_by": "scripts/compare_lblrtm_layers.py",
        "measured": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "profile": str(args.profile), "layer_pressure": "air-weighted mean of the edges, both codes",
        "lblrtm": "12.17, IATM=0 layer input, TAPE3 run_lnfl_igrins (AER 3.9)",
        "tellurix": "AER 3.9 lines within 25 cm-1, direct_sparse with pressure shifts; native MT_CKD 4.3",
        "window_cm1": WINDOW_CM1, "resolving_power": RESOLVING_POWER,
        "samples_per_resolution": SAMPLES_PER_RESOLUTION, "zenith_angle_deg": 0.0,
        "metric": "absolute transmission error where the reference exceeds 0.05",
        "water_column_cm2": float(np.sum(profile.air_column_cm2 * profile.vmr["H2O"])),
        "lines_only": compare(pieces, 0.0, lblrtm_tau["lines"], nu),
        "lines_and_continua": compare(pieces, mt_ckd, lblrtm_tau["continua"], nu),
        "species_lines": species_lines,
        "continuum_mean_optical_depth": {
            "lblrtm_all_continua": float(lblrtm_continuum[inner].mean()),
            "tellurix_mt_ckd": float(mt_ckd[inner].mean()),
        },
    }
    for key in ("lines_only", "lines_and_continua"):
        print(key, json.dumps(report[key]), flush=True)
    print(json.dumps(report["species_lines"]), json.dumps(report["continuum_mean_optical_depth"]))
    output = root / args.output
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
