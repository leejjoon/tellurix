#!/usr/bin/env python
"""tellurix against LBLRTM 12.17 when both integrate the same atmosphere.

LBLRTM is given tellurix's own layers -- pressure, temperature and molecular
column of each, as IATM=0 layer input (``LBLRTMRunConfig.user_layers``) -- so
the comparison measures physics, not two different atmospheres. The layer
pressure is the air-weighted mean of the edges for both codes, lines and
continua alike. Reports the transmission error at R=45,000 before and after
fitting the H2O and CO2 column scales and a linear continuum, the way any fit
would, and the per-species line and continuum ratios behind it.

It then splits the CO2 residual. LBLRTM applies AER's first-order line
coupling to every CO2 line the TAPE3 carries coefficients for, and cuts every
line at 25 cm-1 less a pedestal; tellurix does neither. A CO2-only TAPE3 is
built with LNFL both with and without coupling (NOCPL), and the cutoff is
evaluated with ``tellurix.lblrtm_line_shape_optical_depth``, so each effect is
measured on its own and on the fitted transmission. Writes
``docs/lblrtm_identical_layers.json``.

    UV_CACHE_DIR=.uv-cache uv run python scripts/compare_lblrtm_layers.py

Needs ``bootstrap_lblrtm.sh``. The LBLRTM and LNFL runs take seconds; the
tellurix side a minute on a GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import tellurix  # noqa: F401  (x64 before exojax)
import jax.numpy as jnp
from scipy.optimize import least_squares

from tellurix import (
    AER_MOLECULE_IDS, AERLineDatabase, DataPaths, ExoJAXOpacityBackend, LBLRTMRunConfig,
    MTCKDWaterContinuum, TelluricModel, TelluricParameters, constant_velocity_grid,
    lblrtm_line_shape_optical_depth, load_atmosphere_csv, run_lblrtm, run_lnfl,
)

# The window the template was built for; --v1/--v2 measure another.
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


def compare(tau_species: dict, extra: np.ndarray, target_tau: np.ndarray, nu: np.ndarray,
            window: tuple = WINDOW_CM1) -> dict:
    """Unfitted and fitted error of exp(-tau) against LBLRTM, monochromatic and at R=45,000."""

    inner = (nu >= window[0]) & (nu <= window[1])
    reference = np.exp(-target_tau)
    total = sum(tau_species.values()) + extra
    convolved_reference = convolve(reference)
    keep = inner & (convolved_reference > 0.05)
    result = {
        "unfitted_monochromatic": errors(np.exp(-total), reference, inner & (reference > 0.05)),
        "unfitted_R45000": errors(convolve(np.exp(-total)), convolved_reference, keep),
    }
    others = sum(t for s, t in tau_species.items() if s not in ("H2O", "CO2")) + extra
    x = (nu - 0.5 * sum(window)) / (0.5 * (window[1] - window[0]))

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
    parser.add_argument("--v1", type=float, default=WINDOW_CM1[0])
    parser.add_argument("--v2", type=float, default=WINDOW_CM1[1])
    args = parser.parse_args()
    window = (args.v1, args.v2)

    root = Path(__file__).resolve().parents[1]
    paths = DataPaths.bootstrapped(root)
    reference = root / "data/lblrtm"
    executable = reference / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
    tape3 = reference / "run_lnfl_igrins/TAPE3"
    source = load_atmosphere_csv(root / args.profile)
    # One pressure per layer, as LBLRTM has: the air-weighted mean.
    profile = dataclasses.replace(source, mean_pressure_bar=source.continuum_pressure_bar)
    nu = constant_velocity_grid(1.0e7 / window[1], 1.0e7 / window[0],
                                resolving_power=RESOLVING_POWER,
                                samples_per_resolution=SAMPLES_PER_RESOLUTION,
                                margin_cm1=LINE_MARGIN_CM1)

    def lblrtm(run_profile, continuum_flag, name, line_file=tape3, line_rejection=True):
        config = LBLRTMRunConfig(float(nu[0]) - PAD_CM1, float(nu[-1]) + PAD_CM1,
                                 continuum_flag=continuum_flag, user_layers=True,
                                 line_rejection=line_rejection,
                                 description=f"tellurix identical layers {name}")
        spectrum = run_lblrtm(root / args.run_dir / name, run_profile, config, executable,
                              line_file, paths.mt_ckd)
        print(f"LBLRTM {name} done", flush=True)
        return on_grid(spectrum, nu)

    lblrtm_tau = {
        "continua": lblrtm(profile, 1, "continua"),
        "lines": lblrtm(profile, 0, "lines"),
        **{f"lines_{s.lower()}": lblrtm(dataclasses.replace(profile, vmr={s: profile.vmr[s]}), 0,
                                         f"lines_{s.lower()}")
           for s in ("H2O", "CO2")},
    }

    # CO2 alone from TAPE3s with and without AER's line coupling, over the
    # grid plus LBLRTM's 25 cm-1 reach. Line rejection is off in both: LBLRTM
    # never rejects a coupled line, so with it on the pair would differ by
    # rejection as well as by coupling.
    co2_profile = dataclasses.replace(profile, vmr={"CO2": profile.vmr["CO2"]})
    for coupling in (True, False):
        name = "co2_coupled" if coupling else "co2_uncoupled"
        line_file = run_lnfl(root / args.run_dir / f"lnfl_{name}", ("CO2",),
                             float(nu[0]) - LINE_MARGIN_CM1 - 1.0, float(nu[-1]) + LINE_MARGIN_CM1 + 1.0,
                             reference / "AER_Line_File/aer_v_3.9/line_file/aer_v_3.9",
                             reference / "LNFL/lnfl_v3.2_linux_gnu_sgl", line_coupling=coupling)
        lblrtm_tau[name] = lblrtm(co2_profile, 0, name, line_file, line_rejection=False)
        if coupling:
            # LNFL's TAPE6 lists, per molecule, the lines and how many of them
            # carry coupling coefficients.
            log = (line_file.parent / "TAPE6").read_bytes().replace(b"\0", b"").decode("ascii", "replace")
            lines, coupled_count = re.search(r"CO2\s+=\s+(\d+)\s+(\d+)", log).groups()

    databases = {}
    for species in profile.vmr:
        try:
            # Lines LBLRTM reaches for: everything within 25 cm-1 of the grid,
            # with AER's coupling coefficients for the coupled model.
            databases[species] = AERLineDatabase(paths.line_file(species, AER_MOLECULE_IDS[species]),
                                                 species, (float(nu[0]), float(nu[-1])),
                                                 margin_cm1=LINE_MARGIN_CM1,
                                                 line_coupling=paths.line_coupling)
        except ValueError as exc:
            if "lines found" not in str(exc):
                raise
    continuum = MTCKDWaterContinuum.from_netcdf(paths.mt_ckd, nu)

    def tellurix(line_coupling):
        backend = ExoJAXOpacityBackend.prepare(
            databases, nu, methods="direct_sparse",
            temperature_range_k=(float(profile.temperature_k.min()), float(profile.temperature_k.max())),
            maximum_pressure_bar=float(profile.pressure_layer_bar.max()),
            # A loop over six layers keeps lax.cond a branch; under vmap it becomes a
            # select that also evaluates the dense fallback, 9 GB on a 70 cm-1 order.
            vectorize_layers=False, pressure_shift=True, line_coupling=line_coupling)
        model = TelluricModel(profile, nu, backend, continuum=continuum)
        parameters = TelluricParameters({s: 0.0 for s in model.species}, 0.0, 0.0, 3.0,
                                        jnp.asarray([0.0]), np.log(1.0e-5))
        return {k: -np.log(np.asarray(v)) for k, v in model.species_transmission(parameters).items()}

    pieces, coupled = tellurix(False), tellurix(True)
    mt_ckd = pieces.pop("continuum")
    coupled.pop("continuum")

    inner = (nu >= window[0]) & (nu <= window[1])
    lblrtm_continuum = lblrtm_tau["continua"] - lblrtm_tau["lines"]
    species_lines = {}
    for s in ("H2O", "CO2"):
        theirs, ours = lblrtm_tau[f"lines_{s.lower()}"], pieces[s]
        species_lines[s] = {
            "integrated_ratio_tellurix_over_lblrtm": float(ours[inner].sum() / theirs[inner].sum()),
            "residual_rms": float(np.sqrt(np.mean((theirs - ours)[inner] ** 2))),
        }
    # What the 25 cm-1 cutoff changes for the lines tellurix carries -- all
    # of those within 25 cm-1 of the grid, as LBLRTM carries.
    cutoff = {}
    for species, database in databases.items():
        truncated, full = lblrtm_line_shape_optical_depth(database, profile, nu)
        cutoff[species] = truncated - full

    def ratio_and_rms(ours, theirs):
        return {"integrated_ratio_tellurix_over_lblrtm": float(ours[inner].sum() / theirs[inner].sum()),
                "residual_rms": float(np.sqrt(np.mean((theirs - ours)[inner] ** 2)))}

    their_coupling = lblrtm_tau["co2_coupled"] - lblrtm_tau["co2_uncoupled"]
    our_coupling = coupled["CO2"] - pieces["CO2"]
    co2 = {
        "tape3_lines": int(lines), "tape3_coupled_lines": int(coupled_count),
        "lblrtm_line_rejection": False,
        "uncoupled": ratio_and_rms(pieces["CO2"], lblrtm_tau["co2_uncoupled"]),
        "uncoupled_with_cutoff": ratio_and_rms(pieces["CO2"] + cutoff["CO2"],
                                               lblrtm_tau["co2_uncoupled"]),
        "coupled": ratio_and_rms(coupled["CO2"], lblrtm_tau["co2_coupled"]),
        "coupled_with_cutoff": ratio_and_rms(coupled["CO2"] + cutoff["CO2"], lblrtm_tau["co2_coupled"]),
        "coupling_term_over_uncoupled_co2": {
            "tellurix": float(our_coupling[inner].sum() / pieces["CO2"][inner].sum()),
            "lblrtm": float(their_coupling[inner].sum() / lblrtm_tau["co2_uncoupled"][inner].sum()),
            "rms_of_lblrtm_term": float(np.sqrt(np.mean(their_coupling[inner] ** 2))),
            "rms_of_difference": float(np.sqrt(np.mean((their_coupling - our_coupling)[inner] ** 2))),
        },
    }
    variants = {"as_now": pieces, "with_line_coupling": coupled,
                "with_cutoff": {k: v + cutoff.get(k, 0.0) for k, v in pieces.items()},
                "with_line_coupling_and_cutoff": {k: v + cutoff.get(k, 0.0) for k, v in coupled.items()}}
    co2["fitted_transmission"] = {
        "note": "tellurix lines + MT_CKD against LBLRTM with continua as it runs; tellurix with its "
                "own line coupling (direct_sparse line_coupling=True) and/or LBLRTM's cutoff",
        **{name: compare(taus, mt_ckd, lblrtm_tau["continua"], nu, window) for name, taus in variants.items()}}

    report = {
        "description": __doc__.split("\n\n")[1].replace("\n", " "),
        "generated_by": "scripts/compare_lblrtm_layers.py",
        "measured": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "profile": str(args.profile), "layer_pressure": "air-weighted mean of the edges, both codes",
        "lblrtm": "12.17, IATM=0 layer input, TAPE3 run_lnfl_igrins (AER 3.9)",
        "tellurix": "AER 3.9 lines within 25 cm-1, direct_sparse with pressure shifts; native MT_CKD 4.3; "
                    "line coupling off except where stated",
        "window_cm1": window, "resolving_power": RESOLVING_POWER,
        "samples_per_resolution": SAMPLES_PER_RESOLUTION, "zenith_angle_deg": 0.0,
        "metric": "absolute transmission error where the reference exceeds 0.05",
        "water_column_cm2": float(np.sum(profile.air_column_cm2 * profile.vmr["H2O"])),
        "lines_only": compare(pieces, 0.0, lblrtm_tau["lines"], nu, window),
        "lines_and_continua": compare(pieces, mt_ckd, lblrtm_tau["continua"], nu, window),
        "species_lines": species_lines,
        "continuum_mean_optical_depth": {
            "lblrtm_all_continua": float(lblrtm_continuum[inner].mean()),
            "tellurix_mt_ckd": float(mt_ckd[inner].mean()),
        },
        "co2_attribution": co2,
    }
    for key in ("lines_only", "lines_and_continua"):
        print(key, json.dumps(report[key]), flush=True)
    print(json.dumps(report["species_lines"]), json.dumps(report["continuum_mean_optical_depth"]))
    for key, value in co2.items():
        if isinstance(value, dict) and key != "fitted_transmission":
            print(key, json.dumps(value))
    for key, value in co2["fitted_transmission"].items():
        if key != "note":
            print(key, json.dumps(value["fitted_R45000"]))
    output = root / args.output
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
