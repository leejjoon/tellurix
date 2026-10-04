#!/usr/bin/env python
"""What the fast-mode gap to LBLRTM in the 5000-5020 cm-1 template is made of.

Reads the correction template and the LBLRTM runs ``build_lblrtm_correction.py``
left under ``data/lblrtm/run_corrections/``; runs nothing new. Measures three
things: the column of each gas each code was given (LBLRTM's from its TAPE6),
what LBLRTM's 25 cm-1 line cutoff would change, and the transmission error at
R=45,000 before and after the column scales are fitted. Writes
``docs/lblrtm_gap_attribution.json``.

    UV_CACHE_DIR=.uv-cache uv run python scripts/attribute_lblrtm_gap.py

LBLRTM truncates every line at 25 cm-1 less a pedestal
(``tellurix.lblrtm_line_shape_optical_depth``); tellurix keeps the full Voigt
line and, in this template, carried only lines inside its grid. The predicted
LBLRTM-minus-tellurix optical depth from that alone is compared with the
template's measured residual.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import tellurix  # noqa: F401  (x64 before exojax)
from tellurix import (
    AER_MOLECULE_IDS, AERLineDatabase, DataPaths, lblrtm_line_shape_optical_depth,
    load_atmosphere_csv, read_tape12_single_precision,
)

CUTOFF_CM1 = 25.0
WINDOW_CM1 = (5000.0, 5020.0)
# TAPE6's 'INTEGRATED ABSORBER AMOUNTS' columns after the two boundaries.
TAPE6_COLUMNS = ("AIR", "H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2")


def tape6_total_column(path: Path) -> dict:
    text = path.read_bytes().replace(b"\0", b"").decode("ascii", "replace")
    match = re.search(r"^0TOTAL\s+\S+\s+\S+((?:\s+\S+){8})", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"no TOTAL amounts line in {path}")
    return dict(zip(TAPE6_COLUMNS, (float(v) for v in match.group(1).split())))


def tape12_optical_depth(path: Path, nu: np.ndarray) -> np.ndarray:
    spectrum = read_tape12_single_precision(path)
    transmission = np.clip(np.asarray(spectrum.transmission, float), np.finfo(np.float32).tiny, None)
    return np.interp(nu, np.asarray(spectrum.wavenumber_cm1), -np.log(transmission))


def cutoff_residual(paths, species, profile, nu) -> np.ndarray:
    """LBLRTM minus tellurix optical depth from the cutoff and the grid edges alone."""

    def lines(lower, upper):
        try:
            return AERLineDatabase(paths.line_file(species, AER_MOLECULE_IDS[species]),
                                   species, (lower, upper), margin_cm1=0.0)
        except ValueError as exc:
            if "lines found" not in str(exc):
                raise
            return None

    # The template's backend carried only the lines inside its grid, with
    # pressure shifts: the cutoff changes those, and LBLRTM adds the lines
    # within 25 cm-1 beyond either end.
    carried = lines(nu[0], nu[-1])
    tau = np.zeros_like(nu)
    if carried is not None:
        truncated, full = lblrtm_line_shape_optical_depth(carried, profile, nu)
        tau = truncated - full
    for beyond in (lines(nu[0] - CUTOFF_CM1 - 1.0, nu[0]), lines(nu[-1], nu[-1] + CUTOFF_CM1 + 1.0)):
        if beyond is not None:
            tau = tau + lblrtm_line_shape_optical_depth(beyond, profile, nu)[0]
    return tau


def convolve(transmission: np.ndarray) -> np.ndarray:
    # R=45,000 at the template's 4 samples per resolution element.
    sigma = 4.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    offsets = np.arange(-12, 13)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return np.convolve(transmission, kernel / kernel.sum(), "same")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, default=Path("data/corrections/lblrtm_5000_5020.npz"))
    parser.add_argument("--profile", type=Path, default=Path("data/profiles/example_midlatitude.csv"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/lblrtm/run_corrections"))
    parser.add_argument("--output", type=Path, default=Path("docs/lblrtm_gap_attribution.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = DataPaths.bootstrapped(root)
    template = np.load(root / args.template)
    nu = template["wavenumber_cm1"]
    profile = load_atmosphere_csv(root / args.profile)
    runs = root / args.run_dir
    species = [k.split("__", 1)[1] for k in template.files if k.startswith("line_residual__")]
    inner = (nu >= WINDOW_CM1[0]) & (nu <= WINDOW_CM1[1])

    lblrtm_column = tape6_total_column(runs / "continuum_all/TAPE6")
    tellurix_column = {"AIR": float(np.sum(profile.air_column_cm2))}
    tellurix_column.update({s: float(np.sum(profile.air_column_cm2 * profile.vmr[s])) for s in species})
    columns = {s: {"tellurix": tellurix_column[s], "lblrtm": lblrtm_column[s]}
               for s in ("AIR", "H2O", "CO2")}

    lines_lblrtm = {s: tape12_optical_depth(runs / f"lines_{s.lower()}/TAPE12", nu) for s in species}
    lines_tellurix = {s: lines_lblrtm[s] - template[f"line_residual__{s}"] for s in species}
    wide = (nu > 4980.0) & (nu < 5040.0)
    integrated = {s: float(lines_tellurix[s][wide].sum() / lines_lblrtm[s][wide].sum())
                  for s in ("H2O", "CO2")}

    cutoff = {}
    residual_rms = {}
    for s in species:
        cutoff[s] = cutoff_residual(paths, s, profile, nu)
        residual = template[f"line_residual__{s}"]
        residual_rms[s] = {"before": float(np.sqrt(np.mean(residual[inner] ** 2))),
                           "after_cutoff": float(np.sqrt(np.mean((residual - cutoff[s])[inner] ** 2)))}
        print(s, residual_rms[s], flush=True)

    all_tau = tape12_optical_depth(runs / "continuum_all/TAPE12", nu)
    mt_ckd = template["water_self_optical_depth"] + template["water_foreign_optical_depth"]
    lines_total = sum(lines_tellurix.values())
    cutoff_total = sum(cutoff.values())
    reference = np.exp(-all_tau)
    cases = {"lines": 0.0, "lines + MT_CKD": mt_ckd,
             "lines with cutoff": cutoff_total, "lines with cutoff + MT_CKD": cutoff_total + mt_ckd}

    unfitted = {}
    for name, extra in cases.items():
        unfitted[name] = {}
        for label, operator in (("monochromatic", lambda t: t), ("R45000", convolve)):
            candidate, target = operator(np.exp(-(lines_total + extra))), operator(reference)
            error = np.abs(candidate - target)[inner & (target > 0.05)]
            unfitted[name][label] = {"median": float(np.median(error)),
                                     "p99": float(np.percentile(error, 99)),
                                     "max": float(error.max())}

    convolved_reference = convolve(reference)
    keep = inner & (convolved_reference > 0.05)
    x = (nu - 5010.0) / 10.0
    fitted = {}
    for name, extra in cases.items():
        others = sum(lines_tellurix[s] for s in species if s not in ("H2O", "CO2")) + extra

        def model(p):
            tau = np.exp(p[0]) * lines_tellurix["H2O"] + np.exp(p[1]) * lines_tellurix["CO2"] + others
            return convolve(np.exp(-tau)) * (1.0 + p[2] + p[3] * x)

        fit = least_squares(lambda p: (model(p) - convolved_reference)[keep], np.zeros(4))
        error = np.abs(fit.fun)
        fitted[name] = {"h2o_scale": float(np.exp(fit.x[0])), "co2_scale": float(np.exp(fit.x[1])),
                        "median": float(np.median(error)), "p99": float(np.percentile(error, 99)),
                        "max": float(error.max())}
        print(name, fitted[name], flush=True)

    report = {
        "description": __doc__.split("\n\n")[1].replace("\n", " "),
        "generated_by": "scripts/attribute_lblrtm_gap.py",
        "measured": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "template": str(args.template), "profile": str(args.profile), "runs": str(args.run_dir),
        "window_cm1": WINDOW_CM1,
        "column_cm2": {"note": "LBLRTM's from continuum_all/TAPE6; it auto-layered the levels "
                               "the TAPE5 writer gave it and interpolated water exponentially",
                       **columns},
        "integrated_line_optical_depth_ratio_tellurix_over_lblrtm_4980_5040": integrated,
        "cutoff_residual_rms": residual_rms,
        "transmission_error_unfitted": unfitted,
        "transmission_error_fitted_R45000": {
            "note": "H2O and CO2 column scales and a linear continuum fitted to LBLRTM", **fitted},
        "metric": "absolute transmission error where the reference exceeds 0.05",
    }
    output = root / args.output
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
