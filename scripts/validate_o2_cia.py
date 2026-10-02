#!/usr/bin/env python
"""tellurix's O2 collision-induced continuum against LBLRTM 12.17's, band by band.

LBLRTM's O2 continuum alone is the difference between a run with every continuum
but Rayleigh (ICNTNM=5) and one with none (ICNTNM=0), on an O2-only profile where
nothing else can contribute. Vertical path, so geometry plays no part. Writes
docs/o2_cia_validation.json. Needs an O2 LNFL line file over 7400-21000 cm-1 in
data/lblrtm/run_lnfl_o2_wide (see compare_o2_lblrtm.py for the recipe).

    uv run python scripts/validate_o2_cia.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from tellurix import (LBLRTMRunConfig, O2CollisionInducedContinuum, load_atmosphere_csv,
                      run_lblrtm)

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data/lblrtm"
# One window per term, at its peak, plus the 577 nm visible band.
WINDOWS = {"O2INF1 1.27um": (7800.0, 7950.0), "O2INF2 1.06um": (9330.0, 9480.0),
           "O2INF3 A-band": (13050.0, 13200.0), "O2_VIS 630nm": (15800.0, 15950.0),
           "O2_VIS 577nm": (17250.0, 17400.0)}


def main() -> None:
    full = load_atmosphere_csv(ROOT / "data/profiles/kitt_peak_19830626_file3.csv")
    profile = dataclasses.replace(full, vmr={"O2": full.vmr["O2"]})
    report = {"about": __doc__.split("\n\n")[0], "profile": "kitt_peak_19830626_file3.csv, O2 only",
              "path": "vertical", "windows": {}}
    for label, (v1, v2) in WINDOWS.items():
        taus = {}
        for flag in (0, 5):
            run = run_lblrtm(REF / f"run_o2cia_{int(v1)}_c{flag}", profile,
                             LBLRTMRunConfig(v1 - 5.0, v2 + 5.0, 0.0, flag, f"O2 CIA {label} c{flag}"),
                             REF / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl",
                             REF / "run_lnfl_o2_wide/TAPE3",
                             REF / "LBLRTM/data/absco-ref_wv-mt-ckd.nc")
            taus[flag] = (run.wavenumber_cm1, -np.log(np.clip(run.transmission, 1e-300, None)),
                          run.transmission)
        nu = np.linspace(v1, v2, 1501)
        lbl = np.interp(nu, *taus[5][:2]) - np.interp(nu, *taus[0][:2])
        # In saturated cores LBLRTM's single-precision transmission underflows
        # to zero in both runs, and the difference of the clipped depths reads
        # as no continuum; just above underflow, a 0.02 difference between two
        # depths near 50 is lost to float32. Compare where T > 1e-8 (depth < 18).
        alive = ((np.interp(nu, taus[0][0], taus[0][2]) > 1e-8)
                 & (np.interp(nu, taus[5][0], taus[5][2]) > 1e-8))
        bound = O2CollisionInducedContinuum(nu).bind(profile)
        mine = np.asarray(bound.optical_depth(profile, {"O2": profile.vmr["O2"]})).sum(axis=0)
        nu, lbl, mine = nu[alive], lbl[alive], mine[alive]
        peak = float(np.max(lbl))
        entry = {"pixels_compared": int(alive.sum()), "lblrtm_peak_tau": peak, "tellurix_peak_tau": float(np.max(mine)),
                 "integrated_ratio": float(np.trapz(mine, nu) / np.trapz(lbl, nu)),
                 "max_abs_difference_over_peak": float(np.max(np.abs(mine - lbl)) / peak)}
        report["windows"][label] = entry
        print(f"{label:16} LBLRTM peak tau {peak:.3e}  tellurix {entry['tellurix_peak_tau']:.3e}  "
              f"integrated ratio {entry['integrated_ratio']:.4f}  max |diff|/peak "
              f"{entry['max_abs_difference_over_peak']:.4f}", flush=True)
    (ROOT / "docs/o2_cia_validation.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
