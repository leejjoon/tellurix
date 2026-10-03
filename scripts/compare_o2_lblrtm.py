#!/usr/bin/env python
"""O2 A- and B-band absorption: tellurix against LBLRTM 12.17, same AER lines.

For each window, the equivalent width (integral of 1 - T, unchanged by an
instrument convolution and what a column fit matches) from LBLRTM, and the O2
column scale tellurix needs to reproduce it. A scale above 1 means tellurix
under-absorbs there, which a fit to the sky turns into an O2 excess. See
docs/solar_fit_plan.md §4n and docs/o2_lblrtm_comparison.json.

Needs two O2-only LNFL line files, with and without AER's line coupling
(LNFL option NOCPL), over 12750-14850 cm-1:

    cd data/lblrtm && mkdir run_lnfl_o2 && cd run_lnfl_o2
    printf 'O2\n 12750.000 14850.000\n%-47s    %s\n' "$(printf '0000001%040d' 0)" "" > TAPE5
    ln -s ../LNFL/lnfl_v3.2_linux_gnu_sgl lnfl
    ln -s ../AER_Line_File/aer_v_3.9/line_file/aer_v_3.9 TAPE1 && ./lnfl
    # and again in run_lnfl_o2_nocpl with NOCPL as the option

    uv run python scripts/compare_o2_lblrtm.py OUT.json AIRMASS [cia]

Without `cia`, three LBLRTM variants: coupled lines, uncoupled lines, lines
with all continua (Rayleigh dominates that one). With `cia`, coupled lines and
all continua except Rayleigh (ICNTNM=5), which isolates O2 collision-induced
absorption.
"""
import dataclasses, json, sys
from pathlib import Path
import numpy as np
from tellurix import (AERLineDatabase, ExoJAXOpacityBackend, LBLRTMRunConfig, TelluricModel,
    TelluricParameters, constant_velocity_grid, load_atmosphere_csv, run_lblrtm, trim_wavenumber_grid)
from tellurix_fts.nso import zenith_angle_deg_for_airmass

ROOT = Path("/home/jjlee/work/lblrtm"); REF = ROOT / "data/lblrtm"
AIRMASS = float(sys.argv[2]) if len(sys.argv) > 2 else 5.37; ZEN = zenith_angle_deg_for_airmass(AIRMASS); FWHM = 0.04167
full = load_atmosphere_csv(ROOT / "data/profiles/kitt_peak_19830626_file3.csv")
prof = dataclasses.replace(full, vmr={"O2": full.vmr["O2"]})
WINDOWS = {"A": [13000.0, 13060.0, 13120.0], "B": [14410.0, 14440.0, 14470.0, 14500.0]}
VARIANTS = ({"coupled": ("run_lnfl_o2", 0), "no-rayleigh continua": ("run_lnfl_o2", 5)} if len(sys.argv) > 3
            else {"coupled": ("run_lnfl_o2", 0), "uncoupled": ("run_lnfl_o2_nocpl", 0), "coupled+continua": ("run_lnfl_o2", 1)})
line = AERLineDatabase  # noqa
out = []
for band, starts in WINDOWS.items():
    for v1 in starts:
        v2 = v1 + 30.0
        grid = trim_wavenumber_grid(constant_velocity_grid(1e7/v2, 1e7/v1, resolving_power=0.5*(v1+v2)/FWHM,
                                    samples_per_resolution=4.0, margin_cm1=25.0), v1, v2, 5.0)
        db = {"O2": AERLineDatabase(REF/"AER_Line_File/aer_v_3.9/line_files_By_Molecule/07_O2/07_O2", "O2", (v1, v2), margin_cm1=25.0)}
        opacity = ExoJAXOpacityBackend.prepare(db, grid, methods="direct_sparse",
            temperature_range_k=(float(prof.temperature_k.min()), float(prof.temperature_k.max())),
            maximum_pressure_bar=float(prof.pressure_layer_bar.max()), vectorize_layers=True, mixed_precision=True, pressure_shift=True)
        model = TelluricModel(prof, grid, opacity, accuracy_mode="fast").precompute_opacity(self_broadening="linear")
        nu = np.asarray(model.wavenumber_cm1); inside = (nu >= v1) & (nu <= v2)
        def ew_tx(scale):
            p = TelluricParameters(log_column_scales={"O2": float(np.log(scale))}, velocity_kms=0.0, wavelength_stretch=0.0,
                                   lsf_sigma_kms=0.5, continuum_coeffs=np.zeros(1), log_jitter=0.0)
            t = np.asarray(model.transmission(p, ZEN)); return float(np.trapz(1 - t[inside], nu[inside]))
        scales = np.linspace(0.85, 1.20, 36); ews = np.array([ew_tx(s) for s in scales])
        row = {"band": band, "v1": v1, "ew_tellurix_1": ew_tx(1.0)}
        for name, (tape, cont) in VARIANTS.items():
            lbl = run_lblrtm(REF / f"run_o2cmp_{name.replace(' ','_')}_{int(v1)}_X{AIRMASS:.2f}", prof,
                LBLRTMRunConfig(v1 - 2.0, v2 + 2.0, ZEN, cont, f"O2 {band} {v1:.0f} {name}"),
                REF / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl", REF / tape / "TAPE3", REF / "LBLRTM/data/absco-ref_wv-mt-ckd.nc")
            w = lbl.wavenumber_cm1; m = (w >= v1) & (w <= v2)
            ew = float(np.trapz(1 - lbl.transmission[m], w[m]))
            row[f"ew_{name}"] = ew
            row[f"scale_{name}"] = float(np.interp(ew, ews, scales)) if ews[0] <= ew <= ews[-1] else float("nan")
        out.append(row)
        print(f"{band} {v1:.0f} X={AIRMASS}: EW tellurix {row['ew_tellurix_1']:.4f} | " + " | ".join(
              f"{k}: EW {row['ew_'+k]:.4f} scale {row['scale_'+k]:.3f}" for k in VARIANTS), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
for band in WINDOWS:
    r = [x for x in out if x["band"] == band]
    print(band, {k: round(float(np.nanmedian([x[f"scale_{k}"] for x in r])), 4) for k in VARIANTS})
