#!/usr/bin/env python
"""Can a temperature-profile error explain the O2 excess of the June 1983 fits?

A window's high-J O2 lines (lower-state energy 1000-2000 cm-1) strengthen by
~3.5% per kelvin while a saturated window's lines barely move, so a profile
too cold would show as an O2 scale rising toward the weak windows. For each O2
window fitted in `ftsspec_830626_{2,3}`, this computes the O2 scale with which
tellurix at the profile's temperatures reproduces its own transmission at
+2 K (least squares over pixels above the fits' 0.15 transmission mask), then
fits ln(fitted scale) = factor per file (and band) + sensitivity * dT.
Columns are hydrostatic, so a temperature change moves no molecules. The
weakest windows carry most of the leverage and the most solar contamination,
so the fit is repeated above three depth cuts. docs/solar_fit_plan.md §4n;
writes docs/o2_temperature_sensitivity.json.

    uv run python scripts/o2_temperature_sensitivity.py [OUT.json]
"""
import dataclasses, json, sys
from pathlib import Path
import numpy as np
from tellurix import (AERLineDatabase, ExoJAXOpacityBackend, TelluricModel, TelluricParameters,
    constant_velocity_grid, load_atmosphere_csv, trim_wavenumber_grid)
from tellurix_fts.nso import zenith_angle_deg_for_airmass

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/o2_temperature_sensitivity.json"
AER_O2 = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule/07_O2/07_O2"
FILES = {"ftsspec_830626_2": 5.37, "ftsspec_830626_3": 3.09}
FWHM = 0.04167; DT = 2.0; MASK = 0.15
DEPTH_CUTS = (0.0, 0.005, 0.02)


def band(v1):
    return "A" if v1 < 13300 else ("B" if v1 < 14600 else "gamma")


def regression(rows, per_band):
    keys = sorted({(r["airmass"], band(r["v1"]) if per_band else "all") for r in rows})
    M = np.zeros((len(rows), len(keys) + 1)); y = np.log([r["fitted"] for r in rows])
    for i, r in enumerate(rows):
        M[i, keys.index((r["airmass"], band(r["v1"]) if per_band else "all"))] = 1.0
        M[i, -1] = r["scale_per_k"]
    p, *_ = np.linalg.lstsq(M, y, rcond=None); res = y - M @ p
    q, *_ = np.linalg.lstsq(M[:, :-1], y, rcond=None)
    cov = np.linalg.inv(M.T @ M) * res.var(ddof=len(p))
    return {"windows": len(rows), "delta_t_k": float(p[-1]), "delta_t_sigma_k": float(np.sqrt(cov[-1, -1])),
            "factors": {f"X={k[0]} {k[1]}": float(np.exp(v)) for k, v in zip(keys, p[:-1])},
            "rms": float(res.std()), "rms_without_delta_t": float((y - M[:, :-1] @ q).std())}


def main():
    full = load_atmosphere_csv(ROOT / "data/profiles/kitt_peak_19830626_file3.csv")
    prof = dataclasses.replace(full, vmr={"O2": full.vmr["O2"]})
    warm = dataclasses.replace(prof, temperature_k=prof.temperature_k + DT)
    fitted = {}
    for name, airmass in FILES.items():
        for r in json.load(open(ROOT / f"data/corrected/solar/{name}_summary.json"))["results"]:
            if "O2" in r.get("species", []) and r.get("parameters") and (
                    12900 < r["v1"] < 13200 or 14300 < r["v1"] < 14560 or 15700 < r["v1"] < 15950):
                fitted.setdefault(r["v1"], {})[airmass] = float(np.exp(r["parameters"]["O2"]))
    rows = []
    for v1 in sorted(fitted):
        v2 = v1 + 30.0
        grid = trim_wavenumber_grid(constant_velocity_grid(1e7 / v2, 1e7 / v1, resolving_power=0.5 * (v1 + v2) / FWHM,
                                    samples_per_resolution=4.0, margin_cm1=25.0), v1, v2, 5.0)
        db = {"O2": AERLineDatabase(AER_O2, "O2", (v1, v2), margin_cm1=25.0)}
        models = {}
        for name, p in (("base", prof), ("warm", warm)):
            opacity = ExoJAXOpacityBackend.prepare(db, grid, methods="direct_sparse",
                temperature_range_k=(float(p.temperature_k.min()) - 1, float(p.temperature_k.max()) + 1),
                maximum_pressure_bar=float(p.pressure_layer_bar.max()), vectorize_layers=True,
                mixed_precision=True, pressure_shift=True)
            models[name] = TelluricModel(p, grid, opacity, accuracy_mode="fast").precompute_opacity(self_broadening="linear")
        nu = np.asarray(models["base"].wavenumber_cm1); inside = (nu >= v1) & (nu <= v2)

        def transmission(model, scale, airmass):
            p = TelluricParameters(log_column_scales={"O2": float(np.log(scale))}, velocity_kms=0.0,
                                   wavelength_stretch=0.0, lsf_sigma_kms=0.5, continuum_coeffs=np.zeros(1), log_jitter=0.0)
            return np.asarray(model.transmission(p, zenith_angle_deg_for_airmass(airmass)))[inside]
        scales = np.linspace(0.80, 1.30, 101)
        for airmass, value in fitted[v1].items():
            curve = np.array([transmission(models["base"], s, airmass) for s in scales])
            reference = transmission(models["warm"], 1.0, airmass); keep = reference > MASK
            lsq = ((curve[:, keep] - reference[keep]) ** 2).sum(1); k = int(np.argmin(lsq))
            a, b, c = lsq[k - 1:k + 2]
            best = scales[k] + 0.5 * (a - c) / (a - 2 * b + c) * (scales[1] - scales[0])
            rows.append({"v1": v1, "band": band(v1), "airmass": airmass, "fitted": value,
                         "mean_depth": float(1 - curve[np.argmin(abs(scales - 1))].mean()),
                         "scale_per_k": float((best - 1) / DT)})
            print(json.dumps(rows[-1]), flush=True)
    fits = {}
    for cut in DEPTH_CUTS:
        # Scales outside 0.9-1.25 are windows with almost no O2 whose column absorbed solar error.
        use = [r for r in rows if r["mean_depth"] > cut and 0.9 < r["fitted"] < 1.25]
        fits[f"depth>{cut}"] = {"common": regression(use, False), "per_band": regression(use, True),
                                **{f"{b} only": regression([r for r in use if r["band"] == b], False)
                                   for b in ("A", "B", "gamma") if sum(r["band"] == b for r in use) > 4}}
    OUTPUT.write_text(json.dumps({"about": __doc__.split("\n\n")[0] + " scripts/o2_temperature_sensitivity.py.",
                                  "delta_t_k": DT, "windows": rows, "fits": fits}, indent=1) + "\n")
    print(json.dumps(fits, indent=1))


if __name__ == "__main__":
    main()
