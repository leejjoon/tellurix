#!/usr/bin/env python
"""O2 A- and B-band absorption: AER v3.9 line parameters against HITRAN's.

Tests whether AER's O2 intensities (or widths) explain the O2 excess the solar
fits ask for (docs/solar_fit_plan.md §4n). Two parts, written to
docs/o2_hitran_comparison.json:

- the line parameters of each band, AER over HITRAN, matched line by line on
  isotopologue, position and lower-state energy;
- the O2 column scale that makes tellurix with AER's lines reproduce the
  equivalent width it gives with HITRAN's strengths, widths (with their
  temperature exponent) and shifts substituted -- the windows, profile and
  sinc of scripts/compare_o2_lblrtm.py, at the air masses of niratl and the two
  1983 files. A scale above 1 means HITRAN absorbs more, and would lower the
  fitted O2 excess by that factor.

HITRAN's line-by-line service needs no API key for .par records. The files are
fetched once into data/databases/hitran/ (gitignored), with the date in FETCHED,
since the service returns whatever edition is current.

    uv run python scripts/compare_o2_hitran.py [docs/o2_hitran_comparison.json]
"""
import dataclasses, datetime, json, sys, urllib.request
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from tellurix import (AERLineDatabase, ExoJAXOpacityBackend, TelluricModel, TelluricParameters,
    constant_velocity_grid, load_atmosphere_csv, trim_wavenumber_grid)
from tellurix_fts.nso import zenith_angle_deg_for_airmass

ROOT = Path(__file__).resolve().parents[1]
AER_O2 = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule/07_O2/07_O2"
HITRAN = ROOT / "data/databases/hitran"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/o2_hitran_comparison.json"
BANDS = {"a1Delta 1.06um": (9200.0, 9600.0), "A 760nm": (12950.0, 13200.0),
         "B 687nm": (14300.0, 14600.0), "gamma 628nm": (15700.0, 16100.0)}
FETCH = [(9200, 9600), (12750, 14850), (15700, 16100)]
WINDOWS = {"A": [13000.0, 13060.0, 13120.0], "B": [14410.0, 14440.0, 14470.0, 14500.0]}
AIRMASSES = [1.10, 3.09, 5.37]
VARIANTS = {"strength": ("S",), "width": ("width",), "all": ("S", "width", "shift")}
FWHM = 0.04167


def fetch():
    HITRAN.mkdir(parents=True, exist_ok=True)
    for v1, v2 in FETCH:
        path = HITRAN / f"O2_{v1}_{v2}.par"
        if not path.exists():
            url = f"https://hitran.org/lbl/api?iso_ids_list=36,37,38&numin={v1}&numax={v2}"
            path.write_bytes(urllib.request.urlopen(url, timeout=300).read())
            (HITRAN / "FETCHED").write_text(datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
    return (HITRAN / "FETCHED").read_text().strip()


def read_par(path, molecule="7"):
    # HITRAN 160-character and AER 100-character records share the first 67 columns;
    # AER writes some exponents with D.
    rows = []
    for line in open(path):
        if len(line.rstrip("\n")) < 67 or line[:2].strip() != molecule:
            continue
        f = lambda a, b: float(line[a:b].replace("D", "E"))
        try:
            rows.append((int(line[2]), f(3, 15), f(15, 25), f(35, 40), f(40, 45), f(45, 55), f(55, 59), f(59, 67)))
        except ValueError:  # AER's line-coupling continuation records
            continue
    return np.array(rows)  # iso, nu, S, gamma_air, gamma_self, E'', n_air, delta_air


def match(aer, hit):
    """Index of the HITRAN line for each AER line, -1 where none.

    Position alone is not enough: weak high-J lines moved by up to 0.05 cm-1, so
    the lower-state energy, which both lists share to 0.01 cm-1, disambiguates."""
    index = np.full(len(aer), -1)
    for i, (iso, nu, e) in enumerate(aer[:, [0, 1, 5]]):
        c = np.flatnonzero((hit[:, 0] == iso) & (abs(hit[:, 1] - nu) < 0.06) & (abs(hit[:, 5] - e) < 1.0))
        if len(c):
            index[i] = c[np.argmin(abs(hit[c, 1] - nu))]
    return index


def line_parameters(aer, hit):
    out = {}
    for name, (a, b) in BANDS.items():
        band = {}
        for iso in (1, 2, 3):
            A = aer[(aer[:, 0] == iso) & (aer[:, 1] > a) & (aer[:, 1] < b)]
            H = hit[(hit[:, 0] == iso) & (hit[:, 1] > a) & (hit[:, 1] < b)]
            if not len(A):
                continue
            row = {"lines_aer": len(A), "lines_hitran": len(H),
                   "band_strength_ratio": float(A[:, 2].sum() / H[:, 2].sum())}
            if iso == 1:
                j = match(A, H); ok = j >= 0
                strong = ok & (A[:, 2] > 0.01 * A[:, 2].max()); w = A[strong, 2]
                row["matched"] = int(ok.sum()); row["strong"] = int(strong.sum())
                for k, label in ((2, "strength"), (3, "gamma_air"), (4, "gamma_self"), (6, "n_air")):
                    r = A[strong, k] / H[j[strong], k]
                    row[f"{label}_ratio_weighted"] = float(np.sum(r * w) / w.sum())
                    row[f"{label}_ratio_range"] = [float(r.min()), float(r.max())]
                row["delta_air_difference_weighted"] = float(np.sum((A[strong, 7] - H[j[strong], 7]) * w) / w.sum())
            band[f"iso{iso}"] = row
        out[name] = band
    return out


def substitute(db, hit, fields):
    j = match(np.column_stack([db.isoid, db.nu_lines, db.line_strength_ref_original, db.gamma_air,
                               db.gamma_self, db.elower, db.n_air, db.delta_air]), hit)
    ok = j >= 0
    columns = {"S": [("line_strength_ref_original", 2)],
               "width": [("gamma_air", 3), ("gamma_self", 4), ("n_air", 6)], "shift": [("delta_air", 7)]}
    for field in fields:
        for attribute, k in columns[field]:
            values = np.array(getattr(db, attribute), dtype=float); values[ok] = hit[j[ok], k]
            setattr(db, attribute, values)
    db.logsij0 = jnp.log(jnp.asarray(db.line_strength_ref_original))
    return int(ok.sum())


def main():
    fetched = fetch()
    hit = np.vstack([read_par(p) for p in sorted(HITRAN.glob("O2_*.par"))])
    report = {"about": "O2 line parameters, AER v3.9 over HITRAN, and the O2 column scale that makes tellurix "
                       "with AER's lines reproduce the equivalent width it gives with HITRAN's. "
                       "scripts/compare_o2_hitran.py; docs/solar_fit_plan.md section 4n.",
              "hitran_fetched": fetched, "line_parameters": line_parameters(read_par(AER_O2), hit)}
    full = load_atmosphere_csv(ROOT / "data/profiles/kitt_peak_19830626_file3.csv")
    prof = dataclasses.replace(full, vmr={"O2": full.vmr["O2"]})
    windows = []
    for band, starts in WINDOWS.items():
        for v1 in starts:
            v2 = v1 + 30.0
            grid = trim_wavenumber_grid(constant_velocity_grid(1e7 / v2, 1e7 / v1, resolving_power=0.5 * (v1 + v2) / FWHM,
                                        samples_per_resolution=4.0, margin_cm1=25.0), v1, v2, 5.0)
            models = {}
            for name, fields in [("aer", ())] + list(VARIANTS.items()):
                db = AERLineDatabase(AER_O2, "O2", (v1, v2), margin_cm1=25.0)
                if fields:
                    substitute(db, hit, fields)
                opacity = ExoJAXOpacityBackend.prepare({"O2": db}, grid, methods="direct_sparse",
                    temperature_range_k=(float(prof.temperature_k.min()), float(prof.temperature_k.max())),
                    maximum_pressure_bar=float(prof.pressure_layer_bar.max()), vectorize_layers=True,
                    mixed_precision=True, pressure_shift=True)
                models[name] = TelluricModel(prof, grid, opacity, accuracy_mode="fast").precompute_opacity(self_broadening="linear")
            nu = np.asarray(models["aer"].wavenumber_cm1); inside = (nu >= v1) & (nu <= v2)

            def ew(model, scale, airmass):
                p = TelluricParameters(log_column_scales={"O2": float(np.log(scale))}, velocity_kms=0.0,
                                       wavelength_stretch=0.0, lsf_sigma_kms=0.5, continuum_coeffs=np.zeros(1), log_jitter=0.0)
                t = np.asarray(model.transmission(p, zenith_angle_deg_for_airmass(airmass)))
                return float(np.trapz(1 - t[inside], nu[inside]))
            scales = np.linspace(0.85, 1.25, 41)
            for airmass in AIRMASSES:
                curve = np.array([ew(models["aer"], s, airmass) for s in scales])
                row = {"band": band, "v1": v1, "airmass": airmass, "ew_aer": ew(models["aer"], 1.0, airmass)}
                for name in VARIANTS:
                    row[f"ew_{name}"] = ew(models[name], 1.0, airmass)
                    row[f"scale_{name}"] = float(np.interp(row[f"ew_{name}"], curve, scales))
                windows.append(row)
                print(json.dumps(row), flush=True)
    report["windows"] = windows
    report["summary"] = {f"{band} X={airmass}": {name: float(np.mean([w[f"scale_{name}"] for w in windows
                         if w["band"] == band and w["airmass"] == airmass])) for name in VARIANTS}
                         for band in WINDOWS for airmass in AIRMASSES}
    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
