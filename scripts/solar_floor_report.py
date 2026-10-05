#!/usr/bin/env python
"""Split the solar residual floor into the flux/intensity part and the rest.

Reads the fits scripts/fit_solar_floor.py writes -- niratl (disc centre) and the
Wallace 2011 and IAG flux atlases over the same windows -- and writes
docs/solar_floor_split.json. The method, in docs/solar_fit_plan.md §4p:

- **Where the floor is measured.** Only pixels the atmosphere leaves alone
  (convolved transmission > 0.995, unmasked), so what is measured is the solar
  model, not the telluric one, and the three spectra's different skies drop
  out. Residuals are in units of the fitted continuum. "Line" pixels are those
  where the fitted solar model is more than 5% deep, "continuum" pixels under
  1%. Each floor is reported raw and with the pixel noise removed in
  quadrature.
- **The split, by variance.** Fitted with the same source and settings, a
  disc-centre floor F_dc and a flux floor F_fx over the same window give a
  flux/intensity part sqrt(F_dc^2 - F_fx^2) and a remainder F_fx. Wallace is the
  reference flux spectrum: same McMath FTS and site as niratl. IAG checks that
  the remainder is not Kitt Peak's.
- **The split, by pattern.** The two residuals are put on one solar frame (the
  shift that best aligns the two fitted solar models) and correlated. The
  shared variance is what both spectra disagree with Payne Zero about, whatever
  the quantity observed; what is not shared belongs to one spectrum. If IAG and
  Wallace -- two instruments -- share their residual, the remainder is the
  solar model's.
- **Line depth.** The observed solar depth, 1 - flux / (continuum x effective
  transmission), regressed through the origin on the model's depth over the
  same line pixels: 1 means Payne Zero's lines are as deep as the Sun's in that
  spectrum, under 1 that they are too deep.
"""
import argparse, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "data/corrected/solar/floor"
C_KMS = 299792.458
CLEAN_TRANSMISSION = 0.995
LINE_DEPTH = 0.05
CONTINUUM_DEPTH = 0.01
MINIMUM_PIXELS = 100


def load_run(name):
    results = []
    for summary in sorted((FLOOR / name).glob("summary*.json")):
        meta = json.loads(summary.read_text())
        results += meta["results"]
    return meta, {round(r["v1"], 1): r for r in results}


def window_arrays(name, row):
    z = np.load(FLOOR / name / row["npz"])
    continuum = z["continuum"]
    star = z["stellar_only_pixels"]
    effective = z["model_flux"] / np.maximum(star, 1e-12)
    depth = 1.0 - star / continuum
    clean = z["mask"] & (effective > CLEAN_TRANSMISSION)
    # The batch layout stores pixels descending in wavenumber; interpolation wants them ascending.
    order = np.argsort(z["wavenumber_cm1"])
    arrays = {
        "nu": z["wavenumber_cm1"], "r": z["residual"] / continuum, "noise": z["uncertainty"] / continuum,
        "depth": depth, "observed_depth": 1.0 - z["flux"] / (continuum * effective),
        "clean": clean, "lines": clean & (depth > LINE_DEPTH), "continuum": clean & (depth < CONTINUUM_DEPTH),
        "mask": z["mask"],
    }
    return {key: value[order] for key, value in arrays.items()}


def floors(a):
    """Raw and noise-removed rms of the residual over the clean, line and continuum pixels."""

    out = {"rms_window": float(np.sqrt(np.mean(a["r"][a["mask"]] ** 2)))}
    for key in ("clean", "lines", "continuum"):
        pixels = a[key]
        out[f"pixels_{key}"] = int(pixels.sum())
        if pixels.sum() < MINIMUM_PIXELS:
            out[f"floor_{key}"] = out[f"floor_{key}_net"] = None
            continue
        rms = float(np.sqrt(np.mean(a["r"][pixels] ** 2)))
        noise = float(np.sqrt(np.mean(a["noise"][pixels] ** 2)))
        out[f"floor_{key}"] = rms
        out[f"floor_{key}_net"] = float(np.sqrt(max(rms**2 - noise**2, 0.0)))
    out["noise"] = float(np.median(a["noise"][a["clean"]])) if a["clean"].any() else None
    lines = a["lines"]
    out["depth_scale"] = (float(np.sum(a["observed_depth"][lines] * a["depth"][lines]) / np.sum(a["depth"][lines] ** 2))
                          if lines.sum() >= MINIMUM_PIXELS else None)
    out["mean_model_depth_lines"] = float(np.mean(a["depth"][lines])) if lines.any() else None
    return out


def solar_shift_kms(a, b):
    """The Doppler factor carrying b's pixels onto a's solar frame: the one that best aligns their solar models."""

    def misfit(v):
        moved = np.interp(a["nu"], b["nu"] * (1.0 + v / C_KMS), b["depth"], left=np.nan, right=np.nan)
        good = np.isfinite(moved)
        return np.mean((moved[good] - a["depth"][good]) ** 2)

    coarse = np.arange(-6.0, 6.0001, 0.05)
    best = coarse[np.argmin([misfit(v) for v in coarse])]
    fine = np.arange(best - 0.06, best + 0.0601, 0.002)
    return float(fine[np.argmin([misfit(v) for v in fine])])


def shared_pattern(a, b):
    """Correlation of two residuals over pixels clean in both, on a's solar frame."""

    v = solar_shift_kms(a, b)
    nu_b = b["nu"] * (1.0 + v / C_KMS)
    r_b = np.interp(a["nu"], nu_b, b["r"], left=np.nan, right=np.nan)
    clean_b = np.interp(a["nu"], nu_b, b["clean"].astype(float), left=0.0, right=0.0) > 0.999
    out = {"solar_shift_kms": v}
    for key in ("lines", "clean"):
        both = a[key] & clean_b & np.isfinite(r_b)
        if both.sum() < MINIMUM_PIXELS:
            out[f"correlation_{key}"] = None
            continue
        x, y = a["r"][both], r_b[both]
        out[f"correlation_{key}"] = float(np.corrcoef(x, y)[0, 1])
        out[f"pixels_{key}"] = int(both.sum())
        # rms of what is left of a's residual once b's is taken out at the best scale.
        scale = float(np.sum(x * y) / np.sum(y * y))
        out[f"scale_{key}"] = scale
        out[f"unshared_rms_{key}"] = float(np.sqrt(np.mean((x - scale * y) ** 2)))
    return out


def summarize(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if values.size == 0:
        return None
    return {"n": int(values.size), "median": float(np.median(values)),
            "p16": float(np.percentile(values, 16)), "p84": float(np.percentile(values, 84))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--disc-centre", default="niratl")
    parser.add_argument("--flux", default="wallace2011")
    parser.add_argument("--second-flux", default="iag")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/solar_floor_split.json")
    args = parser.parse_args()

    names = {"disc_centre": args.disc_centre, "flux": args.flux, "second_flux": args.second_flux}
    runs = {role: load_run(name) for role, name in names.items()}
    common = sorted(set.intersection(*(set(k for k, r in rows.items() if "npz" in r) for _, rows in runs.values())))
    windows = []
    for v1 in common:
        rows = {role: runs[role][1][v1] for role in names}
        arrays = {role: window_arrays(names[role], rows[role]) for role in names}
        entry = {"v1": v1, "v2": rows["disc_centre"]["v2"], "page": rows["disc_centre"]["page"],
                 "species": rows["disc_centre"]["species"],
                 "negligible_telluric": rows["disc_centre"]["negligible_telluric"]}
        for role in names:
            entry[role] = {**floors(arrays[role]),
                           "lsf_sigma_kms": rows[role]["parameters"]["lsf_sigma_kms"],
                           "median_transmission": rows[role]["median_transmission"],
                           "at_bound": rows[role]["at_bound"]}
        dc, fx = entry["disc_centre"]["floor_lines_net"], entry["flux"]["floor_lines_net"]
        if dc is not None and fx is not None and dc > 0:
            entry["split_lines"] = {"flux_intensity": float(np.sqrt(max(dc**2 - fx**2, 0.0))), "remainder": fx,
                                    "flux_intensity_variance_fraction": float((dc**2 - fx**2) / dc**2)}
        entry["pattern"] = {"disc_centre_vs_flux": shared_pattern(arrays["disc_centre"], arrays["flux"]),
                            "flux_vs_second_flux": shared_pattern(arrays["flux"], arrays["second_flux"]),
                            "disc_centre_vs_second_flux": shared_pattern(arrays["disc_centre"], arrays["second_flux"])}
        windows.append(entry)

    usable = [w for w in windows if "split_lines" in w]
    # Pooled over windows: the sums of squares, weighted by each window's line pixels.
    def pooled(role, among):
        among = [w for w in among if w[role]["floor_lines_net"] is not None]
        num = sum(w[role]["floor_lines_net"] ** 2 * w[role]["pixels_lines"] for w in among)
        return float(np.sqrt(num / sum(w[role]["pixels_lines"] for w in among)))

    pooled_dc, pooled_fx = pooled("disc_centre", usable), pooled("flux", usable)
    # IAG's sky is heavier, so it has fewer clean line pixels: pooled over the windows where it has enough,
    # beside Wallace over the same windows.
    with_second = [w for w in usable if w["second_flux"]["floor_lines_net"] is not None]
    pooled_fx2, pooled_fx_same = pooled("second_flux", with_second), pooled("flux", with_second)
    summary = {
        "windows_fitted_in_all_three": len(windows),
        "windows_with_line_floor_in_both": len(usable),
        "pooled_line_floor_net": {"disc_centre": pooled_dc, "flux": pooled_fx},
        "pooled_line_floor_net_where_second_flux_has_lines": {"windows": len(with_second), "second_flux": pooled_fx2,
                                                              "flux": pooled_fx_same},
        "pooled_split_lines": {"flux_intensity": float(np.sqrt(max(pooled_dc**2 - pooled_fx**2, 0.0))),
                               "remainder": pooled_fx,
                               "flux_intensity_variance_fraction": float((pooled_dc**2 - pooled_fx**2) / pooled_dc**2)},
        "per_window": {
            **{f"{role}_{key}": summarize(w[role][key] for w in windows)
               for role in names for key in ("rms_window", "floor_clean_net", "floor_lines_net",
                                             "floor_continuum_net", "depth_scale", "lsf_sigma_kms")},
            "flux_intensity_variance_fraction": summarize(w["split_lines"]["flux_intensity_variance_fraction"]
                                                          for w in usable),
            **{f"{key}_{pair}": summarize(w["pattern"][pair].get(key) for w in windows)
               for pair in ("disc_centre_vs_flux", "flux_vs_second_flux", "disc_centre_vs_second_flux")
               for key in ("correlation_lines", "scale_lines", "unshared_rms_lines")},
        },
    }
    meta = {role: runs[role][0] for role in names}
    report = {
        "question": "How much of the solar fits' ~1% residual floor is Payne Zero's flux standing in for "
                    "disc-centre intensity (GitHub #2)",
        "method": __doc__.split("The method, in docs/solar_fit_plan.md §4p:")[1].strip(),
        "spectra": {role: {"name": names[role], "spectrum": meta[role]["spectrum"], "sha256": meta[role]["sha256"],
                           "settings": meta[role]["settings"]} for role in names},
        "thresholds": {"clean_transmission": CLEAN_TRANSMISSION, "line_depth": LINE_DEPTH,
                       "continuum_depth": CONTINUUM_DEPTH, "minimum_pixels": MINIMUM_PIXELS},
        "summary": summary,
        "windows": windows,
        "driver": "solar_floor_report.py",
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
