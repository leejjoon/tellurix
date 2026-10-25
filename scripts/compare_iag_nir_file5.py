#!/usr/bin/env python
"""The IAG NIR co-add's correction against Kitt Peak file 5's, on file 5's pixels.

Scored separately, IAG NIR is less flat than file 5 where the two overlap (1.27% against
0.86% at the median). This asks whether that is the instrument or the sky. Every series
is put on file 5's pixels and scored with `score_telluric_correction.py`'s selection
there (file 5's transmission relative to its 99th percentile, its solar model, photatl's
filled pixels), less any pixel within three file-5 FWHM of a pixel the IAG fit masked:

- `iag_native`: the IAG correction, interpolated.
- `iag_degraded`: IAG's atlas, model and solar model each cut to file 5's sinc
  (`tellurix_fts.flatness.truncate_to_sinc`, exact for IAG's longer boxcar) and the
  correction formed afterwards, which is what a fit of the degraded spectrum would give.
- `f5_ours`, `photatl`: our file 5 correction and the authors'.

Then the median telluric optical depth of each atmosphere at the scored pixels
(-ln T, relative), because a model error that is a fixed fraction of the depth leaves
a correction error that grows with it.

    uv run python scripts/compare_iag_nir_file5.py --output docs/iag_nir_file5_comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tellurix_fts.flatness import (BINS, EDGE_PIXELS, MIN_TRANSMISSION, corrected, photatl_solar, scatter,
                                   truncate_to_sinc)
from tellurix_fts.nso import read_photatl_page

ATLASES = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/nso")
FILE5_FWHM_CM1 = 0.01753
NAMES = ("iag_native", "iag_degraded", "f5_ours", "photatl")
REGIONS = ((4366, 5000), (5000, 6000), (6000, 7000), (7000, 8000), (8000, 9000))


def iag_series(path: Path):
    with np.load(path) as z:
        order = np.argsort(z["wavenumber_cm1"])
        t = {k: np.asarray(z[k])[order] for k in ("wavenumber_cm1", "flux", "model_flux", "stellar_only_pixels",
                                                   "continuum", "mask", "transmission_pixels")}
    nu, mask = t["wavenumber_cm1"], t["mask"].astype(bool)
    # Masked pixels hold 1.0 placeholders; the model stands in so they do not ring.
    flux = np.where(mask, t["flux"], t["model_flux"])
    native = flux / (t["model_flux"] / t["stellar_only_pixels"]) / t["continuum"]
    cut = {k: truncate_to_sinc(nu, v, FILE5_FWHM_CM1)
           for k, v in (("flux", flux), ("model", t["model_flux"]), ("star", t["stellar_only_pixels"]))}
    degraded = cut["flux"] / (cut["model"] / cut["star"]) / t["continuum"]
    reach = int(round(3 * FILE5_FWHM_CM1 / abs(nu[1] - nu[0])))
    near_masked = np.convolve((~mask).astype(float), np.ones(2 * reach + 1), mode="same") > 0
    return nu, native, degraded, near_masked, t["transmission_pixels"], mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file5", type=Path, default=Path("data/corrected/solar"))
    parser.add_argument("--iag", type=Path, default=Path("data/corrected/solar/iag_nir"))
    parser.add_argument("--photatl", type=Path, default=ATLASES / "photatl")
    parser.add_argument("--minimum-pixels", type=int, default=200,
                        help="telluric pixels a window needs on file 5's grid (deep: a sixth of it)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    photatl = [read_photatl_page(p) for p in sorted(args.photatl.glob("wn*")) if p.name[2:].isdigit()]
    pages = {}
    for f5 in sorted(args.file5.glob("ftsspec_901218_5_*.npz")):
        v1 = f5.stem.rsplit("_", 1)[1]
        iag = args.iag / f"iag_nir_{v1}.npz"
        if not iag.exists():
            continue
        nu, flat, solar, transmission, mask = corrected(f5)
        inu, native, degraded, near_masked, iag_t, iag_mask = iag_series(iag)
        authors, filled = photatl_solar(photatl, nu)
        clean = (mask & (solar > 0.995) & (transmission >= MIN_TRANSMISSION) & ~filled & np.isfinite(authors)
                 & (np.interp(nu, inu, near_masked.astype(float)) == 0))
        if not mask.any():
            continue
        transmission = transmission / np.percentile(transmission[mask], 99)
        iag_t = np.interp(nu, inu, iag_t / np.percentile(iag_t[iag_mask], 99))
        reference = clean & (transmission > 0.995)
        if reference.sum() < 50:
            continue
        series = {"iag_native": np.interp(nu, inu, native), "iag_degraded": np.interp(nu, inu, degraded),
                  "f5_ours": flat, "photatl": authors / np.median(authors[reference])}
        interior = np.zeros(nu.size, bool)
        interior[EDGE_PIXELS:-EDGE_PIXELS] = True
        row = {}
        for label, ceiling in BINS.items():
            scored = clean & interior & (transmission < ceiling)
            if scored.sum() < (args.minimum_pixels if label == "telluric" else args.minimum_pixels // 6):
                continue
            depth = lambda t: float(np.median(-np.log(np.clip(t[scored], 1e-3, None))))
            row[label] = {"pixels": int(scored.sum()), "tau_iag": depth(iag_t), "tau_f5": depth(transmission),
                          **{name: scatter(nu, values, reference, scored) for name, values in series.items()}}
        if row:
            pages[f"{int(float(v1))}"] = row

    summary = {}
    for label in BINS:
        keys = [k for k in pages if label in pages[k]]
        get = lambda name: np.array([pages[k][label][name] for k in keys])
        ti, tf = get("tau_iag"), get("tau_f5")
        per_tau_iag, per_tau_f5 = get("iag_degraded") / ti, get("f5_ours") / tf
        summary[label] = {
            "windows": len(keys),
            "median_percent": {n: float(np.median(get(n))) for n in NAMES},
            "median_tau": {"iag": float(np.median(ti)), "f5": float(np.median(tf)),
                           "ratio_per_window": float(np.median(ti / tf))},
            "median_percent_per_unit_tau": {"iag_degraded": float(np.median(per_tau_iag)),
                                            "f5_ours": float(np.median(per_tau_f5))},
            "windows_iag_degraded_at_or_below_f5_ours": int((get("iag_degraded") <= get("f5_ours")).sum()),
            "windows_iag_better_per_unit_tau": int((per_tau_iag <= per_tau_f5).sum()),
            "regions": {},
        }
        print(f"{label}: {len(keys)} windows  " + "  ".join(
            f"{n} {v:.3f}" for n, v in summary[label]["median_percent"].items()))
        print(f"   median tau IAG {np.median(ti):.3f}, file 5 {np.median(tf):.3f};  per unit tau: IAG "
              f"{np.median(per_tau_iag):.2f}, file 5 {np.median(per_tau_f5):.2f}; IAG better in "
              f"{summary[label]['windows_iag_better_per_unit_tau']} of {len(keys)}")
        for lo, hi in REGIONS:
            sel = np.array([lo <= int(k) < hi for k in keys])
            if sel.any():
                summary[label]["regions"][f"{lo}-{hi}"] = {
                    "windows": int(sel.sum()), **{n: float(np.median(get(n)[sel])) for n in NAMES},
                    "tau_ratio": float(np.median((ti / tf)[sel])),
                    "per_unit_tau_iag": float(np.median(per_tau_iag[sel])),
                    "per_unit_tau_f5": float(np.median(per_tau_f5[sel]))}
    if args.output:
        args.output.write_text(json.dumps({"method": __doc__, "file5_fwhm_cm1": FILE5_FWHM_CM1,
                                           "summary": summary, "pages": pages}, indent=1) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
