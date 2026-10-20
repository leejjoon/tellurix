#!/usr/bin/env python
"""Bundle the Baker et al. 2020 telluric fits for the review page, one chunk per window.

Reads what ``fit_baker_telluric.py`` wrote and ``analyse_baker_telluric.py``
measured; numpy and astropy only, no model is rebuilt. A window's spectra share
the atlas's own grid, so one axis serves all 53 of them, and the chunk carries
every spectrum: the page draws one at a time and stacks all their residuals.

Per spectrum: observed and model, both over the fitted continuum, and the fit
mask. Per window: each group's solar model ``stel_mod`` (in the solar frame, as
published) and the solar-clean cut the analysis used. Per feature: the
analysis's mean ratio and offset, and the solar fits' ratios.

Arrays are uint16 with a per-array scale (1.6e-5 of the continuum against
1e-3 of pixel noise); code 65535 is a masked pixel.

    UV_CACHE_DIR=.uv-cache uv run python scripts/export_baker_review.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATLAS = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/iag_telfree")
NAN_CODE = 65535
SOLAR_CLEAN = 0.97
SOLAR_SHIFT_CM1 = 0.1


def json_safe(value):
    """JSON has no NaN: a value that is not measured is null."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def quantize(values) -> tuple[bytes, float, float]:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    lo = float(values[good].min()) if good.any() else 0.0
    hi = float(values[good].max()) if good.any() else 1.0
    if hi <= lo:
        hi = lo + 1e-12
    codes = np.where(good, np.round((np.where(good, values, lo) - lo) / (hi - lo) * (NAN_CODE - 1)), NAN_CODE)
    return codes.astype("<u2").tobytes(), lo, hi


class Chunk:
    def __init__(self):
        self.parts, self.size = [], 0

    def add(self, payload: bytes, align: int = 2) -> int:
        # Every typed view must start at a multiple of its element size.
        pad = (-self.size) % align
        if pad:
            self.parts.append(b"\0" * pad); self.size += pad
        offset = self.size
        self.parts.append(payload); self.size += len(payload)
        return offset

    def array(self, values) -> dict:
        payload, lo, hi = quantize(values)
        return {"offset": self.add(payload), "lo": lo, "hi": hi}

    def bits(self, flags) -> int:
        return self.add(np.packbits(np.asarray(flags, bool)).tobytes(), align=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits", type=Path, default=ROOT / "data/corrected/baker2020")
    parser.add_argument("--analysis", type=Path, default=ROOT / "docs/baker_telluric_ratios.json")
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--output", type=Path, default=ROOT / "data/review_baker2020")
    args = parser.parse_args()

    from astropy.io import fits as pyfits
    from scipy.ndimage import minimum_filter1d
    from tellurix_fts.baker import baker_wavenumber_grid

    analysis = json.loads(args.analysis.read_text())
    features = sorted(analysis["features"], key=lambda f: f["nu"])
    feature_nu = np.array([f["nu"] for f in features])

    by_window = {}
    for path in sorted(glob.glob(str(args.fits / "g*_*_*.json"))):
        meta = json.loads(Path(path).read_text())
        v1 = float(Path(path).stem.rsplit("_", 1)[1])
        by_window.setdefault(v1, []).append((meta, Path(path).with_suffix(".npz")))

    solar, grid = {}, None
    for group in sorted({m["group"] for rows in by_window.values() for m, _ in rows}):
        with pyfits.open(args.atlas / f"telluric_spectra_{group}.fits", memmap=True) as handle:
            table = handle[1].data
            if grid is None:
                grid = baker_wavenumber_grid(table["v"])
            stel = np.asarray(table["stel_mod"], dtype=float)
        reach = int(np.ceil(SOLAR_SHIFT_CM1 / (grid[1] - grid[0])))
        solar[group] = (stel, minimum_filter1d(stel, 2 * reach + 1) > SOLAR_CLEAN)

    args.output.mkdir(parents=True, exist_ok=True)
    windows, total = [], 0
    for number, v1 in enumerate(sorted(by_window)):
        rows = sorted(by_window[v1], key=lambda r: r[0]["airmass"])
        chunk = Chunk()
        axis = None
        spectra = []
        for meta, npz in rows:
            with np.load(npz) as data:
                nu = np.asarray(data["wavenumber_cm1"]); order = np.argsort(nu); nu = nu[order]
                continuum = np.asarray(data["continuum"])[order]
                observed = np.asarray(data["flux"])[order] / continuum
                model = np.asarray(data["model_flux"])[order] / continuum
                mask = np.asarray(data["mask"])[order].astype(bool)
                sigma = float(np.median(np.asarray(data["uncertainty"])[order] / continuum))
            if axis is None:
                axis = nu
            elif nu.size != axis.size or np.abs(nu - axis).max() > 1e-9:
                raise SystemExit(f"{npz.name}: not on the window's grid")
            observed[~mask] = np.nan
            residual = observed[mask] - model[mask]
            report = meta["report"]
            parameters = report["parameters"]
            spectra.append({
                "group": meta["group"], "number": meta["number"], "airmass": meta["airmass"],
                "utc": meta["utc"][:16], "profile": Path(meta["profile"]).stem.replace("goettingen_", ""),
                "observed": chunk.array(observed), "model": chunk.array(model), "mask": chunk.bits(mask),
                "rms": float(np.sqrt(np.mean(residual ** 2))) if residual.size else None,
                "sigma": sigma, "masked": int((~mask).sum()),
                "columns": {k: float(parameters[k]) for k in ("H2O", "O2") if k in parameters},
                "velocity_kms": float(parameters["velocity_kms"]),
                "lsf_sigma_kms": float(parameters["lsf_sigma_kms"]),
                "converged": bool(report.get("all_stages_converged", True)),
            })
        start = int(np.argmin(np.abs(grid - axis[0])))
        if np.abs(grid[start:start + axis.size] - axis).max() > 1e-6:
            raise SystemExit(f"window {v1}: not on the atlas grid")
        groups = {}
        for group in sorted({s["group"] for s in spectra}):
            stel, clean = solar[group]
            groups[str(group)] = {"stel_mod": chunk.array(stel[start:start + axis.size]),
                                  "clean": chunk.bits(clean[start:start + axis.size])}
        inside = (feature_nu > axis[0]) & (feature_nu < axis[-1])
        name = f"chunk_{number:03d}.bin"
        (args.output / name).write_bytes(b"".join(chunk.parts))
        total += chunk.size
        rms = [s["rms"] for s in spectra if s["rms"] is not None]
        windows.append({
            "v1": v1, "v2": v1 + 30.0, "chunk": name, "n": int(axis.size),
            "nu0": float(axis[0]), "dnu": float(grid[1] - grid[0]), "row0": start,
            "spectra": spectra, "groups": groups,
            "median_rms": float(np.median(rms)) if rms else None,
            "worst_rms": float(np.max(rms)) if rms else None,
            "features": [{k: f[k] for k in ("nu", "spectra", "ratio", "ratio_std", "offset_ms",
                                            "offset_std_ms", "inverse_airmass_slope", "inverse_airmass_slope_error",
                                            "solar")}
                         for f, keep in zip(features, inside) if keep],
        })
        print(f"{v1:.0f}: {len(spectra)} spectra, {chunk.size / 1e6:.2f} MB", flush=True)

    (args.output / "manifest.json").write_text(json.dumps(json_safe({
        "source": "Baker, Blake & Reiners 2020 telluric spectra (Zenodo 10.5281/zenodo.3598136), "
                  "fitted by scripts/fit_baker_telluric.py",
        "summary": {k: analysis[k] for k in ("repeatability", "airmass", "solar_fits", "positions")},
        "windows": windows,
    }), separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {len(windows)} windows, {total / 1e6:.1f} MB of chunks, to {args.output}")


if __name__ == "__main__":
    main()
