#!/usr/bin/env python
"""Bundle the solar window fits for review, with each species' own transmission.

The fits save the *total* transmission and nothing per species, so "which
molecule makes this feature" cannot be answered from the cached arrays. This
rebuilds each window's model through `fit_fts_window.prepare_window` -- the same
function the fitter itself uses, not a second description of it -- refreezes the
opacity at the fitted parameters, where that is exact, and splits the
transmission with `TelluricModel.species_transmission`.

**The decomposition is checked, not assumed.** Optical depth is additive, so the
per-species curves must multiply back to the transmission the fit recorded. A
window whose product disagrees is written as failed rather than shipped: a
decomposition shown next to observed data is evidence, and wrong evidence is
worse than none.

Arrays go out as uint16 with a per-array scale, which is 1.5e-5 of their range
against about 1e-3 of pixel noise. Nothing is decimated -- these spectra are
sampled at 1.85 points per resolution element, so dropping points would alias.

    UV_CACHE_DIR=.uv-cache uv run python scripts/export_solar_review.py --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tellurix import TelluricParameters, read_fts_spectrum, read_scan  # noqa: E402
from tellurix.nso import MEASURED_FWHM_CM1  # noqa: E402

from fit_fts_window import prepare_window  # noqa: E402

# Everything the page can offer as a row, whether or not any window fits it.
from tellurix import AER_MOLECULE_IDS  # noqa: E402

CHUNK_BYTES = 1_500_000


def quantize(values: np.ndarray) -> tuple[bytes, float, float]:
    """uint16 with a per-array scale, and the scale needed to read it back."""
    finite = np.asarray(values, dtype=float)
    good = np.isfinite(finite)
    lo = float(np.min(finite[good])) if good.any() else 0.0
    hi = float(np.max(finite[good])) if good.any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0e-12
    scaled = np.where(good, (finite - lo) / (hi - lo), 0.0)
    return np.clip(np.round(scaled * 65535.0), 0, 65535).astype("<u2").tobytes(), lo, hi


def settings_for(row, spectrum_path: Path, args) -> SimpleNamespace:
    """The fit's own configuration, read back from what the run recorded."""
    return SimpleNamespace(
        spectrum=spectrum_path, v1=row["v1"], v2=row["v2"],
        margin_cm1=args.margin_cm1, grid_margin_cm1=args.grid_margin_cm1,
        samples_per_resolution=args.samples_per_resolution, fwhm_cm1=MEASURED_FWHM_CM1,
        profile=args.profile, species=",".join(row["species"]),
        stellar="auto", vsini_kms=0.0, macroturbulence_kms=1.5,
        source_continuum=False, normalize_source=False,
        continuum_degree=args.continuum_degree, stages="continuum", pin=[],
        zenith_angle_deg=None, accuracy_mode="mt_ckd", correction=None,
        gaussian_ils=False, vectorize_layers=True, mixed_precision=True,
        # False on purpose: prepare_window would otherwise precompute the
        # opacity for a fit that never happens here, and this script refreezes
        # at the *fitted* parameters a moment later. That discarded evaluation
        # was 2.3 s of every 6.6 s window.
        precompute_opacity=False, self_broadening="linear", layer_chunk_size=0,
        report=None, diagnostic_npz=None,
    )


def parameters_for(row, species) -> TelluricParameters:
    p = row["parameters"]
    return TelluricParameters(
        log_column_scales={name: float(p[name]) for name in species},
        velocity_kms=float(p["velocity_kms"]),
        wavelength_stretch=0.0,
        lsf_sigma_kms=float(p["lsf_sigma_kms"]),
        continuum_coeffs=np.asarray(p["continuum_coeffs"], dtype=float),
        log_jitter=float(p["log_jitter"]),
        stellar_velocity_kms=float(p["stellar_velocity_kms"]),
    )


def scan_rows(scan_name: str, root: Path, airmass: float, fitted) -> list[dict]:
    """Every molecule the scan considered, and why it is or is not fitted.

    The page greys out what was not fitted, and a greyed row that cannot say
    *why* is the display that let OCS and O3 hide: 'not in the list' and
    'measured and too weak' look identical, and only one of them is a reason.
    """
    report = read_scan(root / "data/scans" / scan_name)
    rows = []
    for entry in report["fit"] + report["rejected"]:
        depth = entry["peak_optical_depth"] * airmass
        rows.append({"species": entry["species"], "state":
                     "fitted" if entry["species"] in fitted else "below cut",
                     "peak_optical_depth": depth, "lines": entry.get("lines")})
    for entry in report["rejected_by_bound"]:
        rows.append({"species": entry["species"], "state": "bounded out",
                     "optical_depth_bound": entry["optical_depth_bound"] * airmass,
                     "lines": entry["lines"]})
    for name in report["no_lines_in_window"]:
        rows.append({"species": name, "state": "no lines here"})
    for name in report["not_in_profile"]:
        rows.append({"species": name, "state": "not in the profile"})
    seen = {r["species"] for r in rows}
    for name in sorted(AER_MOLECULE_IDS):
        if name not in seen:
            rows.append({"species": name, "state": "no line file"})
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, action="append", default=None,
                        help="a run summary; repeat for several. Default: both FTS files.")
    parser.add_argument("--npz-dir", type=Path, default=root / "data/corrected/solar")
    parser.add_argument("--profile", type=Path,
                        default=Path("data/profiles/kitt_peak_19901218_era5_afgl.csv"))
    parser.add_argument("--output", type=Path, default=root / "data/review")
    parser.add_argument("--margin-cm1", type=float, default=25.0)
    parser.add_argument("--grid-margin-cm1", type=float, default=5.0)
    parser.add_argument("--samples-per-resolution", type=float, default=4.0)
    parser.add_argument("--continuum-degree", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=2.0e-6,
                        help="how far the product of the species may sit from the "
                             "transmission the fit recorded, before the window is "
                             "written as failed")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", default=None, metavar="I/N")
    parser.add_argument("--compilation-cache", default=str(root / ".jax-cache"))
    args = parser.parse_args()

    if args.compilation_cache:
        import jax

        Path(args.compilation_cache).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    summaries = args.summary or [
        root / "data/corrected/solar/ftsspec_901218_5_summary.json",
        root / "data/corrected/solar/ftsspec_901218_4_summary.json",
    ]
    jobs = []
    for path in summaries:
        blob = json.loads(Path(path).read_text())
        spectrum_path = Path(blob["spectrum"])
        label = spectrum_path.stem.rsplit("_", 1)[-1]
        for row in blob["results"]:
            if row.get("npz"):
                jobs.append((label, spectrum_path, blob, row))
    jobs.sort(key=lambda j: (j[3]["v1"], j[0]))
    if args.shard:
        index, _, count = args.shard.partition("/")
        jobs = jobs[int(index):: int(count)]
    if args.limit:
        jobs = jobs[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    spectra: dict[str, object] = {}
    manifest, failures = [], []
    chunk, chunk_bytes, chunk_index = [], 0, 0
    started = time.time()

    def flush() -> None:
        nonlocal chunk, chunk_bytes, chunk_index
        if not chunk:
            return
        (args.output / f"chunk_{chunk_index:03d}.bin").write_bytes(b"".join(chunk))
        chunk, chunk_bytes = [], 0
        chunk_index += 1

    for number, (label, spectrum_path, blob, row) in enumerate(jobs, 1):
        try:
            import jax

            key = str(spectrum_path)
            if key not in spectra:
                spectra[key] = read_fts_spectrum(spectrum_path)
            prepared = prepare_window(settings_for(row, spectrum_path, args), root,
                                      spectrum=spectra[key])
            model, order = prepared.model, prepared.order
            parameters = parameters_for(row, model.species)
            exact = model.precompute_opacity(parameters)
            zenith = order.zenith_angle_deg
            pieces = {name: np.asarray(values) for name, values
                      in exact.species_transmission(parameters, zenith).items()}
            total = np.asarray(exact.transmission(parameters, zenith))

            product = np.ones_like(total)
            for values in pieces.values():
                product = product * values
            drift = float(np.max(np.abs(product - total)))
            if drift > args.tolerance:
                raise ValueError(f"species do not multiply back to the total: {drift:.2e}")

            with np.load(args.npz_dir / row["npz"]) as stored:
                flip = slice(None, None, -1)
                nu = np.asarray(stored["wavenumber_cm1"])[flip]
                wavelength = np.asarray(stored["wavelength_vacuum_nm"])[flip]
                arrays = {
                    "observed": np.where(np.asarray(stored["mask"])[flip],
                                         np.asarray(stored["flux"])[flip], np.nan),
                    "model": np.asarray(stored["model_flux"])[flip],
                    "continuum": np.asarray(stored["continuum"])[flip],
                    "stellar": np.asarray(stored["stellar_only_pixels"])[flip],
                    "transmission": np.asarray(stored["transmission_pixels"])[flip],
                }
                residual = np.asarray(stored["residual"])[flip]
                mask = np.asarray(stored["mask"])[flip].astype(bool)
            sigma = float(row["pixel_sigma"])
            arrays["residual"] = np.where(mask, residual / sigma, np.nan)

            # Each species on the pixels, by the same interpolation the fit used
            # for the total; the grid is in ascending wavelength, so reverse.
            grid_nm = (1.0e7 / np.asarray(model.wavenumber_cm1))[::-1]
            for name, values in pieces.items():
                arrays[f"species:{name}"] = np.interp(wavelength, grid_nm, values[::-1])

            entry = {
                "file": label, "v1": row["v1"], "v2": row["v2"],
                "airmass": row["airmass"], "zenith_angle_deg": row["zenith_angle_deg"],
                "n": int(nu.size), "nu0": float(nu[0]),
                "dnu": float((nu[-1] - nu[0]) / (nu.size - 1)),
                "chunk": chunk_index, "arrays": {},
                "mask_offset": None,
                "species": {name: {
                    "log_scale": float(row["parameters"][name]),
                    "at_bound": name in row["at_bound"]} for name in model.species},
                "scan": scan_rows(row["scan"], root, float(row["airmass"]),
                                  set(model.species)),
                "quality": {k: row[k] for k in (
                    "residual_rms_over_noise", "reduced_chi2", "jitter_over_uncertainty",
                    "median_transmission", "reliable", "pixels", "masked",
                    "continuum_level", "condition_number", "all_stages_converged")},
                "at_bound": row["at_bound"],
                "opaque": row["reliable"] == 0,
                "product_drift": drift,
            }
            for name, values in arrays.items():
                payload, lo, hi = quantize(values)
                entry["arrays"][name] = {"offset": chunk_bytes, "lo": lo, "hi": hi}
                chunk.append(payload)
                chunk_bytes += len(payload)
            packed = np.packbits(mask).tobytes()
            entry["mask_offset"] = chunk_bytes
            chunk.append(packed)
            chunk_bytes += len(packed)
            manifest.append(entry)
            status = (f"{len(pieces)} species, drift {drift:.1e}, "
                      f"{chunk_bytes / 1e3:.0f} kB in chunk {chunk_index}")
            if chunk_bytes >= CHUNK_BYTES:
                flush()
            jax.clear_caches()
        except Exception as exc:  # one window must not stop the export
            failures.append({"file": label, "v1": row["v1"], "error": str(exc),
                             "traceback": traceback.format_exc()[-1200:]})
            status = f"FAILED: {exc}"
        print(f"[{number}/{len(jobs)}] file {label} {row['v1']:7.1f}  {status}   "
              f"({(time.time() - started) / 60:.1f} min)", flush=True)

    flush()
    (args.output / "manifest.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "windows": manifest, "failures": failures,
        "species_all": sorted(AER_MOLECULE_IDS),
        "note": ("arrays are uint16; value = lo + code * (hi - lo) / 65535. "
                 "The wavenumber axis is uniform: nu0 + i * dnu, ascending. "
                 "mask is bit-packed MSB-first over n pixels."),
    }, indent=2) + "\n", encoding="utf-8")
    total_bytes = sum(f.stat().st_size for f in args.output.glob("chunk_*.bin"))
    print(f"\n{len(manifest)} windows, {len(failures)} failed, "
          f"{chunk_index} chunks, {total_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
