#!/usr/bin/env python
"""Fit every window of an NSO FTS solar spectrum, species computed per window.

The scaling-up of `fit_fts_window.py`, which this driver *calls* rather than
reimplements. The Arcturus batch driver carries its own copy of the page fit;
that was affordable there because the species list was a six-molecule constant,
and it is not affordable here, because the species list is a per-window result
and a second implementation would be free to compute it differently.

Two costs, and they are paid on different schedules.

**The scan** ranks every absorber AER ships against the window and the
atmosphere, at about six minutes a window. It depends on no spectrum, so one
scan serves file 4 and file 5 and every later refit; `tellurix.cached_scan`
keeps them under `--scan-cache` and re-runs only what is missing. Across ~225
windows that is a one-time ~22 hours, or half that split over two GPUs.

**The fit** is per window *and* per file, and costs about a minute. It is the
cheap half, which is the opposite of the arrangement the plan assumed.

*Failures are recorded, not fatal.* One window that will not converge must not
stop the run, and a resumed run must not repeat completed work. The record is
assembled at the end from the per-window `.npz` files rather than from memory,
so an interrupted run that is resumed still writes a **complete** record --
unlike the Arcturus driver, whose summary drops the per-row record block and so
writes a record covering only the windows of its final pass.

    UV_CACHE_DIR=.uv-cache uv run python scripts/fit_fts_batch.py \\
        --spectrum .../ftsspec_901218_5.txt --v1 4300 --v2 4420
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tellurix import (  # noqa: E402
    ScanIdentity,
    cached_scan,
    file_sha256,
    load_atmosphere_csv,
    read_fts_spectrum,
    species_above,
)
from tellurix.nso import MEASURED_FWHM_CM1, window_continuum_snr  # noqa: E402

from fit_fts_window import fit_window  # noqa: E402

LINE_ROOT = Path("data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule")

# Everything about the fit that is the same for every window. Named here and
# read from here, so the summary records what ran rather than a second
# description of it that can drift.
PHYSICS = {
    "accuracy_mode": "mt_ckd",
    "opacity_method": "direct_sparse",
    "pressure_shift": True,
    "mixed_precision": True,
    "vectorize_layers": True,
    "pixel_integration": "point",
    "instrument": "boxcar FTS sinc",
    "fwhm_cm1": MEASURED_FWHM_CM1,
    "vsini_kms": 0.0,
    "macroturbulence_kms": 1.5,
    "stages": "continuum,velocity,columns,stellar",
    "species": "computed per window by the scan",
}


def enable_compilation_cache(directory: Path) -> None:
    """Persist compiled executables so a repeated shape never recompiles.

    Shapes vary per window, so within one pass this mostly does not hit. It pays
    on a resumed run and across workers sharing one directory.
    """

    import jax

    directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def window_edges(spectrum, v1: float, v2: float, width_cm1: float,
                 minimum_snr: float) -> list[dict]:
    """Tile the requested range, keeping only windows the file actually measured.

    These files are one transform over 1630-9510 cm-1, but the instrument's
    filters pass a fraction of that; outside them the "spectrum" is noise, whose
    99th percentile still looks like a continuum. `window_continuum_snr` is the
    same test the ILS measurement uses, for the same reason.
    """

    nu = spectrum.wavenumber_vacuum_cm1
    edges = np.arange(v1, v2 + 0.5 * width_cm1, width_cm1)
    windows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (nu >= lo) & (nu < hi)
        if inside.sum() < 100:
            continue
        snr = window_continuum_snr(spectrum.flux[inside])
        if snr < minimum_snr:
            continue
        windows.append({"v1": float(lo), "v2": float(hi), "continuum_snr": round(snr, 1)})
    return windows


def physics_record(root: Path) -> dict:
    """What produced these numbers: the fixed physics and the code that ran it.

    Both drivers' hashes, because this one delegates the fit to the other and a
    change in either changes the result.
    """

    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover
        version = None
    driver = Path(__file__).resolve()
    fitter = driver.with_name("fit_fts_window.py")
    return {
        **PHYSICS,
        "driver": driver.name,
        "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        "fitter": fitter.name,
        "fitter_sha256": hashlib.sha256(fitter.read_bytes()).hexdigest(),
        "tellurix": version("tellurix") if version else None,
    }


def scan_species(entry, args, root, profile, airmass: float) -> dict:
    """The window's species list, from the cache when it is already there.

    The scan ranks the vertical column at a low floor; the cut this fit applies
    is `--species-threshold` along **this** file's path. That separation is what
    lets one scan serve file 4 at air mass 4.7 and file 5 at 2.0, which is the
    whole reason a six-minute scan is affordable.
    """

    identity = ScanIdentity.for_profile(
        root / args.profile, (entry["v1"], entry["v2"]),
        threshold=args.scan_threshold, line_budget=args.scan_line_budget,
        margin_cm1=args.margin_cm1, fwhm_cm1=args.fwhm_cm1,
        samples_per_resolution=args.scan_samples_per_resolution)
    cached = cached_scan(
        args.scan_cache, identity, profile, root / LINE_ROOT,
        layer_chunk_size=args.layer_chunk_size,
        progress=(lambda line: print(line, flush=True)) if args.verbose_scan else None)
    species = species_above(cached, args.species_threshold, airmass=airmass)
    return {"species": species, "scan": identity.path(args.scan_cache).name,
            "scan_airmass": airmass,
            "scan_below_cut": [row["species"] for row in [*cached["fit"], *cached["rejected"]]
                               if row["species"] not in species],
            "scan_headroom": cached["headroom"]}


def run_one(entry, args, root, profile, airmass: float) -> dict:
    """Scan, fit and export one window."""

    started = time.time()
    scan = scan_species(entry, args, root, profile, airmass)
    if not scan["species"]:
        # Nothing absorbs here at the scan's threshold. That is a result, not a
        # failure: record it and do not spend a fit proving it.
        return {**entry, **scan, "negligible_telluric": True,
                "seconds": round(time.time() - started, 1)}

    settings = SimpleNamespace(
        spectrum=args.spectrum, v1=entry["v1"], v2=entry["v2"],
        margin_cm1=args.margin_cm1, grid_margin_cm1=args.grid_margin_cm1,
        samples_per_resolution=args.samples_per_resolution, fwhm_cm1=args.fwhm_cm1,
        profile=args.profile, species=",".join(scan["species"]),
        stellar="auto", vsini_kms=PHYSICS["vsini_kms"],
        macroturbulence_kms=PHYSICS["macroturbulence_kms"],
        source_continuum=False, normalize_source=False,
        continuum_degree=args.continuum_degree, stages=PHYSICS["stages"], pin=[],
        zenith_angle_deg=args.zenith_angle_deg,
        accuracy_mode="mt_ckd", correction=None, gaussian_ils=False,
        vectorize_layers=True, mixed_precision=True,
        precompute_opacity=args.precompute_opacity, self_broadening=args.self_broadening,
        layer_chunk_size=args.layer_chunk_size,
        report=None, diagnostic_npz=None,
    )
    report, arrays = fit_window(settings, root)

    mask = arrays["mask"]
    observed = arrays["observed_raw"]
    model_flux = arrays["model_flux"]
    star = arrays["stellar_only_pixels"]
    continuum = arrays["continuum"]
    transmission = arrays["transmission_pixels"]
    reliable = mask & (transmission >= args.min_transmission)
    corrected = (observed / np.maximum(model_flux, 1e-6)) * star
    # Where the correction's level ends up. The fitted continuum cancels out of
    # that ratio, so on a window with no telluric-free pixel the continuum and
    # the column are degenerate: their product still fits the data, the residual
    # looks fine, and the ratio used for the correction is off by the whole
    # degenerate factor. Comparing the two where the star itself is unabsorbed
    # turns that into a number instead of leaving it implicit.
    star_flat = star / np.maximum(continuum, 1e-12)
    clean = reliable & np.isfinite(corrected) & (star_flat > 0.98) & (star > 1e-6)
    level = float(np.median(corrected[clean] / star[clean])) if clean.sum() >= 20 else float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz = args.output_dir / f"{args.spectrum.stem}_{entry['v1']:.0f}.npz"
    np.savez_compressed(
        npz, corrected=corrected,
        corrected_normalized=corrected / np.maximum(continuum, 1e-12),
        reliable=reliable, **arrays)

    parameters = report["parameters"]
    sigma = float(report["spectrum"]["uncertainty"])
    return {
        **entry, **scan,
        "npz": npz.name,
        "negligible_telluric": False,
        "pixels": report["spectrum"]["pixels"],
        "masked": report["spectrum"]["masked"],
        "reliable": int(reliable.sum()),
        "grid_points": report["grid"]["points"],
        "mopd_cm": report["physics"]["mopd_cm"],
        "zenith_angle_deg": report["physics"]["zenith_angle_deg"],
        "airmass": report["physics"]["airmass_used"],
        "stellar_source": Path(report["stellar"]["source"]).name,
        "parameters": parameters,
        "parameter_names": [str(n) for n in arrays["parameter_names"]],
        "at_bound": report["at_bound"],
        "all_stages_converged": report["all_stages_converged"],
        "condition_number": report["condition_number"],
        "pixel_sigma": sigma,
        "residual_rms": report["residuals"]["rms"],
        "residual_rms_over_noise": report["residuals"]["rms"] / sigma,
        "reduced_chi2": report["residuals"]["reduced_chi2"],
        "jitter_over_uncertainty": report["residuals"]["jitter_over_uncertainty"],
        "median_transmission": report["median_transmission"],
        "continuum_level": None if level != level else round(level, 4),
        "continuum_level_pixels": int(clean.sum()),
        "stages": report["stages"],
        "seconds": round(time.time() - started, 1),
    }


def record_row(row, npz_dir: Path, spectrum_sha: str) -> dict:
    """One record row, assembled from the summary and the window's own arrays.

    Read back from disk rather than carried in memory, so a run that was
    interrupted and resumed still writes a record covering every window.
    """

    with np.load(npz_dir / row["npz"]) as arrays:
        parameters = row["parameters"]
        species = [n for n in row["parameter_names"]
                   if n not in ("velocity_kms", "stellar_velocity_kms", "lsf_sigma_kms",
                                "wavelength_stretch", "log_jitter")
                   and not n.startswith("continuum_")]
        return {
            "file": Path(row["file"]).name,
            "window": f"{row['v1']:.0f}-{row['v2']:.0f}",
            "page_sha256": spectrum_sha,
            "log_column_scales": {s: float(parameters[s]) for s in species},
            "continuum_coeffs": np.asarray(parameters["continuum_coeffs"], dtype=float),
            "v1": row["v1"], "v2": row["v2"], "mopd_cm": row["mopd_cm"],
            "pixels": row["pixels"], "grid_points": row["grid_points"],
            "reliable": row["reliable"],
            "velocity_kms": parameters["velocity_kms"],
            "stellar_velocity_kms": parameters["stellar_velocity_kms"],
            "wavelength_stretch": 0.0,
            "lsf_sigma_kms": parameters["lsf_sigma_kms"],
            "log_jitter": parameters["log_jitter"],
            "pixel_sigma": row["pixel_sigma"],
            "residual_rms": row["residual_rms"],
            "residual_rms_over_noise": row["residual_rms_over_noise"],
            "reduced_chi2": row["reduced_chi2"],
            "median_transmission": row["median_transmission"],
            "continuum_level": row["continuum_level"] or 0.0,
            "continuum_level_pixels": row["continuum_level_pixels"],
            "condition_number": row["condition_number"],
            "all_stages_converged": row["all_stages_converged"],
            "negligible_telluric": row["negligible_telluric"],
            "airmass": row["airmass"], "zenith_angle_deg": row["zenith_angle_deg"],
            "free_species": "+".join(species),
            "at_bound": "+".join(row["at_bound"]),
            "parameter_names": row["parameter_names"],
            "sigma": arrays["sigma"], "correlation": arrays["correlation"],
            "ils_velocity_kms": arrays["ils_velocity_kms"],
            "ils_profile": arrays["ils_profile"],
        }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--v1", type=float, default=1876.0,
                        help="the blue edge of the reddest solar source, by default")
    parser.add_argument("--v2", type=float, default=9091.0)
    parser.add_argument("--window-cm1", type=float, default=30.0)
    parser.add_argument("--minimum-snr", type=float, default=30.0,
                        help="continuum over noise below which the file measured nothing here")
    parser.add_argument("--profile", type=Path,
                        default=Path("data/profiles/kitt_peak_19901218_era5_afgl.csv"))
    parser.add_argument("--zenith-angle-deg", type=float, default=None,
                        help="default: the air mass in the file's own header")
    parser.add_argument("--fwhm-cm1", type=float, default=MEASURED_FWHM_CM1)
    parser.add_argument("--samples-per-resolution", type=float, default=4.0)
    parser.add_argument("--margin-cm1", type=float, default=25.0)
    parser.add_argument("--grid-margin-cm1", type=float, default=5.0)
    parser.add_argument("--continuum-degree", type=int, default=3)
    parser.add_argument("--min-transmission", type=float, default=0.15)
    parser.add_argument("--precompute-opacity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-broadening", choices=("linear", "frozen"), default="linear")
    parser.add_argument("--layer-chunk-size", type=int, default=0)
    parser.add_argument("--scan-cache", type=Path, default=root / "data/scans")
    parser.add_argument("--scan-threshold", type=float, default=2.0e-4,
                        help="the floor a scan evaluates down to, on the VERTICAL column. "
                             "It must sit below --species-threshold divided by the largest "
                             "air mass any file will ask for (4.73 for ftsspec_901218_4), "
                             "because rejecting on the bound is the one cut a scan cannot "
                             "undo. Lowering it makes scans slower, not wrong.")
    parser.add_argument("--species-threshold", type=float, default=1.0e-3,
                        help="peak optical depth ALONG THIS FILE'S PATH below which a "
                             "species is not worth a free parameter. About a tenth of the "
                             "NSO FTS noise.")
    parser.add_argument("--scan-line-budget", type=float, default=1.0e-6,
                        help="optical depth discarded by dropping weak lines; two orders "
                             "below --scan-threshold so it cannot move a species across it")
    parser.add_argument("--scan-samples-per-resolution", type=float, default=2.0)
    parser.add_argument("--verbose-scan", action="store_true")
    parser.add_argument("--stride", type=int, default=1, help="take every Nth window")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="take every Nth window starting at I, for one GPU of N. "
                             "Interleaved rather than split by range, because the scan's "
                             "cost is concentrated in the line-rich red end and a "
                             "contiguous split would leave one worker idle for hours.")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many windows")
    parser.add_argument("--output-dir", type=Path, default=root / "data/corrected/solar")
    parser.add_argument("--summary", type=Path, default=None,
                        help="default: <output-dir>/<spectrum stem>_summary.json")
    parser.add_argument("--record", type=Path, default=None,
                        help="default: <output-dir>/<spectrum stem>.h5")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compilation-cache", default=str(root / ".jax-cache"))
    parser.add_argument("--scan-only", action="store_true",
                        help="populate the scan cache and stop, which is the half that "
                             "is shared between files and worth running once")
    args = parser.parse_args()

    stem = args.spectrum.stem
    args.summary = args.summary or args.output_dir / f"{stem}_summary.json"
    args.record = args.record or args.output_dir / f"{stem}.h5"
    if args.compilation_cache:
        enable_compilation_cache(Path(args.compilation_cache))

    spectrum = read_fts_spectrum(args.spectrum)
    profile = load_atmosphere_csv(root / args.profile)
    if args.zenith_angle_deg is not None:
        airmass = 1.0 / np.cos(np.radians(args.zenith_angle_deg))
    elif spectrum.airmass_mean is None:
        raise SystemExit(f"{args.spectrum.name} has no air mass; pass --zenith-angle-deg")
    else:
        airmass = float(spectrum.airmass_mean)
    jobs = window_edges(spectrum, args.v1, args.v2, args.window_cm1, args.minimum_snr)
    jobs = jobs[:: args.stride]
    if args.shard:
        index, _, count = args.shard.partition("/")
        jobs = jobs[int(index):: int(count)]
    if args.limit:
        jobs = jobs[: args.limit]

    done = {}
    if args.resume and args.summary.exists():
        for row in json.loads(args.summary.read_text())["results"]:
            done[round(row["v1"], 3)] = row

    results = list(done.values())
    started = time.time()
    for index, entry in enumerate(jobs, 1):
        key = round(entry["v1"], 3)
        if key in done:
            continue
        entry = {**entry, "file": str(args.spectrum)}
        if args.scan_only:
            row = {**entry, **scan_species(entry, args, root, profile, airmass)}
            status = "scan: " + (", ".join(row["species"]) or "nothing above threshold")
        else:
            try:
                row = run_one(entry, args, root, profile, airmass)
                status = ("negligible telluric" if row["negligible_telluric"] else
                          f"rms/noise {row['residual_rms_over_noise']:5.2f}  "
                          f"{'+'.join(row['species'])}")
            # SystemExit as well as Exception: `fit_window` came from a CLI and
            # reports a user error by raising SystemExit, which is a
            # BaseException and would otherwise take the whole run down. The
            # first window does exactly that -- 1876-1906 cm-1 reaches 5330.5 nm
            # and the reddest Payne Zero band stops at 5330.0.
            except (Exception, SystemExit) as exc:  # a bad window must not stop the run
                row = {**entry, "error": str(exc), "traceback": traceback.format_exc()[-1500:]}
                status = f"FAILED: {exc}"
        results.append(row)
        done[key] = row
        # Every window has its own grid size and species set, so nothing XLA
        # compiled for the last one can be reused for the next; left in place it
        # is retention, and it is what killed the scan pass three windows in.
        import jax

        jax.clear_caches()
        print(f"[{index}/{len(jobs)}] {entry['v1']:7.1f}-{entry['v2']:7.1f}  {status}   "
              f"({(time.time() - started)/60:.1f} min elapsed)", flush=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(
            {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "spectrum": str(args.spectrum), "spectrum_sha256": spectrum.sha256,
             "profile": str(args.profile),
             "physics": physics_record(root),
             "settings": {"window_cm1": args.window_cm1, "minimum_snr": args.minimum_snr,
                          "samples_per_resolution": args.samples_per_resolution,
                          "margin_cm1": args.margin_cm1,
                          "grid_margin_cm1": args.grid_margin_cm1,
                          "continuum_degree": args.continuum_degree,
                          "min_transmission": args.min_transmission,
                          "scan_threshold": args.scan_threshold,
                          "species_threshold": args.species_threshold,
                          "scan_line_budget": args.scan_line_budget,
                          "airmass": airmass},
             "results": sorted(results, key=lambda r: r["v1"])},
            indent=2) + "\n", encoding="utf-8")

    fitted = [r for r in results if r.get("npz")]
    if fitted and not args.scan_only:
        from tellurix import write_record
        from tellurix.fit import _ParameterCodec

        species = sorted({s for r in fitted for s in r["species"]})
        codec = _ParameterCodec(species, args.continuum_degree + 1,
                                include_stellar_velocity=True)
        position = {name: index for index, name in enumerate(codec.names)}
        rows = []
        for row in sorted(fitted, key=lambda r: r["v1"]):
            entry = record_row(row, args.output_dir, spectrum.sha256)
            # A window's parameter vector is not the run's: which species are
            # present depends on the window, so each row's sigma and correlation
            # are remapped into the union rather than assumed to line up.
            local = entry.pop("parameter_names")
            where = np.array([position[name] for name in local])
            sigma = np.zeros(len(codec.names))
            correlation = np.zeros((len(codec.names), len(codec.names)))
            sigma[where] = entry["sigma"]
            correlation[np.ix_(where, where)] = entry["correlation"]
            entry["sigma"], entry["correlation"] = sigma, correlation
            rows.append(entry)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        write_record(
            args.record,
            run={"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 **{k: v for k, v in physics_record(root).items()
                    if k in ("driver", "driver_sha256", "fitter", "fitter_sha256", "tellurix")}},
            config={"window_cm1": args.window_cm1, "minimum_snr": args.minimum_snr,
                    "samples_per_resolution": args.samples_per_resolution,
                    "margin_cm1": args.margin_cm1, "grid_margin_cm1": args.grid_margin_cm1,
                    "continuum_degree": args.continuum_degree,
                    "min_transmission": args.min_transmission,
                    "scan_threshold": args.scan_threshold,
                    "species_threshold": args.species_threshold,
                    "scan_line_budget": args.scan_line_budget, "airmass": airmass},
            physics=physics_record(root),
            inputs={"spectrum": str(args.spectrum), "spectrum_sha256": spectrum.sha256,
                    "profile": str(args.profile),
                    "profile_sha256": file_sha256(root / args.profile),
                    "scan_cache": str(args.scan_cache)},
            parameter_names=codec.names, species=species, pages=rows,
            continuum_degree=args.continuum_degree,
            key_fields=("file", "window"),
            extra_columns=(("airmass", "f8"), ("zenith_angle_deg", "f8")),
        )
        print(f"wrote {args.record} ({args.record.stat().st_size/1e6:.2f} MB)")

    ok = [r for r in results if "error" not in r]
    print(f"\n{len(ok)} of {len(results)} succeeded; wrote {args.summary}")
    if fitted:
        rms = np.array([r["residual_rms_over_noise"] for r in fitted])
        print(f"residual rms / noise: median {np.median(rms):.2f}, "
              f"worst {rms.max():.2f}, best {rms.min():.2f}")


if __name__ == "__main__":
    main()
