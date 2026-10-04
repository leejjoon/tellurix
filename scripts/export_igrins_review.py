#!/usr/bin/env python
"""Bundle a night's IGRINS standard fits for the review page, species by species.

The driver saves the *total* transmission and nothing per species, so this
rebuilds each order through `tellurix_igrins.standard.build_order_context` --
the function the fit itself uses -- refreezes the opacity at each frame's fitted
parameters, where that is exact, and splits the transmission with
`TelluricModel.species_transmission`. One context per order serves every frame,
as it does in the fit: within a night the PLP's wavelength solution is shared.

Two checks run on every order-frame, and a failure writes the entry as failed
rather than shipping it. The species must multiply back to the total on the
model grid (optical depth is additive), and that total, on the pixels, must
reproduce the transmission the fit cached -- which is what proves the
parameters read back from the summary are the ones that made the cached arrays.

Arrays go out as uint16 with a per-array scale, as for the solar bundle. The
pixel axis is not uniform in wavenumber here, so each entry also carries its
own axis (float32 offsets from its first pixel) and its detector columns, which
are what a PLP product indexes.

    UV_CACHE_DIR=.uv-cache uv run python scripts/export_igrins_review.py \\
        --run data/corrected/igrins/ladder_a0v --run data/corrected/igrins/ladder_k_a0v \\
        --shard 0/4 --output data/review_igrins_0
    UV_CACHE_DIR=.uv-cache uv run python scripts/export_igrins_review.py \\
        --merge data/review_igrins_? --output data/review_igrins
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

CHUNK_BYTES = 1_500_000
OVERVIEW_BINS = 192


def quantize(values: np.ndarray) -> tuple[bytes, float, float]:
    """uint16 with a per-array scale. NaN writes code 0; the page blanks
    masked pixels itself, and no array here carries a NaN the mask does not."""
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    lo = float(np.min(values[good])) if good.any() else 0.0
    hi = float(np.max(values[good])) if good.any() else 1.0
    if hi <= lo:
        hi = lo + 1.0e-12
    scaled = np.where(good, (values - lo) / (hi - lo), 0.0)
    return np.clip(np.round(scaled * 65535), 0, 65535).astype("<u2").tobytes(), lo, hi


def read_run(run: Path) -> dict:
    """Every frame's rows, joined across the order shards that wrote them."""
    frames: dict[str, dict] = {}
    for path in sorted(run.glob("*summary.json")):
        blob = json.loads(path.read_text())
        stem = Path(blob["observation"]["path"]).name.split(".")[0]
        frame = frames.setdefault(stem, {"observation": blob["observation"],
                                         "settings": blob["settings"], "rows": {},
                                         "failures": []})
        for row in blob["results"]:
            frame["rows"][int(row["order"])] = row
        frame["failures"].extend(blob.get("failures", []))
    if not frames:
        raise SystemExit(f"{run}: no summaries")
    return frames


def overview(nu, observed, continuum, transmission, mask) -> dict:
    """A coarse picture of the order for the whole-band strip: the envelope of
    observed / continuum per bin, and the mean transmission. uint8 is plenty
    for a strip a few hundred pixels tall."""
    edges = np.linspace(0, nu.size, OVERVIEW_BINS + 1).astype(int)
    ratio = np.where(mask, observed / np.maximum(continuum, 1e-12), np.nan)
    lo, hi, t = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        r = ratio[a:b]
        r = r[np.isfinite(r)]
        lo.append(np.min(r) if r.size else np.nan)
        hi.append(np.max(r) if r.size else np.nan)
        t.append(np.mean(transmission[a:b]))

    def pack(values, top):
        v = np.asarray(values, dtype=float)
        codes = np.where(np.isfinite(v), np.clip(np.round(v / top * 254), 0, 254), 255)
        return base64.b64encode(codes.astype(np.uint8).tobytes()).decode()

    centres = 0.5 * (nu[edges[:-1]] + nu[np.minimum(edges[1:], nu.size) - 1])
    return {"lo": pack(lo, 1.3), "hi": pack(hi, 1.3), "t": pack(t, 1.0),
            "nu": [round(float(v), 2) for v in centres]}


def export(args, root: Path) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    if args.compilation_cache:
        Path(args.compilation_cache).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    from tellurix import StellarSpectrum, TelluricParameters, load_atmosphere_csv
    from tellurix.download import DataPaths
    from tellurix_igrins import read_igrins_observation
    from tellurix_igrins.standard import StandardFitSettings, build_order_context

    jobs = []
    runs = []
    for run in args.run:
        frames = read_run(run)
        first = next(iter(frames.values()))
        settings = first["settings"]
        band = first["observation"]["band"]
        night = first["observation"]["date_obs"][:10]
        runs.append({"run": str(run), "band": band, "settings": settings})
        orders = sorted(set().union(*(f["rows"].keys() for f in frames.values())))
        for number in orders:
            jobs.append((run, band, number, frames, settings))
    if args.shard:
        index, _, count = args.shard.partition("/")
        jobs = jobs[int(index):: int(count)]
    if args.limit:
        jobs = jobs[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("chunk_*.wasm"):
        stale.unlink()
    observations, profiles, stellars = {}, {}, {}
    manifest, frames_out, failures = [], {}, []
    chunk, chunk_bytes, chunk_index = [], 0, 0
    started = time.time()

    def flush() -> None:
        nonlocal chunk, chunk_bytes, chunk_index
        if chunk:
            (args.output / f"chunk_{chunk_index:03d}.wasm").write_bytes(b"".join(chunk))
            chunk, chunk_bytes = [], 0
            chunk_index += 1

    for number_job, (run, band, number, frames, settings) in enumerate(jobs, 1):
        context = None
        for stem, frame in sorted(frames.items()):
            row = frame["rows"].get(number)
            if row is None:
                continue
            obs_meta = frame["observation"]
            frames_out.setdefault(stem, {
                "stem": stem, "band": band, "frame": stem.split("_")[-1],
                "object": obs_meta["object"], "date_obs": obs_meta["date_obs"],
                "mjd": obs_meta["mjd"], "zenith_angle_deg": obs_meta["zenith_angle_deg"],
                "airmass": float(row["airmass"]),
                "failures": frame["failures"]})
            try:
                path = root / obs_meta["path"]
                if path not in observations:
                    observations[path] = read_igrins_observation(path)
                observation = observations[path]
                if context is None:
                    if settings["profile"] not in profiles:
                        profiles[settings["profile"]] = load_atmosphere_csv(root / settings["profile"])
                    stellar_name = settings["stellar"]
                    if stellar_name not in stellars:
                        stellars[stellar_name] = (None if stellar_name == "flat" else
                                                  StellarSpectrum.from_npz(root / stellar_name))
                    fit_settings = StandardFitSettings(**{k: settings[k] for k in (
                        "resolving_power", "samples_per_resolution", "margin_cm1",
                        "grid_margin_cm1", "vsini_kms", "self_broadening",
                        "min_optical_depth")}, precompute_opacity=True,
                        line_coupling=bool(settings.get("line_coupling", False)))
                    context = build_order_context(observation, number, fit_settings,
                                                  DataPaths.bootstrapped(root),
                                                  profiles[settings["profile"]],
                                                  stellars[stellar_name])
                model = context["model"]
                parameters = TelluricParameters(
                    log_column_scales={s: float(row["log_column_scales"].get(s, 0.0))
                                       for s in model.species},
                    velocity_kms=float(row["velocity_kms"]), wavelength_stretch=0.0,
                    lsf_sigma_kms=float(row["lsf_sigma_kms"]),
                    continuum_coeffs=np.asarray(row["continuum_coeffs"], dtype=float),
                    log_jitter=float(row["log_jitter"]),
                    stellar_velocity_kms=float(row["stellar_velocity_kms"]))
                zenith = float(row["zenith_angle_deg"])
                exact = model.precompute_opacity(parameters)
                grid = np.asarray(model.wavenumber_cm1)
                grid_pieces = {k: np.asarray(v) for k, v in
                               exact.species_transmission(parameters, zenith).items()}
                grid_total = np.asarray(exact.transmission(parameters, zenith))
                product = np.prod(np.stack(list(grid_pieces.values())), axis=0)
                split_drift = float(np.max(np.abs(product - grid_total)))
                if split_drift > args.tolerance:
                    raise ValueError(f"species do not multiply back: {split_drift:.2e}")

                npz = run / f"{stem}_{band}{number}.npz"
                with np.load(npz) as stored:
                    saved = {k: np.asarray(stored[k]) for k in stored.files}
                nu = saved["wavenumber_cm1"]
                total = np.interp(nu, grid, grid_total)
                record_drift = float(np.max(np.abs(total - saved["transmission"])))
                if record_drift > args.tolerance:
                    raise ValueError(f"rebuilt transmission misses the cached one by "
                                     f"{record_drift:.2e}: wrong parameters?")
                pieces = {k: np.interp(nu, grid, v) for k, v in grid_pieces.items()}

                # The cache runs in ascending wavenumber, IGRINSOrder in
                # ascending wavelength; match them by wavenumber, never by
                # position, and refuse rather than guess.
                extracted = observation.order(number)
                order_nu = 1.0e7 / np.asarray(extracted.wavelength_vacuum_nm)
                axis = np.argsort(order_nu)
                if order_nu.size != nu.size or not np.allclose(order_nu[axis], nu, atol=1e-6, rtol=0):
                    raise ValueError("cached pixels are not the order's pixels")
                columns = (np.asarray(extracted.pixel)[axis] if extracted.pixel is not None
                           else np.arange(nu.size)[::-1])

                mask = saved["mask"].astype(bool)
                residual_z = np.where(mask, saved["residual"] / saved["uncertainty"], np.nan)
                reliable = saved["reliable"].astype(bool)
                arrays = {
                    "observed": np.where(mask, saved["observed"], np.nan),
                    # Outside the mask the degree-9 continuum is extrapolated
                    # and reaches 5e9 at the order ends; left in, it sets the
                    # uint16 scale and erases everything inside the mask.
                    "model": np.where(mask, saved["model_flux"], np.nan),
                    "continuum": np.where(mask, saved["continuum"], np.nan),
                    "stellar": np.where(mask, saved["stellar_only"], np.nan),
                    "transmission": saved["transmission"],
                    "residual": residual_z,
                }
                for optional in ("plp_telluric", "response_pattern", "stellar_continuum"):
                    if optional in saved:
                        arrays[optional] = saved[optional]
                for name, values in pieces.items():
                    arrays[f"species:{name}"] = values

                entry = {
                    "band": band, "order": number, "name": f"{band}{number}", "frame": stem,
                    "n": int(nu.size), "nu0": float(nu[0]), "arrays": {},
                    "chunk": chunk_index,
                    "v1": float(row["v1"]), "v2": float(row["v2"]),
                    "airmass": float(row["airmass"]), "zenith_angle_deg": zenith,
                    "free_species": row["free_species"],
                    "absent_species": row["absent_species"],
                    "lines": row["lines"],
                    "max_optical_depth": row["max_optical_depth"],
                    "log_column_scales": row["log_column_scales"],
                    "sigma": {k: v for k, v in row["sigma"].items()},
                    "at_bound": row["at_bound"],
                    "quality": {
                        **{k: row.get(k) for k in (
                            "residual_rms_over_noise", "residual_rms_over_noise_uncorrected",
                            "residual_rms", "pixel_sigma", "median_transmission", "reliable",
                            "kept", "pixels", "all_stages_converged", "lsf_sigma_kms",
                            "resolving_power_fitted", "velocity_kms", "stellar_velocity_kms",
                            "response_pattern_rms", "plp_rms_difference")},
                        "residual_z_rms": float(np.sqrt(np.nanmean(residual_z[reliable] ** 2)))
                        if reliable.any() else None,
                    },
                    "split_drift": split_drift, "record_drift": record_drift,
                    "overview": overview(nu, saved["observed"], saved["continuum"],
                                         saved["transmission"], mask),
                }
                payload = (nu - nu[0]).astype("<f4").tobytes()
                entry["axis_offset"] = chunk_bytes
                chunk.append(payload); chunk_bytes += len(payload)
                payload = columns.astype("<u2").tobytes()
                entry["column_offset"] = chunk_bytes
                chunk.append(payload); chunk_bytes += len(payload)
                for name, values in arrays.items():
                    payload, lo, hi = quantize(values)
                    entry["arrays"][name] = {"offset": chunk_bytes, "lo": lo, "hi": hi}
                    chunk.append(payload); chunk_bytes += len(payload)
                packed = np.packbits(mask).tobytes()
                entry["mask_offset"] = chunk_bytes
                chunk.append(packed); chunk_bytes += len(packed)
                packed = np.packbits(reliable).tobytes()
                entry["reliable_offset"] = chunk_bytes
                chunk.append(packed); chunk_bytes += len(packed)
                # uint16 arrays need 2-byte alignment for a typed-array view.
                if chunk_bytes % 4:
                    pad = 4 - chunk_bytes % 4
                    chunk.append(b"\0" * pad); chunk_bytes += pad
                manifest.append(entry)
                status = f"split {split_drift:.1e}, record {record_drift:.1e}"
            except Exception as exc:  # one order-frame must not stop the export
                failures.append({"band": band, "order": number, "frame": stem,
                                 "error": str(exc), "traceback": traceback.format_exc()[-1200:]})
                status = f"FAILED: {exc}"
            print(f"[{number_job}/{len(jobs)}] {band}{number} {stem}  {status}  "
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)
        if chunk_bytes >= CHUNK_BYTES:
            flush()
        context = None
        jax.clear_caches()

    flush()
    (args.output / "manifest.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": runs, "entries": manifest, "frames": list(frames_out.values()),
        "failures": failures,
    }) + "\n", encoding="utf-8")
    print(f"\n{len(manifest)} entries, {len(failures)} failed, {chunk_index} chunks")


def merge(args) -> None:
    """Renumber each shard's chunks and concatenate the manifests. Chunk
    contents and the offsets inside them are untouched."""
    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("chunk_*.wasm"):
        stale.unlink()
    entries, frames, failures, runs, offset = [], {}, [], [], 0
    for shard in args.merge:
        blob = json.loads((shard / "manifest.json").read_text())
        runs = runs or blob["runs"]
        chunks = sorted(shard.glob("chunk_*.wasm"))
        for index, path in enumerate(chunks):
            shutil.copyfile(path, args.output / f"chunk_{offset + index:03d}.wasm")
        entries.extend({**e, "chunk": e["chunk"] + offset} for e in blob["entries"])
        for f in blob["frames"]:
            frames.setdefault((f["band"], f["stem"]), f)
        failures.extend(blob["failures"])
        offset += len(chunks)
    entries.sort(key=lambda e: (e["band"], e["order"], e["frame"]))
    # The overview's bin centres belong to the order, not the frame: every
    # frame of a night shares the wavelength solution. Ship them once.
    overview_nu = {}
    for e in entries:
        overview_nu.setdefault(e["name"], e["overview"].pop("nu"))
    (args.output / "manifest.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": runs,
        "entries": entries, "overview_nu": overview_nu, "frames": sorted(frames.values(), key=lambda f: (f["band"], f["mjd"])),
        "failures": failures,
        "note": ("arrays are uint16, value = lo + code * (hi - lo) / 65535; the axis is "
                 "float32 offsets from nu0 in cm-1, ascending; columns are uint16 detector "
                 "columns; mask and reliable are bit-packed MSB-first; overview arrays are "
                 "uint8 base64, code / 254 * top (1.3 for flux, 1.0 for transmission), 255 = none."),
    }, separators=(",", ":")) + "\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in args.output.glob("chunk_*.wasm"))
    print(f"{len(entries)} entries, {len(failures)} failed, {offset} chunks, "
          f"{size / 1e6:.1f} MB -> {args.output}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", default=None,
                        help="a fit_igrins_standard output directory; repeat for H and K")
    parser.add_argument("--merge", type=Path, nargs="+", default=None,
                        help="join these shard bundles instead of exporting")
    parser.add_argument("--output", type=Path, default=root / "data/review_igrins")
    parser.add_argument("--tolerance", type=float, default=2.0e-6)
    parser.add_argument("--shard", default=None, metavar="I/N")
    parser.add_argument("--limit", type=int, default=0, help="orders, for a trial run")
    parser.add_argument("--compilation-cache", default=str(root / ".jax-cache"))
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        args.run = args.run or [root / "data/corrected/igrins/ladder_a0v",
                                root / "data/corrected/igrins/ladder_k_a0v"]
        export(args, root)


if __name__ == "__main__":
    main()
