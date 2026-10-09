#!/usr/bin/env python
"""Bundle the Arcturus atlas fit for the review page, page-epoch by page-epoch.

Reads the committed full-coverage record (`data/corrected/atlas/arcturus_atlas.h5`,
822,928 pixels on 598 page-epochs) and the per-page `.npz` cache beside it --
never `docs/arcturus_atlas_summary.json`, which is the superseded trimmed run.

The cache holds the total transmission and nothing per species, so each page
*window* is rebuilt from the record alone -- its grid, line selection, opacity
backend and continuum, exactly as `export_transmission_hdf5.py` does, whose
`--check` reproduces the cached transmission to 3e-16 -- and the transmission is
split with `TelluricModel.species_transmission`. Both epochs of a page share one
window, so one compilation serves two rows. Two checks run on every row and a
failure writes the row as failed rather than shipping it: the species must
multiply back to the total on the model grid (optical depth is additive), and
that total, on the pixels, must reproduce the cached `transmission` -- which is
what proves the record's parameters made the cached arrays.

What ships per row, uint16 with a per-array scale (1.5e-5 of the range against a
median 0.0045 pixel sigma): observed, the fitted model, the stellar-only model,
the fitted Chebyshev continuum (all four blanked outside the fit mask before
quantizing -- see `docs/review_page/CLAUDE.md`), the effective transmission
`model_flux / stellar_only` (the operator the correction divides by, kept on
every pixel so saturated cores show), the unconvolved transmission, residual over
pixel sigma, the atlas authors' own telluric spectrum (NaN above the fit's
`telluric_ceiling`, where their division blew up), each species' unconvolved
transmission, the pixel axis as float32 offsets (the atlas prints wavenumbers
to 0.01 cm-1, so steps vary by 3%), the atlas row index of every pixel, and the
fit and reliable masks. The page derives the corrected spectrum and the
continuum-normalized stellar model from those, as the solar page does.

    JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache uv run python scripts/export_arcturus_review.py \\
        --shard 0/4 --output data/review_arcturus_0
    UV_CACHE_DIR=.uv-cache uv run python scripts/export_arcturus_review.py \\
        --merge data/review_arcturus_? --output data/review_arcturus

`--no-species` skips the rebuild (numpy only, a minute) and ships the total.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

CHUNK_BYTES = 1_500_000
# Arrays whose NaN means something the mask does not carry keep it: finite
# values use codes 0-65534 and NaN is 65535. Elsewhere NaN writes code 0, which
# is safe only because the page blanks masked pixels itself.
NAN_CODE = 65535
NAN_ARRAYS = ("atlas_telluric",)
PARAMETER_FACTS = ("velocity_kms", "stellar_velocity_kms", "lsf_sigma_kms",
                   "wavelength_stretch", "log_jitter")


def quantize(values: np.ndarray, keep_nan: bool = False) -> tuple[bytes, float, float]:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    lo = float(np.min(values[good])) if good.any() else 0.0
    hi = float(np.max(values[good])) if good.any() else 1.0
    if hi <= lo:
        hi = lo + 1.0e-12
    top = NAN_CODE - 1 if keep_nan else 65535
    codes = np.clip(np.round(np.where(good, (values - lo) / (hi - lo), 0.0) * top), 0, top)
    codes = codes.astype("<u2")
    if keep_nan:
        codes[~good] = NAN_CODE
    return codes.tobytes(), lo, hi


def rounded(value: float, digits: int = 6):
    """Significant digits for the manifest; JSON has no NaN, so None."""
    if not math.isfinite(value):
        return None
    if value == 0.0:
        return 0.0
    return round(value, -int(math.floor(math.log10(abs(value)))) + (digits - 1))


def text_list(value) -> list[str]:
    text = value.decode() if isinstance(value, bytes) else str(value)
    return [s for s in text.replace("+", ",").split(",") if s]


class WindowRebuilder:
    """The forward model of one page window, rebuilt from the record alone.

    Same construction as `export_transmission_hdf5.py` (which `--check`s it
    against the cache to 3e-16); repeated here rather than imported because no
    script imports another, and checked here again on every row.
    """

    def __init__(self, record, root: Path):
        import jax

        from tellurix import load_atmosphere_csv, file_sha256

        self.jax = jax
        self.record = record
        self.config, self.physics, self.inputs = record.config, record.physics, record.inputs
        if self.physics.get("line_coupling") or self.physics.get("o2_cia"):
            raise SystemExit("this record uses physics the rebuild below does not add")
        profile_path = Path(self.inputs["profile"])
        if not profile_path.exists():
            profile_path = root / "data/profiles" / profile_path.name
        if file_sha256(profile_path) != self.inputs["profile_sha256"]:
            raise SystemExit(f"{profile_path} is not the profile the run used")
        self.profile = load_atmosphere_csv(profile_path)
        self.line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
        self.mt_ckd = root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc"
        if file_sha256(self.mt_ckd) != self.inputs["mt_ckd_sha256"]:
            raise SystemExit("the MT_CKD file is not the one the run used")
        self.species_all = [n[len("log_column_"):] for n in record.pages.dtype.names
                            if n.startswith("log_column_")]

    def build(self, v1: float, v2: float):
        from tellurix import (AER_MOLECULE_IDS, AERLineDatabase, ExoJAXOpacityBackend,
                              MTCKDWaterContinuum, TelluricModel, constant_velocity_grid,
                              trim_wavenumber_grid)

        config, physics, profile = self.config, self.physics, self.profile
        grid = trim_wavenumber_grid(
            constant_velocity_grid(1.0e7 / v2, 1.0e7 / v1,
                                   resolving_power=float(config["resolving_power"]),
                                   samples_per_resolution=float(config["samples_per_resolution"]),
                                   margin_cm1=float(config["margin_cm1"])),
            v1, v2, float(config["grid_margin_cm1"]))
        if float(config.get("line_budget", 0.0)) > 0.0:
            raise SystemExit("this record pruned lines; the rebuild below does not")
        databases = {}
        for name in sorted(self.species_all):
            stem = f"{AER_MOLECULE_IDS[name]:02d}_{name}"
            try:
                databases[name] = AERLineDatabase(self.line_root / stem / stem, name, (v1, v2),
                                                  margin_cm1=float(config["margin_cm1"]))
            except ValueError as exc:
                if not str(exc).startswith(f"no {name} lines found"):
                    raise
        opacity = ExoJAXOpacityBackend.prepare(
            databases, grid, methods=physics["opacity_method"],
            temperature_range_k=(float(np.min(profile.temperature_k)),
                                 float(np.max(profile.temperature_k))),
            maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
            vectorize_layers=bool(physics["vectorize_layers"]),
            mixed_precision=bool(physics["mixed_precision"]),
            pressure_shift=bool(physics["pressure_shift"]))
        continuum = (None if physics["accuracy_mode"] == "fast"
                     else MTCKDWaterContinuum.from_netcdf(self.mt_ckd, grid))
        model = TelluricModel(profile, grid, opacity, continuum=continuum,
                              accuracy_mode=physics["accuracy_mode"],
                              max_lsf_sigma_kms=float(physics["max_lsf_sigma_kms"]),
                              pixel_integration=physics["pixel_integration"])

        # One compile per window: the scales are operands, so both epochs reuse
        # it. Total and pieces in one function so the cross sections are shared.
        @self.jax.jit
        def evaluate(scales, angle, _model=model):
            from tellurix import TelluricParameters

            p = TelluricParameters(log_column_scales=scales, velocity_kms=0.0,
                                   wavelength_stretch=0.0, lsf_sigma_kms=1.0,
                                   continuum_coeffs=np.zeros(1), log_jitter=0.0,
                                   stellar_velocity_kms=0.0)
            return _model.transmission(p, angle), _model.species_transmission(p, angle)

        return grid, model, evaluate


def export(args, root: Path) -> None:
    from tellurix import read_record
    from tellurix_fts import read_arcturus_page
    from tellurix.model import BoxcarFTSInstrumentProfile

    record = read_record(args.record)
    pages = record.pages
    names, sigma_all = list(record.parameter_names), record.sigma
    config = record.config
    ceiling = float(config.get("telluric_ceiling", 1.05))
    atlas_root = Path(record.inputs["atlas_root"])

    windows: dict[tuple[float, float], list[int]] = {}
    for index in range(len(pages)):
        key = (round(float(pages["v1"][index]), 6), round(float(pages["v2"][index]), 6))
        windows.setdefault(key, []).append(index)
    ordered = sorted(windows)
    if args.shard:
        i, _, n = args.shard.partition("/")
        ordered = ordered[int(i):: int(n)]
    if args.limit:
        ordered = ordered[: args.limit]

    rebuilder = None
    if args.species:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax

        if args.compilation_cache:
            Path(args.compilation_cache).mkdir(parents=True, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache))
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        rebuilder = WindowRebuilder(record, root)

    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("chunk_*.wasm"):
        stale.unlink()
    manifest, failures = [], []
    chunk, chunk_bytes, chunk_index = [], 0, 0
    started = time.time()

    def flush() -> None:
        nonlocal chunk, chunk_bytes, chunk_index
        if chunk:
            (args.output / f"chunk_{chunk_index:03d}.wasm").write_bytes(b"".join(chunk))
            chunk, chunk_bytes = [], 0
            chunk_index += 1

    def put(payload: bytes) -> int:
        nonlocal chunk_bytes
        offset = chunk_bytes
        chunk.append(payload)
        chunk_bytes += len(payload)
        # Typed-array views need their element size's alignment.
        if chunk_bytes % 4:
            pad = 4 - chunk_bytes % 4
            chunk.append(b"\0" * pad)
            chunk_bytes += pad
        return offset

    for position, key in enumerate(ordered, 1):
        v1, v2 = key
        built = None
        if rebuilder is not None:
            try:
                built = rebuilder.build(v1, v2)
            except Exception as exc:
                for index in windows[key]:
                    failures.append({"key": ":".join(record.key(index)),
                                     "error": f"rebuild: {exc}",
                                     "traceback": traceback.format_exc()[-1200:]})
                print(f"[{position}/{len(ordered)}] {v1:.1f} FAILED rebuild: {exc}", flush=True)
                continue
        for index in windows[key]:
            row = pages[index]
            page, epoch = record.key(index)
            try:
                with np.load(args.record.parent / f"{page}_{epoch}.npz") as stored:
                    saved = {k: np.asarray(stored[k]) for k in stored.files}
                nu = saved["wavenumber_cm1"]
                mask = saved["mask"].astype(bool)
                reliable = saved["reliable"].astype(bool)
                free = text_list(row["free_species"])
                at_bound = text_list(row["at_bound"])

                pieces, split_drift, record_drift = {}, None, None
                if built is not None:
                    grid, model, evaluate = built
                    present = [s for s in rebuilder.species_all if s in model.species]
                    scales = {s: float(row[f"log_column_{s}"]) for s in present}
                    total, grid_pieces = evaluate(scales, float(config["zenith_angle_deg"]))
                    total = np.asarray(total)
                    grid_pieces = {k: np.asarray(v) for k, v in grid_pieces.items()}
                    product = np.prod(np.stack(list(grid_pieces.values())), axis=0)
                    split_drift = float(np.max(np.abs(product - total)))
                    if split_drift > args.tolerance:
                        raise ValueError(f"species do not multiply back: {split_drift:.2e}")
                    again = np.interp(nu, grid, total)
                    record_drift = float(np.max(np.abs(again - saved["transmission"])))
                    if record_drift > args.tolerance:
                        raise ValueError(f"rebuilt transmission misses the cached one by "
                                         f"{record_drift:.2e}: wrong parameters?")
                    pieces = {k: np.interp(nu, grid, v) for k, v in grid_pieces.items()}

                # The atlas row of every pixel, for the top axis. The full-coverage
                # run fits every row, so this is 0..n-1 -- but it is matched by
                # wavenumber against the page file, never assumed.
                pixel_index, pixel_source = np.arange(nu.size), "assumed"
                page_path = atlas_root / page
                if page_path.exists():
                    atlas = read_arcturus_page(page_path, epoch)
                    start = int(np.searchsorted(atlas.wavenumber_vacuum_cm1, nu[0] - 1e-6))
                    stop = start + nu.size
                    if (stop > atlas.wavenumber_vacuum_cm1.size or not np.allclose(
                            atlas.wavenumber_vacuum_cm1[start:stop], nu, atol=1e-6, rtol=0)):
                        raise ValueError("cached pixels are not a run of the page's rows")
                    pixel_index, pixel_source = np.arange(start, stop), "atlas page"
                    page_rows = int(atlas.wavenumber_vacuum_cm1.size)
                else:
                    page_rows = None

                star = saved["stellar_only"]
                effective = np.where(np.abs(star) > 1e-12,
                                     saved["model_flux"] / np.where(np.abs(star) > 1e-12, star, 1.0),
                                     np.nan)
                sigma = float(row["pixel_sigma"])
                atlas_telluric = np.where(saved["atlas_telluric"] > ceiling, np.nan,
                                          saved["atlas_telluric"])
                arrays = {
                    "observed": np.where(mask, saved["observed"], np.nan),
                    "model": np.where(mask, saved["model_flux"], np.nan),
                    "stellar": np.where(mask, star, np.nan),
                    "continuum": np.where(mask, saved["continuum"], np.nan),
                    "effective": effective,
                    "transmission": saved["transmission"],
                    "residual": np.where(mask, saved["residual"] / sigma, np.nan),
                    "atlas_telluric": atlas_telluric,
                }
                for name, values in pieces.items():
                    arrays[f"species:{name}"] = values

                center = 0.5 * (float(nu[0]) + float(nu[-1]))
                sinc_r = BoxcarFTSInstrumentProfile(float(row["mopd_cm"]), center).resolving_power
                # The fitted Gaussian in quadrature with the sinc, as
                # docs/arcturus_fit.md combines them to reach R = 100,504.
                c_kms = 299792.458
                fwhm_kms = math.hypot(c_kms / sinc_r, 2.354820045 * float(row["lsf_sigma_kms"]))
                free_set = set(free)
                species_facts = {}
                for s in rebuilder.species_all if rebuilder else [
                        n[len("log_column_"):] for n in pages.dtype.names
                        if n.startswith("log_column_")]:
                    if s not in free_set and f"species:{s}" not in arrays:
                        continue
                    j = names.index(s)
                    species_facts[s] = {
                        "log_scale": rounded(float(row[f"log_column_{s}"])),
                        "sigma": rounded(float(sigma_all[index, j]), 3) if s in free_set else None,
                        "free": s in free_set, "at_bound": s in at_bound}
                residual_z = arrays["residual"]
                entry = {
                    "key": f"{page}:{epoch}", "page": page, "epoch": epoch,
                    "n": int(nu.size), "nu0": float(nu[0]),
                    "v1": float(nu[0]), "v2": float(nu[-1]),
                    "pixel_first": int(pixel_index[0]), "pixel_source": pixel_source,
                    "page_rows": page_rows,
                    "chunk": chunk_index, "arrays": {},
                    "mopd_cm": rounded(float(row["mopd_cm"])),
                    "sinc_resolving_power": round(sinc_r),
                    "resolving_power": round(c_kms / fwhm_kms),
                    "parameters": {k: rounded(float(row[k])) for k in PARAMETER_FACTS},
                    "parameter_sigma": {k: rounded(float(sigma_all[index, names.index(k)]), 3)
                                        for k in PARAMETER_FACTS if k in names},
                    "continuum_coeffs": [rounded(float(c)) for c in row["continuum_coeffs"]],
                    "free_species": free, "species": species_facts, "at_bound": at_bound,
                    "quality": {
                        "residual_rms_over_noise": rounded(float(row["residual_rms_over_noise"])),
                        "residual_rms": rounded(float(row["residual_rms"])),
                        "pixel_sigma": rounded(sigma),
                        "reduced_chi2": rounded(float(row["reduced_chi2"])),
                        "median_transmission": rounded(float(row["median_transmission"])),
                        "continuum_level": rounded(float(row["continuum_level"])),
                        "condition_number": rounded(float(row["condition_number"]), 3),
                        "pixels": int(row["pixels"]), "fitted": int(mask.sum()),
                        "reliable": int(row["reliable"]),
                        "all_stages_converged": bool(row["all_stages_converged"]),
                        "negligible_telluric": bool(row["negligible_telluric"]),
                        "residual_z_rms_reliable": rounded(float(np.sqrt(np.nanmean(
                            residual_z[reliable] ** 2)))) if reliable.any() else None,
                    },
                    "split_drift": split_drift, "record_drift": record_drift,
                }
                entry["axis_offset"] = put((nu - nu[0]).astype("<f4").tobytes())
                for name, values in arrays.items():
                    keep = name in NAN_ARRAYS
                    payload, lo, hi = quantize(values, keep_nan=keep)
                    entry["arrays"][name] = {"offset": put(payload), "lo": lo, "hi": hi}
                    if keep:
                        entry["arrays"][name]["nan_code"] = NAN_CODE
                entry["mask_offset"] = put(np.packbits(mask).tobytes())
                entry["reliable_offset"] = put(np.packbits(reliable).tobytes())
                manifest.append(entry)
                status = (f"{len(pieces)} pieces, split {split_drift:.1e}, record {record_drift:.1e}"
                          if built is not None else "total only")
            except Exception as exc:  # one row must not stop the export
                failures.append({"key": f"{page}:{epoch}", "error": str(exc),
                                 "traceback": traceback.format_exc()[-1200:]})
                status = f"FAILED: {exc}"
            print(f"[{position}/{len(ordered)}] {page} {epoch}  {status}  "
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)
        if chunk_bytes >= CHUNK_BYTES:
            flush()
        if built is not None:
            built = None
            rebuilder.jax.clear_caches()

    flush()
    (args.output / "manifest.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "record": str(args.record.relative_to(root) if args.record.is_relative_to(root)
                      else args.record),
        "species_split": bool(args.species),
        "entries": manifest, "failures": failures,
    }) + "\n", encoding="utf-8")
    print(f"\n{len(manifest)} entries, {len(failures)} failed, {chunk_index} chunks")


def merge(args, root: Path) -> None:
    """Renumber each shard's chunks and concatenate the manifests. Chunk
    contents and the offsets inside them are untouched."""
    import h5py

    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("chunk_*.wasm"):
        stale.unlink()
    entries, failures, offset, head = [], [], 0, None
    for shard in args.merge:
        blob = json.loads((shard / "manifest.json").read_text())
        head = head or blob
        chunks = sorted(shard.glob("chunk_*.wasm"))
        for index, path in enumerate(chunks):
            shutil.copyfile(path, args.output / f"chunk_{offset + index:03d}.wasm")
        entries.extend({**e, "chunk": e["chunk"] + offset} for e in blob["entries"])
        failures.extend(blob["failures"])
        offset += len(chunks)
    keys = [e["key"] for e in entries]
    if len(set(keys)) != len(keys):
        raise SystemExit("a page-epoch appears in more than one shard")
    entries.sort(key=lambda e: (e["v1"], e["epoch"]))

    # Run-level facts the page shows once, from the record rather than a shard.
    with h5py.File(args.record) as handle:
        rows = handle["pages"][:]
        config = {k: (v.decode() if isinstance(v, bytes) else v)
                  for k, v in handle["config"].attrs.items()}
    pixels = int(rows["pixels"].sum())
    run = {
        "page_epochs": int(rows.size),
        "pages": int(np.unique(rows["page"]).size),
        "pixels": pixels,
        "median_residual_rms_over_noise": round(float(np.median(rows["residual_rms_over_noise"])), 3),
        "at_bound_fraction": round(float(np.mean(rows["at_bound"] != b"")), 4),
        "min_transmission": float(config.get("min_transmission", 0.15)),
        "telluric_ceiling": float(config.get("telluric_ceiling", 1.05)),
    }
    if pixels != 822_928:
        print(f"note: {pixels} pixels -- not the full-coverage record's 822,928")
    total = sum(f.stat().st_size for f in args.output.glob("chunk_*.wasm"))
    (args.output / "manifest.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "record": head["record"],
        "species_split": head["species_split"], "run": run,
        "entries": entries, "failures": failures,
        "note": ("arrays are uint16, value = lo + code * (hi - lo) / 65535 (codes up to "
                 "65534 and 65535 = NaN where nan_code is given); the axis is float32 "
                 "offsets from nu0 in cm-1, ascending; the atlas row of pixel i is "
                 "pixel_first + i; mask and reliable are bit-packed MSB-first."),
    }, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest_size = (args.output / "manifest.json").stat().st_size
    print(f"{len(entries)} entries, {len(failures)} failed, {offset} chunks, "
          f"{total / 1e6:.1f} MB + {manifest_size / 1e6:.2f} MB manifest -> {args.output}")
    if total + manifest_size > 50e6:
        raise SystemExit("the bundle is over 50 MB")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path,
                        default=root / "data/corrected/atlas/arcturus_atlas.h5")
    parser.add_argument("--merge", type=Path, nargs="+", default=None,
                        help="join these shard bundles instead of exporting")
    parser.add_argument("--output", type=Path, default=root / "data/review_arcturus")
    parser.add_argument("--species", action=argparse.BooleanOptionalAction, default=True,
                        help="rebuild each window to split the transmission by species")
    parser.add_argument("--tolerance", type=float, default=2.0e-6)
    parser.add_argument("--shard", default=None, metavar="I/N")
    parser.add_argument("--limit", type=int, default=0, help="windows, for a trial run")
    parser.add_argument("--compilation-cache", default=str(root / ".jax-cache"))
    args = parser.parse_args()
    args.record = args.record.resolve()
    if args.merge:
        merge(args, root)
    else:
        export(args, root)


if __name__ == "__main__":
    main()
