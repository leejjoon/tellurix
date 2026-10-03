#!/usr/bin/env python
"""Bundle a night's IGRINS science-mode corrections for the target review page.

The page asks a different question from the standards' review: on a target the
residual is mostly the star's own lines, so "does the model fit" says little.
What it shows instead is whether the telluric absorption is *gone* -- the
corrected spectrum, the operator that was divided out, where the deepest
telluric lines fall, which pixels the clip set aside, and how the same order
reads in its neighbours and under another calibration.

Everything comes from the science driver's cached arrays and its record; no
model is rebuilt. Two things are reconstructed, both from saved quantities:

* the clip. `fit_igrins_science.py --clip-sigma` drops pixels more than
  ``clip_sigma`` below the model, plus ``clip_pixels`` either side, when it
  measures the frame's shifts; the mask is not saved. It is recomputed here
  from the *final* model and the record's jitter, so it is what the clip would
  remove now, which differs from what it removed by however far the shifts
  moved (0.01 in log water at most on this night).
* the observed flux, as corrected x effective transmission, which is exact to
  the quantization.

Variants of one frame -- the full calibration, a target model in the source
slot, a three-standard calibration -- are separate entries, so the page can lay
one over another. Within a night every frame of an order shares the PLP's
wavelength solution, so each order's axis and detector columns are shipped
once, after checking that they really are shared.

    UV_CACHE_DIR=.uv-cache uv run python scripts/export_igrins_target_review.py
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHUNK_BYTES = 1_500_000
OVERVIEW_BINS = 192
TICKS_PER_ORDER = 40
TICK_MINIMUM_DEPTH = 0.05

# (variant key, label, run directory). One run may hold both targets and
# standards; the frame's own OBJTYPE says which.
DEFAULT_VARIANTS = [
    ("full", "full calibration (10 standards)", "data/corrected/igrins/science_dct2018"),
    ("full", "full calibration (10 standards)", "data/corrected/igrins/science_dct2018_standards"),
    ("model", "full calibration, target model as source", "data/corrected/igrins/science_dct2018_model"),
    ("sparse", "sparse calibration (3 standards + master pattern)",
     "data/corrected/igrins/science_dct2018_sparse"),
]


def quantize(values: np.ndarray, lo: float | None = None, hi: float | None = None):
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if lo is None:
        lo = float(np.min(values[good])) if good.any() else 0.0
    if hi is None:
        hi = float(np.max(values[good])) if good.any() else 1.0
    if hi <= lo:
        hi = lo + 1.0e-12
    scaled = np.where(good, (np.clip(values, lo, hi) - lo) / (hi - lo), 0.0)
    return np.round(scaled * 65535).astype("<u2").tobytes(), float(lo), float(hi)


def robust_range(values: np.ndarray, where: np.ndarray, pad: float = 0.5):
    """The span of the reliable pixels, widened by ``pad`` of itself. A pixel
    outside it -- a corrected value in a saturated core, where the division is
    by almost nothing -- is clipped to the edge instead of setting the scale."""
    v = values[where & np.isfinite(values)]
    if v.size < 16:
        v = values[np.isfinite(values)]
    if not v.size:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [0.1, 99.9])
    span = max(hi - lo, 1e-6)
    return float(lo - pad * span), float(hi + pad * span)


def telluric_ticks(effective: np.ndarray, mask: np.ndarray) -> list[int]:
    """The deepest local minima of the operator that was divided out."""
    t = np.where(mask, effective, np.nan)
    inner = t[1:-1]
    minima = np.flatnonzero((inner < t[:-2]) & (inner <= t[2:])
                            & (1.0 - inner > TICK_MINIMUM_DEPTH)) + 1
    minima = minima[np.argsort(t[minima])][:TICKS_PER_ORDER]
    return sorted(int(i) for i in minima)


def overview(corrected, continuum, effective, reliable) -> dict:
    n = corrected.size
    edges = np.linspace(0, n, OVERVIEW_BINS + 1).astype(int)
    ratio = np.where(reliable, corrected / np.maximum(continuum, 1e-12), np.nan)
    lo, hi, t = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        r = ratio[a:b]
        r = r[np.isfinite(r)]
        lo.append(np.percentile(r, 2) if r.size else np.nan)
        hi.append(np.percentile(r, 98) if r.size else np.nan)
        t.append(np.nanmean(effective[a:b]))

    def pack(values, top):
        v = np.asarray(values, dtype=float)
        codes = np.where(np.isfinite(v), np.clip(np.round(v / top * 254), 0, 254), 255)
        return base64.b64encode(codes.astype(np.uint8).tobytes()).decode()

    return {"lo": pack(lo, 1.3), "hi": pack(hi, 1.3), "t": pack(t, 1.0)}


def plp_a0v(spec_path: Path, cache: dict):
    """The PLP's own A0V-divided spectrum for this frame, if it was fetched
    (download_rrisa_standard.py --extra spec_a0v.fits), and the standard it
    divided by. It shares none of this package's machinery, which is the point."""
    path = spec_path.with_name(spec_path.name.replace(".spec.fits", ".spec_a0v.fits"))
    if path not in cache:
        if not path.exists():
            cache[path] = None
        else:
            from astropy.io import fits

            with fits.open(path) as hdus:
                header = hdus["A0V_SPEC"].header
                cache[path] = {
                    "wavelength_um": np.asarray(hdus["WAVELENGTH"].data, dtype=float),
                    "divided": np.asarray(hdus["SPEC_DIVIDE_A0V"].data, dtype=float),
                    "normalized": np.asarray(hdus["SPEC_DIVIDE_CONT"].data, dtype=float),
                    "standard": {"object": str(header.get("OBJECT", "")).strip(),
                                 "date_obs": str(header.get("DATE-OBS", "")),
                                 "file": path.name},
                }
    return cache[path]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=root / "data/review_igrins_targets")
    args = parser.parse_args()

    from tellurix_igrins import read_igrins_observation
    from tellurix.record import read_record

    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("*.wasm"):
        stale.unlink()

    jobs, frames, variants = [], {}, {}
    for variant, label, run in DEFAULT_VARIANTS:
        run = root / run
        variants[variant] = label
        records = {}
        for band in ("H", "K"):
            path = run / f"record_{band}.h5"
            if path.exists():
                pages = read_record(path).pages
                records.update({(p["frame"].decode(), p["order"].decode()): p for p in pages})
        for summary in sorted(run.glob("*_science_summary.json")):
            blob = json.loads(summary.read_text())
            obs = blob["observation"]
            stem = Path(obs["path"]).name.split(".")[0]
            band, number = obs["band"], stem.split("_")[-1]
            frames.setdefault(number, {
                "frame": number, "object": obs["object"],
                "kind": "target" if obs.get("object_type") == "TAR" else "standard",
                "date_obs": obs["date_obs"], "mjd": obs["mjd"], "bands": {}, "variants": {}})
            frames[number]["bands"][band] = {"airmass": obs["airmass"], "path": obs["path"]}
            final = blob["shifts"][blob["shifts"].get("final", "frame")]
            frames[number]["variants"].setdefault(variant, {})[band] = {
                "velocity_shift_kms": final["velocity_kms"],
                "water_shift": final["log_column_H2O"],
                "co2_shift": final.get("log_column_CO2"),
                "band_agreement": blob.get("band_agreement"),
                "source": blob["settings"]["stellar"],
                "clip_sigma": blob["settings"].get("clip_sigma"),
                "clip_pixels": blob["settings"].get("clip_pixels", 2),
                "calibration": blob["calibration"]["path"],
                "standards": [s.split("_")[-1] for s in blob["calibration"].get("standards", [])],
            }
            for row in blob["results"]:
                jobs.append((variant, run, blob, stem, band, number, row,
                             records.get((stem, row["name"]))))
    # Neighbouring orders of one frame and variant land in the same chunk,
    # which is what the overlap check loads together.
    jobs.sort(key=lambda j: (j[0], j[3][5:], j[4], j[6]["order"]))

    observations, axes, plp_cache = {}, {}, {}
    entries, chunk, chunk_bytes, chunk_index = [], [], 0, 0
    started = time.time()

    def flush():
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
        if chunk_bytes % 4:
            pad = 4 - chunk_bytes % 4
            chunk.append(b"\0" * pad)
            chunk_bytes += pad
        return offset

    for variant, run, blob, stem, band, number, row, record in jobs:
        name = row["name"]
        with np.load(run / f"{stem}_{name}.npz") as stored:
            d = {k: np.asarray(stored[k]) for k in stored.files}
        nu, mask, reliable = d["wavenumber_cm1"], d["mask"].astype(bool), d["reliable"].astype(bool)

        # The order's axis and columns, shared by every frame of the night.
        path = Path(blob["observation"]["path"])
        path = path if path.is_absolute() else root / path
        if (band, row["order"]) not in axes:
            if path not in observations:
                observations[path] = read_igrins_observation(path)
            extracted = observations[path].order(row["order"])
            order_nu = 1.0e7 / np.asarray(extracted.wavelength_vacuum_nm)
            axis = np.argsort(order_nu)
            if order_nu.size != nu.size or not np.allclose(order_nu[axis], nu, atol=1e-6, rtol=0):
                raise SystemExit(f"{stem} {name}: cached pixels are not the order's pixels")
            columns = (np.asarray(extracted.pixel)[axis] if extracted.pixel is not None
                       else np.arange(nu.size)[::-1])
            axes[(band, row["order"])] = (nu, columns)
        elif not (axes[(band, row["order"])][0].size == nu.size
                  and np.allclose(axes[(band, row["order"])][0], nu, atol=1e-6, rtol=0)):
            raise SystemExit(f"{stem} {name}: wavelength solution differs from the night's; "
                             "this bundle assumes one per order")

        effective = d["effective_transmission"]
        corrected = np.where(mask, d["corrected"], np.nan)
        uncertainty = np.where(mask, d["corrected_uncertainty"], np.nan)
        continuum = np.where(mask, d["continuum"], np.nan)
        source = np.where(mask, d["stellar_only"], np.nan)

        # The clip: the driver's own mask where it saved one, otherwise
        # recomputed from the final model and the fit's jitter -- which flags
        # about 1.7x as many, because the final continuum was refitted without
        # the clipped pixels and sits higher.
        settings = blob["settings"]
        clip = np.zeros(nu.size, dtype=bool)
        clip_source = "none"
        if "clipped" in d:
            clip = d["clipped"].astype(bool)
            clip_source = "saved"
        elif settings.get("clip_sigma") not in (None, "") and record is not None:
            clip_source = "recomputed"
            sigma = np.sqrt(d["uncertainty"] ** 2 + np.exp(2.0 * float(record["log_jitter"])))
            low = mask & ((d["observed"] - d["model_flux"]) / sigma < -float(settings["clip_sigma"]))
            width = 2 * int(settings.get("clip_pixels", 2)) + 1
            clip = mask & (np.convolve(low.astype(float), np.ones(width), mode="same") > 0)

        # The PLP's product on the same pixels: its arrays are indexed by
        # detector column, so take our columns and check the wavelengths agree.
        plp = plp_a0v(path, plp_cache)
        plp_normalized = None
        if plp is not None:
            columns = axes[(band, row["order"])][1]
            wavelength = 1.0e4 / nu
            rows = [r for r in range(plp["wavelength_um"].shape[0])
                    if np.allclose(plp["wavelength_um"][r, columns], wavelength, atol=1e-9, rtol=0)]
            if len(rows) == 1:
                values = plp["normalized"][rows[0], columns]
                plp_normalized = np.where(mask & np.isfinite(values), values, np.nan)
                frames[number].setdefault("plp", {})[band] = plp["standard"]

        lo, hi = robust_range(corrected, reliable)
        entry = {
            "variant": variant, "band": band, "order": int(row["order"]), "name": name,
            "frame": number, "n": int(nu.size), "chunk": chunk_index, "arrays": {},
            "v1": float(nu[0]), "v2": float(nu[-1]),
            "source": settings["stellar"],
            "quality": {k: row.get(k) for k in (
                "residual_rms_over_noise", "median_transmission", "reliable", "pixels",
                "clipped_fraction", "velocity_kms", "log_column_H2O",
                "log_column_H2O_interpolated", "log_column_H2O_per_order", "water_free")},
            "clip_fraction": float(clip.sum() / max(mask.sum(), 1)), "clip_source": clip_source,
            "ticks": telluric_ticks(effective, mask),
            "overview": overview(corrected, d["continuum"], effective, reliable),
        }
        if record is not None:
            entry["quality"]["lsf_sigma_kms"] = float(record["lsf_sigma_kms"])
            entry["quality"]["negligible_telluric"] = bool(record["negligible_telluric"])
            entry["quality"]["free_species"] = record["free_species"].decode() \
                if isinstance(record["free_species"], bytes) else str(record["free_species"])
        arrays = {
            "corrected": (corrected, lo, hi),
            "uncertainty": (uncertainty, 0.0, float(np.nanpercentile(
                uncertainty[reliable], 99.9) * 4) if reliable.any() else None),
            "effective": (effective, None, None),
            "continuum": (continuum, None, None),
            "source": (source, None, None),
            "model": (np.where(mask, d["model_flux"], np.nan), None, None),
        }
        if plp_normalized is not None:
            arrays["plp"] = (plp_normalized, *robust_range(plp_normalized, reliable))
        for key, (values, a, b) in arrays.items():
            payload, a, b = quantize(values, a, b)
            entry["arrays"][key] = {"offset": put(payload), "lo": a, "hi": b}
        entry["mask_offset"] = put(np.packbits(mask).tobytes())
        entry["reliable_offset"] = put(np.packbits(reliable).tobytes())
        entry["clip_offset"] = put(np.packbits(clip).tobytes())
        entries.append(entry)
        if chunk_bytes >= CHUNK_BYTES:
            flush()
    flush()

    # One file of axes and columns: float64 wavenumber, uint16 column, per order.
    axis_blob, axis_index, offset = [], {}, 0
    for (band, order), (nu, columns) in sorted(axes.items()):
        a, c = nu.astype("<f8").tobytes(), columns.astype("<u2").tobytes()
        pad = b"\0" * ((-len(c)) % 8)
        axis_index[f"{band}{order}"] = {"n": int(nu.size), "nu_offset": offset,
                                        "column_offset": offset + len(a)}
        axis_blob += [a, c, pad]
        offset += len(a) + len(c) + len(pad)
    (args.output / "axes.wasm").write_bytes(b"".join(axis_blob))

    overview_nu = {}
    for key, (nu, _) in axes.items():
        edges = np.linspace(0, nu.size, OVERVIEW_BINS + 1).astype(int)
        overview_nu[f"{key[0]}{key[1]}"] = [
            round(float(0.5 * (nu[a] + nu[min(b, nu.size) - 1])), 2)
            for a, b in zip(edges[:-1], edges[1:])]

    (args.output / "manifest.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "night": "2018-12-20 DCT",
        "variants": variants, "frames": sorted(frames.values(), key=lambda f: f["mjd"]),
        "entries": entries, "axes": axis_index, "overview_nu": overview_nu,
        "note": ("arrays are uint16, value = lo + code * (hi - lo) / 65535, clipped to [lo, hi]; "
                 "axes.wasm holds each order's float64 wavenumbers (ascending) and uint16 "
                 "detector columns; mask, reliable and clip are bit-packed MSB-first; "
                 "observed = corrected x effective."),
    }, separators=(",", ":")) + "\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in args.output.glob("*.wasm"))
    print(f"{len(entries)} entries, {len(frames)} frames, {chunk_index} chunks, "
          f"{size / 1e6:.1f} MB, {(time.time() - started):.0f} s -> {args.output}")


if __name__ == "__main__":
    main()
