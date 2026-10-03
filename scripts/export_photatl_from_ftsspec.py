#!/usr/bin/env python
"""Telluric-correct photatl from the ftsspec_901218_5 fit, without refitting it.

**photatl's observed column is ftsspec_901218_5.** On all 258 pages its
``total`` equals ``gain * ftsspec_5 + offset``, one gain and one offset per
page, on the same sample grid to 1.6e-5 cm-1 (photatl's single precision), with
a residual of at most 0.06 sigma of that file's own noise. Its air mass fitted
from photatl pages independently agrees: 1.99-2.01 from each of CO2, CH4, N2O
and O2 against 1.985 in file 5's header -- and water, which varies by the hour,
gives 1.991 too. docs/solar_fit_plan.md has the measurement.

So fitting photatl would be refitting file 5. This script transfers that fit
instead: every photatl pixel is a file-5 pixel, so the effective transmission
the file-5 fit applied carries over sample for sample, with no interpolation.

    corrected = total / effective_transmission

``effective_transmission`` is ``model_flux / stellar_only`` from the window of
the file-5 fit containing that pixel, as in every other tellurix export. It is
the *convolved* operator. The unconvolved transmission -- the multiplicand for
someone synthesizing their own spectrum -- is not repeated here, because it is
already on disk for exactly this atmosphere and wavenumber scale:
``ftsspec_901218_5_transmission.h5``.

The one thing photatl adds is the ``offset``: negative on 239 of 258 pages, so
the atlas authors lowered the zero level, most (to -1.3% of continuum) across
the 1.9 um water band. It is stored per page and not undone.

**Except across 5298-5402 cm-1** (wn5300-wn5375), where that zero level is the difference. There
file 5's saturated cores sit 0.6-1.4% above zero, photatl's sit at zero, and a
fit of photatl is measurably better (residual 0.70-0.98 of the fit without the
offset, where everywhere else the ratio is 0.997-1.003). A transmission fitted
to file 5 and divided into photatl there is inconsistent by 10-84 sigma. So
those pages come from a fit of photatl itself, passed as ``--patch``; everywhere
else a refit would only exchange file 5's windows for photatl's pages, which
moves the answer by about a sigma and improves nothing. ``source`` says which
fit each page came from.

    UV_CACHE_DIR=.uv-cache uv run python scripts/export_photatl_from_ftsspec.py \\
        --patch data/corrected/solar/photatl_band/photatl_band_summary.json --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tellurix import file_sha256  # noqa: E402
from tellurix_fts import read_fts_spectrum, read_photatl_page, robust_noise  # noqa: E402
from tellurix_fts.nso import is_photatl_page  # noqa: E402
from tellurix.quality import UPPER_BOUND_NOTE, species_at_upper_bound  # noqa: E402

NSO = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/nso")

# Per pixel: where its correction came from.
STATUS = {
    0: "fitted: the file-5 window containing this pixel was fitted and is in the record",
    1: "opaque: that window had no pixel above the transmission floor and is not in "
       "the record; effective_transmission is the fit's but must not be trusted",
    2: "not fitted: file 5 has no fitted window here (below 1876 cm-1 on wn1850)",
    4: "not usable: the file-5 window here had " + UPPER_BOUND_NOTE,
    3: "refitted: this page was fitted as photatl itself (see `source`), not "
       "transferred from file 5",
}
# The same sample to single precision. photatl stores wavenumbers as float32,
# whose step at 9000 cm-1 is 4.9e-4; the measured worst case is 1.6e-5.
MATCH_TOLERANCE_CM1 = 1.0e-4

ABOUT = """\
Telluric-corrected NSO photatl (Livingston & Wallace 1991, NSO TR 91-001), one
row per page, produced by tellurix from its fit of ftsspec_901218_5 -- which is
photatl's observed column, rescaled per page (see gain, offset) -- except the
1.9 um band pages `source` names, fitted as photatl itself. corrected =
observed / effective_transmission, where observed is photatl's `total` column in
the page's own intensity units. Use `reliable` before anything else: it is
False where the correction is untrustworthy (transmission below the floor,
saturated cores, windows the fit did not cover). The unconvolved transmission
for this atmosphere is ftsspec_901218_5_transmission.h5, on the same
wavenumber scale.
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_windows(summary_path: Path, npz_dir: Path) -> list[dict]:
    rows = json.loads(summary_path.read_text())["results"]
    windows = []
    for row in sorted(rows, key=lambda r: r["v1"]):
        if not row.get("npz"):
            continue
        with np.load(npz_dir / row["npz"]) as z:
            nu = z["wavenumber_cm1"]
            order = np.argsort(nu)
            model, star = z["model_flux"][order], z["stellar_only_pixels"][order]
            windows.append({
                "v1": row["v1"], "v2": row["v2"], "npz": row["npz"],
                "opaque": bool(row.get("opaque")),
                "railed": species_at_upper_bound(
                    row["at_bound"], {s: row["parameters"][s] for s in row["species"]}),
                "wavenumber_cm1": nu[order],
                "effective_transmission": model / star,
                "transmission_unconvolved": z["transmission_pixels"][order],
                "reliable": z["reliable"][order],
                "corrected_file5": z["corrected"][order],
                "model_flux": model,
            })
    return windows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photatl", type=Path, default=NSO / "photatl")
    parser.add_argument("--ftsspec", type=Path, default=NSO / "telluric_near_ir/ftsspec_901218_5.txt")
    parser.add_argument("--summary", type=Path,
                        default=root / "data/corrected/solar/ftsspec_901218_5_summary.json")
    parser.add_argument("--record", type=Path,
                        default=root / "data/corrected/solar/ftsspec_901218_5.h5")
    parser.add_argument("--output", type=Path, default=root / "data/corrected/photatl_corrected.h5")
    parser.add_argument("--patch", type=Path, action="append", default=[],
                        help="summary of a batch run over photatl pages; those pages take "
                             "their transmission from it instead of from file 5")
    parser.add_argument("--check", action="store_true",
                        help="verify the transfer against the file-5 fit's own corrected spectrum")
    args = parser.parse_args()

    import h5py

    fts = read_fts_spectrum(args.ftsspec)
    windows = load_windows(args.summary, args.summary.parent)
    starts = np.array([w["v1"] for w in windows])
    ends = np.array([w["v2"] for w in windows])
    paths = sorted((p for p in args.photatl.iterdir() if is_photatl_page(p)),
                   key=lambda p: int(p.name[2:]))
    patches = {}
    for summary in args.patch:
        for row in json.loads(summary.read_text())["results"]:
            if row.get("npz") and not row.get("opaque") and "error" not in row:
                patches[Path(row["file"]).name] = (summary, row)

    rows, worst_match, worst_check, worst_patch = [], 0.0, 0.0, 0.0
    for path in paths:
        page = read_photatl_page(path)
        nu, total = page.wavenumber_vacuum_cm1, page.total
        index = np.clip(np.searchsorted(fts.wavenumber_vacuum_cm1, nu), 1,
                        fts.wavenumber_vacuum_cm1.size - 1)
        index = np.where(np.abs(fts.wavenumber_vacuum_cm1[index - 1] - nu)
                         < np.abs(fts.wavenumber_vacuum_cm1[index] - nu), index - 1, index)
        match = float(np.max(np.abs(fts.wavenumber_vacuum_cm1[index] - nu)))
        if match > MATCH_TOLERANCE_CM1:
            raise SystemExit(f"{path.name} is not on the ftsspec_5 sample grid ({match:.2e} cm-1)")
        worst_match = max(worst_match, match)
        flux5 = fts.flux[index]

        # photatl = gain * file 5 + offset, by least squares over the whole page.
        design = np.column_stack([flux5, np.ones_like(flux5)])
        (gain, offset), *_ = np.linalg.lstsq(design, total, rcond=None)
        relation = float(np.std(total - design @ (gain, offset)) / (gain * robust_noise(flux5)))

        n = nu.size
        te = np.full(n, np.nan)
        tu = np.full(n, np.nan)
        reliable = np.zeros(n, bool)
        status = np.full(n, 2, np.int8)
        source = np.full(n, -1, np.int16)
        which = np.searchsorted(starts, nu, side="right") - 1
        for w in np.unique(which[which >= 0]):
            pixels = np.flatnonzero((which == w) & (nu < ends[w]))
            if not pixels.size:
                continue
            win = windows[w]
            k = np.searchsorted(win["wavenumber_cm1"], nu[pixels] - MATCH_TOLERANCE_CM1)
            k = np.clip(k, 0, win["wavenumber_cm1"].size - 1)
            hit = np.abs(win["wavenumber_cm1"][k] - nu[pixels]) <= MATCH_TOLERANCE_CM1
            pixels, k = pixels[hit], k[hit]
            te[pixels] = win["effective_transmission"][k]
            tu[pixels] = win["transmission_unconvolved"][k]
            reliable[pixels] = win["reliable"][k] & ~win["opaque"] & (not win["railed"])
            status[pixels] = 1 if win["opaque"] else 4 if win["railed"] else 0
            source[pixels] = w
            if args.check:
                # The file-5 fit's own corrected spectrum, rescaled by this page's
                # relation, must be what dividing photatl gives -- to within what
                # the relation leaves over, and exactly for the offset-free part.
                ok = reliable[pixels] & (win["model_flux"][k] > 1e-3)
                if ok.any():
                    mine = (total[pixels][ok] - offset) / gain / te[pixels][ok]
                    theirs = win["corrected_file5"][k][ok]
                    worst_check = max(worst_check, float(np.max(
                        np.abs(mine - theirs) / (robust_noise(flux5) / te[pixels][ok]))))
        source_name = "ftsspec_901218_5 fit"
        if path.name in patches:
            summary, row = patches[path.name]
            if row["spectrum_sha256"] != page.sha256:
                raise SystemExit(f"--patch fitted a different {path.name} than this one")
            with np.load(summary.parent / row["npz"]) as z:
                order = np.argsort(z["wavenumber_cm1"])
                if not np.allclose(z["wavenumber_cm1"][order], nu, atol=MATCH_TOLERANCE_CM1, rtol=0):
                    raise SystemExit(f"--patch {path.name} is not on this page's samples")
                te = (z["model_flux"] / z["stellar_only_pixels"])[order]
                tu = z["transmission_pixels"][order]
                reliable = z["reliable"][order].copy()
                if args.check:
                    # The patch run's own corrected spectrum is this division, exactly.
                    ok = reliable & (z["model_flux"][order] > 1e-3)
                    worst_patch = max(worst_patch, float(np.max(np.abs(
                        total[ok] / te[ok] - z["corrected"][order][ok]) / np.abs(z["corrected"][order][ok]))))
            status[:] = 3
            source[:] = -1
            if species_at_upper_bound(row["at_bound"],
                                      {s: row["parameters"][s] for s in row["species"]}):
                reliable[:] = False
                status[:] = 4
            source_name = f"photatl refit ({summary.name})"
        corrected = total / te
        reliable &= np.isfinite(corrected)
        rows.append({"page": path.name, "sha256": page.sha256, "wavenumber_cm1": nu,
                     "source": source_name,
                     "observed": total, "effective_transmission": te,
                     "transmission_unconvolved": tu, "corrected": corrected,
                     "reliable": reliable, "status": status, "source_window": source,
                     "gain": float(gain), "offset": float(offset),
                     "relation_residual_over_noise": relation})

    width = max(r["wavenumber_cm1"].size for r in rows)

    def pack(name, dtype, fill):
        out = np.full((len(rows), width), fill, dtype=dtype)
        for i, r in enumerate(rows):
            out[i, : r[name].size] = r[name]
        return out

    columns = {
        "wavenumber_cm1": ("f8", np.nan, "cm-1", "vacuum wavenumber, ascending"),
        "observed": ("f4", np.nan, "page intensity units",
                     "photatl's `total` column, as published"),
        "corrected": ("f4", np.nan, "page intensity units",
                      "THE PRODUCT: observed / effective_transmission -- the telluric "
                      "absorption removed, the solar spectrum retained"),
        "effective_transmission": ("f4", np.nan, "fraction",
                                   "model_flux / stellar_only of the ftsspec_901218_5 fit "
                                   "at this very sample -- the operator the correction "
                                   "applied. NOT the unconvolved transmission"),
        "transmission_unconvolved": ("f4", np.nan, "fraction",
                                     "atmospheric transmission before the instrument "
                                     "profile; diagnostic only, do not divide by this"),
        "reliable": ("?", False, "", "True where corrected may be used"),
        "status": ("i1", 2, "", "; ".join(f"{k} = {v}" for k, v in STATUS.items())),
        "source_window": ("i2", -1, "",
                          "row of `windows` whose fit supplied this pixel, -1 for none"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.attrs["about"] = ABOUT
        handle.attrs["format"] = "tellurix photatl 1"
        handle.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        handle.attrs["photatl"] = str(args.photatl)
        handle.attrs["ftsspec"] = str(args.ftsspec)
        handle.attrs["ftsspec_sha256"] = fts.sha256
        handle.attrs["summary"] = str(args.summary)
        handle.attrs["summary_sha256"] = file_sha256(args.summary)
        if args.record.exists():
            handle.attrs["record"] = str(args.record)
            handle.attrs["record_sha256"] = file_sha256(args.record)
        handle.attrs["transmission_file"] = "ftsspec_901218_5_transmission.h5"
        handle.attrs["airmass"] = 1.985
        handle.attrs["driver"] = Path(__file__).name
        handle.attrs["driver_sha256"] = sha256_bytes(Path(__file__).read_bytes())
        handle.create_dataset("page", data=[r["page"].encode() for r in rows])
        handle.create_dataset("page_sha256", data=[r["sha256"].encode() for r in rows])
        handle.create_dataset("source", data=[r["source"].encode() for r in rows])
        handle["source"].attrs["description"] = "which fit supplied this page's transmission"
        for number, summary in enumerate(args.patch):
            handle.attrs[f"patch_{number}"] = str(summary)
            handle.attrs[f"patch_{number}_sha256"] = file_sha256(summary)
        handle.create_dataset("pixels", data=np.array([r["wavenumber_cm1"].size for r in rows], "i4"))
        for name, description in (
                ("gain", "photatl total = gain * ftsspec_5 flux + offset, fitted per page"),
                ("offset", "the zero level the atlas authors applied, in page units; "
                           "negative means lowered"),
                ("relation_residual_over_noise",
                 "what gain and offset leave over, in units of file 5's own noise")):
            handle.create_dataset(name, data=np.array([r[name] for r in rows]))
            handle[name].attrs["description"] = description
        for name, (dtype, fill, units, description) in columns.items():
            data = pack(name, dtype, fill)
            dataset = handle.create_dataset(name, data=data, compression="gzip", shuffle=True)
            dataset.attrs["units"] = units
            dataset.attrs["description"] = description
        table = handle.create_group("windows")
        table.create_dataset("v1", data=starts)
        table.create_dataset("v2", data=ends)
        table.create_dataset("npz", data=[w["npz"].encode() for w in windows])
        table.create_dataset("opaque", data=np.array([w["opaque"] for w in windows]))

    reliable = sum(int(r["reliable"].sum()) for r in rows)
    pixels = sum(r["wavenumber_cm1"].size for r in rows)
    relation = np.array([r["relation_residual_over_noise"] for r in rows])
    offsets = np.array([r["offset"] / r["gain"] for r in rows])
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB): {len(rows)} pages, "
          f"{reliable}/{pixels} pixels reliable ({reliable / pixels:.1%})")
    print(f"grid match: worst {worst_match:.2e} cm-1")
    print(f"gain/offset relation: residual/noise median {np.median(relation):.4f}, "
          f"max {relation.max():.4f}; offset < 0 on {(offsets < 0).sum()}/{len(rows)} pages")
    if patches:
        print(f"patched from photatl refits: {', '.join(sorted(patches, key=lambda n: int(n[2:])))}")
    if args.check:
        print(f"check: photatl, un-rescaled and divided, against the file-5 fit's own "
              f"corrected spectrum: worst {worst_check:.3f} sigma")
        if patches:
            print(f"check: patched pages against their own fit's corrected spectrum: "
                  f"worst relative difference {worst_patch:.1e}")


if __name__ == "__main__":
    main()
