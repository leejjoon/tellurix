#!/usr/bin/env python
"""Pack a batch run over atlas pages into one telluric-corrected file, a row per page.

For an atlas fitted page by page (niratl; photatl's four band pages), where each
page's transmission comes from that page's own fit. photatl as a whole comes
from the ftsspec_901218_5 fit instead -- see export_photatl_from_ftsspec.py --
and the two files share this layout, so a reader handles both the same way.

    corrected = observed / effective_transmission

``effective_transmission`` is ``model_flux / stellar_only``, the *convolved*
operator the fit applied; ``transmission_unconvolved`` is the atmosphere before
the instrument and is diagnostic only. The unconvolved transmission on the
model's own grid, for a consumer's synthesis, is export_transmission_hdf5.py.

A page the scan found nothing on is **not corrected**, not clean: every species
fell below the run's slant optical-depth cut, which still allows lines up to
~1% deep. That is invisible at ftsspec's noise and 50 sigma at niratl's, so
those pages carry ``status`` 1 and ``reliable`` False.

    UV_CACHE_DIR=.uv-cache uv run python scripts/export_atlas_corrected.py \\
        --summary data/corrected/solar/niratl/niratl_summary.json \\
        --output data/corrected/niratl_corrected.h5 --check

A run over windows of one long spectrum (the IAG atlases) is packed the same way,
a row per window, with `page` naming the window and `v1`/`v2` its limits, cut from
the atlas as the fit cut it. Such a run may be a co-add, which the file says
instead of quoting an air mass.

A window whose saturated cores sit far above zero is withheld (status 4). Where
the model transmits under 2%, the IAG atlases sit 0.1-0.4% of the continuum above
it, but 12-19% at 7216 and 7336 cm-1: a pedestal no transmission explains, which a
division turns into fake emission.

    UV_CACHE_DIR=.uv-cache uv run python scripts/export_atlas_corrected.py \\
        --summary data/corrected/solar/iag_nir/iag_nir_summary.json \\
        --output data/corrected/iag_nir_corrected.h5 --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tellurix import file_sha256  # noqa: E402
from tellurix_fts import read_solar_spectrum  # noqa: E402
from tellurix.quality import UPPER_BOUND_NOTE, species_at_upper_bound  # noqa: E402

STATUS = {
    0: "fitted: this page's own fit supplied the correction",
    1: "not corrected: every species was below the run's slant optical-depth cut, "
       "which still allows lines up to ~1% deep; observed is kept, corrected is NaN",
    2: "not fitted: the page's fit failed or was opaque",
    3: "not usable: " + UPPER_BOUND_NOTE + "; reliable is False on the whole page",
    4: "not usable: the atlas's saturated cores sit more than 5% of the continuum above "
       "the model, a zero-level pedestal the division would turn into emission; reliable "
       "is False on the whole page",
}
# Where the model transmits under CORE_TRANSMISSION, the median of observed minus model,
# over the continuum, above which a page is withheld; it needs CORE_PIXELS such pixels.
PEDESTAL_LIMIT = 0.05
CORE_TRANSMISSION = 0.02
CORE_PIXELS = 200


def core_pedestal(z) -> float:
    """Observed minus model over the continuum in saturated cores, or NaN without enough."""

    mask = np.asarray(z["mask"]).astype(bool)
    flux, raw = np.asarray(z["flux"]), np.asarray(z["observed_raw"])
    if not mask.any():
        return float("nan")
    # `flux` is the atlas over the window's level, with placeholders where masked;
    # the saturated cores are masked, so rescale the raw atlas there instead.
    level = float(np.median(raw[mask] / flux[mask]))
    model, continuum = np.asarray(z["model_flux"]), np.asarray(z["continuum"])
    core = model / continuum < CORE_TRANSMISSION
    if core.sum() < CORE_PIXELS:
        return float("nan")
    return float(np.median((raw[core] / level - model[core]) / continuum[core]))


def read_whole(path: Path):
    """A whole spectrum as the batch driver read it: a flux atlas through its cache."""

    from tellurix_fts.flux_atlas import IAG_NIR, IAG_VIS, WALLACE2011_REGIONS, read_flux_atlas

    root = Path(__file__).resolve().parents[1]
    if path.is_dir():
        # The Kurucz et al. 1984 pages, the one flux atlas kept as a directory.
        return read_flux_atlas("fluxatl", path, (0.0, 1.0e6), cache_directory=root / "data/databases/fluxatl")
    for atlas, files in (("iag", (IAG_VIS["file"], IAG_NIR["file"])),
                         ("wallace2011", tuple(r["file"] for r in WALLACE2011_REGIONS.values()))):
        if path.name in files:
            return read_flux_atlas(atlas, path, (0.0, 1.0e6), cache_directory=root / f"data/databases/{atlas}")
    return read_solar_spectrum(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--record", type=Path, default=None,
                        help="default: the summary's sibling .h5 with the same stem")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true",
                        help="verify each page against its fit's own corrected spectrum")
    args = parser.parse_args()

    import h5py

    summary = json.loads(args.summary.read_text())
    record = args.record or args.summary.with_name(
        args.summary.name.replace("_summary.json", ".h5"))
    results = sorted(summary["results"], key=lambda r: r["v1"])
    floor = float(summary["settings"]["min_transmission"])
    cut = float(summary["settings"]["species_threshold"])

    windowed = len({row["file"] for row in results}) < len(results)
    whole = {}
    coadd = summary["physics"].get("coadd")
    rows, worst = [], 0.0
    for row in results:
        path = Path(row["file"])
        if windowed:
            if str(path) not in whole:
                whole[str(path)] = read_whole(path)
                if whole[str(path)].sha256 != row["spectrum_sha256"]:
                    raise SystemExit(f"{path.name} has changed since it was fitted")
            window = whole[str(path)].select(row["v1"], row["v2"])
            nu, observed = window.wavenumber_vacuum_cm1, window.flux
            label = f"{path.name} {row['v1']:.0f}-{row['v2']:.0f}"
            sha = row["spectrum_sha256"]
        else:
            spectrum = read_solar_spectrum(path)
            if spectrum.sha256 != row["spectrum_sha256"]:
                raise SystemExit(f"{path.name} has changed since it was fitted")
            nu, observed = spectrum.wavenumber_vacuum_cm1, spectrum.flux
            label, sha = path.name, spectrum.sha256
        n = nu.size
        pedestal = float("nan")
        te = np.full(n, np.nan)
        tu = np.full(n, np.nan)
        reliable = np.zeros(n, bool)
        if row.get("npz") and not row.get("opaque") and "error" not in row:
            with np.load(args.summary.parent / row["npz"]) as z:
                order = np.argsort(z["wavenumber_cm1"])
                if not np.allclose(z["wavenumber_cm1"][order], nu, atol=1e-4, rtol=0):
                    raise SystemExit(f"{path.name}: the fit is not on this page's samples")
                pedestal = core_pedestal(z)
                te = (z["model_flux"] / z["stellar_only_pixels"])[order]
                tu = z["transmission_pixels"][order]
                reliable = z["reliable"][order].copy()
                if args.check:
                    ok = reliable & (z["model_flux"][order] > 1e-3)
                    theirs = z["corrected"][order][ok]
                    worst = max(worst, float(np.max(np.abs(observed[ok] / te[ok] - theirs)
                                                    / np.abs(theirs))))
            status = 0
            railed = species_at_upper_bound(
                row["at_bound"], {s: row["parameters"][s] for s in row["species"]})
            if railed:
                reliable[:] = False
                status = 3
            elif pedestal > PEDESTAL_LIMIT:
                reliable[:] = False
                status = 4
        elif row.get("negligible_telluric"):
            status = 1
        else:
            status = 2
        corrected = observed / te
        reliable &= np.isfinite(corrected)
        rows.append({"page": label, "sha256": sha, "v1": row["v1"], "v2": row["v2"], "page_status": status,
                     "pedestal": pedestal, "wavenumber_cm1": nu,
                     "observed": observed, "effective_transmission": te,
                     "transmission_unconvolved": tu, "corrected": corrected,
                     "reliable": reliable, "status": np.full(n, status, np.int8)})

    width = max(r["wavenumber_cm1"].size for r in rows)
    pedestals = [(r["page"], r["pedestal"]) for r in rows if r["page_status"] == 4]

    def pack(name, dtype, fill):
        out = np.full((len(rows), width), fill, dtype=dtype)
        for i, r in enumerate(rows):
            out[i, : r[name].size] = r[name]
        return out

    columns = {
        "wavenumber_cm1": ("f8", np.nan, "cm-1", "vacuum wavenumber, ascending"),
        "observed": ("f4", np.nan, "page intensity units", "the atlas's observed column"),
        "corrected": ("f4", np.nan, "page intensity units",
                      "THE PRODUCT: observed / effective_transmission -- the telluric "
                      "absorption removed, the solar spectrum retained"),
        "effective_transmission": ("f4", np.nan, "fraction",
                                   "model_flux / stellar_only of this page's fit -- the "
                                   "operator the correction applied. NOT the unconvolved "
                                   "transmission"),
        "transmission_unconvolved": ("f4", np.nan, "fraction",
                                     "atmospheric transmission before the instrument "
                                     "profile; diagnostic only, do not divide by this"),
        "reliable": ("?", False, "",
                     f"True where corrected may be used: fitted, transmission >= {floor}, "
                     "not a masked saturated core"),
        "status": ("i1", 2, "", "; ".join(f"{k} = {v}" for k, v in STATUS.items())),
    }
    # A window with no pixels still has a status; give it one from its row.
    statuses = np.array([r["page_status"] for r in rows], dtype="i1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        unit = "window" if windowed else "page"
        geometry = (f"as a co-add of several days' atmospheres ({Path(coadd).name})" if coadd
                    else f"at air mass {summary['settings']['airmass']}")
        handle.attrs["about"] = (
            f"Telluric-corrected {Path(summary['spectrum']).name}, one row per {unit}, from "
            f"tellurix's fit of each {unit} {geometry}. "
            "corrected = observed / effective_transmission. Use `reliable` before anything "
            f"else. Pages with every species below the slant optical-depth cut ({cut}) are "
            "status 1, not corrected.")
        handle.attrs["format"] = "tellurix atlas 1"
        handle.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        handle.attrs["summary"] = str(args.summary)
        handle.attrs["summary_sha256"] = file_sha256(args.summary)
        if record.exists():
            handle.attrs["record"] = str(record)
            handle.attrs["record_sha256"] = file_sha256(record)
        if coadd:
            handle.attrs["coadd"] = str(coadd)
            handle.attrs["coadd_sha256"] = str(summary["physics"].get("coadd_sha256", ""))
        else:
            handle.attrs["airmass"] = float(summary["settings"]["airmass"])
        handle.attrs["species_threshold"] = cut
        handle.attrs["fwhm_cm1"] = float(summary["physics"]["fwhm_cm1"])
        handle.attrs["driver"] = Path(__file__).name
        handle.attrs["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        handle.create_dataset("page", data=[r["page"].encode() for r in rows])
        handle.create_dataset("page_sha256", data=[r["sha256"].encode() for r in rows])
        handle.create_dataset("pixels", data=np.array([r["wavenumber_cm1"].size for r in rows], "i4"))
        handle.create_dataset("page_status", data=statuses)
        handle["page_status"].attrs["description"] = "status of the whole row, for rows with no pixels too"
        if windowed:
            handle.create_dataset("v1", data=np.array([r["v1"] for r in rows], "f8"))
            handle.create_dataset("v2", data=np.array([r["v2"] for r in rows], "f8"))
        handle.create_dataset("core_pedestal", data=np.array([r["pedestal"] for r in rows], "f4"))
        handle["core_pedestal"].attrs["description"] = (
            f"median (observed - model) / continuum where the model transmits under "
            f"{CORE_TRANSMISSION}; NaN with fewer than {CORE_PIXELS} such pixels. Above "
            f"{PEDESTAL_LIMIT} the row is status 4")
        for name, (dtype, fill, units, description) in columns.items():
            dataset = handle.create_dataset(name, data=pack(name, dtype, fill),
                                            compression="gzip", shuffle=True)
            dataset.attrs["units"] = units
            dataset.attrs["description"] = description

    counts = {k: int((statuses == k).sum()) for k in STATUS}
    print(f"{counts[3]} pages withheld for a column at its upper bound, {counts[4]} for a pedestal"
          + "".join(f"; {page} {value:+.3f}" for page, value in pedestals))
    reliable = sum(int(r["reliable"].sum()) for r in rows)
    pixels = sum(r["wavenumber_cm1"].size for r in rows)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB): {len(rows)} pages "
          f"({counts[0]} fitted, {counts[1]} not corrected, {counts[2]} not fitted), "
          f"{reliable}/{pixels} pixels reliable ({reliable / pixels:.1%})")
    if args.check:
        print(f"check: against each fit's own corrected spectrum, worst relative difference {worst:.1e}")


if __name__ == "__main__":
    main()
