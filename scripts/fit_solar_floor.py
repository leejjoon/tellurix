#!/usr/bin/env python
"""Fit niratl's pages in a flux atlas and in niratl itself, for the residual-floor split.

Every solar run stops at ~1% rms, and the suspect is that Payne Zero's Eddington
flux stands in for disc-centre intensity (GitHub #2, docs/solar_fit_plan.md
§4p). The test is like for like: the same wavenumbers fitted in a disc-centre
spectrum and in a flux spectrum, with the same solar source and settings, so
what differs is the quantity observed.

- ``niratl``: disc-centre intensity, the Kitt Peak McMath FTS, air mass 1.10.
  Refitted here, not read from its record, because the batch skips the pages
  with no telluric absorption -- the ones that measure the solar model alone.
- ``wallace2011``: Kitt Peak flux atlas #2, region 1 (1989/10/13 #8, air mass
  1.5): disc-integrated, **the same instrument and site** as niratl, at nearly
  the same telluric load. The pair that isolates the quantity.
- ``iag``: the IAG flux atlas (Göttingen 2014, nine days co-added, zenith 0):
  a different instrument, for whether what remains belongs to Kitt Peak.

Windows are niratl's pages inside ``--v1/--v2`` and each takes the species
niratl's scan found for it ("H2O" where it found none, so every window has the
same freedom). Disc centre is fitted at vsini 0, flux at 1.9 km/s, as Phase 1
prescribes; ``--macroturbulence-kms`` is the one broadening left open.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/fit_solar_floor.py wallace2011 --shard 1/2

Writes data/corrected/solar/floor/<atlas>[_<tag>]/ (one npz per window, in the
batch driver's array layout, and summary[_shard].json); scripts/solar_floor_report.py
reads them.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NSO = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/nso")
NIRATL_SUMMARY = ROOT / "data/corrected/solar/niratl/niratl_summary.json"
ATLASES = {
    "niratl": {"profile": "data/profiles/kitt_peak_19830626_era5_afgl.csv", "fwhm_cm1": 0.01859,
               "airmass": 1.10, "vsini_kms": 0.0, "samples_per_resolution": 4.0},
    "wallace2011": {"profile": "data/profiles/kitt_peak_19891013_era5.csv", "fwhm_cm1": None,
                    "airmass": None, "vsini_kms": 1.9, "samples_per_resolution": 4.0},
    # Payne Zero's nir band is sampled for 0.01859 cm-1 at 4 per element, 3.2 at IAG's 0.0145.
    "iag": {"profile": "data/profiles/goettingen_20140417_era5.csv", "fwhm_cm1": None,
            "airmass": 0.0, "vsini_kms": 1.9, "samples_per_resolution": 3.0},
}


def niratl_pages(v1, v2):
    rows = json.loads(NIRATL_SUMMARY.read_text())["results"]
    return [{"v1": r["v1"], "v2": r["v2"], "page": Path(r["file"]).name,
             "species": r["species"] or ["H2O"], "negligible_telluric": bool(r.get("negligible_telluric"))}
            for r in rows if r["v1"] >= v1 and r["v2"] <= v2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("atlas", choices=sorted(ATLASES))
    parser.add_argument("--v1", type=float, default=10800.0)
    parser.add_argument("--v2", type=float, default=13500.0)
    parser.add_argument("--pages", default=None, help="comma-separated niratl page names: only these")
    parser.add_argument("--macroturbulence-kms", type=float, default=1.5)
    parser.add_argument("--tag", default="", help="suffix for the output directory")
    parser.add_argument("--shard", default=None, metavar="I/N")
    parser.add_argument("--resume", action="store_true", help="skip windows whose npz exists")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"

    from tellurix.download import DataPaths
    from tellurix_fts import read_solar_spectrum
    from tellurix_fts.flux_atlas import IAG_FWHM_CM1, WALLACE2011_REGIONS, read_flux_atlas
    from tellurix_fts.nso import zenith_angle_deg_for_airmass
    from tellurix_fts.window import WindowSettings, fit_window

    config = dict(ATLASES[args.atlas])
    pages = niratl_pages(args.v1, args.v2)
    if args.pages:
        wanted = set(args.pages.split(","))
        pages = [p for p in pages if p["page"] in wanted]
    if args.shard:
        index, count = (int(v) for v in args.shard.split("/"))
        pages = pages[index - 1::count]
    keep = (args.v1 - 30.0, args.v2 + 30.0)
    if args.atlas == "wallace2011":
        region = WALLACE2011_REGIONS[1]
        config.update(fwhm_cm1=region["fwhm_cm1"], airmass=region["airmass"])
        path = NSO / "wallace2011_flux" / region["file"]
        shared = read_flux_atlas("wallace2011", path, keep, cache_directory=ROOT / "data/databases/wallace2011",
                                 airmass=region["airmass"], fwhm_cm1=region["fwhm_cm1"],
                                 source_name=f"Kitt Peak flux atlas #2, {region['spectrum']}, observed flux")
    elif args.atlas == "iag":
        config.update(fwhm_cm1=IAG_FWHM_CM1)
        path = ROOT / "data/databases/iag/spvis.dat.gz"
        shared = read_flux_atlas("iag", path, keep, cache_directory=ROOT / "data/databases/iag",
                                 fwhm_cm1=IAG_FWHM_CM1, source_name="IAG solar flux atlas, VIS")
    zenith = 0.0 if not config["airmass"] else float(zenith_angle_deg_for_airmass(config["airmass"]))

    name = args.atlas + (f"_{args.tag}" if args.tag else "")
    output = ROOT / "data/corrected/solar/floor" / name
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / (f"summary_{args.shard.replace('/', 'of')}.json" if args.shard else "summary.json")
    results = json.loads(summary_path.read_text())["results"] if args.resume and summary_path.exists() else []
    done = {r["v1"] for r in results}
    stages = "continuum,velocity,columns,stellar"
    for page in pages:
        npz = output / f"{args.atlas}_{page['v1']:.1f}.npz"
        if args.resume and npz.exists() and page["v1"] in done:
            continue
        started = time.time()
        if args.atlas == "niratl":
            spectrum_path = NSO / "niratl" / page["page"]
            spectrum = read_solar_spectrum(spectrum_path)
        else:
            spectrum_path, spectrum = path, shared
        settings = WindowSettings(
            spectrum=spectrum_path, v1=page["v1"], v2=page["v2"], profile=Path(config["profile"]),
            fwhm_cm1=config["fwhm_cm1"], samples_per_resolution=config["samples_per_resolution"],
            species=",".join(page["species"]), stellar="auto", vsini_kms=config["vsini_kms"],
            macroturbulence_kms=args.macroturbulence_kms, stages=stages, zenith_angle_deg=zenith,
            accuracy_mode="mt_ckd", o2_cia=True)
        try:
            report, arrays = fit_window(settings, DataPaths.bootstrapped(ROOT), stellar_directory=ROOT / "data/stellar",
                                        base=ROOT, spectrum=spectrum)
        except ValueError as error:
            # A flux atlas can have a gap or too few unmasked pixels where niratl had none.
            results.append({**page, "error": str(error), "seconds": round(time.time() - started, 1)})
            print(f"{page['v1']:.1f}: {error}", flush=True)
            continue
        np.savez_compressed(npz, **arrays)
        results = [r for r in results if r["v1"] != page["v1"]]
        results.append({**page, "npz": npz.name, "parameters": report["parameters"], "at_bound": report["at_bound"],
                        "residuals": report["residuals"], "pixel_sigma": report["spectrum"]["uncertainty"],
                        "median_transmission": report["median_transmission"],
                        "all_stages_converged": report["all_stages_converged"],
                        "seconds": round(time.time() - started, 1)})
        print(f"{page['v1']:.1f}: rms {report['residuals']['rms']:.4f} "
              f"T {report['median_transmission']:.3f} {time.time() - started:.0f} s", flush=True)
        summary_path.write_text(json.dumps({
            "atlas": args.atlas, "spectrum": str(spectrum_path if args.atlas == "niratl" else path),
            "sha256": spectrum.sha256 if args.atlas != "niratl" else None,
            "settings": {**config, "zenith_angle_deg": zenith, "macroturbulence_kms": args.macroturbulence_kms,
                         "stages": stages, "accuracy_mode": "mt_ckd", "o2_cia": True, "stellar": "auto",
                         "v1": args.v1, "v2": args.v2},
            "driver": "fit_solar_floor.py", "results": sorted(results, key=lambda r: r["v1"])}, indent=1) + "\n")


if __name__ == "__main__":
    main()
