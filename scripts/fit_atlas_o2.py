#!/usr/bin/env python
"""Fit the O2 windows of a solar flux atlas, as checks on the 1983 O2 excess.

The June 1983 Kitt Peak spectra ask for ~5% more O2 in the B-band, relative to
the A-band, than the laboratory intensities allow (docs/solar_fit_plan.md
§4n). Two flux atlases test whether that belongs to those spectra:

- ``iag``: the IAG solar flux atlas (Reiners et al. 2016, A&A 587, A65; CDS
  J/A+A/587/A65), Göttingen 2014 -- a different FTS, site and decade. It
  co-adds 1190 scans from nine days of March-July 2014, so it has no one air
  mass; the fit puts zenith at 0 and lets the column scale carry the mean.
  What survives the averaging is the B/A ratio of weak-line scales, since a
  weak line's depth is linear in the column. The FTS was scanned at 0.01 cm-1
  but its optics limit it to R ~ 1e6 (the paper's Table 1), 0.0145 cm-1 at the
  B-band; that is the sinc used here.
- ``wallace2011``: Kitt Peak atlas #2 (Wallace, Hinkle, Livingston & Davis
  2011), whose A- and B-band regions are single McMath integrated-sun spectra,
  1989/10/13 #8 at air mass 1.5 and #7 at 1.4 -- the 1983 instrument, six years
  later. The observed-flux column (the fourth) is used, not the
  telluric-corrected one, whose transmission came from the 1983-06-26 pair. The
  air masses are given to one decimal, which alone puts ~2.6% on a B/A ratio.
  The sinc is each region's stated resolving power at the band.

Both: the grid is 2 samples per resolution element at most, because Payne
Zero's solar model is sampled at 0.146 km/s; the source is Payne Zero's flux,
the right quantity for a flux atlas, rotationally broadened by 1.9 km/s.

    uv run python scripts/fit_atlas_o2.py iag --spectrum data/databases/iag/spvis.dat.gz
    uv run python scripts/fit_atlas_o2.py wallace2011 \\
        --spectrum .../differentiable_stellar_spectroscopy/data/atlases/nso/wallace2011_flux

Writes data/corrected/solar/<atlas>_o2/<name>_o2_summary.json and one .npz per
window, in the batch driver's layout, so scripts/o2_curve_of_growth.py reads it.
"""
import argparse, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A_BAND = [12970.0 + 30 * k for k in range(7)]
B_BAND = [14380.0 + 30 * k for k in range(6)]
GAMMA = [15730.0 + 30 * k for k in range(7)]
# Each atlas as (file, wavenumber range kept, air mass or None, sinc FWHM, windows).
ATLASES = {
    "iag": {
        "name": "iag_vis", "profile": "data/profiles/goettingen_20140417_era5.csv",
        "source": "IAG solar flux atlas, VIS (Reiners et al. 2016), normalised flux",
        # The VIS setting starts at 9387 cm-1, on the 1.06 um band.
        "parts": [("spvis.dat.gz", (9387.0, 16000.0), None, 0.0145, [9400.0, 9430.0] + A_BAND + B_BAND + GAMMA)],
    },
    "wallace2011": {
        "name": "wallace2011", "profile": "data/profiles/kitt_peak_19891013_era5.csv",
        "source": "Kitt Peak flux atlas #2 (Wallace et al. 2011), observed flux",
        # Region 1 is 1989/10/13 #8, R 676,000; region 2 is #7, R 698,000. Region 1's own
        # and region 2's interferograms say 0.0227 and 0.0328 cm-1
        # (flux_atlas.WALLACE2011_REGIONS), not the 0.0193 and 0.0207 below, which are what
        # the committed O2 numbers (solar_fit_plan.md §4n) were fitted with.
        "parts": [("sptr.reg1", (12800.0, 13300.0), 1.5, 13060.0 / 676000.0, A_BAND),
                  ("sptr.reg2", (14300.0, 16000.0), 1.4, 14455.0 / 698000.0, B_BAND + GAMMA)],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("atlas", choices=sorted(ATLASES))
    parser.add_argument("--spectrum", type=Path, required=True, help="the atlas file (iag) or directory (wallace2011)")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"

    from tellurix.download import DataPaths
    from tellurix_fts.flux_atlas import read_flux_atlas
    from tellurix_fts.nso import zenith_angle_deg_for_airmass
    from tellurix_fts.window import WindowSettings, fit_window

    atlas = ATLASES[args.atlas]
    output = ROOT / "data/corrected/solar" / f"{args.atlas}_o2"
    output.mkdir(parents=True, exist_ok=True)
    stages = "continuum,velocity,columns,stellar"
    results, files = [], {}
    for filename, keep, airmass, fwhm, windows in atlas["parts"]:
        path = args.spectrum if args.spectrum.is_file() else args.spectrum / filename
        spectrum = read_flux_atlas(args.atlas, path, keep, cache_directory=ROOT / "data/databases" / args.atlas,
                                   airmass=airmass, fwhm_cm1=fwhm, source_name=atlas["source"])
        files[filename] = spectrum.sha256
        zenith = 0.0 if airmass is None else float(zenith_angle_deg_for_airmass(airmass))
        for v1 in windows:
            settings = WindowSettings(
                spectrum=path, v1=v1, v2=v1 + 30.0, profile=Path(atlas["profile"]), fwhm_cm1=fwhm,
                samples_per_resolution=2.0, species="O2,H2O", stellar="auto", vsini_kms=1.9,
                macroturbulence_kms=1.5, stages=stages, zenith_angle_deg=zenith, accuracy_mode="mt_ckd")
            report, arrays = fit_window(settings, DataPaths.bootstrapped(ROOT), stellar_directory=ROOT / "data/stellar",
                                        base=ROOT, spectrum=spectrum)
            npz = f"{atlas['name']}_{int(v1)}.npz"
            np.savez(output / npz, **arrays)
            results.append({"v1": v1, "v2": v1 + 30.0, "file": filename, "species": ["O2", "H2O"],
                            "npz": f"{args.atlas}_o2/{npz}", "airmass": airmass, "zenith_angle_deg": zenith,
                            "fwhm_cm1": fwhm, "parameters": report["parameters"], "residuals": report["residuals"]})
            print(f"{v1:.0f}: O2 {np.exp(report['parameters']['O2']):.4f} H2O {np.exp(report['parameters']['H2O']):.3f} "
                  f"rms {report['residuals']['rms']:.4f}", flush=True)
    physics = {"accuracy_mode": "mt_ckd", "samples_per_resolution": 2.0, "vsini_kms": 1.9,
               "macroturbulence_kms": 1.5, "stages": stages, "species": "O2,H2O", "driver": "fit_atlas_o2.py"}
    (output / f"{atlas['name']}_o2_summary.json").write_text(json.dumps(
        {"atlas": args.atlas, "spectrum": str(args.spectrum), "files_sha256": files, "profile": atlas["profile"],
         "physics": physics, "results": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
