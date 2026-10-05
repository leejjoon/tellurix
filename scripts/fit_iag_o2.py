#!/usr/bin/env python
"""Fit the O2 A- and B-band windows of the IAG solar flux atlas.

An independent check on the O2 excess of the June 1983 Kitt Peak spectra
(docs/solar_fit_plan.md §4n): a different FTS, site and decade. The atlas
(Reiners et al. 2016, A&A 587, A65; CDS J/A+A/587/A65) co-adds 1190 scans from
nine days of March-July 2014, so it has no one air mass and its absolute O2
column means nothing; the fit puts zenith at 0 and lets the column scale carry
the mean air mass. What survives the averaging is the B/A ratio of weak-line
scales, since a weak line's depth is linear in the column.

The FTS was scanned at 0.01 cm-1 but its optics limit it to R ~ 1e6 (the
paper's Table 1), 0.0145 cm-1 at the B-band; that is the sinc used here, below
O2's ~0.03 cm-1 Doppler width, with the fit's Gaussian free on top. It also
sets the grid, at 2 samples per element: Payne Zero's solar model is sampled at
0.146 km/s, which forbids anything finer. The source is Payne Zero's flux,
which is the right quantity for a flux atlas, rotationally broadened by 1.9 km/s.

    uv run python scripts/fit_iag_o2.py --spectrum data/databases/iag/spvis.dat.gz \\
        --profile data/profiles/goettingen_20140417_era5.csv

Writes data/corrected/solar/iag_o2/iag_vis_o2_summary.json and one .npz per
window, in the batch driver's layout, so scripts/o2_curve_of_growth.py reads it.
"""
import argparse, gzip, hashlib, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = [12970.0 + 30 * k for k in range(7)] + [14380.0 + 30 * k for k in range(6)]
FWHM_CM1 = 0.0145


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectrum", type=Path, default=ROOT / "data/databases/iag/spvis.dat.gz")
    parser.add_argument("--profile", type=Path, default=Path("data/profiles/goettingen_20140417_era5.csv"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/corrected/solar/iag_o2")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"

    from tellurix.download import DataPaths
    from tellurix_fts.nso import FTSSpectrum
    from tellurix_fts.window import WindowSettings, fit_window

    # 4 million rows; only the two bands are kept, and cached beside the download.
    cache = args.spectrum.with_name(args.spectrum.name.split(".")[0] + "_o2_bands.npz")
    if not cache.exists():
        data = np.loadtxt(gzip.open(args.spectrum), usecols=(0, 1))
        keep = (data[:, 0] > 12800.0) & (data[:, 0] < 14700.0)
        np.savez(cache, nu=data[keep, 0], flux=data[keep, 1])
    z = np.load(cache)
    sha = hashlib.sha256(args.spectrum.read_bytes()).hexdigest()
    spectrum = FTSSpectrum(path=args.spectrum, sha256=sha, source_name="IAG solar flux atlas, VIS",
                           comment="Reiners et al. 2016, normalised flux column", date_mst="2014-03..07",
                           julian_day=0, universal_time_start="", universal_time_stop="",
                           airmass_start=None, airmass_stop=None, stated_resolution_cm1=0.01,
                           transform_samples=0, point_of_center=0, wavenumber_vacuum_cm1=z["nu"],
                           flux=z["flux"], grid_residual_cm1=0.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    physics = {"accuracy_mode": "mt_ckd", "fwhm_cm1": FWHM_CM1, "samples_per_resolution": 2.0,
               "vsini_kms": 1.9, "macroturbulence_kms": 1.5, "stages": "continuum,velocity,columns,stellar",
               "zenith_angle_deg": 0.0, "species": "O2,H2O", "driver": "fit_iag_o2.py"}
    results = []
    for v1 in WINDOWS:
        settings = WindowSettings(
            spectrum=args.spectrum, v1=v1, v2=v1 + 30.0, profile=args.profile, fwhm_cm1=FWHM_CM1,
            samples_per_resolution=2.0, species="O2,H2O", stellar="auto", vsini_kms=1.9,
            macroturbulence_kms=1.5, stages=physics["stages"], zenith_angle_deg=0.0, accuracy_mode="mt_ckd")
        report, arrays = fit_window(settings, DataPaths.bootstrapped(ROOT), stellar_directory=ROOT / "data/stellar",
                                    base=ROOT, spectrum=spectrum)
        npz = f"iag_vis_{int(v1)}.npz"
        np.savez(args.output_dir / npz, **arrays)
        results.append({"v1": v1, "v2": v1 + 30.0, "species": ["O2", "H2O"], "npz": f"iag_o2/{npz}",
                        "airmass": None, "parameters": report["parameters"], "residuals": report["residuals"]})
        print(f"{v1:.0f}: O2 {np.exp(report['parameters']['O2']):.4f} H2O {np.exp(report['parameters']['H2O']):.3f} "
              f"rms {report['residuals']['rms']:.4f}", flush=True)
    (args.output_dir / "iag_vis_o2_summary.json").write_text(json.dumps(
        {"spectrum": str(args.spectrum), "spectrum_sha256": sha, "profile": str(args.profile),
         "physics": physics, "results": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
