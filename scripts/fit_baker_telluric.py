#!/usr/bin/env python
"""Fit Baker, Blake & Reiners 2020's telluric spectra with a flat Sun.

Their 141 Goettingen spectra (``tellurix_fts.baker``) have the Sun and the iodine
cell divided out already, so a telluric fit of them involves no solar model of
ours. What it is for is the line list: the per-line strength and position errors
measured in the Kitt Peak solar fits (docs/line_intensity_corrections.md) should
reappear here, at another site, instrument and decade, if they are the line
list's. ``analyse_baker_telluric.py`` measures them.

Each spectrum gets the ERA5 profile of its own hour. Make those first; the
driver lists the hours its spectra need:

    for t in $(uv run python scripts/fit_baker_telluric.py --list-hours); do
      uv run python scripts/era5_site_profile.py --site Goettingen --time $t \\
          --output data/profiles/baker2020/goettingen_$(echo $t | tr -d :-).csv
    done

Then fit, four workers a GPU; a rerun skips the windows already written:

    for i in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$((i % 2)) XLA_PYTHON_CLIENT_PREALLOCATE=false \
          uv run python scripts/fit_baker_telluric.py --shard $i/8 &
    done

By default one spectrum per ERA5 hour, the first distinct one of each: 53 of the
106 distinct spectra (35 appear in more than one group, there divided by another
group's solar model). And the 30 cm-1 windows over 9020-17600 cm-1 holding at
least three features the solar fits measured (docs/line_depth_ratios.json): 146.
H2O and O2 only, the instrument the IAG atlas's 0.0145 cm-1 sinc, MT_CKD and O2
CIA on, HITRAN's H2O pressure shifts, no covariance (``WindowSettings.covariance``),
AER's widths unless ``--pressure-width-scale``, and Voigt lines unless
``--speed-dependence`` (Gamma2/Gamma0 of H2O's speed-dependent Voigt):

    ... fit_baker_telluric.py --pressure-width-scale 1.06 --output-dir data/corrected/baker2020_w106
    ... fit_baker_telluric.py --speed-dependence 0.13 --output-dir data/corrected/baker2020_sdv13

A fit is mostly compilation, on the CPU, and the GPU idles: so run several workers
a GPU with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (a fit peaks at 0.5 GB). The
objective's compilation is shared by every spectrum on a window -- the model's
arrays are its operands (``tellurix.compiled``) and the workers share
``--compilation-cache`` -- while the opacity precompute's (1.9 s) is not.

Each window writes ``gGG_NNNN_V1.npz`` (fit_window's arrays) and ``.json`` (its
report and the spectrum's header entries). These are measurements, not products:
no record and no results-archive entry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATLAS = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/iag_telfree")
WIDTH_CM1 = 30.0


def profile_for(directory: Path, observation) -> Path:
    return directory / f"goettingen_{observation.era5_hour:%Y%m%dT%H00}.csv"


def choose_spectra(observations, every: bool):
    first = {}
    for observation in observations:
        first.setdefault(observation.number, observation)
    if every:
        return sorted(first.values(), key=lambda o: o.number)
    hours = {}
    for observation in sorted(first.values(), key=lambda o: o.number):
        hours.setdefault(observation.era5_hour, observation)
    return sorted(hours.values(), key=lambda o: o.number)


def choose_windows(report: Path, v1: float, v2: float, min_features: int):
    data = json.loads(report.read_text())
    measured = {int(i) for fit in data["ratio_and_peak_tau"].values() for i in fit}
    starts = np.floor((np.array([data["features"][i]["nu"] for i in measured]) - v1) / WIDTH_CM1)
    counts = np.bincount(starts[starts >= 0].astype(int))
    return [v1 + WIDTH_CM1 * k for k in np.where(counts >= min_features)[0]
            if v1 + WIDTH_CM1 * (k + 1) <= v2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--profiles", type=Path, default=ROOT / "data/profiles/baker2020")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/corrected/baker2020")
    parser.add_argument("--all-spectra", action="store_true",
                        help="every distinct spectrum, not one per ERA5 hour")
    parser.add_argument("--number", action="append", default=[], help="only these spectra (NNNN)")
    parser.add_argument("--window", type=float, action="append", default=[],
                        help="a window's v1; repeat. Default: chosen from --features-from")
    parser.add_argument("--features-from", type=Path, default=ROOT / "docs/line_depth_ratios.json")
    parser.add_argument("--min-features", type=int, default=3)
    parser.add_argument("--v1", type=float, default=9020.0)
    parser.add_argument("--v2", type=float, default=17600.0)
    parser.add_argument("--shard", default="0/1", help="I/N: this process takes every Nth window")
    parser.add_argument("--compilation-cache", default=str(ROOT / ".jax-cache"),
                        help="shared by the workers; empty string to disable")
    parser.add_argument("--pressure-width-scale", type=float, default=1.0,
                        help="multiply H2O's and O2's gamma_air and gamma_self, as the solar fits "
                             "did (x1.06-1.11); write to another --output-dir")
    parser.add_argument("--speed-dependence", type=float, default=0.0,
                        help="Gamma2/Gamma0 of a speed-dependent Voigt for H2O; 0 keeps the Voigt. "
                             "Write to another --output-dir")
    parser.add_argument("--list-hours", action="store_true",
                        help="print the ERA5 hours the chosen spectra need, and stop")
    args = parser.parse_args()

    from tellurix_fts.baker import baker_observations, read_baker_telluric

    spectra = choose_spectra(baker_observations(args.atlas), args.all_spectra)
    if args.number:
        spectra = [o for o in spectra if o.number in set(args.number)]
    if args.list_hours:
        for hour in sorted({o.era5_hour for o in spectra}):
            print(f"{hour:%Y-%m-%dT%H:00}")
        return
    missing = [str(profile_for(args.profiles, o)) for o in spectra if not profile_for(args.profiles, o).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} profiles missing, e.g. {missing[0]}; see --list-hours")

    from tellurix import DataPaths
    from tellurix.compiled import enable_compilation_cache
    from tellurix_fts.flux_atlas import IAG_FWHM_CM1
    from tellurix_fts.window import WindowSettings, fit_window

    windows = args.window or choose_windows(args.features_from, args.v1, args.v2, args.min_features)
    shard, shards = (int(x) for x in args.shard.split("/"))
    windows = windows[shard::shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(spectra)} spectra x {len(windows)} windows (shard {args.shard})", flush=True)
    paths = DataPaths.bootstrapped(ROOT)
    if args.compilation_cache:
        enable_compilation_cache(args.compilation_cache)
    failures = []
    for observation in spectra:
        stem = f"g{observation.group:02d}_{observation.number}"
        todo = [v1 for v1 in windows if not (args.output_dir / f"{stem}_{v1:.0f}.json").exists()]
        if not todo:
            continue
        # Read once per spectrum: the table is 40-55 MB and a window takes a slice.
        spectrum = read_baker_telluric(args.atlas, observation)
        for v1 in todo:
            settings = WindowSettings(
                spectrum=spectrum.path, profile=profile_for(args.profiles, observation),
                v1=v1, v2=v1 + WIDTH_CM1, fwhm_cm1=IAG_FWHM_CM1, species="H2O,O2",
                stellar="flat", o2_cia=True, h2o_pressure_shift="hitran", covariance=False,
                pressure_width_scale=args.pressure_width_scale,
                speed_dependence=args.speed_dependence)
            try:
                report, arrays = fit_window(settings, paths, stellar_directory=ROOT / "data/stellar",
                                            base=ROOT, spectrum=spectrum)
            except Exception as exc:  # one window must not stop the run
                failures.append({"spectrum": stem, "v1": v1, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {stem} {v1:.0f}: {exc}", flush=True)
                continue
            np.savez_compressed(args.output_dir / f"{stem}_{v1:.0f}.npz", **arrays)
            (args.output_dir / f"{stem}_{v1:.0f}.json").write_text(json.dumps({
                "group": observation.group, "number": observation.number,
                "airmass": observation.airmass, "julian_day": observation.julian_day,
                "utc": observation.utc.isoformat(), "h2o_optical_depth": observation.h2o_optical_depth,
                "profile": str(settings.profile.relative_to(ROOT)), "report": report,
            }, default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n")
            columns = {k: round(v, 3) for k, v in report["parameters"].items()
                       if k in ("H2O", "O2", "velocity_kms", "lsf_sigma_kms")}
            print(f"{stem} X {observation.airmass:.2f} {v1:.0f} {columns}", flush=True)
    if failures:
        (args.output_dir / f"failures_{shard}of{shards}.json").write_text(json.dumps(failures, indent=1) + "\n")
    print(f"done: {len(failures)} failed", flush=True)


if __name__ == "__main__":
    main()
