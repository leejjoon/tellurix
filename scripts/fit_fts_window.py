#!/usr/bin/env python
"""Fit telluric absorption and a continuum to one window of an NSO FTS spectrum.

The fit is ``tellurix_fts.window``; this is its command line, reading data from
this checkout. Uses ``accuracy_mode="mt_ckd"`` with the native differentiable
continuum, so LBLRTM is not involved at fit time.

Three things differ from ``fit_arcturus_page.py`` and all three are
load-bearing.

**The resolving power is not a free choice.** These spectra hold the
interferogram truncation fixed rather than the resolving power, so the
resolution element is constant in wavenumber (0.017532 cm-1, measured; see
docs/solar_ils.md) and R rises with it -- 114,075 at 2000 cm-1 to 513,338 at
9000. The grid is sized from the window's own centre. A single atlas-wide
velocity step, which is what the Arcturus driver uses, would oversample the red
end by a factor of 4.5.

**The MOPD is one constant, not a per-window measurement.** The page-to-page
spread is 1.010, inside the FFT bin, so fitting it per window would be fitting
noise. This is the ``--sinc-resolving-power`` lesson from Arcturus, on firmer
ground.

**The zenith angle comes from the header.** The atlas fits at zero and folds
air mass into the column scale; these spectra record it, so the fit can use it.
``--zenith-angle-deg 0`` restores the Arcturus configuration, which is the
diagnostic half of the slant-path test: run both ways, and the two files' column
scales must agree at the header zenith and differ by their air-mass ratio at
zero.

    UV_CACHE_DIR=.uv-cache uv run python scripts/fit_fts_window.py \\
        --spectrum .../telluric_near_ir/ftsspec_901218_5.txt --v1 6000 --v2 6030
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tellurix.download import DataPaths
from tellurix_fts.nso import MEASURED_FWHM_CM1
from tellurix_fts.window import WindowSettings, fit_window

NSO_ROOT = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/nso")
DEFAULT_SPECTRUM = NSO_ROOT / "telluric_near_ir/ftsspec_901218_5.txt"


def settings_from(args) -> WindowSettings:
    """The fit's settings from the command line: every option except the outputs."""

    return WindowSettings(
        spectrum=args.spectrum, profile=args.profile, v1=args.v1, v2=args.v2,
        margin_cm1=args.margin_cm1, grid_margin_cm1=args.grid_margin_cm1,
        samples_per_resolution=args.samples_per_resolution, fwhm_cm1=args.fwhm_cm1,
        species=args.species, stellar=args.stellar, vsini_kms=args.vsini_kms,
        macroturbulence_kms=args.macroturbulence_kms, source_continuum=args.source_continuum,
        normalize_source=args.normalize_source, continuum_degree=args.continuum_degree,
        stages=args.stages, pin=tuple(args.pin), zenith_angle_deg=args.zenith_angle_deg,
        accuracy_mode=args.accuracy_mode, correction=args.correction,
        gaussian_ils=args.gaussian_ils, vectorize_layers=args.vectorize_layers,
        mixed_precision=args.mixed_precision, precompute_opacity=args.precompute_opacity,
        self_broadening=args.self_broadening, layer_chunk_size=args.layer_chunk_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectrum", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--v1", type=float, default=6000.0)
    parser.add_argument("--v2", type=float, default=6030.0)
    parser.add_argument("--margin-cm1", type=float, default=25.0,
                        help="how far outside the window a line may still contribute")
    parser.add_argument("--grid-margin-cm1", type=float, default=5.0,
                        help="how far outside the window the model grid extends")
    parser.add_argument("--samples-per-resolution", type=float, default=4.0)
    parser.add_argument("--fwhm-cm1", type=float, default=MEASURED_FWHM_CM1,
                        help="measured sinc FWHM; sets both the grid and the instrument")
    parser.add_argument("--profile", type=Path,
                        default=Path("data/profiles/kitt_peak_19901218_file5.csv"))
    parser.add_argument("--species", default="H2O,CO2,CH4",
                        help="comma-separated molecules, or 'all'. The right set is "
                             "per window: O2 has no lines at 6000 cm-1 and 818 at 7874.")
    parser.add_argument("--stellar", default="auto",
                        help="'flat', 'auto' to pick the band covering the window, or a path")
    parser.add_argument("--vsini-kms", type=float, default=0.0,
                        help="zero for disc centre, where solar rotation is transverse")
    parser.add_argument("--macroturbulence-kms", type=float, default=1.5)
    parser.add_argument("--source-continuum", action="store_true",
                        help="feed the source as flux_total, so it carries Payne Zero's own "
                             "predicted continuum, instead of flux = flux_total/flux_continuum "
                             "with that continuum already divided out. The fitted Chebyshev then "
                             "represents the instrument response and any grey telluric absorption "
                             "alone, rather than those times the stellar continuum slope -- which "
                             "is -5.7% across 2030-2060 cm-1 and -2.5% across 4350-4380. Implies "
                             "--normalize-source, because flux_total carries the star's units and "
                             "continuum_0 is bounded to +-2 in the log.")
    parser.add_argument("--normalize-source", action=argparse.BooleanOptionalAction, default=False,
                        help="Payne Zero already ships flux/flux_continuum, sitting at 1 where "
                             "there is no line. Dividing by the median replaces that physical "
                             "zero point with an arbitrary one; the scale is degenerate with "
                             "continuum_0 so it cannot change the fit, only where the split falls.")
    parser.add_argument("--continuum-degree", type=int, default=3)
    parser.add_argument("--stages", default="continuum,velocity,columns")
    parser.add_argument("--pin", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--zenith-angle-deg", type=float, default=None,
                        help="default is the header's mean air mass; 0 folds it into the "
                             "column scales, which is the diagnostic half of the slant-path test")
    parser.add_argument("--accuracy-mode", choices=("mt_ckd", "fast", "lblrtm_corrected"),
                        default="mt_ckd",
                        help="'fast' fits lines with no continuum at all, for windows past the "
                             "end of the MT_CKD table (20000 cm-1), where its coefficients are "
                             "~1e-31 and falling. "
                             "'lblrtm_corrected' replaces the runtime MT_CKD continuum with an "
                             "LBLRTM correction template built by build_lblrtm_correction.py. "
                             "The template is valid only for the profile and grid it was built "
                             "for, so --profile, --v1/--v2 and the grid options must match it.")
    parser.add_argument("--correction", type=Path, default=None,
                        help="the template npz; required by --accuracy-mode lblrtm_corrected")
    parser.add_argument("--gaussian-ils", action="store_true")
    parser.add_argument("--vectorize-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precompute-opacity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-broadening", choices=("linear", "frozen"), default="linear")
    parser.add_argument("--layer-chunk-size", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("docs/solar_fts_window_fit.json"))
    parser.add_argument("--diagnostic-npz", type=Path,
                        default=Path("benchmarks/results/solar_fts_window.npz"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    try:
        report, arrays = fit_window(settings_from(args), DataPaths.bootstrapped(root),
                                    stellar_directory=root / "data/stellar", base=root)
    except ValueError as exc:
        raise SystemExit(str(exc))

    (root / args.report).parent.mkdir(parents=True, exist_ok=True)
    (root / args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / args.diagnostic_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / args.diagnostic_npz, **arrays)
    print(json.dumps(report["residuals"], indent=2))
    print(f"wrote {args.report} and {args.diagnostic_npz}")


if __name__ == "__main__":
    main()
