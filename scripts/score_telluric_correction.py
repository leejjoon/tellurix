#!/usr/bin/env python
"""How flat corrected solar spectra are where only the atmosphere absorbs, against the authors'.

niratl ships the authors' own corrected solar spectrum (Wallace, Hinkle &
Livingston 1993). Ours minus theirs mixes both corrections' errors, so instead
each is scored on its own, the same way: at pixels where the model transmission
is below 0.97 and the solar model is flat (above 0.995 of its continuum), the
rms of the corrected spectrum about a running mean of nearby pixels with
neither solar nor telluric absorption. A perfect correction leaves noise there.
The first and last 50 pixels are not scored: an edge pixel is inside the
neighbouring page too (pages overlap by 5 cm-1), and a fit that leaves edges
out should not be judged on them. Nor are pixels below a transmission of 0.15, which
the corrected products blank: dividing by a saturated core turns a percent of model
error into a spike that can outweigh the rest of a window. Two bins: every telluric pixel (T < 0.97) and
the deep ones (T < 0.7).

    uv run python scripts/score_telluric_correction.py \\
        production=data/corrected/superseded_pre_widthscale/solar/niratl clean=data/corrected/solar/niratl \\
        --output docs/niratl_correction_score.json

Each RUN is NAME=DIRECTORY holding a fit_fts_batch.py run's window npz files;
the first run's transmission and solar model choose the scored pixels for all of
them, so every run is judged on the same pixels.

The raw Kitt Peak files have no corrected spectrum of the authors', except file 5,
which photatl is (rescaled per page; its `solar` column is the authors'
correction). There pixels photatl interpolated (atmospheric < dg) are not scored
for any run, being flat by construction. At air mass 2-5 the continua keep the
transmission below 0.995 even between lines, so `--relative-transmission` divides
it by each window's 99th percentile first:

    uv run python scripts/score_telluric_correction.py --stem ftsspec_901218_5 \\
        --authors photatl --relative-transmission old=data/corrected/solar new=...

Wallace 2011 regions 1-2 carry the authors' correction of the same pixels, but only
where their transmission is above about 0.8 (zero below, absent over the 0.94 um band
and the A-band), so ``--authors wallace2011`` scores only the pixels they corrected,
and ``--authors none`` on the same runs scores the rest:

    uv run python scripts/score_telluric_correction.py --stem wallace2011_reg1 \\
        --authors wallace2011 --wallace2011-region 1 w100=data/corrected/solar/wallace2011
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tellurix_fts.flatness import BINS, EDGE_PIXELS, MIN_TRANSMISSION, corrected, photatl_solar, scatter

ATLASES = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/nso")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", metavar="NAME=DIRECTORY")
    parser.add_argument("--stem", default="ph*", help="window files are <stem>_<v1>.npz (default: niratl pages)")
    parser.add_argument("--authors", choices=("niratl", "photatl", "wallace2011", "none"), default="niratl")
    parser.add_argument("--wallace2011-region", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--wallace2011", type=Path, default=ATLASES / "wallace2011_flux")
    parser.add_argument("--niratl", type=Path, default=ATLASES / "niratl")
    parser.add_argument("--photatl", type=Path, default=ATLASES / "photatl")
    parser.add_argument("--relative-transmission", action="store_true",
                        help="measure transmission against each window's 99th percentile")
    parser.add_argument("--minimum-pixels", type=int, default=30,
                        help="telluric pixels a page needs to be scored (deep: a sixth of it)")
    parser.add_argument("--telluric-below", type=float, default=BINS["telluric"],
                        help="the telluric bin's transmission ceiling; Wallace 2011 region 3's "
                             "water lines are 1-4%% deep, so few of its pixels reach 0.97")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    bins = {**BINS, "telluric": args.telluric_below}

    from tellurix_fts.flux_atlas import WALLACE2011_REGIONS, read_wallace2011_correction
    from tellurix_fts.nso import read_niratl_page, read_photatl_page

    runs = dict(entry.split("=", 1) for entry in args.runs)
    first, *_ = runs
    photatl = ([read_photatl_page(p) for p in sorted(args.photatl.glob("wn*")) if p.name[2:].isdigit()]
               if args.authors == "photatl" else [])
    wallace = (read_wallace2011_correction(args.wallace2011 / WALLACE2011_REGIONS[args.wallace2011_region]["file"])
               if args.authors == "wallace2011" else None)
    pages = {}
    for path in sorted(Path(runs[first]).glob(f"{args.stem}_*.npz")):
        page = path.stem if args.authors != "niratl" else path.stem.split("_")[0]
        others = {name: sorted(Path(d).glob(f"{page}.npz" if args.authors != "niratl" else f"{page}_*.npz"))
                  for name, d in runs.items()}
        if not all(others.values()):
            continue
        nu, flat, solar, transmission, mask = corrected(path)
        series = {name: np.interp(nu, *corrected(found[0])[:2]) for name, found in others.items()}
        clean = mask & (solar > 0.995) & (transmission >= MIN_TRANSMISSION)
        authors = None
        if args.authors == "niratl":
            niratl = read_niratl_page(args.niratl / page)
            authors = np.interp(nu, niratl.wavenumber_vacuum_cm1,
                                np.where(niratl.solar == -1.0, np.nan, niratl.solar))
        elif args.authors == "photatl":
            authors, filled = photatl_solar(photatl, nu)
            clean &= ~filled
        elif args.authors == "wallace2011":
            w_nu, w_corrected, _ = wallace
            authors = np.interp(nu, w_nu, np.nan_to_num(w_corrected))
            # A pixel next to one they did not correct is partly interpolated from it.
            authors[np.interp(nu, w_nu, np.isfinite(w_corrected).astype(float)) < 0.999] = np.nan
        if authors is not None:
            clean &= np.isfinite(authors)
        if args.relative_transmission and mask.any():
            transmission = transmission / np.percentile(transmission[mask], 99)
        reference = clean & (transmission > 0.995)
        if reference.sum() < (50 if args.relative_transmission else 100):
            continue
        if authors is not None:
            # The authors' level is arbitrary; put it on ours where nothing absorbs.
            series["authors"] = authors / np.median(authors[reference])
        interior = np.zeros(nu.size, bool)
        interior[EDGE_PIXELS:-EDGE_PIXELS] = True
        scores = {}
        for label, ceiling in bins.items():
            scored = clean & interior & (transmission < ceiling)
            if scored.sum() < (args.minimum_pixels if label == "telluric" else args.minimum_pixels // 6):
                continue
            scores[label] = {"pixels": int(scored.sum()),
                             **{name: scatter(nu, values, reference, scored)
                                for name, values in series.items()}}
        if scores:
            pages[page] = scores

    authored = args.authors != "none"
    names = [*runs, *(["authors"] if authored else [])]
    summary = {}
    for label in bins:
        table = np.array([[pages[p][label][n] for n in names] for p in pages if label in pages[p]])
        if not table.size:
            continue
        summary[label] = {
            "pages": int(len(table)),
            "median_percent": {n: float(v) for n, v in zip(names, np.median(table, axis=0))},
            f"pages_flatter_than_{first}": {n: int(np.sum(table[:, i] < table[:, 0]))
                                            for i, n in enumerate(names) if i},
        }
        if authored:
            summary[label]["pages_at_or_below_authors"] = {
                n: int(np.sum(table[:, i] <= table[:, -1])) for i, n in enumerate(names[:-1])}
            summary[label]["pages_worse_than_authors_by_0p2"] = {
                n: int(np.sum(table[:, i] - table[:, -1] > 0.2)) for i, n in enumerate(names[:-1])}
        print(f"{label} (T < {bins[label]}), {len(table)} pages, median %: "
              + "  ".join(f"{n} {v:.3f}" for n, v in summary[label]["median_percent"].items()))
        print(f"  flatter than {first}: " + "  ".join(
            f"{n} {v}" for n, v in summary[label][f"pages_flatter_than_{first}"].items()))
        if authored:
            print("  at or below the authors: " + "  ".join(
                f"{n} {v}" for n, v in summary[label]["pages_at_or_below_authors"].items()))
    if args.output:
        args.output.write_text(json.dumps({
            "method": __doc__.split("\n\n")[1].replace("\n", " "),
            "settings": {"stem": args.stem, "authors": args.authors,
                         **({"wallace2011_region": args.wallace2011_region} if args.authors == "wallace2011" else {}),
                         "relative_transmission": args.relative_transmission,
                         **({"telluric_below": args.telluric_below}
                            if args.telluric_below != BINS["telluric"] else {})},
            "runs": runs, "summary": summary, "pages": pages}, indent=1) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
