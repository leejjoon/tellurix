#!/usr/bin/env python
"""Self-consistency of a raw FTS pair: the same Sun corrected from two air masses.

ftsspec_830626_2 (air mass 5.37) and _3 (3.09) are two scans of the same Sun on one day. If the
telluric correction is right their corrected spectra agree; where they do not at telluric
pixels, that is correction error, measured without anyone else's product. #3 is put on #2's
pixels after one velocity shift per run that aligns their solar lines (the README warns of a
wavenumber-scale shift, and the Sun's line-of-sight velocity changes between scans). The ratio
is normalized by a running mean over pixels where neither file has telluric absorption, then
scored as score_telluric_correction.py scores flatness: rms of (ratio - 1) at pixels clear of
solar lines where the first file's relative transmission is below 0.97 and both are at least
0.15, interior only. An error identical in both corrected spectra cancels, so this bounds the
correction error from below; since the error grows with air mass, most of it shows.

    uv run python scripts/check_fts_pair_consistency.py \\
        before=data/corrected/superseded_pre_widthscale/solar/ftsspec_830626_2:data/corrected/superseded_pre_widthscale/solar/ftsspec_830626_3 \\
        refit=data/corrected/solar:data/corrected/solar \\
        --output docs/june1983_pair_consistency.json --figures /tmp/figs

Each RUN is NAME=DIR_A:DIR_B, the directories holding the two files' window npz files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

C_KMS = 299792.458
EDGE_PIXELS = 50
MIN_TRANSMISSION = 0.15
MEAN_PIXELS = 51


def load(path):
    with np.load(path) as z:
        nu = np.asarray(z["wavenumber_cm1"]); o = np.argsort(nu)
        g = lambda k: np.asarray(z[k])[o]
        c = g("continuum"); eff = g("model_flux") / g("stellar_only_pixels")
        return dict(nu=nu[o], flat=g("flux") / eff / c, solar=g("stellar_only_pixels") / c,
                    T=g("transmission_pixels"), mask=g("mask").astype(bool),
                    noise=g("uncertainty") / c / eff)


def windows(dir_a, dir_b, stem_a, stem_b):
    out = {}
    for pa in sorted(Path(dir_a).glob(f"{stem_a}_*.npz")):
        v1 = pa.stem.split("_")[-1]
        pb = Path(dir_b) / f"{stem_b}_{v1}.npz"
        if pb.exists():
            out[float(v1)] = (load(pa), load(pb))
    return out


def on_a(a, b, v):
    """b's arrays on a's pixels, b shifted by v km/s."""
    nu = b["nu"] * (1 + v / C_KMS)
    out = {k: np.interp(a["nu"], nu, b[k]) for k in ("flat", "solar", "T", "noise")}
    out["mask"] = np.interp(a["nu"], nu, b["mask"].astype(float)) > 0.999
    return out


def running_reference(nu, ratio, ref):
    # normalized, so a short reference set or a window end does not pull the baseline down
    n = int(ref.sum()); k = np.ones(min(MEAN_PIXELS, n))
    sm = np.convolve(ratio[ref], k, "same") / np.convolve(np.ones(n), k, "same")
    return np.interp(nu, nu[ref], sm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", metavar="NAME=DIR_A:DIR_B")
    parser.add_argument("--stems", default="ftsspec_830626_2,ftsspec_830626_3")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None, help="write two PNGs here")
    parser.add_argument("--example-window", type=float, default=10240.0)
    args = parser.parse_args()
    stem_a, stem_b = args.stems.split(",")
    RUNS = {name: spec.split(":") for name, spec in (r.split("=", 1) for r in args.runs)}
    VG = np.arange(-1.0, 1.0001, 0.01)  # km/s
    summary, perwin, keep = {}, {}, {}
    for run in RUNS:
        W = windows(*RUNS[run], stem_a, stem_b)
        # one velocity: align solar lines where both skies are clear
        tot = np.zeros(VG.size); used = 0
        for a, b in W.values():
            if not a["mask"].any() or not b["mask"].any():
                continue
            t2 = a["T"] / np.percentile(a["T"][a["mask"]], 99)
            cost = []
            for v in VG:
                bb = on_a(a, b, v)
                t3 = bb["T"] / np.percentile(b["T"][b["mask"]], 99)
                g = a["mask"] & bb["mask"] & (t2 > 0.99) & (t3 > 0.99) & (a["solar"] < 0.97)
                cost.append(np.mean((a["flat"][g] - bb["flat"][g]) ** 2) if g.sum() > 50 else np.nan)
            cost = np.array(cost)
            if np.isfinite(cost).all():
                tot += cost / cost.min(); used += 1
        v = float(VG[np.argmin(tot)])
        rows = {}
        for v1, (a, b) in W.items():
            bb = on_a(a, b, v)
            if not (a["mask"].any() and bb["mask"].any()):
                continue
            t2 = a["T"] / np.percentile(a["T"][a["mask"]], 99)
            t3 = bb["T"] / np.percentile(bb["T"][bb["mask"]], 99)
            both = a["mask"] & bb["mask"] & (a["T"] >= MIN_TRANSMISSION) & (bb["T"] >= MIN_TRANSMISSION) \
                & (a["solar"] > 0.995)
            ref = both & (t2 > 0.995) & (t3 > 0.995)
            if ref.sum() < 50:
                continue
            ratio = a["flat"] / bb["flat"]
            ratio = ratio / running_reference(a["nu"], ratio, ref)
            s = both & (t2 < 0.97); s[:EDGE_PIXELS] = False; s[-EDGE_PIXELS:] = False
            if s.sum() < 30:
                continue
            noise = np.sqrt(a["noise"] ** 2 + bb["noise"] ** 2)
            rows[v1] = dict(pixels=int(s.sum()), rms=float(100 * np.sqrt(np.mean((ratio[s] - 1) ** 2))),
                            noise=float(100 * np.sqrt(np.mean(noise[s] ** 2))),
                            tau2=float(np.mean(-np.log(t2[s]))))
            keep[(run, v1)] = (a["nu"], a["flat"], bb["flat"], ratio, s, t2, t3, a["solar"])
        perwin[run] = rows
        tab = np.array([[r["rms"], r["noise"]] for r in rows.values()])
        summary[run] = dict(shift_kms=v, shift_windows=used, windows=len(rows),
                            median_rms=float(np.median(tab[:, 0])), median_noise=float(np.median(tab[:, 1])))
        print(run, summary[run])

    first, last = list(RUNS)[0], list(RUNS)[-1]
    common = sorted(set(perwin[first]) & set(perwin[last]))
    b = np.array([perwin[first][k]["rms"] for k in common]); r = np.array([perwin[last][k]["rms"] for k in common])
    nz = np.array([perwin[last][k]["noise"] for k in common])
    net = lambda x: float(np.median(np.sqrt(np.clip(x ** 2 - nz ** 2, 0, None))))
    summary["common"] = {"windows": len(common), f"median_{first}": float(np.median(b)),
                         f"median_{last}": float(np.median(r)), f"{last}_agrees_better": int((r < b).sum()),
                         "median_noise": float(np.median(nz)),
                         f"median_{first}_net_of_noise": net(b), f"median_{last}_net_of_noise": net(r)}
    print("common", summary["common"])
    if args.output:
        args.output.write_text(json.dumps({
            "method": __doc__.split("\n\n")[1].replace("\n", " "),
            "runs": {k: v for k, v in RUNS.items()}, "summary": summary,
            "windows": {k: {str(w): x for w, x in v.items()} for k, v in perwin.items()}}, indent=1) + "\n")
        print(f"wrote {args.output}")
    if args.figures:
        figures(args.figures, perwin, keep, common, b, r, nz, first, last, args.example_window)


def figures(OUT, perwin, keep, common, b, r, nz, first, last, example):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)

    # per-window agreement
    v1s = np.array(common)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    ax[0].scatter(b, r, s=14, c=v1s, cmap="viridis")
    lim = [0.1, max(b.max(), r.max()) * 1.2]
    ax[0].plot(lim, lim, "k-", lw=.6); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(f"disagreement, {first} (rms %)"); ax[0].set_ylabel(f"{last} (rms %)")
    ax[0].set_title(f"{len(common)} windows; {last} agrees better in {int((r < b).sum())}")
    sc = ax[0].collections[0]; fig.colorbar(sc, ax=ax[0], label="window start (cm$^{-1}$)")
    ax[1].plot(v1s, b, "C7.", ms=5, label=f"{first} (median {np.median(b):.2f}%)")
    ax[1].plot(v1s, r, "C0.", ms=5, label=f"{last} (median {np.median(r):.2f}%)")
    ax[1].plot(v1s, nz, "k-", lw=.8, alpha=.6, label=f"pixel-noise floor (median {np.median(nz):.2f}%)")
    ax[1].set_yscale("log"); ax[1].set_xlabel("window start (cm$^{-1}$)"); ax[1].set_ylabel("rms of corrected #2 / corrected #3 - 1 (%)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "pair_consistency.png", dpi=110)

    # ---- figure: one window ---------------------------------------------------
    pick = example if ((last, example) in keep and (first, example) in keep) else common[len(common) // 2]
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [1.6, 3, 2]})
    nu, f2, f3, rat, s, t2, t3, solar = keep[(last, pick)]
    ax[0].plot(nu, t2, "C2", lw=.8, label="#2 transmission, X 5.37"); ax[0].plot(nu, t3, "C1", lw=.8, label="#3 transmission, X 3.09")
    ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=8, loc="lower left"); ax[0].set_ylabel("transmission")
    bl = lambda y: np.where(np.isfinite(rat), y, np.nan)
    ax[1].plot(nu, solar, "0.65", lw=1.3, label="solar model")
    ax[1].plot(nu, f2, "C2", lw=.8, label=f"#2 corrected ({last})"); ax[1].plot(nu, f3, "C1", lw=.8, label=f"#3 corrected ({last})")
    ax[1].set_ylim(0.6, 1.1); ax[1].legend(fontsize=8, loc="lower left", ncol=3); ax[1].set_ylabel("corrected / continuum")
    for run, col in [(first, "C7"), (last, "C0")]:
        nu_, _, _, rat_, s_, *_ = keep[(run, pick)]
        ax[2].plot(nu_, np.where(s_, 100 * (rat_ - 1), np.nan), ".", color=col, ms=2.5,
                   label=f"{run}: rms {perwin[run][pick]['rms']:.2f}%")
    ax[2].axhline(0, color="0.5", lw=.5); ax[2].set_ylabel("scored pixels:\n#2 / #3 - 1 (%)")
    ax[2].legend(fontsize=8, loc="lower left"); ax[2].set_xlabel("vacuum wavenumber (cm$^{-1}$)")
    n = nu.size
    sec = ax[0].secondary_xaxis("top", functions=(lambda x: np.interp(x, nu, np.arange(n)), lambda i: np.interp(i, np.arange(n), nu)))
    sec.set_xlabel("pixel index in #2's window")
    ax[0].set_title(f"{pick:.0f}-{pick + 30:.0f} cm$^{{-1}}$: the same Sun through 5.37 and 3.09 air masses")
    fig.tight_layout(); fig.savefig(OUT / "pair_window.png", dpi=110)
    print("example window", pick)


if __name__ == "__main__":
    main()
