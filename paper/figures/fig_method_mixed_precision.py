#!/usr/bin/env python
"""Paper figure: float32 cancellation in the Voigt wing derivative, and its cure.

ExoJAX evaluates the Voigt-Hjerting function H(x, a) = Re w(x + ia) in the far
wing (x^2 + a^2 >= 111) with Zaghloul's truncated asymptotic series, and
differentiates it through the Faddeeva identity w'(z) = -2 z w(z) + 2i/sqrt(pi):

    dH/dx = 2 a L - 2 x H,        dH/da = 2 x L + 2 a H - 2/sqrt(pi),   L = Im w.

In the wing x L -> 1/sqrt(pi) and the derivatives are O(1/x^2) and O(1/x^3)
remainders of O(1) terms, so in float32 they cancel catastrophically.
``tellurix.direct._mixed_wing_jvp`` substitutes the same truncated series into
the identity and simplifies it algebraically: with u = 1/z^2,

    dH/dx = Im(u + 1.5 u^2 + 3.75 u^3)/sqrt(pi),  dH/da = Re(...)/sqrt(pi),

which has no cancellation. This script evaluates the value and both partials
in float32 (ExoJAX's form and the rewrite) and float64 (ExoJAX's form) against
an extended-precision (numpy longdouble) evaluation of the same truncated
series, and histograms the wing-pair offsets x of a real workload (AER 3.9 H2O,
5000-5100 cm-1, Kitt Peak 1994 profile, bottom and top layers).

    UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_method_mixed_precision.py

CPU only. Writes PDF and PNG (200 dpi) to ``paper/figures/output/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/figures/output"
CACHE = ROOT / "paper/figures/cache/method_mixed_precision_x.npz"
LINES = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule/01_H2O/01_H2O"
PROFILE = ROOT / "data/profiles/kitt_peak_1994.csv"
WINDOW_CM1 = (5000.0, 5100.0)

BLUE, VERMILLION, GREY, ORANGE, GREEN = "#0072B2", "#D55E00", "#666666", "#E69F00", "#009E73"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42,
})
SQRT_PI = np.sqrt(np.pi)


def asymptotic(x, a, dtype):
    """ExoJAX's asymptotic_wofz, evaluated entirely in ``dtype`` (complex)."""
    ctype = {np.float32: np.complex64, np.float64: np.complex128, np.longdouble: np.clongdouble}[dtype]
    z = (x.astype(dtype) + 1j * a.astype(dtype)).astype(ctype)
    half = ctype(1) / (ctype(2) * z * z)
    sp = ctype(np.sqrt(dtype(np.pi)) if dtype is not np.longdouble else np.sqrt(np.longdouble(np.pi)))
    return ctype(1j) / (z * sp) * (1 + half * (1 + half * (3 + half * 15)))


def naive_derivatives(x, a, dtype):
    """ExoJAX's hjert_jvp identity, in ``dtype``."""
    w = asymptotic(x, a, dtype)
    h, l = w.real, w.imag
    xs, as_ = x.astype(dtype), a.astype(dtype)
    two_over = dtype(2) / np.sqrt(dtype(np.pi))
    return h, dtype(2) * as_ * l - dtype(2) * xs * h, dtype(2) * xs * l + dtype(2) * as_ * h - two_over


def rewritten_derivatives(x, a, dtype):
    """tellurix's _mixed_wing_jvp: the same identity, simplified with u = 1/z^2."""
    ctype = {np.float32: np.complex64, np.longdouble: np.clongdouble, np.float64: np.complex128}[dtype]
    z = (x.astype(dtype) + 1j * a.astype(dtype)).astype(ctype)
    u = ctype(1) / (z * z)
    d = u * (1 + u * (ctype(1.5) + ctype(3.75) * u)) / np.sqrt(dtype(np.pi))
    return asymptotic(x, a, dtype).real, d.imag, d.real


def workload_offsets():
    """|x| of every wing pair for the bottom and top layer of the real workload."""
    if CACHE.exists():
        data = np.load(CACHE)
        return {key: data[key] for key in data.files}
    import tellurix  # noqa: F401  (x64 before ExoJAX)
    from tellurix import AERLineDatabase, constant_velocity_grid, load_atmosphere_csv
    from exojax.database.core.broadening import doppler_sigma, gamma_hitran

    profile = load_atmosphere_csv(PROFILE)
    grid = constant_velocity_grid(1.0e7 / WINDOW_CM1[1], 1.0e7 / WINDOW_CM1[0])
    db = AERLineDatabase(LINES, "H2O", WINDOW_CM1)
    nu0 = np.asarray(db.nu_lines)
    edges = np.linspace(0.0, 4.5, 181)  # log10 |x|
    result = {"edges": edges, "lines": np.array(nu0.size), "samples": np.array(grid.size)}
    for label, index in (("bottom", -1), ("top", 0)):
        t, p = float(profile.temperature_k[index]), float(profile.pressure_layer_bar[index])
        pself = p * float(profile.vmr["H2O"][index])
        sigma = np.asarray(doppler_sigma(db.nu_lines, t, db.molmass))
        gamma = np.asarray(gamma_hitran(p, t, pself, db.n_air, db.gamma_air, db.gamma_self))
        scale = 1.0 / (np.sqrt(2.0) * sigma)
        a = scale * gamma
        counts = np.zeros(edges.size - 1)
        core = 0
        for start in range(0, nu0.size, 500):
            sl = slice(start, start + 500)
            x = np.abs(grid[None, :] - nu0[sl, None]) * scale[sl, None]
            wing = x * x + a[sl, None] ** 2 >= 111.0
            core += int((~wing).sum())
            counts += np.histogram(np.log10(np.maximum(x[wing], 1e-12)), bins=edges)[0]
        result[f"{label}_counts"] = counts
        result[f"{label}_a_median"] = np.array(np.median(a))
        result[f"{label}_core"] = np.array(core)
        result[f"{label}_T"] = np.array(t)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **result)
    return result


def main() -> None:
    x = np.logspace(np.log10(10.6), 4.4, 8000)
    bins = np.logspace(np.log10(10.6), 4.4, 71)
    centres_x = np.sqrt(bins[1:] * bins[:-1])
    which = np.digitize(x, bins) - 1
    work = workload_offsets()
    a_values = (float(work["top_a_median"]), float(work["bottom_a_median"]))
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 4.9), constrained_layout=True)
    flat = axes.ravel()
    titles = ("(a) value  $H=\\mathrm{Re}\\,w$", "(b) $\\partial H/\\partial x$",
              "(c) $\\partial H/\\partial a$")
    numbers = {}
    for (a_value, style) in zip(a_values, ("-", "--")):
        a = np.full_like(x, a_value)
        a = np.where(x * x + a * a >= 111.0, a, np.nan)
        truth = rewritten_derivatives(x.astype(np.longdouble), a.astype(np.longdouble), np.longdouble)
        cases = (("ExoJAX identity, float32", naive_derivatives(x, a, np.float32), VERMILLION, 1.3),
                 ("ExoJAX identity, float64", naive_derivatives(x, a, np.float64), GREY, 0.9),
                 ("tellurix rewrite, float32", rewritten_derivatives(x, a, np.float32), BLUE, 1.3))
        for label, values, color, lw in cases:
            for k, ax in enumerate(flat[:3]):
                rel = np.abs(np.asarray(values[k], dtype=np.longdouble) - truth[k]) / np.abs(truth[k])
                rel = np.maximum(rel.astype(float), 1e-19)
                # Round-off scatters by orders of magnitude from one x to the
                # next; the rms within narrow log bins is what the eye needs.
                rms = np.sqrt(np.array([np.nanmean(rel[which == b] ** 2) for b in range(centres_x.size)]))
                ax.loglog(centres_x, rms, style, color=color, lw=lw,
                          label=label if style == "-" else None)
                key = f"{label} | {titles[k].split()[0]} | a={a_value:.3g}"
                at = {f"x={v:g}": float(np.interp(np.log(v), np.log(centres_x), np.log(rms))) for v in (100, 1000, 3000, 10000)}
                numbers[key] = {k2: float(np.exp(v)) for k2, v in at.items()}
    for k, ax in enumerate(flat[:3]):
        ax.set_title(titles[k], loc="left")
        ax.set_xlabel("offset from line centre $x$ (Doppler widths)")
        ax.set_ylabel("relative error (rms)")
        ax.set_ylim(1e-17, 1e2)
        ax.set_xlim(10, x[-1])
        ax.axhline(np.finfo(np.float32).eps, color="k", lw=0.6, ls=":")
        ax.axhline(np.finfo(np.float64).eps, color="k", lw=0.6, ls=":")
        ax.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
    flat[0].text(12, np.finfo(np.float32).eps * 3, "float32 $\\epsilon$", fontsize=6.5)
    flat[0].text(12, np.finfo(np.float64).eps * 3, "float64 $\\epsilon$", fontsize=6.5)
    flat[2].text(150, 3e-1, "$\\propto x^2\\epsilon_{32}$", color=VERMILLION, fontsize=7)
    flat[2].text(400, 1e-13, "$\\propto x^2\\epsilon_{64}$", color=GREY, fontsize=7)
    handles = flat[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    extra = [Line2D([], [], color="k", lw=0.9, ls="-"), Line2D([], [], color="k", lw=0.9, ls="--")]
    figure.legend(handles[0] + extra, handles[1] + [f"$a={a_values[0]:.2f}$ (top layer median)",
                                                    f"$a={a_values[1]:.1f}$ (bottom layer median)"],
                  loc="outside upper center", ncol=3, frameon=False, fontsize=6.5)
    # (d) where the wing pairs of a real workload actually sit
    ax = flat[3]
    edges = work["edges"]
    centres = 10 ** (0.5 * (edges[1:] + edges[:-1]))
    for label, color, style in (("bottom", GREEN, "-"), ("top", ORANGE, "--")):
        counts = work[f"{label}_counts"]
        cumulative = np.cumsum(counts) / counts.sum()
        ax.semilogx(centres, 1 - cumulative, style, color=color, lw=1.3,
                    label=f"{label} layer, $T={float(work[f'{label}_T']):.0f}$ K")
        for v in (100, 1000):
            numbers[f"fraction of {label}-layer wing pairs with x > {v}"] = float(
                np.interp(np.log10(v), np.log10(centres), 1 - cumulative))
    ax.set_xlim(10, x[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("offset from line centre $x$ (Doppler widths)")
    ax.set_ylabel("fraction of wing pairs beyond $x$")
    ax.set_title("(d) wing pairs, H$_2$O 5000–5100 cm$^{-1}$", loc="left")
    ax.text(0.03, 0.42, f"{int(work['lines']):,} lines $\\times$ {int(work['samples']):,} samples",
            transform=ax.transAxes, fontsize=6.5)
    ax.legend(loc="lower left", frameon=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT / f"fig_method_mixed_precision.{suffix}", dpi=200)
    numbers["a_values"] = a_values
    numbers["core_pairs"] = {k: int(work[f"{k}_core"]) for k in ("bottom", "top")}
    (OUTPUT / "fig_method_mixed_precision.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
