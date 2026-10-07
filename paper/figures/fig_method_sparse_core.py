#!/usr/bin/env python
"""Paper figure: the sparse core list of ``SparseCoreDirect``.

ExoJAX's ``hjert`` picks Algorithm 916 (a 27-term series) where
x^2 + a^2 < 111 and Zaghloul's asymptotic series elsewhere, with ``jnp.where``.
Under ``vmap`` over the dense (line, grid-sample) plane ``where`` evaluates
*both* branches at every pair and then selects, so the expensive series is paid
everywhere although almost every pair is in a far wing. ``SparseCoreDirect``
(``tellurix.direct``) computes, once and outside the kernel, a per-line half
width sqrt(222) sigma_D(T_max) (+ the largest pressure shift) inside which a
pair *can* reach the core branch; because grid and line centres are sorted each
line's core is one contiguous run of samples, found with ``searchsorted``.
Inside the kernel the asymptotic branch is evaluated over the whole plane
(recomputing the offsets from the two 1-D vectors, so XLA fuses it into one
reduction) and ``hjert`` only over the compact core list, scatter-added.

Workload: AER 3.9 H2O, 5000-5100 cm-1, the default constant-velocity grid of
``benchmarks/benchmark.py``, the 12-layer Kitt Peak 1994 profile. Timings are
on one GPU (forward cross sections of all 12 layers, vmapped), median of 20.

    CUDA_VISIBLE_DEVICES=2 UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_method_sparse_core.py

Measurements are cached in ``paper/figures/cache/method_sparse_core.npz`` and
``.json``; ``--recompute`` redoes them. Writes PDF and PNG (200 dpi) to
``paper/figures/output/``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/figures/output"
CACHE = ROOT / "paper/figures/cache/method_sparse_core"
LINE_ROOT = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
PROFILE = ROOT / "data/profiles/kitt_peak_1994.csv"
WINDOW_CM1 = (5000.0, 5100.0)
ZOOM_CM1 = (5049.85, 5050.25)

BLUE, VERMILLION, GREY, ORANGE, GREEN, SKY = ("#0072B2", "#D55E00", "#666666", "#E69F00",
                                              "#009E73", "#56B4E9")
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42,
})


def timed(function, *args, iterations=20):
    import jax

    started = time.perf_counter()
    jax.block_until_ready(function(*args))
    first = time.perf_counter() - started
    for _ in range(2):
        jax.block_until_ready(function(*args))
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        jax.block_until_ready(function(*args))
        samples.append(time.perf_counter() - started)
    return first, statistics.median(samples)


def held_bytes_for(databases, grid):
    """Device bytes of ExoJAX's dense offset matrix (+ the old boolean mask) vs
    the 1-D vectors SparseCoreDirect keeps, for the given workload."""
    from tellurix.direct import SparseCoreDirect

    dense = mask = vectors = 0
    for db in databases:
        lines = int(np.asarray(db.nu_lines).size)
        dense += lines * grid.size * 8
        mask += lines * grid.size
        calc = SparseCoreDirect(db, grid)
        vectors += sum(int(np.asarray(v).nbytes) for v in (
            calc._nu_grid, calc._nu_lines, calc._core_half_width, calc._core_grid, calc._core_line))
    return dense, mask, vectors


def measure() -> dict:
    import jax
    import jax.numpy as jnp

    import tellurix  # noqa: F401  (float64 before ExoJAX)
    from tellurix import (AERLineDatabase, ExoJAXOpacityBackend, constant_velocity_grid,
                          load_atmosphere_csv)
    from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
    from exojax.database.core.line_strength import line_strength
    from exojax.opacity.lpf.lpf import hjert
    from exojax.special.faddeeva import asymptotic_wofz, rewofz
    from exojax.utils.constants import Tref_original

    profile = load_atmosphere_csv(PROFILE)
    temperature = jnp.asarray(profile.temperature_k)
    pressure = jnp.asarray(profile.pressure_layer_bar)
    pself = pressure * jnp.asarray(profile.vmr["H2O"])
    t_range = (float(np.min(profile.temperature_k)), float(np.max(profile.temperature_k)))
    grid = constant_velocity_grid(1.0e7 / WINDOW_CM1[1], 1.0e7 / WINDOW_CM1[0])
    db = AERLineDatabase(LINE_ROOT / "01_H2O/01_H2O", "H2O", WINDOW_CM1)
    nu0 = np.asarray(db.nu_lines)
    out: dict = {"device": str(jax.devices()[0]), "lines": int(nu0.size), "samples": int(grid.size),
                 "layers": int(temperature.size), "pairs": int(nu0.size * grid.size)}

    calcs = {}
    for label, method, mixed in (("direct", "direct", False), ("sparse", "direct_sparse", False),
                                 ("sparse_mixed", "direct_sparse", True)):
        backend = ExoJAXOpacityBackend.prepare(
            {"H2O": db}, grid, methods=method, temperature_range_k=t_range,
            maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
            vectorize_layers=True, mixed_precision=mixed)
        calcs[label] = backend.calculators["H2O"]
        # The profile is closed over, as TelluricModel does: it is then a
        # compile-time constant and XLA folds SparseCoreDirect's per-layer
        # range test. Passed as traced arguments instead, the batched
        # lax.cond becomes a select and the dense fallback runs as well.
        evaluate = jax.jit(lambda b=backend: b.cross_sections(temperature, pressure, {"H2O": pself})["H2O"])
        first, median = timed(evaluate)
        traced = jax.jit(lambda t, p, s, b=backend: b.cross_sections(t, p, {"H2O": s})["H2O"])
        out[f"time_{label}_traced_profile"] = dict(zip(("compile_and_first_s", "median_ms"),
                                                       np.array(timed(traced, temperature, pressure, pself))
                                                       * (1, 1e3)))
        out[f"time_{label}"] = {"compile_and_first_s": first, "median_ms": 1e3 * median}
        if label == "direct":
            reference = np.asarray(evaluate())
        else:
            value = np.asarray(evaluate())
            out[f"max_relative_difference_{label}"] = float(
                np.max(np.abs(value - reference)) / np.max(np.abs(reference)))
        print(label, out[f"time_{label}"], flush=True)

    sparse = calcs["sparse"]
    out["core_pairs"] = int(sparse.core_line.size)
    out["core_fraction"] = out["core_pairs"] / out["pairs"]

    # Isolated branches, each reduced to a (layer, grid) cross section exactly
    # as the kernels do, so XLA can fuse them the same way.
    grid_j, nu0_j = jnp.asarray(grid), jnp.asarray(nu0)
    core_line, core_grid = jnp.asarray(sparse.core_line), jnp.asarray(sparse.core_grid)

    def line_parameters(t, p, s):
        sigma = doppler_sigma(db.nu_lines, t, db.molmass)
        gamma = gamma_hitran(p, t, s, db.n_air, db.gamma_air, db.gamma_self) + gamma_natural(db.A)
        strength = line_strength(t, db.logsij0, db.nu_lines, db.elower,
                                 db.qr_interp(db.isotope, t, Tref_original), Tref_original)
        scale = 1 / (jnp.sqrt(2.0) * sigma)
        return scale, scale * gamma, strength * scale / jnp.sqrt(jnp.pi)

    def dense_branch(kind):
        def one(t, p, s):
            scale, a, weight = line_parameters(t, p, s)
            x = (grid_j[None, :] - nu0_j[:, None]) * scale[:, None]
            aa = jnp.broadcast_to(a[:, None], x.shape)
            if kind == "alg916":
                values = jax.vmap(jax.vmap(rewofz))(jnp.minimum(x, 9.0), aa)
            elif kind == "asymptotic64":
                values = jnp.real(asymptotic_wofz(jnp.maximum(jnp.abs(x), 11.0), aa))
            else:
                values = jnp.real(asymptotic_wofz(jnp.maximum(jnp.abs(x), 11.0).astype(jnp.float32),
                                                  aa.astype(jnp.float32))).astype(x.dtype)
            return jnp.sum(values * weight[:, None], axis=0)
        return jax.jit(lambda: jax.vmap(one)(temperature, pressure, pself))

    def core_branch(t, p, s):
        scale, a, weight = line_parameters(t, p, s)
        x = (grid_j[core_grid] - nu0_j[core_line]) * scale[core_line]
        values = jax.vmap(hjert)(x, a[core_line]) * weight[core_line]
        return jnp.zeros(grid_j.size).at[core_grid].add(values)

    for label, function in (("alg916_all_pairs", dense_branch("alg916")),
                            ("asymptotic_all_pairs_f64", dense_branch("asymptotic64")),
                            ("asymptotic_all_pairs_f32", dense_branch("asymptotic32")),
                            ("hjert_core_list", jax.jit(lambda: jax.vmap(core_branch)(temperature, pressure, pself)))):
        first, median = timed(function)
        out[f"component_{label}"] = {"compile_and_first_s": first, "median_ms": 1e3 * median}
        print(label, out[f"component_{label}"], flush=True)

    # Memory held, for this workload and for the documented Arcturus fit
    # workload (5005-5025 cm-1, H2O + CO2, R = 1e5, 4 samples per resolution).
    dense, mask, vectors = held_bytes_for([db], grid)
    out["held_bytes"] = {"dense_offsets": dense, "dense_mask": mask, "sparse_vectors": vectors}
    fit_grid = constant_velocity_grid(1.0e7 / 5025.0, 1.0e7 / 5005.0, resolving_power=100_000.0,
                                      samples_per_resolution=4.0, margin_cm1=25.0)
    fit_dbs = [AERLineDatabase(LINE_ROOT / f"{stem}/{stem}", name, (float(fit_grid[0]), float(fit_grid[-1])),
                               margin_cm1=0.0) for stem, name in (("01_H2O", "H2O"), ("02_CO2", "CO2"))]
    dense, mask, vectors = held_bytes_for(fit_dbs, fit_grid)
    out["held_bytes_fit_workload"] = {"dense_offsets": dense, "dense_mask": mask, "sparse_vectors": vectors,
                                      "lines": [int(np.asarray(d.nu_lines).size) for d in fit_dbs],
                                      "samples": int(fit_grid.size)}

    # The (line, grid) plane for the bottom and top layers: |x| and the branch.
    arrays = {"grid": grid, "nu0": nu0, "core_line": sparse.core_line, "core_grid": sparse.core_grid}
    for name, index in (("bottom", -1), ("top", 0)):
        scale, a, _ = (np.asarray(v) for v in line_parameters(temperature[index], pressure[index], pself[index]))
        arrays[f"scale_{name}"], arrays[f"a_{name}"] = scale, a
    arrays["strength_ref"] = np.asarray(db.logsij0)
    np.savez(CACHE.with_suffix(".npz"), **arrays)
    CACHE.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def plot(out: dict) -> None:
    data = np.load(CACHE.with_suffix(".npz"))
    grid, nu0 = data["grid"], data["nu0"]
    core_line, core_grid = data["core_line"], data["core_grid"]
    scale, a = data["scale_bottom"], data["a_bottom"]

    figure = plt.figure(figsize=(7.1, 5.6), constrained_layout=True)
    top, bottom = figure.subfigures(2, 1, height_ratios=(1.15, 1.0))
    ax_plane, ax_zoom = top.subplots(1, 2, gridspec_kw={"width_ratios": (1.0, 1.25)})
    ax_cost, ax_held = bottom.subplots(1, 2, gridspec_kw={"width_ratios": (1.6, 1.0)})

    # (a) the whole plane, block-minimum of log10|x| at the bottom layer
    blocks_l, blocks_g = 300, 300
    li = np.linspace(0, nu0.size, blocks_l + 1).astype(int)
    gi = np.linspace(0, grid.size, blocks_g + 1).astype(int)
    image = np.empty((blocks_l, blocks_g))
    for i in range(blocks_l):
        sl = slice(li[i], li[i + 1])
        x = np.abs(grid[None, :] - nu0[sl, None]) * scale[sl, None]
        reduced = np.minimum.reduceat(x, gi[:-1], axis=1)
        image[i] = np.log10(np.maximum(reduced.min(axis=0), 1e-2))
    mesh = ax_plane.pcolormesh(grid[gi[:-1]], li[:-1], image, cmap="Greys_r", vmin=0, vmax=4.3,
                               shading="auto", rasterized=True)
    ax_plane.scatter(grid[core_grid[::7]], core_line[::7], s=0.15, color=VERMILLION, lw=0,
                     rasterized=True)
    colorbar = top.colorbar(mesh, ax=ax_plane, location="right", pad=0.01, shrink=0.9)
    colorbar.set_label("block min. $\\log_{10}|x|$", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)
    zoom_lines = np.flatnonzero((nu0 > ZOOM_CM1[0] - 0.04) & (nu0 < ZOOM_CM1[1] + 0.04))
    ax_plane.add_patch(plt.Rectangle((ZOOM_CM1[0], zoom_lines[0]), ZOOM_CM1[1] - ZOOM_CM1[0],
                                     zoom_lines[-1] - zoom_lines[0], fill=False, ec=BLUE, lw=1.0))
    ax_plane.set_xlabel("grid wavenumber (cm$^{-1}$)")
    ax_plane.set_ylabel("line index (sorted by $\\nu_0$)")
    ax_plane.set_title(f"(a) {out['lines']:,} lines $\\times$ {out['samples']:,} samples", loc="left")
    ax_plane.grid(False)
    ax_plane.text(0.03, 0.97, f"core list: {out['core_pairs']:,} pairs\n= {100 * out['core_fraction']:.3f}%",
                  transform=ax_plane.transAxes, va="top", fontsize=6.5, color=VERMILLION,
                  bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    # (b) zoom: which branch each pair takes at the bottom layer
    gsel = np.flatnonzero((grid >= ZOOM_CM1[0]) & (grid <= ZOOM_CM1[1]))
    lsel = zoom_lines
    in_list = np.zeros((lsel.size, gsel.size), bool)
    lookup = {(int(l), int(g)) for l, g in zip(core_line, core_grid)
              if lsel[0] <= l <= lsel[-1] and gsel[0] <= g <= gsel[-1]}
    for i, l in enumerate(lsel):
        for j, g in enumerate(gsel):
            in_list[i, j] = (int(l), int(g)) in lookup
    x = (grid[gsel][None, :] - nu0[lsel][:, None]) * scale[lsel][:, None]
    takes_core = x * x + a[lsel][:, None] ** 2 < 111.0
    category = np.where(takes_core & in_list, 2, np.where(in_list, 1, 0))
    assert not np.any(takes_core & ~in_list), "a core pair outside the static list"
    cmap = ListedColormap(["#f2f2f2", "#F5C9A6", VERMILLION])
    ax_zoom.pcolormesh(grid[gsel], np.arange(lsel.size), category, cmap=cmap, vmin=-0.5, vmax=2.5,
                       shading="nearest", rasterized=True)
    ax_zoom.set_yticks(np.arange(lsel.size))
    ax_zoom.set_yticklabels([f"{v:.3f}" for v in nu0[lsel]], fontsize=5.5)
    ax_zoom.xaxis.get_major_formatter().set_useOffset(False)
    ax_zoom.set_xticks(np.arange(5049.9, 5050.25, 0.1))
    ax_zoom.set_ylabel("line centre $\\nu_0$ (cm$^{-1}$)")
    ax_zoom.set_xlabel("grid wavenumber (cm$^{-1}$)")
    ax_zoom.grid(False)
    ax_zoom.set_title("(b) zoom, bottom layer ($T=282$ K, $P=0.77$ bar)", loc="left")
    from matplotlib.patches import Patch
    ax_zoom.legend(handles=[Patch(color=VERMILLION, label="$x^2+a^2<111$: Algorithm 916"),
                            Patch(color="#F5C9A6", label="in core list, asymptotic at this layer"),
                            Patch(color="#f2f2f2", label="wing: asymptotic series only")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6.3)

    # (c) cost: what is evaluated, and measured time (log axis, so not stacked)
    groups = [
        ("ExoJAX Direct", out["time_direct"]["median_ms"],
         [("alg916_all_pairs", "Algorithm 916, all pairs", VERMILLION),
          ("asymptotic_all_pairs_f64", "asymptotic, all pairs, f64", SKY)]),
        ("SparseCoreDirect", out["time_sparse"]["median_ms"],
         [("hjert_core_list", "hjert, core list only", ORANGE),
          ("asymptotic_all_pairs_f64", None, SKY)]),
        ("+ mixed precision", out["time_sparse_mixed"]["median_ms"],
         [("hjert_core_list", None, ORANGE),
          ("asymptotic_all_pairs_f32", "asymptotic, all pairs, f32", BLUE)]),
    ]
    ticks = []
    for g, (label, total, parts) in enumerate(groups):
        base = g * 4.0
        ax_cost.barh(base, total, height=0.8, color=GREY, label="whole calculator" if g == 0 else None)
        ax_cost.text(total * 1.08, base, f"{total:.1f} ms", va="center", fontsize=6.5)
        for k, (key, part_label, color) in enumerate(parts):
            width = out[f"component_{key}"]["median_ms"]
            ax_cost.barh(base + 1 + k, width, height=0.8, color=color, label=part_label)
            ax_cost.text(width * 1.08, base + 1 + k, f"{width:.1f}", va="center", fontsize=6)
        ticks.append(base + 1)
    ax_cost.set_yticks(ticks)
    ax_cost.set_yticklabels([g[0] for g in groups])
    ax_cost.tick_params(axis="y", length=0)
    ax_cost.invert_yaxis()
    ax_cost.set_xscale("log")
    ax_cost.set_xlim(1, 5e3)
    ax_cost.set_xlabel(f"GPU time per call, {out['layers']} layers (ms)")
    ax_cost.set_title("(c) forward cost: whole calculator and its branches", loc="left")
    ax_cost.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=6.3)

    # (d) memory held and compile time
    held = out["held_bytes_fit_workload"]
    before = (held["dense_offsets"] + held["dense_mask"]) / 1e6
    after = held["sparse_vectors"] / 1e6
    ax_held.bar([0], [held["dense_offsets"] / 1e6], color=GREY, width=0.6, label="dense offsets (f64)")
    ax_held.bar([0], [held["dense_mask"] / 1e6], bottom=[held["dense_offsets"] / 1e6], color="#aaaaaa",
                width=0.6, label="core mask (bool)")
    ax_held.bar([1], [after], color=BLUE, width=0.6, label="1-D vectors + core list")
    ax_held.set_yscale("log")
    ax_held.set_ylim(0.5, 2e3)
    ax_held.set_xticks([0, 1])
    ax_held.set_xticklabels(["(line, grid)\nmatrices", "recomputed\nin kernel"])
    ax_held.set_xlabel("5005–5025 cm$^{-1}$, H$_2$O + CO$_2$", fontsize=6.5)
    ax_held.set_ylabel("held on device (MB)")
    ax_held.text(0, before * 1.15, f"{before:.0f} MB", ha="center", fontsize=6.5)
    ax_held.text(1, after * 1.15, f"{after:.2f} MB", ha="center", fontsize=6.5)
    ax_held.set_title("(d) memory held", loc="left")
    ax_held.text(0.98, 0.97, "compile + first call:\n"
                 f"Direct {out['time_direct']['compile_and_first_s']:.1f} s, "
                 f"sparse {out['time_sparse']['compile_and_first_s']:.1f} s\n(5000–5100 cm$^{{-1}}$)",
                 transform=ax_held.transAxes, ha="right", va="top", fontsize=6.3)
    ax_held.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False, fontsize=6.3)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT / f"fig_method_sparse_core.{suffix}", dpi=200)
    print(f"wrote {OUTPUT / 'fig_method_sparse_core.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if args.recompute or not CACHE.with_suffix(".json").exists():
        out = measure()
    else:
        out = json.loads(CACHE.with_suffix(".json").read_text())
    print(json.dumps(out, indent=1))
    plot(out)


if __name__ == "__main__":
    main()
