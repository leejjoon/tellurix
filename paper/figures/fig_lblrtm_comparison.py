#!/usr/bin/env python
"""Paper figures: tellurix against LBLRTM 12.17 given identical layers.

Rebuilds, for plotting, what ``scripts/compare_lblrtm_layers.py`` measures
(``docs/lblrtm_identical_layers.json`` and ``..._k_orders.json``) and what
``scripts/validate_aer_co.py`` measures (``aer_co_validation.json``). LBLRTM is
not rerun: its TAPE12s are read from the run directories those scripts left
under ``data/lblrtm/``. Only the tellurix side is computed, once, and cached in
``paper/figures/cache/`` (``--recompute`` redoes it). The fit -- H2O and CO2
column scales and a linear continuum against LBLRTM at R=45,000 -- is
reproduced line for line, so the annotated medians and 99th percentiles are
this script's own numbers, checked against the committed reports.

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_lblrtm_comparison.py

Writes PDF and PNG (200 dpi) to ``paper/figures/output/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper/figures/cache"
OUTPUT = ROOT / "paper/figures/output"
REFERENCE = ROOT / "data/lblrtm"

RESOLVING_POWER = 45_000.0
SAMPLES_PER_RESOLUTION = 4.0
LINE_MARGIN_CM1 = 25.0
# Half-width of the line-level zoom; the zoom grid is ~5x finer than LBLRTM's
# own DV (0.0014 cm-1) so interpolating tellurix onto LBLRTM's samples is exact
# to well below the differences shown.
ZOOM_HALF_WIDTH_CM1 = 0.6
ZOOM_SAMPLES_PER_RESOLUTION = 400.0

# Window and LBLRTM run directory of every case; the K-order windows are the
# ones recorded in docs/lblrtm_identical_layers_k_orders.json.
CASES = {
    "W5000": ((5000.0, 5020.0), "run_identical_layers"),
    "K86": ((4794.2, 4861.0), "run_identical_layers_K86"),
    "K89": ((4958.1, 5027.1), "run_identical_layers_K89"),
    "K92": ((5121.6, 5192.9), "run_identical_layers_K92"),
}

LBLRTM_COLOR = "black"
COUPLED_COLOR = "#0072B2"     # Okabe-Ito blue
UNCOUPLED_COLOR = "#D55E00"   # Okabe-Ito vermillion


# --------------------------------------------------------------------------
# Numerics shared with scripts/compare_lblrtm_layers.py (copied, not imported).

def convolve(transmission: np.ndarray) -> np.ndarray:
    sigma = SAMPLES_PER_RESOLUTION / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    offsets = np.arange(-int(6 * sigma) - 1, int(6 * sigma) + 2)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return np.convolve(transmission, kernel / kernel.sum(), "same")


def tau_on_grid(wavenumber, transmission, nu):
    transmission = np.clip(np.asarray(transmission, float), np.finfo(np.float32).tiny, None)
    return np.interp(nu, np.asarray(wavenumber), -np.log(transmission))


def fit(tau: dict, extra: np.ndarray, target_tau: np.ndarray, nu: np.ndarray, window) -> dict:
    """The fit of compare_lblrtm_layers.compare: H2O, CO2 scales and a linear continuum."""

    from scipy.optimize import least_squares

    inner = (nu >= window[0]) & (nu <= window[1])
    reference = convolve(np.exp(-target_tau))
    keep = inner & (reference > 0.05)
    others = sum(t for s, t in tau.items() if s not in ("H2O", "CO2")) + extra
    x = (nu - 0.5 * sum(window)) / (0.5 * (window[1] - window[0]))

    def model(p):
        total = np.exp(p[0]) * tau["H2O"] + np.exp(p[1]) * tau["CO2"] + others
        return convolve(np.exp(-total)) * (1.0 + p[2] + p[3] * x)

    p = least_squares(lambda p: (model(p) - reference)[keep], np.zeros(4)).x
    residual = model(p) - reference
    error = np.abs(residual[keep])
    return {"p": p, "model": model(p), "reference": reference, "residual": residual,
            "keep": keep, "inner": inner, "median": float(np.median(error)),
            "p99": float(np.percentile(error, 99)), "max": float(error.max()),
            "h2o_scale": float(np.exp(p[0])), "co2_scale": float(np.exp(p[1]))}


def monochromatic(tau: dict, extra, p, nu, window):
    """tellurix's unconvolved transmission with a fit's scales and continuum applied."""

    others = sum(t for s, t in tau.items() if s not in ("H2O", "CO2")) + extra
    x = (nu - 0.5 * sum(window)) / (0.5 * (window[1] - window[0]))
    total = np.exp(p[0]) * tau["H2O"] + np.exp(p[1]) * tau["CO2"] + others
    return np.exp(-total) * (1.0 + p[2] + p[3] * x)


def worst_centre(result, nu, window) -> float:
    away = (nu > window[0] + ZOOM_HALF_WIDTH_CM1) & (nu < window[1] - ZOOM_HALF_WIDTH_CM1)
    score = np.where(result["keep"] & away, np.abs(result["residual"]), -1.0)
    return float(nu[np.argmax(score)])


# --------------------------------------------------------------------------
# The tellurix side, computed once and cached.

def compute_case(name: str) -> None:
    import tellurix  # noqa: F401  (x64 before exojax)
    import jax.numpy as jnp
    from tellurix import (
        AER_MOLECULE_IDS, AERLineDatabase, DataPaths, ExoJAXOpacityBackend, MTCKDWaterContinuum,
        TelluricModel, TelluricParameters, constant_velocity_grid, load_atmosphere_csv,
    )
    from tellurix.reference import read_tape12_single_precision

    window, run_dir = CASES[name]
    paths = DataPaths.bootstrapped(ROOT)
    source = load_atmosphere_csv(ROOT / "data/profiles/example_midlatitude.csv")
    profile = dataclasses.replace(source, mean_pressure_bar=source.continuum_pressure_bar)
    nu = constant_velocity_grid(1.0e7 / window[1], 1.0e7 / window[0],
                                resolving_power=RESOLVING_POWER,
                                samples_per_resolution=SAMPLES_PER_RESOLUTION,
                                margin_cm1=LINE_MARGIN_CM1)
    lblrtm = read_tape12_single_precision(REFERENCE / run_dir / "continua/TAPE12")
    lblrtm_tau = tau_on_grid(lblrtm.wavenumber_cm1, lblrtm.transmission, nu)

    databases = {}
    for species in profile.vmr:
        try:
            databases[species] = AERLineDatabase(paths.line_file(species, AER_MOLECULE_IDS[species]),
                                                 species, (float(nu[0]), float(nu[-1])),
                                                 margin_cm1=LINE_MARGIN_CM1,
                                                 line_coupling=paths.line_coupling)
        except ValueError as exc:
            if "lines found" not in str(exc):
                raise

    def tellurix(grid, line_coupling):
        backend = ExoJAXOpacityBackend.prepare(
            databases, grid, methods="direct_sparse",
            temperature_range_k=(float(profile.temperature_k.min()), float(profile.temperature_k.max())),
            maximum_pressure_bar=float(profile.pressure_layer_bar.max()),
            vectorize_layers=False, pressure_shift=True, line_coupling=line_coupling)
        model = TelluricModel(profile, grid, backend,
                              continuum=MTCKDWaterContinuum.from_netcdf(paths.mt_ckd, grid))
        parameters = TelluricParameters({s: 0.0 for s in model.species}, 0.0, 0.0, 3.0,
                                        jnp.asarray([0.0]), np.log(1.0e-5))
        taus = {k: -np.log(np.asarray(v)) for k, v in model.species_transmission(parameters).items()}
        mt_ckd = taus.pop("continuum")
        others = sum(t for s, t in taus.items() if s not in ("H2O", "CO2"))
        return {"H2O": taus["H2O"], "CO2": taus["CO2"], "others": others}, mt_ckd

    uncoupled, mt_ckd = tellurix(nu, False)
    coupled, _ = tellurix(nu, True)
    print(f"{name}: tellurix on the R=45,000 grid done", flush=True)

    # The zoom goes where the residual is worst: without coupling in the
    # 5000-5020 window (what coupling fixes), with it in the K orders (what is
    # left after it).
    chosen = coupled if name.startswith("K") else uncoupled
    result = fit({"H2O": chosen["H2O"], "CO2": chosen["CO2"], "others": chosen["others"]},
                 mt_ckd, lblrtm_tau, nu, window)
    centre = worst_centre(result, nu, window)
    fine = constant_velocity_grid(1.0e7 / (centre + ZOOM_HALF_WIDTH_CM1),
                                  1.0e7 / (centre - ZOOM_HALF_WIDTH_CM1),
                                  resolving_power=RESOLVING_POWER,
                                  samples_per_resolution=ZOOM_SAMPLES_PER_RESOLUTION, margin_cm1=0.0)
    fine_uncoupled, fine_mt_ckd = tellurix(fine, False)
    fine_coupled, _ = tellurix(fine, True)
    print(f"{name}: zoom at {centre:.3f} cm-1 done", flush=True)

    native = (lblrtm.wavenumber_cm1 >= window[0] - 1.0) & (lblrtm.wavenumber_cm1 <= window[1] + 1.0)
    arrays = {"window": np.asarray(window), "nu": nu, "lblrtm_tau": lblrtm_tau, "mt_ckd": mt_ckd,
              "zoom_centre": np.asarray(centre), "fine_nu": fine, "fine_mt_ckd": fine_mt_ckd,
              "lblrtm_native_nu": lblrtm.wavenumber_cm1[native],
              "lblrtm_native_transmission": lblrtm.transmission[native]}
    for label, pieces in (("uncoupled", uncoupled), ("coupled", coupled),
                          ("fine_uncoupled", fine_uncoupled), ("fine_coupled", fine_coupled)):
        for species, tau in pieces.items():
            arrays[f"{label}_{species}"] = np.asarray(tau)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"lblrtm_{name}.npz", **arrays)


def compute_co() -> None:
    """The CO-only rung, exactly as scripts/validate_aer_co.py runs it."""

    import tellurix  # noqa: F401
    from tellurix import (AERLineDatabase, AtmosphereProfile, ExoJAXOpacityBackend, TelluricModel,
                          TelluricParameters, constant_velocity_grid, load_atmosphere_csv)
    from tellurix.reference import read_tape12_single_precision

    source = load_atmosphere_csv(ROOT / "data/profiles/example_midlatitude.csv")
    profile = AtmosphereProfile(source.pressure_edges_bar, source.temperature_k, source.altitude_km,
                                {"CO": source.vmr["CO"]})
    limits = (4290.0, 4310.0)
    lblrtm = read_tape12_single_precision(REFERENCE / "run_compare_co/TAPE12")
    nu = constant_velocity_grid(1.0e7 / limits[1], 1.0e7 / limits[0],
                                resolving_power=120_000.0, samples_per_resolution=4.0)
    database = AERLineDatabase(REFERENCE / "AER_Line_File/aer_v_3.9/line_files_By_Molecule/05_CO/05_CO",
                               "CO", limits)
    model = TelluricModel(profile, nu, ExoJAXOpacityBackend.prepare({"CO": database}, nu, methods="direct"))
    parameters = TelluricParameters({"CO": 0.0}, 0.0, 0.0, 2.8, np.asarray([0.0]), np.log(1.0e-5))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "lblrtm_CO.npz", nu=nu,
                        transmission=np.asarray(model.transmission(parameters)),
                        lblrtm_nu=lblrtm.wavenumber_cm1, lblrtm_transmission=lblrtm.transmission)
    print("CO done", flush=True)


# --------------------------------------------------------------------------
# Plotting.

def load(name: str) -> dict:
    data = dict(np.load(CACHE / f"lblrtm_{name}.npz"))
    for label in ("uncoupled", "coupled", "fine_uncoupled", "fine_coupled"):
        data[label] = {s: data[f"{label}_{s}"] for s in ("H2O", "CO2", "others")}
    data["window"] = tuple(data["window"])
    data["fits"] = {label: fit(data[label], data["mt_ckd"], data["lblrtm_tau"], data["nu"], data["window"])
                    for label in ("uncoupled", "coupled")}
    return data


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": False, "ytick.right": True, "lines.linewidth": 0.7,
        "pdf.fonttype": 42, "savefig.dpi": 200, "legend.frameon": False,
    })


def top_axes(ax, nu_window, nu_grid, index_label):
    """Wavelength on the top spine and, outside it, the grid's sample index."""

    sec = ax.secondary_xaxis("top", functions=(lambda v: 1.0e4 / np.asarray(v, float),
                                               lambda w: 1.0e4 / np.asarray(w, float)))
    sec.set_xlabel(r"Wavelength ($\mu$m)", labelpad=2)
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useOffset=False)
    sec.xaxis.set_major_formatter(formatter)
    sec.tick_params(direction="in", pad=1.5)
    origin = nu_window[0]
    # Index of the constant-velocity grid counted from the first sample at or
    # above the window's lower edge.
    first = float(nu_grid[np.searchsorted(nu_grid, origin)])
    dlog = float(np.log(nu_grid[1] / nu_grid[0]))
    idx = ax.secondary_xaxis("top", functions=(lambda v: np.log(np.asarray(v, float) / first) / dlog,
                                               lambda i: first * np.exp(np.asarray(i, float) * dlog)))
    idx.spines["top"].set_position(("outward", 24))
    idx.set_xlabel(index_label, labelpad=2)
    idx.tick_params(direction="out", pad=1.0, labelsize=7)
    return sec, idx


def save(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.pdf")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=200)
    print("wrote", OUTPUT / f"{stem}.png")


def pct(value):
    return f"{100 * value:.3f}%" if value < 1e-3 else f"{100 * value:.2f}%"


def figure_window(data, stem, index_label, legend_loc):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    nu, window = data["nu"], data["window"]
    fits = data["fits"]
    inner = fits["coupled"]["inner"]
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0, 1.0], "hspace": 0.06})
    ax = axes[0]
    ax.plot(nu[inner], fits["coupled"]["reference"][inner], color=LBLRTM_COLOR, lw=1.1,
            label="LBLRTM 12.17")
    ax.plot(nu[inner], fits["coupled"]["model"][inner], color=COUPLED_COLOR, lw=0.7, ls="--",
            label="tellurix, line coupling")
    ax.set_ylabel("Transmission\n" + r"($R=45{,}000$, fitted)")
    ax.legend(loc=legend_loc, ncol=1, handlelength=2.2)
    ax.set_xlim(*window)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    top_axes(ax, window, nu, index_label)

    limit = 0.0
    for label in ("uncoupled", "coupled"):
        r = fits[label]["residual"]
        limit = max(limit, np.abs(r[fits[label]["keep"]]).max())
    limit = 1.15 * 100 * limit
    for ax, label, color, text in ((axes[1], "uncoupled", UNCOUPLED_COLOR, "no line coupling"),
                                   (axes[2], "coupled", COUPLED_COLOR, "line coupling")):
        f = fits[label]
        r = np.where(f["keep"], f["residual"], np.nan) * 100
        ax.axhspan(-100 * f["p99"], 100 * f["p99"], color=color, alpha=0.15, lw=0)
        ax.axhline(0.0, color="0.5", lw=0.5)
        ax.plot(nu[inner], r[inner], color=color, lw=0.7)
        ax.set_ylim(-limit, 1.6 * limit)
        ax.set_ylabel(r"$\Delta T$ (%)")
        box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0}
        ax.text(0.005, 0.95, f"tellurix $-$ LBLRTM, {text}", transform=ax.transAxes, va="top",
                ha="left", fontsize=7.5, bbox=box)
        ax.text(0.995, 0.95, f"median {pct(f['median'])}, 99th percentile {pct(f['p99'])} (band)",
                transform=ax.transAxes, va="top", ha="right", fontsize=7.5, bbox=box)
    axes[-1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.08)
    save(fig, stem)
    plt.close(fig)


def figure_zoom(data, stem, index_label):
    """Monochromatic and R=45,000 agreement around the worst residual."""

    import matplotlib.pyplot as plt

    nu, window, fits = data["nu"], data["window"], data["fits"]
    centre = float(data["zoom_centre"])
    lo, hi = centre - ZOOM_HALF_WIDTH_CM1, centre + ZOOM_HALF_WIDTH_CM1
    fine = data["fine_nu"]
    lnu, lt = data["lblrtm_native_nu"], data["lblrtm_native_transmission"]
    sel = (lnu >= lo) & (lnu <= hi)
    lnu, lt = lnu[sel], lt[sel]

    mono = {label: monochromatic(data[f"fine_{label}"], data["fine_mt_ckd"], fits[label]["p"], fine, window)
            for label in ("uncoupled", "coupled")}

    fig, axes = plt.subplots(4, 1, figsize=(7.1, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.8, 1.0, 1.3, 1.0], "hspace": 0.07})
    ax = axes[0]
    ax.plot(lnu, lt, color=LBLRTM_COLOR, lw=1.3, label="LBLRTM 12.17")
    ax.plot(fine, mono["uncoupled"], color=UNCOUPLED_COLOR, lw=0.8, ls=(0, (4, 2)),
            label="tellurix, no coupling")
    ax.plot(fine, mono["coupled"], color=COUPLED_COLOR, lw=0.8, ls=(0, (1.5, 1.2)),
            label="tellurix, line coupling")
    ax.set_ylabel("Transmission\n(monochromatic)")
    ax.legend(loc="lower left", ncol=1, handlelength=2.6)
    ax.set_xlim(lo, hi)
    top_axes(ax, window, nu, index_label)

    ax = axes[1]
    ax.axhline(0.0, color="0.5", lw=0.5)
    for label, color in (("uncoupled", UNCOUPLED_COLOR), ("coupled", COUPLED_COLOR)):
        diff = (np.interp(lnu, fine, mono[label]) - lt) * 100
        ax.plot(lnu, diff, color=color, lw=0.8)
    ax.set_ylabel(r"$\Delta T$ (%)" + "\nmonochr.")
    ax.text(0.005, 0.93, "tellurix $-$ LBLRTM, monochromatic (LBLRTM samples)", transform=ax.transAxes,
            va="top", fontsize=7.5)

    window_sel = (nu >= lo) & (nu <= hi)
    ax = axes[2]
    ax.plot(nu[window_sel], fits["coupled"]["reference"][window_sel], color=LBLRTM_COLOR, lw=1.3,
            marker="o", ms=2.2, label="LBLRTM")
    ax.plot(nu[window_sel], fits["uncoupled"]["model"][window_sel], color=UNCOUPLED_COLOR, lw=0.8,
            ls=(0, (4, 2)), marker="s", ms=1.8)
    ax.plot(nu[window_sel], fits["coupled"]["model"][window_sel], color=COUPLED_COLOR, lw=0.8,
            ls=(0, (1.5, 1.2)), marker="^", ms=1.8)
    ax.set_ylabel("Transmission\n" + r"($R=45{,}000$)")

    ax = axes[3]
    ax.axhline(0.0, color="0.5", lw=0.5)
    for label, color, marker in (("uncoupled", UNCOUPLED_COLOR, "s"), ("coupled", COUPLED_COLOR, "^")):
        f = fits[label]
        ax.axhspan(-100 * f["p99"], 100 * f["p99"], color=color, alpha=0.12, lw=0)
        ax.plot(nu[window_sel], 100 * f["residual"][window_sel], color=color, lw=0.8, marker=marker, ms=2.0)
    ax.set_ylabel(r"$\Delta T$ (%)" + "\n" + r"$R=45{,}000$")
    ax.text(0.005, 0.93, "tellurix $-$ LBLRTM, fitted, one marker per grid sample; bands: window p99",
            transform=ax.transAxes, va="top", fontsize=7.5)
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    for a in axes[1:]:
        a.yaxis.set_major_locator(plt.MaxNLocator(4))
    for a in (axes[1], axes[3]):
        top = max(abs(v) for v in a.get_ylim())
        a.set_ylim(-1.25 * top, 1.6 * top)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.07)
    save(fig, stem)
    plt.close(fig)


def figure_co(stem):
    import matplotlib.pyplot as plt
    from tellurix.reference import LBLRTMSpectrum, compare_transmission, degrade_to_resolving_power

    d = np.load(CACHE / "lblrtm_CO.npz")
    ref = degrade_to_resolving_power(LBLRTMSpectrum(d["lblrtm_nu"], d["lblrtm_transmission"]))
    ours = degrade_to_resolving_power(LBLRTMSpectrum(d["nu"], d["transmission"]))
    m = compare_transmission(ref, ours)
    rnu = np.asarray(ref.wavenumber_cm1)
    model = np.interp(rnu, ours.wavenumber_cm1, ours.transmission)
    window = (4290.0, 4310.0)
    inside = (rnu >= window[0]) & (rnu <= window[1])
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.07})
    axes[0].plot(rnu[inside], ref.transmission[inside], color=LBLRTM_COLOR, lw=1.1, label="LBLRTM")
    axes[0].plot(rnu[inside], model[inside], color=COUPLED_COLOR, lw=0.7, ls="--", label="tellurix")
    axes[0].set_ylabel("Transmission\n(CO only, $R=45{,}000$)")
    axes[0].legend(loc="lower left")
    axes[0].set_xlim(*window)
    top_axes(axes[0], window, rnu, "Sample index ($R=45{,}000$, 3 per res. el.)")
    keep = inside & (np.asarray(ref.transmission) > 0.05)
    axes[1].axhspan(-100 * m.percentile_99_absolute_error, 100 * m.percentile_99_absolute_error,
                    color=COUPLED_COLOR, alpha=0.15, lw=0)
    axes[1].axhline(0, color="0.5", lw=0.5)
    axes[1].plot(rnu[inside], 100 * np.where(keep, model - ref.transmission, np.nan)[inside],
                 color=COUPLED_COLOR, lw=0.7)
    axes[1].set_ylabel(r"$\Delta T$ (%)")
    axes[1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    top = 100 * m.percentile_99_absolute_error
    axes[1].set_ylim(-1.35 * top, 1.9 * top)
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(5))
    axes[1].text(0.97, 0.94, f"median {pct(m.median_absolute_error)}\np99 {pct(m.percentile_99_absolute_error)} (band)",
                 transform=axes[1].transAxes, fontsize=7, ha="right", va="top")
    fig.subplots_adjust(left=0.20, right=0.96, top=0.80, bottom=0.12)
    save(fig, stem)
    plt.close(fig)
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    for name in CASES:
        if args.recompute or not (CACHE / f"lblrtm_{name}.npz").exists():
            compute_case(name)
    if args.recompute or not (CACHE / "lblrtm_CO.npz").exists():
        compute_co()

    style()
    committed = json.loads((ROOT / "docs/lblrtm_identical_layers.json").read_text())
    k_orders = json.loads((ROOT / "docs/lblrtm_identical_layers_k_orders.json").read_text())["orders"]
    summary = {}
    datasets = {name: load(name) for name in CASES}
    for name, data in datasets.items():
        report = committed if name == "W5000" else k_orders[name]
        ft = report["co2_attribution"]["fitted_transmission"]
        for label, key in (("uncoupled", "as_now"), ("coupled", "with_line_coupling")):
            f = data["fits"][label]
            ref = ft[key]["fitted_R45000"]
            summary[f"{name} {label}"] = {
                "median": f["median"], "p99": f["p99"], "max": f["max"],
                "committed_median": ref["median"], "committed_p99": ref["p99"],
                "h2o_scale": f["h2o_scale"], "co2_scale": f["co2_scale"],
            }
            print(f"{name:6s} {label:10s} median {100*f['median']:.4f}% (committed {100*ref['median']:.4f}%) "
                  f"p99 {100*f['p99']:.4f}% (committed {100*ref['p99']:.4f}%) max {100*f['max']:.4f}%")

    index_label = r"Sample index ($R=45{,}000$ grid, 4 per res. el.)"
    w = datasets["W5000"]
    figure_window(w, "fig_lblrtm_5000", index_label, "lower right")
    figure_zoom(w, "fig_lblrtm_5000_zoom", index_label)
    worst = max(("K86", "K89", "K92"), key=lambda k: datasets[k]["fits"]["coupled"]["p99"])
    print("worst K order with coupling:", worst)
    figure_window(datasets[worst], f"fig_lblrtm_{worst}", index_label, "upper right")
    figure_zoom(datasets[worst], f"fig_lblrtm_{worst}_zoom", index_label)
    m = figure_co("fig_lblrtm_co")
    print(f"CO: median {100*m.median_absolute_error:.4f}% p99 {100*m.percentile_99_absolute_error:.4f}%")
    summary["CO"] = dataclasses.asdict(m)
    summary["zoom_centres_cm1"] = {k: float(d["zoom_centre"]) for k, d in datasets.items()}
    (OUTPUT / "fig_lblrtm_numbers.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
