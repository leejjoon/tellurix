#!/usr/bin/env python
"""Paper figures: what each stage costs in LBLRTM 12.17 and in tellurix.

Reads ``docs/lblrtm_tellurix_performance.json``, written by
``benchmarks/benchmark_lblrtm_stages.py --part merge``; nothing is recomputed.

    UV_CACHE_DIR=.uv-cache uv run python paper/figures/fig_performance.py

Writes PDF and PNG (200 dpi) to ``paper/figures/output/``:

  fig_performance_stages   stage by stage, LBLRTM against tellurix on GPU and CPU
  fig_performance_scaling  forward model and gradient against window width
  fig_performance_gradient gradient cost against the number of free parameters
  fig_performance_fit      the end-to-end fit: wall clock and recovered columns
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/lblrtm_tellurix_performance.json"
OUTPUT = ROOT / "paper/figures/output"
WIDTH_IN = 7.1

# Okabe-Ito: LBLRTM is the neutral reference, the two tellurix devices the
# blue/vermillion pair; hatching repeats the identity for print and CVD.
LBLRTM = "#555555"
GPU = "#0072B2"
CPU = "#D55E00"
SERIES = (("lblrtm", "LBLRTM 12.17 (1 CPU core)", LBLRTM, "//"),
          ("tellurix_gpu", "tellurix, 1 GPU", GPU, ""),
          ("tellurix_cpu", "tellurix, CPU", CPU, ".."))

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42,
})


def save(figure, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUTPUT / name}.png")


def stage_rows(entry):
    """(label, LBLRTM warm, LBLRTM cold, {device: (warm, cold)}) per comparable stage."""

    lbl = entry["lblrtm"]
    tape6 = lbl["L1_lblrtm_run"]["tape6_median"]
    lines = sum(m["time_s"] for m in tape6["modules"].values())
    other = tape6["opdpth_other_incl_contnm_s"] + tape6["layer_merge_s"] + tape6["emission_init_s"]
    run = lbl["L1_lblrtm_run"]
    post = lbl["L2_postprocess"]["total_median_s"]
    rows = [
        ("Line data preparation\n(LNFL | AER read + core lists)", lbl["L0_lnfl"]["uncoupled"]["median_s"], None,
         lambda t: (t["T0_line_read"]["cached"]["median_s"] + t["T1_prepare"]["median_s"],
                    t["T0_line_read"]["cold_index_build_s"] + t["T1_prepare"]["first_call_s"])),
        ("Line-by-line optical depth", lines, None,
         lambda t: (t["T2_opacity_kernel"]["median_s"], t["T2_opacity_kernel"]["first_call_s"])),
        ("Continuum + layers\n-> slant transmission", other, None,
         lambda t: (t["T3_slant_transmission_given_opacity"]["median_s"],
                    t["T3_slant_transmission_given_opacity"]["first_call_s"])),
        ("Instrument: LSF, pixels,\ncontinuum", post, None,
         lambda t: (t["T4_instrument"]["median_s"], t["T4_instrument"]["first_call_s"])),
        ("Forward model", run["median_s"] + post, run["first_call_s"] + post,
         lambda t: (t["T5_forward_live"]["median_s"], t["T5_forward_live"]["first_call_s"])),
        ("Forward, precomputed\nopacity", None, None,
         lambda t: (t["T5_forward_precomputed"]["median_s"], t["T5_forward_precomputed"]["first_call_s"])),
        ("Gradient (central FD |\nautodiff value+grad)", lbl["A2_fd_gradient"]["central"]["wall_s"], None,
         lambda t: (t["T6_value_and_grad_live"]["median_s"], t["T6_value_and_grad_live"]["first_call_s"])),
        ("Gradient, precomputed\nopacity", None, None,
         # Its compile happened inside the cold fit, which built this objective.
         lambda t: (t["T7_value_and_grad_precomputed"]["median_s"],
                    t["E1_fit_cold"]["objective_compile_and_first_s"])),
    ]
    out = []
    for label, lbl_warm, lbl_cold, ours in rows:
        values = {"lblrtm": (lbl_warm, lbl_cold)}
        for key in ("tellurix_gpu", "tellurix_cpu"):
            if key in entry:
                values[key] = ours(entry[key])
        out.append((label, values))
    return out


def figure_stages(report):
    names = list(report["workloads"])
    figure, axes = plt.subplots(1, len(names), figsize=(WIDTH_IN, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    height = 0.26
    for axis, name in zip(axes, names):
        entry = report["workloads"][name]
        rows = stage_rows(entry)
        for row_index, (label, values) in enumerate(rows):
            for offset, (key, _, color, hatch) in enumerate(SERIES):
                warm, cold = values.get(key, (None, None))
                y = row_index + (offset - 1) * height
                if warm:
                    axis.barh(y, warm, height=height * 0.92, color=color, hatch=hatch, edgecolor="white",
                              linewidth=0.0, left=1e-5)
                if cold and warm and cold > 1.05 * warm:
                    axis.plot([warm, cold], [y, y], color=color, lw=0.8, alpha=0.7)
                    axis.plot(cold, y, marker="|", color=color, ms=5, mew=1.2)
        axis.set_xscale("log")
        axis.set_xlim(1e-4, 100)
        axis.set_title(f"{name}: {entry['label']}\n{entry['lblrtm']['pixels']} px, "
                       f"{entry['tellurix_gpu']['grid_samples'] if 'tellurix_gpu' in entry else '?'} grid, "
                       f"{entry['lblrtm']['layers']} layers")
        axis.set_xlabel("time per call (s)")
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for label, _ in rows])
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, hatch=h, ec="white") for _, _, c, h in SERIES]
    handles.append(plt.Line2D([], [], color="#333333", marker="|", lw=0.8, ms=5))
    figure.legend(handles, [label for _, label, _, _ in SERIES] + ["first call incl. XLA compile"],
                  loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.55, -0.04))
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save(figure, "fig_performance_stages")


def width_series(report):
    names = sorted(report["workloads"], key=lambda n: np.diff(report["workloads"][n]["window_cm1"])[0])
    widths = [float(np.diff(report["workloads"][n]["window_cm1"])[0]) for n in names]
    return names, np.asarray(widths)


def figure_scaling(report):
    names, widths = width_series(report)
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.8), sharey=True)
    entries = [report["workloads"][n] for n in names]

    def get(key, path):
        values = []
        for e in entries:
            value = e.get(key)
            for p in path:
                value = None if value is None else value.get(p)
            values.append(np.nan if value is None else value)
        return np.asarray(values, float)

    lbl_forward = np.asarray([e["lblrtm"]["L1_lblrtm_run"]["median_s"] + e["lblrtm"]["L2_postprocess"]["total_median_s"]
                              for e in entries])
    lbl_gradient = np.asarray([e["lblrtm"]["A2_fd_gradient"]["central"]["wall_s"] for e in entries])
    panels = (
        (axes[0], "Forward model", lbl_forward, "LBLRTM run + post-process",
         ("T5_forward_live", "T5_forward_precomputed")),
        (axes[1], "Gradient of $\\chi^2$", lbl_gradient, "LBLRTM central FD (2P+1 runs)",
         ("T6_value_and_grad_live", "T7_value_and_grad_precomputed")),
    )
    for axis, title, lbl_values, lbl_label, (live, pre) in panels:
        axis.plot(widths, lbl_values, color=LBLRTM, marker="s", ms=5, lw=1.5, label=lbl_label)
        for key, color, device in (("tellurix_gpu", GPU, "GPU"), ("tellurix_cpu", CPU, "CPU")):
            axis.plot(widths, get(key, (live, "median_s")), color=color, marker="o", ms=5, lw=1.5,
                      label=f"tellurix {device}, live opacity")
            axis.plot(widths, get(key, (pre, "median_s")), color=color, marker="o", ms=5, lw=1.2, ls="--",
                      mfc="white", label=f"tellurix {device}, precomputed")
        axis.set_yscale("log")
        axis.set_xlabel("window width (cm$^{-1}$)")
        axis.set_title(title, pad=12)
        for name, width in zip(names, widths):
            axis.text(width, 1.01, name, transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                      fontsize=6.5, color="#666666")
    axes[0].set_ylabel("time per evaluation (s)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    figure.tight_layout()
    save(figure, "fig_performance_scaling")


def figure_gradient(report):
    names = list(report["workloads"])
    name = "W3" if "W3" in names else names[-1]
    entry = report["workloads"][name]
    lbl = entry["lblrtm"]
    per_run = lbl["A2_fd_gradient"]["per_evaluation_s"]
    figure, axis = plt.subplots(figsize=(WIDTH_IN * 0.78, 3.0))
    total = len(entry["tellurix_gpu"]["parameters"]) if "tellurix_gpu" in entry else 16
    p = np.arange(1, total + 1)
    axis.plot(p, (p + 1) * per_run, color=LBLRTM, lw=1.5, label="LBLRTM forward FD, (P+1) runs\n(measured: open square)")
    axis.plot(p, (2 * p + 1) * per_run, color=LBLRTM, lw=1.5, ls="--", label="LBLRTM central FD, (2P+1) runs\n(measured: open diamond)")
    for scheme, marker in (("forward", "s"), ("central", "D")):
        a2 = lbl["A2_fd_gradient"][scheme]
        axis.plot(a2["parameters_needing_reruns"], a2["wall_s"], marker=marker, color=LBLRTM, ls="none", ms=6,
                  mfc="white", mew=1.3)
    for key, color, device in (("tellurix_gpu", GPU, "GPU"), ("tellurix_cpu", CPU, "CPU")):
        sweep = entry.get(key, {}).get("gradient_vs_parameters")
        if not sweep:
            continue
        for label, style in (("live", "-"), ("precomputed", ":")):
            rows = [r for r in sweep.get(label, []) if "value_and_grad" in r]
            if rows:
                axis.plot([r["k"] for r in rows], [r["value_and_grad"]["median_s"] for r in rows], color=color,
                          ls=style, marker="o", ms=4, lw=1.4, label=f"tellurix {device} autodiff, {label}")
            rows = [r for r in sweep.get(label, []) if "jacfwd" in r]
            if rows and label == "live":
                axis.plot([r["k"] for r in rows], [r["jacfwd"]["median_s"] for r in rows], color=color,
                          ls="-.", marker="^", ms=4, lw=1.0, alpha=0.8,
                          label=f"tellurix {device} jacfwd (full Jacobian), live")
    axis.set_yscale("log")
    axis.set_xlim(0.5, total + 0.5)
    axis.set_xlabel("free parameters P (for LBLRTM: column scales, one rerun each)")
    axis.set_ylabel("time per gradient (s)")
    axis.set_title(f"{name} ({entry['label']}): gradient cost against parameter count")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    figure.tight_layout()
    save(figure, "fig_performance_gradient")


def figure_fit(report):
    names = list(report["workloads"])
    figure, (left, right) = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
    bars = (("LBLRTM in the loop\n(least_squares, FD)", LBLRTM, "//"),
            ("tellurix GPU, cold", GPU, ""), ("tellurix GPU, warm", GPU, "xx"),
            ("tellurix CPU, cold", CPU, ".."), ("tellurix CPU, warm", CPU, "oo"))
    width = 0.16
    for index, name in enumerate(names):
        entry = report["workloads"][name]
        e2 = entry["lblrtm"].get("E2_lblrtm_fit", {})

        def cold(key):
            t = entry.get(key)
            if not t:
                return None
            return (t["T0_line_read"]["cold_index_build_s"] + t["T1_prepare"]["first_call_s"]
                    + t["continuum_load_s"] + t["E1_fit_cold"]["total_after_prepare_s"])

        values = (e2.get("wall_s"), cold("tellurix_gpu"),
                  entry.get("tellurix_gpu", {}).get("E1_fit_warm", {}).get("median_s"),
                  cold("tellurix_cpu"), entry.get("tellurix_cpu", {}).get("E1_fit_warm", {}).get("median_s"))
        for offset, ((label, color, hatch), value) in enumerate(zip(bars, values)):
            if value is None:
                continue
            x = index + (offset - 2) * width
            left.bar(x, value, width * 0.92, color=color, hatch=hatch, edgecolor="white", linewidth=0,
                     label=label if index == 0 else None)
            if offset == 0:
                left.annotate(f"{e2['lblrtm_runs']} runs", (x, value), xytext=(0, 2), textcoords="offset points",
                              ha="center", fontsize=6, rotation=90, va="bottom")
    left.set_xticks(range(len(names)))
    left.set_xticklabels([f"{n}\n{report['workloads'][n]['label']}" for n in names])
    left.set_yscale("log")
    left.set_ylabel("wall clock (s)")
    left.set_title("Fit to S/N 200 synthetic data")
    left.grid(axis="x", visible=False)
    left.legend(loc="upper left", frameon=False, fontsize=6.2, ncol=2)
    top = left.get_ylim()[1]
    left.set_ylim(top=top * 8)

    # Recovered column scales, minus truth, on the LBLRTM-made data.
    species = ("H2O", "CO2")
    methods = (("LBLRTM fit", LBLRTM, "s", lambda e: e["lblrtm"].get("E2_lblrtm_fit", {}).get("error")),
               ("tellurix fit, LBLRTM data", GPU, "o", lambda e: e.get("tellurix_gpu", {})
                .get("E1_fit_lblrtm_data", {}).get("error")),
               ("tellurix fit, tellurix data", GPU, "D", lambda e: e.get("tellurix_gpu", {})
                .get("E1_fit_cold", {}).get("tellurix_data", {}).get("error")))
    positions = np.arange(len(names))
    for m_index, (label, color, marker, get) in enumerate(methods):
        for s_index, s in enumerate(species):
            values = []
            for name in names:
                error = get(report["workloads"][name]) or {}
                values.append(100.0 * error.get(s, np.nan))
            x = positions + (s_index - 0.5) * 0.36 + (m_index - 1) * 0.1
            right.plot(x, values, ls="none", marker=marker, color=color, ms=5,
                       mfc=color if s_index == 0 else "white", mew=1.2,
                       label=f"{label}" if s_index == 0 else None)
    right.axhline(0.0, color="#999999", lw=0.8)
    right.set_xticks(positions)
    right.set_xticklabels(names)
    right.set_ylabel("ln column - truth (%)")
    right.set_title("Recovered columns (filled H$_2$O, open CO$_2$)")
    right.grid(axis="x", visible=False)
    right.legend(loc="best", frameon=False, fontsize=6.2)
    figure.tight_layout()
    save(figure, "fig_performance_fit")


def main():
    report = json.loads(REPORT.read_text())
    figure_stages(report)
    figure_scaling(report)
    figure_gradient(report)
    figure_fit(report)


if __name__ == "__main__":
    main()
