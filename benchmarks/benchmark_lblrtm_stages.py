#!/usr/bin/env python
"""Stage-by-stage cost of LBLRTM 12.17 and tellurix on the same atmosphere.

Both codes integrate identical layers -- LBLRTM is given tellurix's own
pressure, temperature and molecular column of each layer as IATM=0 input
(``LBLRTMRunConfig.user_layers``) -- with the same AER 3.9 lines, the H2O and
CO2 (and CH4 where it has lines) of a 12-layer ERA5 profile, MT_CKD continua
and pressure shifts on in both. What still differs is listed under
``comparability`` in the report: LBLRTM's own monochromatic grid, line
rejection and 25 cm-1 cutoff, its single-precision output, and continua beyond
H2O.

Three parts, each its own process because JAX's platform is fixed at import:

  --part lblrtm               LNFL (L0), LBLRTM runs (L1, with the TAPE6 timing
                              breakdown), TelFit-like post-processing (L2),
                              finite-difference gradients (A2), synthetic data
                              made with LBLRTM, and an LBLRTM-in-the-loop fit (E2)
  --part tellurix --platform  T0-T8, autodiff (A1, A3, A4), staged fit (E1)
  --part merge                cross-code comparisons and the committed report

    UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark_lblrtm_stages.py --part lblrtm
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 UV_CACHE_DIR=.uv-cache uv run python \\
        benchmarks/benchmark_lblrtm_stages.py --part tellurix --platform gpu
    UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark_lblrtm_stages.py --part tellurix --platform cpu
    UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark_lblrtm_stages.py --part merge

Needs ``bootstrap_lblrtm.sh``. Partial results go to ``benchmarks/results/``
(gitignored); the merged report to ``docs/lblrtm_tellurix_performance.json``.
"""

from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import json
import os
import platform as platform_module
import re
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
WORK = ROOT / "data/lblrtm/run_stage_benchmark"
REFERENCE = ROOT / "data/lblrtm"
LBLRTM_EXECUTABLE = REFERENCE / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
LNFL_EXECUTABLE = REFERENCE / "LNFL/lnfl_v3.2_linux_gnu_sgl"
AER_LINE_FILE = REFERENCE / "AER_Line_File/aer_v_3.9/line_file/aer_v_3.9"
PROFILE = "data/profiles/gemini_2021_era5.csv"

C_KMS = 299792.458
RESOLVING_POWER = 45_000.0
SAMPLES_PER_RESOLUTION = 4.0
LINE_MARGIN_CM1 = 25.0
GRID_MARGIN_CM1 = 5.0
# LBLRTM lays its own grid inside the request and can fall short of the ends.
LBLRTM_PAD_CM1 = 0.5
ZENITH_ANGLE_DEG = 30.0
SNR = 200.0
CANDIDATE_SPECIES = ("H2O", "CO2", "CH4")
MOLECULE_IDS = {"H2O": 1, "CO2": 2, "CH4": 6}
# A species enters both codes when it has at least this many lines within the
# line margin; CH4 has one line near 5000 cm-1, which would make a column
# parameter that nothing constrains.
MIN_LINES = 20
# IGRINS K92: 2048 pixels over 71.3 cm-1. The other windows use the same
# dispersion, so the pixel count follows the width.
K92_CM1 = (5121.6, 5192.9)
IGRINS_PIXEL_CM1 = (K92_CM1[1] - K92_CM1[0]) / 2048.0
WORKLOADS = {
    "W1": {"window_cm1": (5000.0, 5020.0), "label": "5000-5020 cm-1", "continuum_degree": 3},
    "W2": {"window_cm1": (5000.0, 5100.0), "label": "5000-5100 cm-1", "continuum_degree": 5},
    "W3": {"window_cm1": K92_CM1, "label": "IGRINS K92", "continuum_degree": 9},
}
TRUTH_COLUMNS = {"H2O": 0.3, "CO2": -0.05, "CH4": 0.1}
TRUTH_VELOCITY_KMS = 0.4
TRUTH_LSF_SIGMA_KMS = 2.8
TRUTH_CONTINUUM = (0.02, -0.01, 0.005)
START_LSF_SIGMA_KMS = 3.5
FD_STEP = 1.0e-2       # ln column step of the LBLRTM finite differences
STAGES = ("continuum", "velocity", "columns")


# --------------------------------------------------------------------------
# Shared set-up


def window_pixels(window) -> int:
    return int(round((window[1] - window[0]) / IGRINS_PIXEL_CM1))


def pixel_wavelengths(window):
    import numpy as np
    return np.linspace(1.0e7 / window[1], 1.0e7 / window[0], window_pixels(window))


def summarize(samples: list[float], first: float | None = None) -> dict:
    ordered = sorted(samples)
    result = {
        "median_s": statistics.median(samples),
        "min_s": ordered[0],
        "p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "mean_s": statistics.mean(samples),
        "repetitions": len(samples),
    }
    if first is not None:
        result["first_call_s"] = first
    return result


def hardware() -> dict:
    def command(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    lscpu = command("lscpu") or ""
    field = lambda name: next((line.split(":", 1)[1].strip() for line in lscpu.splitlines()  # noqa: E731
                               if line.startswith(name + ":")), None)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu = None
    if visible not in (None, ""):
        gpu = command("nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                      "--format=csv,noheader", "-i", visible.split(",")[0])
    return {
        "cpu_model": field("Model name"), "cpu_sockets": field("Socket(s)"),
        "cpu_cores_per_socket": field("Core(s) per socket"),
        "cpu_threads_per_core": field("Thread(s) per core"), "cpu_logical": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "cuda_visible_devices": visible, "gpu": gpu,
        "host": platform_module.node(), "python": platform_module.python_version(),
        # The machine is shared; the load at the start says how quiet it was.
        "load_average_1_5_15": list(os.getloadavg()),
        "gpu_processes_at_start": command("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                          "--format=csv,noheader"),
    }


def load_case(name: str) -> dict:
    """Window, profile, grid, species and pixels of one workload, for both codes."""

    import numpy as np
    from tellurix import (AERLineDatabase, DataPaths, constant_velocity_grid, load_atmosphere_csv,
                          trim_wavenumber_grid)

    spec = WORKLOADS[name]
    window = spec["window_cm1"]
    source = load_atmosphere_csv(ROOT / PROFILE)
    grid = trim_wavenumber_grid(
        constant_velocity_grid(1.0e7 / window[1], 1.0e7 / window[0], resolving_power=RESOLVING_POWER,
                               samples_per_resolution=SAMPLES_PER_RESOLUTION,
                               margin_cm1=LINE_MARGIN_CM1),
        window[0], window[1], GRID_MARGIN_CM1)
    paths = DataPaths.bootstrapped(ROOT)
    species, counts = [], {}
    for candidate in CANDIDATE_SPECIES:
        try:
            database = AERLineDatabase(paths.line_file(candidate, MOLECULE_IDS[candidate]), candidate,
                                       (float(grid[0]), float(grid[-1])), margin_cm1=LINE_MARGIN_CM1)
            counts[candidate] = int(np.asarray(database.nu_lines).size)
        except ValueError as exc:
            if "lines found" not in str(exc):
                raise
            counts[candidate] = 0
        if counts[candidate] >= MIN_LINES:
            species.append(candidate)
    # One pressure per layer, as LBLRTM has: the air-weighted mean, for lines
    # and continua alike in both codes.
    profile = dataclasses.replace(source, vmr={s: source.vmr[s] for s in species},
                                  mean_pressure_bar=source.continuum_pressure_bar)
    degree = spec["continuum_degree"]
    continuum = np.zeros(degree + 1)
    continuum[:len(TRUTH_CONTINUUM)] = TRUTH_CONTINUUM
    return {
        "name": name, "window": window, "label": spec["label"], "profile": profile, "grid": grid,
        "species": tuple(species), "candidate_line_counts": counts, "paths": paths,
        "wavelength": pixel_wavelengths(window), "degree": degree, "truth_continuum": continuum,
        "lblrtm_range": (float(grid[0]) - LBLRTM_PAD_CM1, float(grid[-1]) + LBLRTM_PAD_CM1),
    }


def truth_columns(case) -> dict:
    return {s: TRUTH_COLUMNS[s] for s in case["species"]}


# --------------------------------------------------------------------------
# The TelFit-like instrument model applied to LBLRTM's monochromatic output.
# Same parameters and conventions as TelluricModel.predict: velocity shifts
# the pixel wavelengths, a Gaussian LSF of constant velocity width, Simpson
# averaging over each pixel, an exp(Chebyshev) continuum.


def to_log_grid(nu, transmission):
    """Resample LBLRTM's output, uniform in wavenumber, to uniform in log at its native step."""

    import numpy as np
    nu = np.asarray(nu, float)
    step = float(np.median(np.diff(nu)))
    dlog = step / nu[-1]
    count = int(np.floor(np.log(nu[-1] / nu[0]) / dlog)) + 1
    log_nu = nu[0] * np.exp(dlog * np.arange(count))
    return log_nu, np.interp(log_nu, nu, np.asarray(transmission, float)), dlog


def convolve_log(transmission, dlog, sigma_kms):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(transmission, sigma_kms / (dlog * C_KMS), mode="nearest", truncate=5.0)


def pixel_integrate(log_nu, convolved, wavelength, velocity_kms, coefficients):
    import numpy as np
    from numpy.polynomial import chebyshev
    x = np.linspace(-1.0, 1.0, wavelength.size)
    shifted = wavelength * (1.0 + velocity_kms / C_KMS)
    edges = np.concatenate([shifted[:1] - 0.5 * (shifted[1:2] - shifted[:1]),
                            0.5 * (shifted[:-1] + shifted[1:]),
                            shifted[-1:] + 0.5 * (shifted[-1:] - shifted[-2:-1])])
    wavelength_hi = (1.0e7 / log_nu)[::-1]
    spectrum = convolved[::-1]
    center = np.interp(shifted, wavelength_hi, spectrum)
    left = np.interp(edges[:-1], wavelength_hi, spectrum)
    right = np.interp(edges[1:], wavelength_hi, spectrum)
    return np.exp(chebyshev.chebval(x, coefficients)) * (left + 4.0 * center + right) / 6.0


# --------------------------------------------------------------------------
# LBLRTM part


class LBLRTMRunner:
    """One run directory, reused: TAPE5 rewritten, the binary executed, TAPE12 read."""

    def __init__(self, directory: Path, tape3: Path, mt_ckd: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        for source, name in ((tape3, "TAPE3"), (mt_ckd, "absco-ref_wv-mt-ckd.nc")):
            target = self.directory / name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(Path(source).resolve())
        self.binary = self.directory / "lblrtm"
        shutil.copy2(LBLRTM_EXECUTABLE, self.binary)
        self.runs = 0

    def run(self, profile, config) -> float:
        """Write TAPE5 and execute; returns the process wall clock of the binary alone."""

        from tellurix import write_tape5
        write_tape5(self.directory / "TAPE5", profile, config)
        tape12 = self.directory / "TAPE12"
        if tape12.exists():
            tape12.unlink()
        started = time.perf_counter()
        result = subprocess.run([str(self.binary)], cwd=self.directory, capture_output=True, text=True)
        wall = time.perf_counter() - started
        if result.returncode != 0 or not tape12.exists():
            raise RuntimeError(f"LBLRTM failed: {(result.stdout + result.stderr)[-2000:]}")
        self.runs += 1
        return wall

    def read(self):
        from tellurix.reference import read_tape12_single_precision
        return read_tape12_single_precision(self.directory / "TAPE12")

    def tape6(self) -> str:
        return (self.directory / "TAPE6").read_bytes().replace(b"\0", b"").decode("ascii", "replace")


def parse_tape6(text: str) -> dict:
    """LBLRTM's own timing summary: per-module CPU seconds, layers, lines."""

    floats = lambda pattern: [float(v) for v in re.findall(pattern, text)]  # noqa: E731
    result = {
        "opdpth_total_s": sum(floats(r"TOTAL FOR LAYER\s+([\d.]+)")),
        "layers": len(floats(r"TOTAL FOR LAYER\s+([\d.]+)")),
        "emission_init_s": sum(floats(r"TIME REQUIRED FOR --EMINIT--\s+([\d.]+)")),
        "layer_merge_s": sum(floats(r"([\d.]+) SECS WERE REQUIRED FOR THIS MERGE")),
    }
    entering = re.search(r"TIME ENTERING LBLRTM\s+([\d.]+)", text)
    leaving = re.search(r"TIME\s+LEAVING LBLRTM\s+([\d.]+)\s+TOTAL\s+([\d.]+)", text)
    result["startup_s"] = float(entering.group(1)) if entering else None
    result["cpu_total_s"] = float(leaving.group(1)) if leaving else None
    accumulated = text.rsplit("Total Accumulated Times", 1)
    modules = {}
    if len(accumulated) == 2:
        for name in ("LINF4", "XSECT", "LBLF4", "HIRAC1"):
            match = re.search(rf"^\s*{name}\s+([\d.]+)\s+([\d.]+)", accumulated[1], re.MULTILINE)
            if match:
                modules[name] = {"time_s": float(match.group(1)), "read_s": float(match.group(2))}
    result["modules"] = modules
    module_total = sum(m["time_s"] for m in modules.values())
    # CONTNM has no timer of its own: it runs inside OPDPTH, between the line
    # modules, together with the panel bookkeeping.
    result["opdpth_other_incl_contnm_s"] = result["opdpth_total_s"] - module_total
    hirac = re.findall(r"^\s*HIRAC1\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)\s+\d+",
                       text, re.MULTILINE)
    linf4 = re.findall(r"^\s*LINF4\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)\s*$", text, re.MULTILINE)
    result["hirac1_lines_per_layer"] = [int(a) for a, _ in hirac]
    result["hirac1_lines_after_reject_per_layer"] = [int(b) for _, b in hirac]
    result["linf4_lines_per_layer"] = [int(a) for a, _ in linf4]
    result["linf4_after_shrink_per_layer"] = [int(b) for _, b in linf4]
    result["hirac1_dv_per_layer_cm1"] = floats(r"HIRAC1 \*\s+OUTPUT ON FILE\s+\d+\s+DV =\s+([\d.]+)")
    return result


def parse_lnfl_tape6(text: str) -> dict:
    counts = {}
    for name in CANDIDATE_SPECIES:
        match = re.search(rf"{name}\s+=\s+(\d+)\s+(\d+)", text)
        if match:
            counts[name] = {"lines": int(match.group(1)), "coupled_lines": int(match.group(2))}
    total = re.search(r"TOTAL TIME =\s+([\d.]+)", text)
    return {"lines": counts, "cpu_total_s": float(total.group(1)) if total else None}


def lblrtm_part(args) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    from scipy.optimize import least_squares

    from tellurix import LBLRTMRunConfig

    output = {"part": "lblrtm", "hardware": hardware(), "workloads": {},
              "lblrtm": {"version": "12.17", "binary": LBLRTM_EXECUTABLE.name,
                         "lnfl": LNFL_EXECUTABLE.name, "threads": 1}}
    arrays = {}
    for name in args.workloads:
        case = load_case(name)
        print(f"== {name} {case['label']} species {case['species']}", flush=True)
        profile, species, wavelength = case["profile"], case["species"], case["wavelength"]
        v1, v2 = case["lblrtm_range"]
        record = {"species": species, "candidate_line_counts": case["candidate_line_counts"],
                  "lblrtm_request_cm1": [v1, v2], "layers": int(profile.temperature_k.size),
                  "pixels": int(wavelength.size)}
        base = WORK / name

        # L0: LNFL, the one-time TAPE3 for the window. Uncoupled is the main
        # configuration: tellurix runs without line coupling by default.
        lnfl = {}
        for coupling, repetitions in ((False, args.lnfl_repeats), (True, 1)):
            label = "coupled" if coupling else "uncoupled"
            samples = []
            for _ in range(repetitions):
                from tellurix import run_lnfl
                started = time.perf_counter()
                tape3 = run_lnfl(base / f"lnfl_{label}", species, v1 - LINE_MARGIN_CM1 - 1.0,
                                 v2 + LINE_MARGIN_CM1 + 1.0, AER_LINE_FILE, LNFL_EXECUTABLE,
                                 line_coupling=coupling)
                samples.append(time.perf_counter() - started)
            lnfl[label] = {**summarize(samples), "tape3": str(tape3.relative_to(ROOT)),
                           **parse_lnfl_tape6((tape3.parent / "TAPE6").read_bytes()
                                              .replace(b"\0", b"").decode("ascii", "replace"))}
            print(f"L0 LNFL {label}: {statistics.median(samples):.2f} s", flush=True)
        record["L0_lnfl"] = lnfl
        tape3 = ROOT / lnfl["uncoupled"]["tape3"]

        def config(continuum_flag=1):
            return LBLRTMRunConfig(v1, v2, zenith_angle_deg=ZENITH_ANGLE_DEG, continuum_flag=continuum_flag,
                                   user_layers=True, description=f"tellurix stage benchmark {name}")

        def scaled(columns):
            return dataclasses.replace(profile, vmr={s: profile.vmr[s] * np.exp(columns.get(s, 0.0))
                                                     for s in profile.vmr})

        # L1: one LBLRTM run, process level, with TAPE6's breakdown.
        runner = LBLRTMRunner(base / "run", tape3, case["paths"].mt_ckd)
        truth = truth_columns(case)
        cold = runner.run(scaled(truth), config())
        walls, breakdowns = [], []
        for _ in range(args.repeats):
            walls.append(runner.run(scaled(truth), config()))
            breakdowns.append(parse_tape6(runner.tape6()))
        breakdown = breakdowns[-1]
        median_breakdown = {
            key: statistics.median(b[key] for b in breakdowns)
            for key in ("opdpth_total_s", "emission_init_s", "layer_merge_s", "startup_s", "cpu_total_s",
                        "opdpth_other_incl_contnm_s")}
        median_breakdown["modules"] = {
            module: {k: statistics.median(b["modules"][module][k] for b in breakdowns)
                     for k in ("time_s", "read_s")} for module in breakdown["modules"]}
        spectrum = runner.read()
        nu_mono, t_mono = np.asarray(spectrum.wavenumber_cm1), np.asarray(spectrum.transmission)
        record["L1_lblrtm_run"] = {
            **summarize(walls, cold), "tape6_median": median_breakdown,
            "tape6_last": {k: v for k, v in breakdown.items()},
            "output_samples": int(nu_mono.size), "output_dv_cm1": float(np.median(np.diff(nu_mono))),
        }
        print(f"L1 LBLRTM run: cold {cold:.3f} s, median {statistics.median(walls):.3f} s; "
              f"{nu_mono.size} samples", flush=True)
        variants = {}
        for label, run_config, line_file in (("continuum_off", config(0), tape3),
                                             ("coupled_tape3", config(), ROOT / lnfl["coupled"]["tape3"])):
            variant_runner = LBLRTMRunner(base / f"run_{label}", line_file, case["paths"].mt_ckd)
            variant_runner.run(scaled(truth), run_config)
            samples = [variant_runner.run(scaled(truth), run_config) for _ in range(args.variant_repeats)]
            variants[label] = {**summarize(samples), "tape6": parse_tape6(variant_runner.tape6())}
            print(f"L1 variant {label}: {statistics.median(samples):.3f} s", flush=True)
        record["L1_variants"] = variants

        # L2: what a TelFit-like code does after LBLRTM.
        instrument_truth = (TRUTH_VELOCITY_KMS, TRUTH_LSF_SIGMA_KMS, case["truth_continuum"])
        timings = collections.defaultdict(list)
        for _ in range(args.repeats):
            started = time.perf_counter()
            read = runner.read()
            timings["read_tape12"].append(time.perf_counter() - started)
            started = time.perf_counter()
            log_nu, t_log, dlog = to_log_grid(read.wavenumber_cm1, read.transmission)
            timings["resample_to_log_grid"].append(time.perf_counter() - started)
            started = time.perf_counter()
            convolved = convolve_log(t_log, dlog, TRUTH_LSF_SIGMA_KMS)
            timings["lsf_convolution"].append(time.perf_counter() - started)
            started = time.perf_counter()
            flux_truth = pixel_integrate(log_nu, convolved, wavelength, TRUTH_VELOCITY_KMS,
                                         case["truth_continuum"])
            timings["shift_pixel_integration_continuum"].append(time.perf_counter() - started)
        record["L2_postprocess"] = {key: summarize(value) for key, value in timings.items()}
        record["L2_postprocess"]["total_median_s"] = sum(statistics.median(v) for v in timings.values())
        record["L2_postprocess"]["log_grid_samples"] = int(log_nu.size)
        print(f"L2 post-process: {record['L2_postprocess']['total_median_s'] * 1e3:.1f} ms", flush=True)

        def lblrtm_flux(columns, velocity=TRUTH_VELOCITY_KMS, sigma=TRUTH_LSF_SIGMA_KMS,
                        coefficients=case["truth_continuum"]):
            runner.run(scaled(columns), config())
            read = runner.read()
            log_nu, t_log, dlog = to_log_grid(read.wavenumber_cm1, read.transmission)
            return pixel_integrate(log_nu, convolve_log(t_log, dlog, sigma), wavelength, velocity,
                                   coefficients)

        # Synthetic data made with LBLRTM at the truth, S/N 200.
        uncertainty = np.full(wavelength.size, 1.0 / SNR)
        observed = flux_truth + np.random.default_rng(11).normal(0.0, 1.0 / SNR, wavelength.size)

        # A2: finite-difference gradients over the P column scales -- the only
        # parameters that need LBLRTM rerun; velocity, LSF and continuum act on
        # the stored monochromatic spectrum.
        fd = {}
        for scheme in ("forward", "central"):
            started = time.perf_counter()
            runs_before = runner.runs
            centre = lblrtm_flux(truth)
            jacobian = {}
            for s in species:
                up = lblrtm_flux({**truth, s: truth[s] + FD_STEP})
                if scheme == "central":
                    down = lblrtm_flux({**truth, s: truth[s] - FD_STEP})
                    jacobian[s] = (up - down) / (2.0 * FD_STEP)
                else:
                    jacobian[s] = (up - centre) / FD_STEP
            fd[scheme] = {"wall_s": time.perf_counter() - started, "lblrtm_runs": runner.runs - runs_before,
                          "parameters_needing_reruns": len(species), "step_ln_column": FD_STEP}
            residual = (observed - centre) / uncertainty ** 2
            fd[scheme]["chi2_gradient_columns"] = {s: float(-np.sum(residual * jacobian[s])) for s in species}
            for s in species:
                arrays[f"{name}_lblrtm_jacobian_{scheme}_{s}"] = jacobian[s]
            print(f"A2 {scheme} FD gradient: {fd[scheme]['lblrtm_runs']} runs, "
                  f"{fd[scheme]['wall_s']:.2f} s", flush=True)
        per_run = statistics.median(walls) + record["L2_postprocess"]["total_median_s"]
        record["A2_fd_gradient"] = {
            **fd, "per_evaluation_s": per_run,
            "predicted_forward_s": (len(species) + 1) * per_run,
            "predicted_central_s": (2 * len(species) + 1) * per_run,
        }
        arrays[f"{name}_lblrtm_flux_truth"] = flux_truth
        arrays[f"{name}_observed_lblrtm"] = observed
        arrays[f"{name}_wavelength"] = wavelength
        arrays[f"{name}_uncertainty"] = uncertainty
        arrays[f"{name}_nu_mono"] = nu_mono
        arrays[f"{name}_t_mono"] = t_mono

        # E2: the LBLRTM-in-the-loop fit, TelFit-like: scipy least_squares with
        # 2-point finite differences, LBLRTM rerun whenever a column changes.
        if name in args.e2_workloads:
            record["E2_lblrtm_fit"] = lblrtm_fit(case, runner, config, scaled, observed, uncertainty,
                                                 args.e2_cap, least_squares)
            e2 = record["E2_lblrtm_fit"]
            print(f"E2 LBLRTM fit: {e2['lblrtm_runs']} runs, {e2['wall_s']:.1f} s, {e2['status']}",
                  flush=True)
        output["workloads"][name] = record
        write_partial(args, "lblrtm", output, arrays)


def lblrtm_fit(case, runner, config, scaled, observed, uncertainty, cap, least_squares) -> dict:
    import numpy as np

    species, wavelength, degree = case["species"], case["wavelength"], case["degree"]
    count = len(species)
    cache: collections.OrderedDict = collections.OrderedDict()
    state = {"best_cost": np.inf, "best_x": None, "evaluations": 0, "lblrtm_wall_s": 0.0}

    class Budget(Exception):
        pass

    def monochromatic(columns):
        key = tuple(np.round(columns, 12))
        if key not in cache:
            if state["enforce"] and runner.runs - state["runs_at_start"] >= cap:
                raise Budget
            started = time.perf_counter()
            runner.run(scaled(dict(zip(species, columns))), config())
            read = runner.read()
            cache[key] = to_log_grid(read.wavenumber_cm1, read.transmission)
            state["lblrtm_wall_s"] += time.perf_counter() - started
            while len(cache) > 8:
                cache.popitem(last=False)
        return cache[key]

    def residual(u):
        x = u - SHIFT
        log_nu, t_log, dlog = monochromatic(x[:count])
        model = pixel_integrate(log_nu, convolve_log(t_log, dlog, x[count + 1]), wavelength, x[count],
                                x[count + 2:])
        r = (observed - model) / uncertainty
        cost = 0.5 * float(r @ r)
        state["evaluations"] += 1
        if cost < state["best_cost"]:
            state["best_cost"], state["best_x"] = cost, np.array(x)
        return r

    # scipy's relative difference step is zero at a parameter that is exactly
    # zero and falls back to 1.5e-8, below LBLRTM's single-precision output and
    # its E15.7 layer columns: the column derivative comes out identically zero.
    # Shifting every parameter by a constant makes the step ~1e-2 throughout.
    SHIFT = 10.0

    x0 = np.concatenate([np.zeros(count), [0.0, START_LSF_SIGMA_KMS], np.zeros(degree + 1)])
    lower = np.concatenate([np.full(count, -2.0), [-8.0, 1.0], [-2.0], np.full(degree, -0.5)])
    upper = np.concatenate([np.full(count, 2.0), [8.0, 6.0], [2.0], np.full(degree, 0.5)])
    state["runs_at_start"], state["enforce"] = runner.runs, True
    started = time.perf_counter()
    try:
        fit = least_squares(residual, x0 + SHIFT, jac="2-point", diff_step=1.0e-3,
                            bounds=(lower + SHIFT, upper + SHIFT), method="trf", x_scale="jac", max_nfev=10_000)
        status, x, message, nfev, njev = "converged" if fit.success else "stopped", fit.x - SHIFT, fit.message, \
            int(fit.nfev), int(fit.njev or 0)
    except Budget:
        status, x, message, nfev, njev = "budget_exhausted", state["best_x"], f"cap of {cap} LBLRTM runs", \
            None, None
    wall = time.perf_counter() - started
    runs = runner.runs - state["runs_at_start"]
    state["enforce"] = False
    truth = np.concatenate([[TRUTH_COLUMNS[s] for s in species], [TRUTH_VELOCITY_KMS, TRUTH_LSF_SIGMA_KMS],
                            case["truth_continuum"]])
    names = [*species, "velocity_kms", "lsf_sigma_kms", *(f"continuum_{i}" for i in range(degree + 1))]
    final = residual(x + SHIFT)
    return {
        "method": "scipy.optimize.least_squares(trf, jac='2-point', diff_step=1e-3 on parameters offset by 10, "
                  "i.e. steps ~1e-2, x_scale='jac'), "
                  "LBLRTM rerun for every new column vector, others on the stored monochromatic spectrum",
        "data": "LBLRTM at the truth + Gaussian noise, S/N 200",
        "status": status, "message": str(message), "wall_s": wall, "lblrtm_runs": runs,
        "lblrtm_wall_s": state["lblrtm_wall_s"], "residual_evaluations": state["evaluations"],
        "nfev": nfev, "njev": njev, "cap": cap,
        "parameters": dict(zip(names, map(float, x))), "truth": dict(zip(names, map(float, truth))),
        "error": dict(zip(names, map(float, x - truth))),
        "chi2_per_pixel": float(final @ final / wavelength.size),
    }


# --------------------------------------------------------------------------
# tellurix part


def tellurix_part(args) -> None:
    # Must be selected before importing JAX or tellurix.
    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.cpu_threads:
        os.sched_setaffinity(0, set(range(args.cpu_threads)))

    import jax
    import jax.numpy as jnp
    import numpy as np

    import tellurix.aer as aer_module
    from tellurix import (AERLineDatabase, ExoJAXOpacityBackend, MTCKDWaterContinuum, OrderObjective,
                          SpectralOrder, TelluricModel, TelluricParameters, fit_order)

    devices = jax.devices()
    lbl_arrays = {}
    lbl_file = RESULTS / "lblrtm_stages_lblrtm.npz"
    if lbl_file.exists():
        lbl_arrays = dict(np.load(lbl_file))

    def block(tree):
        for leaf in jax.tree_util.tree_leaves(tree):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        return tree

    def measure(function, *arguments, repetitions=args.repeats, warmup=2):
        started = time.perf_counter()
        block(function(*arguments))
        first = time.perf_counter() - started
        for _ in range(warmup):
            block(function(*arguments))
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter()
            block(function(*arguments))
            samples.append(time.perf_counter() - started)
        return summarize(samples, first)

    def timed(function):
        started = time.perf_counter()
        value = function()
        block(value)
        return value, time.perf_counter() - started

    output = {"part": f"tellurix_{args.label}", "platform": args.platform, "hardware": hardware(),
              "jax": {"version": jax.__version__, "jaxlib": jax.lib.__version__,
                      "device": str(devices[0]), "device_kind": devices[0].device_kind,
                      "x64": bool(jax.config.x64_enabled),
                      "compilation_cache_dir": jax.config.jax_compilation_cache_dir,
                      "xla_flags": os.environ.get("XLA_FLAGS")},
              "workloads": {}}
    try:
        import exojax
        output["jax"]["exojax"] = exojax.__version__
    except Exception:  # noqa: BLE001
        pass
    arrays = {}

    for name in args.workloads:
        case = load_case(name)
        print(f"== {name} {case['label']} species {case['species']} on {devices[0]}", flush=True)
        profile, grid, species, paths = case["profile"], case["grid"], case["species"], case["paths"]
        wavelength, degree = case["wavelength"], case["degree"]
        temperature_range = (float(np.min(profile.temperature_k)), float(np.max(profile.temperature_k)))
        maximum_pressure = float(np.max(profile.pressure_layer_bar))
        record = {"species": species, "layers": int(profile.temperature_k.size),
                  "grid_samples": int(grid.size), "pixels": int(wavelength.size),
                  "grid_cm1": [float(grid[0]), float(grid[-1])],
                  "velocity_step_kms": float(np.log(grid[1] / grid[0]) * C_KMS)}

        # T0: the AER line files, a cold index build against the cached index.
        def read_lines():
            return {s: AERLineDatabase(paths.line_file(s, MOLECULE_IDS[s]), s, (float(grid[0]), float(grid[-1])),
                                       margin_cm1=LINE_MARGIN_CM1) for s in species}
        aer_module._FILE_INDEX.clear()
        started = time.perf_counter()
        databases = read_lines()
        cold_read = time.perf_counter() - started
        cached = []
        for _ in range(5):
            started = time.perf_counter()
            read_lines()
            cached.append(time.perf_counter() - started)
        record["lines"] = {s: int(np.asarray(d.nu_lines).size) for s, d in databases.items()}
        record["T0_line_read"] = {"cold_index_build_s": cold_read, "cached": summarize(cached),
                                  "note": "in-process index cache cleared; the OS page cache is warm"}
        print(f"T0 line read: cold {cold_read:.2f} s, cached {statistics.median(cached) * 1e3:.1f} ms", flush=True)

        # T1: sparse core lists.
        def prepare(mixed=True):
            return ExoJAXOpacityBackend.prepare(
                databases, grid, methods="direct_sparse", temperature_range_k=temperature_range,
                maximum_pressure_bar=maximum_pressure, vectorize_layers=True, mixed_precision=mixed,
                pressure_shift=True)
        prepare_samples = []
        for _ in range(3):
            started = time.perf_counter()
            backend = prepare()
            prepare_samples.append(time.perf_counter() - started)
        record["T1_prepare"] = {**summarize(prepare_samples[1:], prepare_samples[0]),
                                "core_pairs": {s: int(len(c.core_line)) for s, c in backend.calculators.items()
                                               if hasattr(c, "core_line")}}
        started = time.perf_counter()
        mt_ckd = MTCKDWaterContinuum.from_netcdf(paths.mt_ckd, grid)
        record["continuum_load_s"] = time.perf_counter() - started

        def build(opacity):
            return TelluricModel(profile, grid, opacity, continuum=mt_ckd, accuracy_mode="mt_ckd",
                                 max_lsf_sigma_kms=10.0, pixel_integration="simpson")

        model = build(backend)
        uncertainty = np.full(wavelength.size, 1.0 / SNR)
        blank = SpectralOrder(wavelength, np.ones(wavelength.size), uncertainty,
                              source_flux=np.ones(wavelength.size), zenith_angle_deg=ZENITH_ANGLE_DEG)
        codec = OrderObjective(model, blank, degree + 1).codec
        names = list(codec.names)
        truth = TelluricParameters(truth_columns(case), TRUTH_VELOCITY_KMS, 0.0, TRUTH_LSF_SIGMA_KMS,
                                   case["truth_continuum"], np.log(1.0e-5))
        truth_vector = jnp.asarray(codec.pack(truth))
        record["parameters"] = names

        # T5: the full forward model, live opacity.
        forward = jax.jit(lambda vector: model.predict(blank, codec.unpack(vector)))
        record["T5_forward_live"] = measure(forward, truth_vector)
        print(f"T5 forward live: first {record['T5_forward_live']['first_call_s']:.1f} s, "
              f"{record['T5_forward_live']['median_s'] * 1e3:.2f} ms", flush=True)
        flux_truth = np.asarray(forward(truth_vector))
        observed = flux_truth + np.random.default_rng(7).normal(0.0, 1.0 / SNR, wavelength.size)
        order = SpectralOrder(wavelength, observed, uncertainty, source_flux=np.ones(wavelength.size),
                              zenith_angle_deg=ZENITH_ANGLE_DEG)

        # E1 cold: a fresh model through precompute, compile and the staged fit,
        # as a pipeline meets an order. Lines, prepare and continuum are timed
        # above and added in the merge.
        start = TelluricParameters({s: 0.0 for s in species}, 0.0, 0.0, START_LSF_SIGMA_KMS,
                                   np.zeros(degree + 1), np.log(1.0e-3))

        def staged_fit(fit_model, objective, data):
            current, iterations, stage_s = start, 0, {}
            for stage in STAGES:
                started = time.perf_counter()
                result = fit_order(fit_model, data, current, stage_bounds(stage, names, species, current, degree),
                                   objective=objective, covariance=stage == STAGES[-1])
                stage_s[stage] = time.perf_counter() - started
                current, iterations = result.parameters, iterations + result.iterations
            return result, iterations, stage_s

        fresh = build(backend)
        started = time.perf_counter()
        precomputed = fresh.precompute_opacity()
        block(precomputed.opacity.values)
        precompute_cold = time.perf_counter() - started
        started = time.perf_counter()
        objective = OrderObjective(precomputed, order, degree + 1)
        objective(codec.pack(start))
        objective_compile = time.perf_counter() - started
        started = time.perf_counter()
        result, iterations, stage_s = staged_fit(precomputed, objective, order)
        fit_s = time.perf_counter() - started
        record["E1_fit_cold"] = {
            "precompute_s": precompute_cold, "objective_compile_and_first_s": objective_compile,
            "staged_fit_s": fit_s, "stage_s": stage_s, "iterations": iterations,
            "total_after_prepare_s": precompute_cold + objective_compile + fit_s,
            "note": "the final stage includes the Hessian compile for the formal covariance",
        }
        record["E1_fit_cold"]["tellurix_data"] = fit_summary(result, truth, names, species, model, order, block)
        print(f"E1 cold: precompute {precompute_cold:.2f} s, compile {objective_compile:.2f} s, "
              f"fit {fit_s:.2f} s ({iterations} it)", flush=True)
        warm = []
        for _ in range(args.fit_repeats):
            started = time.perf_counter()
            staged_fit(precomputed, objective, order)
            warm.append(time.perf_counter() - started)
        record["E1_fit_warm"] = summarize(warm)
        # The same fit on LBLRTM's synthetic data: same pixels and source, so the
        # compiled objective is rebound rather than rebuilt.
        if f"{name}_observed_lblrtm" in lbl_arrays:
            if not np.allclose(lbl_arrays[f"{name}_wavelength"], wavelength, rtol=0, atol=1e-9):
                raise RuntimeError("LBLRTM part used different pixels")
            lbl_order = SpectralOrder(wavelength, lbl_arrays[f"{name}_observed_lblrtm"], uncertainty,
                                      source_flux=np.ones(wavelength.size), zenith_angle_deg=ZENITH_ANGLE_DEG)
            objective.rebind(lbl_order)
            started = time.perf_counter()
            lbl_result, lbl_iterations, _ = staged_fit(precomputed, objective, lbl_order)
            record["E1_fit_lblrtm_data"] = {
                "staged_fit_s": time.perf_counter() - started, "iterations": lbl_iterations,
                **fit_summary(lbl_result, truth, names, species, model, lbl_order, block)}
            objective.rebind(order)
            print(f"E1 on LBLRTM data: {record['E1_fit_lblrtm_data']['staged_fit_s']:.2f} s, "
                  f"errors {record['E1_fit_lblrtm_data']['error']}", flush=True)

        # T2: the opacity kernel alone, monochromatic cross sections per layer.
        temperature = jnp.asarray(profile.temperature_k)
        pressure = jnp.asarray(profile.pressure_layer_bar)
        partial = {s: pressure * jnp.asarray(profile.vmr[s]) * np.exp(TRUTH_COLUMNS[s]) for s in species}

        def kernel_of(opacity):
            return jax.jit(lambda partial_pressure: opacity.cross_sections(temperature, pressure, partial_pressure))
        record["T2_opacity_kernel"] = measure(kernel_of(backend), partial)
        print(f"T2 kernel: first {record['T2_opacity_kernel']['first_call_s']:.1f} s, "
              f"{record['T2_opacity_kernel']['median_s'] * 1e3:.2f} ms", flush=True)

        # T3: continuum, column scaling and the slant path with the opacity fixed.
        vmr = {s: jnp.asarray(v) for s, v in profile.vmr.items()}
        continuum_only = jax.jit(lambda scale: model.continuum.optical_depth(
            profile, {s: v * jnp.exp(scale) if s == "H2O" else v for s, v in vmr.items()}))
        record["T3_continuum_mt_ckd"] = measure(continuum_only, jnp.asarray(TRUTH_COLUMNS["H2O"]))
        transmission_pre = jax.jit(lambda vector: precomputed.transmission(codec.unpack(vector), ZENITH_ANGLE_DEG))
        record["T3_slant_transmission_given_opacity"] = measure(transmission_pre, truth_vector)

        # T4: the instrument alone -- velocity, LSF, Simpson pixels, continuum.
        def instrument_only(t_hi, vector):
            replaced = copy.copy(model)
            replaced.transmission = lambda parameters, zenith_angle_deg=0.0: t_hi
            return replaced.predict(blank, codec.unpack(vector))
        t_hi = block(transmission_pre(truth_vector))
        record["T4_instrument"] = measure(jax.jit(instrument_only), t_hi, truth_vector)
        print(f"T3 continuum {record['T3_continuum_mt_ckd']['median_s'] * 1e3:.3f} ms, slant "
              f"{record['T3_slant_transmission_given_opacity']['median_s'] * 1e3:.3f} ms; T4 instrument "
              f"{record['T4_instrument']['median_s'] * 1e3:.3f} ms", flush=True)

        # T5 with precomputed opacity, T6 and T7 value and gradient.
        record["T5_forward_precomputed"] = measure(
            jax.jit(lambda vector: precomputed.predict(blank, codec.unpack(vector))), truth_vector)
        live_objective = OrderObjective(model, order, degree + 1)
        record["T6_value_and_grad_live"] = measure(live_objective, np.asarray(truth_vector))
        record["T7_value_and_grad_precomputed"] = measure(objective, np.asarray(truth_vector))
        warm_precompute = []
        for _ in range(3):
            started = time.perf_counter()
            block(fresh.precompute_opacity().opacity.values)
            warm_precompute.append(time.perf_counter() - started)
        record["T7_precompute"] = summarize(warm_precompute, precompute_cold)
        print(f"T6 vag live {record['T6_value_and_grad_live']['median_s'] * 1e3:.2f} ms "
              f"(first {record['T6_value_and_grad_live']['first_call_s']:.1f} s); T7 precompute "
              f"{statistics.median(warm_precompute):.3f} s, vag pre "
              f"{record['T7_value_and_grad_precomputed']['median_s'] * 1e3:.3f} ms", flush=True)

        # A1: gradient over forward.
        record["A1_grad_over_forward"] = {
            "live": record["T6_value_and_grad_live"]["median_s"] / record["T5_forward_live"]["median_s"],
            "precomputed": (record["T7_value_and_grad_precomputed"]["median_s"]
                            / record["T5_forward_precomputed"]["median_s"]),
        }

        # A3: autodiff against finite differences of tellurix itself.
        def chi2_of(fit_model):
            return lambda vector: 0.5 * jnp.sum(((jnp.asarray(observed) - fit_model.predict(
                order, codec.unpack(vector))) / uncertainty) ** 2)
        chi2_live = jax.jit(chi2_of(model))
        grad_live = jax.jit(jax.grad(chi2_of(model)))
        autodiff = np.asarray(grad_live(truth_vector))
        accuracy = {"chi2_gradient": {}, "flux_jacobian": {}}
        for index, parameter in enumerate(names):
            if parameter == "wavelength_stretch" or parameter == "log_jitter":
                continue
            step = 1.0e-4 if parameter not in species else 1.0e-3
            unit = np.zeros(len(names))
            unit[index] = step
            finite = (float(chi2_live(truth_vector + unit)) - float(chi2_live(truth_vector - unit))) / (2.0 * step)
            accuracy["chi2_gradient"][parameter] = {
                "autodiff": float(autodiff[index]), "central_fd": finite,
                "relative_difference": abs(float(autodiff[index]) - finite) / max(abs(finite), 1e-300),
                "step": step}
        jacobian_live = jax.jit(jax.jacfwd(lambda vector: model.predict(blank, codec.unpack(vector))))
        jacobian_pre = jax.jit(jax.jacfwd(lambda vector: precomputed.predict(blank, codec.unpack(vector))))
        try:
            live_jac, live_jac_s = timed(lambda: jacobian_live(truth_vector))
            live_jac = np.asarray(live_jac)
        except Exception as exc:  # noqa: BLE001
            live_jac, live_jac_s = None, None
            accuracy["jacfwd_live_error"] = repr(exc)[:400]
        pre_jac = np.asarray(jacobian_pre(truth_vector))
        for s in species:
            index = names.index(s)
            unit = np.zeros(len(names))
            unit[index] = FD_STEP
            fd = (np.asarray(forward(truth_vector + unit)) - np.asarray(forward(truth_vector - unit))) / (2 * FD_STEP)
            entry = {"step": FD_STEP}
            reference = live_jac[:, index] if live_jac is not None else pre_jac[:, index]
            norm = float(np.sqrt(np.mean(reference ** 2)))
            entry["rms_autodiff"] = norm
            entry["autodiff_vs_tellurix_central_fd_rel_rms"] = float(np.sqrt(np.mean((reference - fd) ** 2))) / norm
            entry["precomputed_vs_live_autodiff_rel_rms"] = float(
                np.sqrt(np.mean((pre_jac[:, index] - reference) ** 2))) / norm
            arrays[f"{name}_tellurix_jacobian_{s}"] = reference
            arrays[f"{name}_tellurix_fd_jacobian_{s}"] = fd
            accuracy["flux_jacobian"][s] = entry
        arrays[f"{name}_tellurix_flux_truth"] = flux_truth
        record["A3_gradient_accuracy"] = accuracy

        # A4: the full Jacobian.
        a4 = {"parameters": len(names)}
        if live_jac is not None and name in args.jacfwd_live_workloads:
            try:
                a4["jacfwd_live"] = measure(jacobian_live, truth_vector, repetitions=max(3, args.repeats // 3))
                a4["jacfwd_live"]["first_call_s"] = live_jac_s
            except Exception as exc:  # noqa: BLE001
                a4["jacfwd_live_error"] = repr(exc)[:400]
        elif live_jac is not None:
            a4["jacfwd_live"] = {"first_call_s": live_jac_s}
        a4["jacfwd_precomputed"] = measure(jacobian_pre, truth_vector)
        record["A4_jacobian"] = a4
        print(f"A4 jacfwd: live {a4.get('jacfwd_live', {}).get('median_s', float('nan')) * 1e3:.1f} ms, "
              f"pre {a4['jacfwd_precomputed']['median_s'] * 1e3:.2f} ms", flush=True)

        # Gradient cost against the number of free parameters: value and
        # gradient (reverse mode) and the Jacobian (forward mode) over the first
        # k parameters, the rest held.
        if name in args.sweep_workloads:
            sweep = {}
            counts = sorted({1, 2, 4, 8, len(names)} & set(range(1, len(names) + 1)))
            for label, fit_model in (("precomputed", precomputed), ("live", model)):
                chi2 = chi2_of(fit_model)
                rows = []
                for k in counts:
                    if label == "live" and k not in (1, len(names)):
                        continue

                    def partial_chi2(sub, k=k, chi2=chi2):
                        return chi2(jnp.concatenate([sub, truth_vector[k:]]))

                    def partial_flux(sub, k=k, fit_model=fit_model):
                        return fit_model.predict(blank, codec.unpack(jnp.concatenate([sub, truth_vector[k:]])))
                    sub = truth_vector[:k]
                    row = {"k": k, "value_and_grad": measure(jax.jit(jax.value_and_grad(partial_chi2)), sub,
                                                             repetitions=max(5, args.repeats // 2))}
                    if not (label == "live" and name not in args.jacfwd_live_workloads):
                        try:
                            row["jacfwd"] = measure(jax.jit(jax.jacfwd(partial_flux)), sub,
                                                    repetitions=max(3, args.repeats // 3))
                        except Exception as exc:  # noqa: BLE001  (device memory)
                            row["jacfwd_error"] = repr(exc)[:400]
                    rows.append(row)
                sweep[label] = rows
            record["gradient_vs_parameters"] = sweep

        # Mixed precision off: kernel, forward, value and gradient.
        if name in args.mixed_off_workloads:
            started = time.perf_counter()
            backend64 = prepare(mixed=False)
            model64 = build(backend64)
            mixed_off = {"prepare_s": time.perf_counter() - started}
            mixed_off["T2_opacity_kernel"] = measure(kernel_of(backend64), partial)
            forward64 = jax.jit(lambda vector: model64.predict(blank, codec.unpack(vector)))
            mixed_off["T5_forward_live"] = measure(forward64, truth_vector)
            try:
                mixed_off["T6_value_and_grad_live"] = measure(OrderObjective(model64, order, degree + 1),
                                                              np.asarray(truth_vector))
            except Exception as exc:  # noqa: BLE001  (float64 wing derivatives exhaust device memory)
                mixed_off["T6_value_and_grad_live_error"] = repr(exc)[:300]
            mixed_off["max_flux_difference_vs_mixed"] = float(np.max(np.abs(
                np.asarray(forward64(truth_vector)) - flux_truth)))
            record["mixed_precision_off"] = mixed_off
            print(f"mixed off: kernel {mixed_off['T2_opacity_kernel']['median_s'] * 1e3:.2f} ms, forward "
                  f"{mixed_off['T5_forward_live']['median_s'] * 1e3:.2f} ms; vag "
                  f"{mixed_off.get('T6_value_and_grad_live', {}).get('median_s', float('nan')) * 1e3:.2f} ms",
                  flush=True)

        record["process_threads"] = process_threads()
        output["workloads"][name] = record
        write_partial(args, f"tellurix_{args.label}", output, arrays)


def process_threads() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def stage_bounds(stage, names, species, current, degree) -> dict:
    """The IGRINS schedule: continuum, then velocity and LSF, then columns."""

    import numpy as np
    pin = lambda value: (float(value), float(value))  # noqa: E731
    free_columns = stage == "columns"
    free_velocity = stage in ("velocity", "columns")
    bounds = {"wavelength_stretch": (0.0, 0.0), "log_jitter": (np.log(1e-5), np.log(0.5))}
    for s in species:
        bounds[s] = (-2.0, 2.0) if free_columns else pin(current.log_column_scales[s])
    bounds["velocity_kms"] = (-8.0, 8.0) if free_velocity else pin(current.velocity_kms)
    bounds["lsf_sigma_kms"] = (1.0, 6.0) if free_velocity else pin(current.lsf_sigma_kms)
    for index in range(degree + 1):
        bounds[f"continuum_{index}"] = (-2.0, 2.0) if index == 0 else (-0.5, 0.5)
    missing = set(names) - set(bounds)
    if missing:
        raise ValueError(f"no bounds for {missing}")
    return bounds


def fit_summary(result, truth, names, species, model, order, block) -> dict:
    import numpy as np
    fitted = {**{s: float(result.parameters.log_column_scales[s]) for s in species},
              "velocity_kms": float(result.parameters.velocity_kms),
              "lsf_sigma_kms": float(result.parameters.lsf_sigma_kms)}
    expected = {**{s: float(truth.log_column_scales[s]) for s in species},
                "velocity_kms": float(truth.velocity_kms), "lsf_sigma_kms": float(truth.lsf_sigma_kms)}
    return {"parameters": fitted, "truth": expected,
            "error": {k: fitted[k] - expected[k] for k in fitted},
            "success": bool(result.success), "message": result.message,
            "chi2_per_pixel": float(np.mean((np.asarray(result.residuals) / np.asarray(order.uncertainty)) ** 2))}


# --------------------------------------------------------------------------
# Output


def write_partial(args, label, output, arrays) -> None:
    import numpy as np
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"lblrtm_stages_{label}.json").write_text(json.dumps(output, indent=1, default=float) + "\n")
    np.savez_compressed(RESULTS / f"lblrtm_stages_{label}.npz", **arrays)


def merge_part(args) -> None:
    import numpy as np

    parts, arrays = {}, {}
    for path in sorted(RESULTS.glob("lblrtm_stages_*.json")):
        label = path.stem.removeprefix("lblrtm_stages_")
        parts[label] = json.loads(path.read_text())
        npz = path.with_suffix(".npz")
        arrays[label] = dict(np.load(npz)) if npz.exists() else {}
    lbl = parts["lblrtm"]
    report = {
        "description": __doc__.split("\n\n")[1].replace("\n", " "),
        "generated_by": "benchmarks/benchmark_lblrtm_stages.py",
        "measured": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "profile": PROFILE, "zenith_angle_deg": ZENITH_ANGLE_DEG, "resolving_power": RESOLVING_POWER,
        "samples_per_resolution": SAMPLES_PER_RESOLUTION, "line_margin_cm1": LINE_MARGIN_CM1,
        "grid_margin_cm1": GRID_MARGIN_CM1, "snr": SNR,
        "truth": {"log_column_scales": TRUTH_COLUMNS, "velocity_kms": TRUTH_VELOCITY_KMS,
                  "lsf_sigma_kms": TRUTH_LSF_SIGMA_KMS, "continuum_leading": TRUTH_CONTINUUM},
        "comparability": COMPARABILITY,
        "timing_method": ("warm = median of >= 10 synchronized calls after two warm-ups "
                          "(jax.block_until_ready on every output leaf); first_call_s = the first call in "
                          "the process, including XLA compilation; no persistent compilation cache. LBLRTM: "
                          "process wall clock of the binary per run, TAPE5 written before the clock starts."),
        "parts": {k: {kk: vv for kk, vv in v.items() if kk != "workloads"} for k, v in parts.items()},
        "workloads": {},
    }
    for name in WORKLOADS:
        if name not in lbl["workloads"]:
            continue
        entry = {"window_cm1": list(WORKLOADS[name]["window_cm1"]), "label": WORKLOADS[name]["label"],
                 "lblrtm": lbl["workloads"][name]}
        for label, part in parts.items():
            if label != "lblrtm" and name in part["workloads"]:
                entry[label] = part["workloads"][name]
        # A3 across codes: tellurix's autodiff Jacobian column against
        # LBLRTM's central difference, both at the truth and on the same pixels.
        cross = {}
        lbl_arrays = arrays["lblrtm"]
        for label in parts:
            if label == "lblrtm":
                continue
            for s in lbl["workloads"][name]["species"]:
                ours = arrays[label].get(f"{name}_tellurix_jacobian_{s}")
                theirs = lbl_arrays.get(f"{name}_lblrtm_jacobian_central_{s}")
                if ours is None or theirs is None:
                    continue
                flux_ours = arrays[label][f"{name}_tellurix_flux_truth"]
                flux_theirs = lbl_arrays[f"{name}_lblrtm_flux_truth"]
                norm = float(np.sqrt(np.mean(theirs ** 2)))
                cross.setdefault(label, {})[s] = {
                    "jacobian_rel_rms_difference": float(np.sqrt(np.mean((ours - theirs) ** 2))) / norm,
                    "jacobian_correlation": float(np.corrcoef(ours, theirs)[0, 1]),
                    "jacobian_sum_ratio_tellurix_over_lblrtm": float(np.sum(ours) / np.sum(theirs)),
                    "flux_rms_difference": float(np.sqrt(np.mean((flux_ours - flux_theirs) ** 2))),
                    "flux_absorption_rel_rms_difference": float(
                        np.sqrt(np.mean((flux_ours - flux_theirs) ** 2))
                        / np.sqrt(np.mean((flux_theirs - np.median(flux_theirs)) ** 2))),
                }
        entry["A3_cross_code"] = cross
        entry["summary"] = summary_table(entry, parts)
        report["workloads"][name] = entry
    output = ROOT / args.output
    output.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"wrote {output}")
    for name, entry in report["workloads"].items():
        print(f"== {name}")
        for row in entry["summary"]:
            cells = "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
            print("  " + cells)


def summary_table(entry, parts) -> list[dict]:
    """Equivalent stages side by side, warm medians in seconds."""

    lbl = entry["lblrtm"]
    tape6 = lbl["L1_lblrtm_run"]["tape6_median"]
    modules = tape6["modules"]
    lines_s = sum(modules.get(m, {}).get("time_s", 0.0) for m in ("LINF4", "LBLF4", "HIRAC1", "XSECT"))
    run_s = lbl["L1_lblrtm_run"]["median_s"]
    post_s = lbl["L2_postprocess"]["total_median_s"]
    rows = [
        ("line data preparation (one-time)", lbl["L0_lnfl"]["uncoupled"]["median_s"],
         lambda t: t["T0_line_read"]["cold_index_build_s"] + t["T1_prepare"]["median_s"]),
        ("line-by-line optical depth", lines_s, lambda t: t["T2_opacity_kernel"]["median_s"]),
        ("continuum + layer merge / slant transmission",
         tape6["opdpth_other_incl_contnm_s"] + tape6["layer_merge_s"] + tape6["emission_init_s"],
         lambda t: t["T3_slant_transmission_given_opacity"]["median_s"]),
        ("instrument (LSF, pixels, continuum)", post_s, lambda t: t["T4_instrument"]["median_s"]),
        ("forward model, one evaluation", run_s + post_s, lambda t: t["T5_forward_live"]["median_s"]),
        ("forward model, precomputed opacity", run_s + post_s, lambda t: t["T5_forward_precomputed"]["median_s"]),
        ("gradient (central FD vs autodiff)", lbl["A2_fd_gradient"]["central"]["wall_s"],
         lambda t: t["T6_value_and_grad_live"]["median_s"]),
        ("gradient, precomputed opacity", lbl["A2_fd_gradient"]["central"]["wall_s"],
         lambda t: t["T7_value_and_grad_precomputed"]["median_s"]),
        ("fit, end to end", lbl.get("E2_lblrtm_fit", {}).get("wall_s"),
         lambda t: (t["T0_line_read"]["cold_index_build_s"] + t["T1_prepare"]["first_call_s"]
                    + t["continuum_load_s"] + t["E1_fit_cold"]["total_after_prepare_s"])),
    ]
    table = []
    for stage, lbl_s, ours in rows:
        row = {"stage": stage, "lblrtm_s": lbl_s}
        for label in parts:
            if label == "lblrtm" or label not in entry:
                continue
            try:
                value = ours(entry[label])
            except (KeyError, TypeError):
                continue
            row[f"{label}_s"] = value
            if lbl_s:
                row[f"speedup_{label}"] = lbl_s / value
        table.append(row)
    return table


COMPARABILITY = {
    "same": [
        "identical layers: LBLRTM IATM=0 input of tellurix's pressure, temperature and column per layer "
        "(LBLRTMRunConfig.user_layers), air-weighted mean layer pressure in both",
        "AER 3.9 line parameters (LNFL TAPE3 for LBLRTM, the per-molecule files for tellurix), "
        "lines within 25 cm-1 of the computed range",
        "Voigt lines with HITRAN-style air pressure shifts in both",
        "MT_CKD 4.3 H2O self and foreign continuum from the same netCDF file",
        "the same synthetic instrument: Gaussian LSF of constant velocity width, velocity shift of the "
        "pixel wavelengths, Simpson pixel integration, exp(Chebyshev) continuum",
    ],
    "different": [
        "LBLRTM computes on its own monochromatic grid (DV set per layer from the Doppler width, merged to "
        "the finest; ~1e-3 cm-1) and splits lines into a fine near-line part (HIRAC1) and a coarse far-wing "
        "part (LBLF4); tellurix evaluates every line on the R=45,000 x 4 constant-velocity grid (~0.028 cm-1)",
        "LBLRTM rejects lines below its DPTMIN/DPTFAC optical depth thresholds per layer and cuts each line "
        "at 25 cm-1 less a pedestal; tellurix keeps every line at full Voigt shape",
        "LBLRTM ICNTNM=1 adds every continuum it has (CO2, N2, O2, Rayleigh) to MT_CKD H2O; tellurix runs "
        "MT_CKD H2O only",
        "LBLRTM's output (TAPE12) is single precision; tellurix is float64 except the mixed-precision far "
        "wings",
        "line coupling is off in both main runs (LNFL NOCPL TAPE3); the coupled-TAPE3 LBLRTM run is a timing "
        "variant only",
        "LBLRTM is a single-threaded Fortran process per run (file I/O included); tellurix runs in one Python "
        "process on one GPU, or on the CPU with every core XLA can use",
        "the instrument stage after LBLRTM is numpy/scipy on LBLRTM's fine grid (TelFit-like); tellurix's is "
        "on its own model grid",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("lblrtm", "tellurix", "merge"), required=True)
    parser.add_argument("--platform", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--label", help="name of the tellurix partial (default: the platform)")
    parser.add_argument("--cpu-threads", type=int, help="restrict the process to this many CPUs")
    parser.add_argument("--workloads", default="W1,W2,W3")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--fit-repeats", type=int, default=3)
    parser.add_argument("--lnfl-repeats", type=int, default=3)
    parser.add_argument("--variant-repeats", type=int, default=5)
    parser.add_argument("--e2-workloads", default="W1,W2,W3")
    parser.add_argument("--e2-cap", type=int, default=300)
    parser.add_argument("--jacfwd-live-workloads", default="W1,W2,W3")
    parser.add_argument("--sweep-workloads", default="W1,W2,W3")
    parser.add_argument("--mixed-off-workloads", default="W1,W2,W3")
    parser.add_argument("--output", default="docs/lblrtm_tellurix_performance.json")
    args = parser.parse_args()
    for key in ("workloads", "e2_workloads", "jacfwd_live_workloads", "sweep_workloads", "mixed_off_workloads"):
        setattr(args, key, [w for w in getattr(args, key).split(",") if w])
    args.label = args.label or args.platform
    if args.part == "lblrtm":
        lblrtm_part(args)
    elif args.part == "tellurix":
        tellurix_part(args)
    else:
        merge_part(args)


if __name__ == "__main__":
    main()
