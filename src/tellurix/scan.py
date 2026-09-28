"""Addressing and storage for species scans.

A scan asks which absorbers are worth fitting in a window. It costs about six
minutes, and it is the reason a hand-written species list can no longer hide an
absorber the way OCS and O3 hid in the 4.9 um solar window (see
`docs/solar_fts_residual.md`). That cost is affordable only because a scan
depends on the **window and the atmosphere alone** -- not on any spectrum, any
observation date, or any fitted parameter -- so a single entry serves every file
that looks through that window, every refit, and any later re-analysis.

The addressing lives here rather than in the script because the scanner and its
consumers must agree on when two scans are the same scan. Anything that can
change the answer goes into `ScanIdentity`, including the profile's *contents*
rather than its path: a profile rebuilt with a different AFGL model keeps its
filename and would otherwise silently reuse the old ranking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["ScanIdentity", "cached_scan", "load_scan", "read_scan",
           "save_scan", "scan_window"]


@dataclass(frozen=True)
class ScanIdentity:
    """Everything a scan's answer depends on."""

    window_cm1: tuple[float, float]
    profile_sha256: str
    threshold: float
    line_budget: float
    margin_cm1: float
    fwhm_cm1: float
    samples_per_resolution: float
    line_files: str = "aer_v_3.9"

    def __post_init__(self) -> None:
        v1, v2 = (float(v) for v in self.window_cm1)
        if not 0.0 < v1 < v2:
            raise ValueError("require 0 < v1 < v2")
        object.__setattr__(self, "window_cm1", (v1, v2))
        if len(self.profile_sha256) != 64:
            raise ValueError("profile_sha256 must be a full sha256 hex digest")

    @classmethod
    def for_profile(cls, profile_path: Path, window_cm1, **kwargs: Any) -> "ScanIdentity":
        digest = hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()
        return cls(window_cm1=tuple(window_cm1), profile_sha256=digest, **kwargs)

    def as_dict(self) -> dict:
        row = asdict(self)
        row["window_cm1"] = list(self.window_cm1)
        return row

    @property
    def digest(self) -> str:
        """A short, stable name for this scan.

        Twelve hex characters, which is far more than the couple of hundred
        windows an atlas needs; `read_scan` re-checks the stored identity in
        full anyway, so the digest only has to make collisions rare, not
        impossible.
        """
        blob = json.dumps(self.as_dict(), sort_keys=True).encode("ascii")
        return hashlib.sha256(blob).hexdigest()[:12]

    def path(self, cache_dir: Path) -> Path:
        v1, v2 = self.window_cm1
        return Path(cache_dir) / f"scan_{v1:.0f}_{v2:.0f}_{self.digest}.json"


def read_scan(path: Path) -> dict:
    """Read one entry, refusing anything that is not the scan it is filed as.

    An entry is a plain JSON file and may be copied, edited or truncated by
    hand, so the name is checked against the identity the file itself records.
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = report.get("identity")
    if stored is None:
        raise ValueError(f"{path} predates cache identities and cannot be trusted")
    identity = ScanIdentity(**{**stored, "window_cm1": tuple(stored["window_cm1"])})
    if Path(path).name != identity.path(Path(path).parent).name:
        raise ValueError(f"{path} is filed under a name its identity does not produce")
    if "species" not in report:
        raise ValueError(f"{path} carries no species list")
    return report


def load_scan(cache_dir: Path, identity: ScanIdentity) -> dict | None:
    """The cached scan for this identity, or None if it has not been run."""
    path = identity.path(cache_dir)
    return read_scan(path) if path.exists() else None


def save_scan(cache_dir: Path, identity: ScanIdentity, report: dict) -> Path:
    path = identity.path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = dict(report, identity=identity.as_dict())
    # Written whole rather than appended to, so a killed scan leaves no entry
    # that a later run would trust.
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def scan_window(profile, line_root: Path, identity: ScanIdentity, *,
                zenith_angle_deg: float = 0.0, layer_chunk_size: int | None = None,
                progress=None) -> dict:
    """Rank every absorber AER ships against one window and one atmosphere.

    The expensive half of a scan, and the reason a hand-written species list can
    no longer hide an absorber. `zenith_angle_deg` scales every species alike
    and so cannot change the ranking; it is kept out of `ScanIdentity` for that
    reason, and recorded only so the printed optical depths mean something.
    """
    import numpy as np

    # Imported here rather than at module scope: the cache is useful to a
    # consumer that only reads entries, and that consumer should not have to
    # pay for ExoJAX.
    import jax

    from .aer import (AER_MOLECULE_IDS, AERLineDatabase, line_optical_depth_bound,
                      select_significant_lines)
    from .exojax_backend import ExoJAXOpacityBackend
    from .model import constant_velocity_grid, trim_wavenumber_grid

    say = progress if progress is not None else (lambda *a, **k: None)
    v1, v2 = identity.window_cm1
    centre = 0.5 * (v1 + v2)
    grid = constant_velocity_grid(
        1.0e7 / v2, 1.0e7 / v1,
        resolving_power=centre / identity.fwhm_cm1,
        samples_per_resolution=identity.samples_per_resolution,
        margin_cm1=identity.margin_cm1)
    grid = trim_wavenumber_grid(grid, v1, v2, 5.0)
    column = np.asarray(profile.air_column_cm2)
    secant = 1.0 / np.cos(np.radians(zenith_angle_deg))

    found, absent, missing, reject_by_bound = [], [], [], []
    for name, molecule_id in sorted(AER_MOLECULE_IDS.items(), key=lambda kv: kv[1]):
        directory = next((d for d in Path(line_root).glob(f"{molecule_id:02d}_*")
                          if d.is_dir()), None)
        if directory is None:
            continue
        if name not in profile.vmr:
            missing.append(name)
            continue
        try:
            database = AERLineDatabase(directory / directory.name, name,
                                       (v1, v2), margin_cm1=identity.margin_cm1)
        except ValueError as exc:
            if str(exc).startswith(f"no {name} lines found"):
                absent.append(name)
                continue
            raise
        # Reject analytically before touching the kernel. The sum of the
        # per-line bounds is an upper bound on the peak optical depth anywhere
        # -- it assumes every line peaks at the same wavenumber, which they do
        # not -- so a species whose bound is below the threshold provably
        # cannot reach it. This is numpy, not XLA: it costs nothing next to a
        # compile, and it removes most of the periodic table from every window.
        # It is a *reject* test only. The bound's conservatism spans 1.3x to
        # 1.05e8x across species, so it cannot be used to rank the survivors.
        bound = float(np.sum(line_optical_depth_bound(database, profile, name)))
        if bound < identity.threshold:
            reject_by_bound.append({"species": name, "molecule": molecule_id,
                                    "lines": int(database.nu_lines.size),
                                    "optical_depth_bound": bound})
            say(f"  {name:8s} {database.nu_lines.size:7d} lines  "
                f"bound {bound:.3e}  -- below threshold, not evaluated")
            continue

        # A ranking does not need every line, and some species carry tens of
        # thousands: O3 has 38,011 at 2030-2060, whose dense line-by-grid
        # intermediates ask for 31 GB. select_significant_lines bounds the
        # optical-depth error by its budget, which is two orders below the
        # threshold being tested, so it cannot change which side of the cut a
        # species falls on.
        full = int(database.nu_lines.size)
        database = select_significant_lines(database, profile, name,
                                            optical_depth_budget=identity.line_budget)
        opacity = ExoJAXOpacityBackend.prepare(
            {name: database}, grid, methods="direct_sparse",
            temperature_range_k=(float(np.min(profile.temperature_k)),
                                 float(np.max(profile.temperature_k))),
            maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
            vectorize_layers=True, mixed_precision=True, pressure_shift=True,
            layer_chunk_size=layer_chunk_size or None)
        # Never call this eagerly. An uncompiled kernel dispatches operation by
        # operation and materialises the dense line-by-grid offset matrix, which
        # asks for 16 GB on a 20,000-line species like O3; jit fuses it away.
        # The backend takes the self-broadening partial pressure per species,
        # keyed like the cross sections it returns.
        pressure = np.asarray(profile.pressure_layer_bar)
        evaluate = jax.jit(lambda t, p_, s: opacity.cross_sections(t, p_, {name: s})[name])
        cross_section = np.asarray(evaluate(
            np.asarray(profile.temperature_k), pressure,
            pressure * np.asarray(profile.vmr[name])))
        # Optical depth is the cross section times the species column, summed
        # over layers -- the profile's own column, which is the whole point.
        tau = np.sum(cross_section * (np.asarray(profile.vmr[name]) * column)[:, None], axis=0)
        found.append({"species": name, "molecule": molecule_id,
                      "lines": full, "lines_kept": int(database.nu_lines.size),
                      "peak_optical_depth": float(np.max(tau) * secant),
                      "median_optical_depth": float(np.median(tau) * secant)})
        say(f"  {name:8s} {full:7d} lines ({database.nu_lines.size:6d} kept)  "
            f"peak tau {found[-1]['peak_optical_depth']:.3e}")

    found.sort(key=lambda row: -row["peak_optical_depth"])
    keep = [row for row in found if row["peak_optical_depth"] >= identity.threshold]
    reject = [row for row in found if row["peak_optical_depth"] < identity.threshold]
    reject_by_bound.sort(key=lambda row: -row["optical_depth_bound"])
    return {
        "window_cm1": [v1, v2],
        "zenith_angle_deg": zenith_angle_deg,
        "threshold_peak_optical_depth": identity.threshold,
        "species": [row["species"] for row in keep],
        "fit": keep,
        "rejected": reject,
        "rejected_by_bound": reject_by_bound,
        "no_lines_in_window": absent,
        "not_in_profile": missing,
        "evaluated": len(found),
        "rejected_without_evaluating": len(reject_by_bound),
        "headroom": (None if not reject else
                     identity.threshold / reject[0]["peak_optical_depth"]),
    }


def cached_scan(cache_dir: Path, identity: ScanIdentity, profile, line_root: Path,
                *, refresh: bool = False, **kwargs) -> dict:
    """The scan for this identity, computed only if it is not already stored.

    Note that an entry is **not** byte-reproducible: the GPU kernel's reduction
    order varies between runs, so a re-scan moves an optical depth in its last
    ulp (measured: 0.07874137491448768 against ...67). That is 1e-16 against a
    1e-3 threshold, so the verdict is reproducible even though the file is not
    -- do not diff two entries and conclude something changed.
    """
    if not refresh:
        cached = load_scan(cache_dir, identity)
        if cached is not None:
            return cached
    report = scan_window(profile, line_root, identity, **kwargs)
    save_scan(cache_dir, identity, report)
    return report
