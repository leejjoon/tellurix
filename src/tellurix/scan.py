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

__all__ = ["ScanIdentity", "load_scan", "save_scan", "read_scan"]


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
