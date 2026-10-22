"""Per-line strength factors measured from observed spectra, applied at read time.

The line list is AER v3.9, and where its strengths are wrong HITRAN2024 agrees with
it (docs/line_intensity_corrections.md): the factors have to come from the
observations. A feature's observed/model optical-depth ratio, repeated across
spectra, instruments and air masses, is a strength error of its lines
(``scripts/vet_baker_line_corrections.py``); ``scripts/build_line_strength_table.py``
turns the ratios into one factor per member line, and :class:`AERLineDatabase`
multiplies ``line_strength_ref_original`` by it. A blend's factor goes to every
line in its cluster: these spectra cannot split a blend.

Off by default. A run records the table's SHA-256 and what it was built from
(:func:`line_strength_provenance`); a record without the key used AER's strengths.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

# The table is built from the same AER records it is matched against, so a line
# matches itself to the precision AER prints (1e-6 cm-1) and its lower-state
# energy (1e-4 cm-1); the tolerances only absorb float round trips.
MATCH_TOLERANCE_CM1 = 1e-5
ELOWER_TOLERANCE_CM1 = 1e-3

_TABLES: dict = {}


@dataclass(frozen=True)
class LineStrengthTable:
    """One strength factor per line, for any number of molecules, sorted by wavenumber."""

    molecule_id: np.ndarray
    isotope: np.ndarray
    wavenumber_cm1: np.ndarray
    elower_cm1: np.ndarray
    factor: np.ndarray
    provenance: dict

    def __post_init__(self) -> None:
        arrays = (self.molecule_id, self.isotope, self.wavenumber_cm1, self.elower_cm1, self.factor)
        if len({np.shape(a) for a in arrays}) != 1 or np.ndim(self.wavenumber_cm1) != 1:
            raise ValueError("a line-strength table needs five arrays of one length")
        if np.any(np.diff(self.wavenumber_cm1) < 0.0):
            raise ValueError("a line-strength table must be sorted by wavenumber")
        if np.any(~np.isfinite(self.factor)) or np.any(np.asarray(self.factor) <= 0.0):
            raise ValueError("strength factors must be finite and positive")

    @classmethod
    def load(cls, path: str | Path) -> "LineStrengthTable":
        """Read a table written by :meth:`save`, cached by path, size and mtime."""

        path = Path(path)
        status = path.stat()
        key = (str(path.resolve()), status.st_size, status.st_mtime_ns)
        if key not in _TABLES:
            with np.load(path, allow_pickle=False) as data:
                _TABLES[key] = cls(
                    molecule_id=data["molecule_id"], isotope=data["isotope"],
                    wavenumber_cm1=data["wavenumber_cm1"], elower_cm1=data["elower_cm1"],
                    factor=data["factor"], provenance=json.loads(str(data["provenance"])))
        return _TABLES[key]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends .npz to a name that lacks it; write where we were told.
        with path.open("wb") as stream:
            np.savez_compressed(
                stream, molecule_id=self.molecule_id, isotope=self.isotope,
                wavenumber_cm1=self.wavenumber_cm1, elower_cm1=self.elower_cm1,
                factor=self.factor, provenance=json.dumps(self.provenance, sort_keys=True))
        return path

    def factors(self, molecule_id: int, isotope, wavenumber_cm1, elower_cm1):
        """Each line's factor, 1 where the table has no entry, and the mask of lines found."""

        isotope = np.asarray(isotope)
        nu = np.asarray(wavenumber_cm1, dtype=float)
        elower = np.asarray(elower_cm1, dtype=float)
        result = np.ones(nu.size)
        found = np.zeros(nu.size, dtype=bool)
        start = np.searchsorted(self.wavenumber_cm1, nu - MATCH_TOLERANCE_CM1, side="left")
        stop = np.searchsorted(self.wavenumber_cm1, nu + MATCH_TOLERANCE_CM1, side="right")
        # Lines sharing a position to 1e-5 cm-1 are a handful at most; walk them as
        # a vector. Isotopologue and lower-state energy tell such a pair apart.
        for step in range(int(np.max(stop - start, initial=0))):
            candidate = start + step
            inside = candidate < stop
            index = np.where(inside, candidate, 0)
            good = (inside & ~found & (self.molecule_id[index] == molecule_id)
                    & (self.isotope[index] == isotope)
                    & (np.abs(self.elower_cm1[index] - elower) <= ELOWER_TOLERANCE_CM1))
            result = np.where(good, self.factor[index], result)
            found |= good
        return result, found


def line_strength_provenance(table: str | Path | None) -> dict:
    """What a run record says about its line strengths: nothing when it used AER's.

    A record without ``line_strength_table`` used AER's strengths, which is what a
    rebuild of it must use. The build description is a JSON string, because a
    batch record keeps its physics as HDF5 attributes, which hold no dict.
    """

    if table is None:
        return {}
    loaded = LineStrengthTable.load(table)
    return {"line_strength_table": Path(table).name,
            "line_strength_table_sha256": hashlib.sha256(Path(table).read_bytes()).hexdigest(),
            "line_strength_built_from": json.dumps(loaded.provenance.get("built_from"), sort_keys=True)}


def recorded_line_strength_table(physics, directory: str | Path) -> Path | None:
    """The table a record used, found by name in ``directory`` and held to its SHA-256.

    None for a record that used AER's strengths. A table rebuilt since the run --
    same name, other factors -- is refused rather than silently used.
    """

    name = physics.get("line_strength_table")
    if not name:
        return None
    path = Path(directory) / name
    if not path.exists():
        raise FileNotFoundError(f"the record used line-strength table {name}, not found in {directory}")
    recorded = physics.get("line_strength_table_sha256")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if recorded is not None and digest != recorded:
        raise ValueError(f"{path} is not the table the fit used: {digest[:12]} against {recorded[:12]}")
    return path
