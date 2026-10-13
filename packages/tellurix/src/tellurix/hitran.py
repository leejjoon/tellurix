"""HITRAN line parameters that replace AER's, one field at a time.

AER v3.9 is LBLRTM's line list and stays the base: positions, strengths,
widths and coupling all come from it. HITRAN is consulted for single fields
where the observations say AER is wrong. The first is the H2O air pressure
shift: AER gives 24 of 350 isolated 0.8-1.1 um water lines a *positive* shift
that HITRAN gives as negative, and those lines sit 100-300 m/s off in the
niratl solar atlas. Fitted across 34 windows, HITRAN's shifts want a pressure
scale of 0.98 and AER's 0.67; across all 148 niratl water windows
0.91 and 0.63 (docs/solar_fit_plan.md §4q).

The table is built from HITRANonline's ``.par`` records by
``tellurix-download-data hitran-h2o``. HITRANonline serves the current edition
and corrects it in place, so the download cannot be pinned by checksum the way
AER and MT_CKD are; the table carries the edition and fetch date it was built
from, and a run records the table's SHA-256 instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


# AER and HITRAN record the same line to within a few 1e-4 cm-1 (0.38 mK rms
# for isolated 0.8-1.1 um water lines); the lower-state energy tells apart the
# rare pair of lines closer than that.
MATCH_TOLERANCE_CM1 = 0.003
ELOWER_TOLERANCE_CM1 = 0.5

_TABLES: dict = {}


def _isotope_code(character: str) -> int:
    """HITRAN's one-character local isotopologue number: 1-9, then 0 for 10, A for 11."""

    if character.isdigit():
        return int(character) or 10
    return 11 + ord(character.upper()) - ord("A")


def parse_par(lines, molecule_id: int) -> dict:
    """Isotopologue, position, lower-state energy and air shift of one molecule's records."""

    isotope, wavenumber, elower, delta = [], [], [], []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("ascii", errors="replace")
        line = line.rstrip("\r\n")
        if len(line) < 67 or not line[:2].strip() or int(line[:2]) != molecule_id:
            continue
        isotope.append(_isotope_code(line[2]))
        wavenumber.append(float(line[3:15]))
        elower.append(float(line[45:55]))
        delta.append(float(line[59:67]))
    order = np.argsort(wavenumber, kind="stable")
    return {
        "isotope": np.asarray(isotope, dtype=np.int16)[order],
        "wavenumber_cm1": np.asarray(wavenumber)[order],
        "elower_cm1": np.asarray(elower)[order],
        "delta_air": np.asarray(delta)[order],
    }


@dataclass(frozen=True)
class PressureShiftTable:
    """HITRAN air pressure shifts of one molecule, to replace AER's line by line."""

    molecule_id: int
    isotope: np.ndarray
    wavenumber_cm1: np.ndarray
    elower_cm1: np.ndarray
    delta_air: np.ndarray
    provenance: dict

    def __post_init__(self) -> None:
        arrays = (self.isotope, self.wavenumber_cm1, self.elower_cm1, self.delta_air)
        if len({np.shape(a) for a in arrays}) != 1 or np.ndim(self.wavenumber_cm1) != 1:
            raise ValueError("a pressure-shift table needs four arrays of one length")
        if np.any(np.diff(self.wavenumber_cm1) < 0.0):
            raise ValueError("a pressure-shift table must be sorted by wavenumber")

    @classmethod
    def load(cls, path: str | Path) -> "PressureShiftTable":
        """Read a table written by :meth:`save`, cached by path, size and mtime."""

        path = Path(path)
        status = path.stat()
        key = (str(path.resolve()), status.st_size, status.st_mtime_ns)
        if key not in _TABLES:
            with np.load(path, allow_pickle=False) as data:
                _TABLES[key] = cls(
                    molecule_id=int(data["molecule_id"]), isotope=data["isotope"],
                    wavenumber_cm1=data["wavenumber_cm1"], elower_cm1=data["elower_cm1"],
                    delta_air=data["delta_air"], provenance=json.loads(str(data["provenance"])))
        return _TABLES[key]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends .npz to a name that lacks it; write where we were told.
        with path.open("wb") as stream:
            np.savez_compressed(
                stream, molecule_id=self.molecule_id, isotope=self.isotope,
                wavenumber_cm1=self.wavenumber_cm1, elower_cm1=self.elower_cm1,
                delta_air=self.delta_air, provenance=json.dumps(self.provenance, sort_keys=True))
        return path

    def replace(self, isotope, wavenumber_cm1, elower_cm1, delta_air):
        """AER's shifts with HITRAN's wherever the same line is in both.

        A line matches when its isotopologue agrees, its position is within
        ``MATCH_TOLERANCE_CM1`` and its lower-state energy within
        ``ELOWER_TOLERANCE_CM1``; the closest in both, each in units of its
        tolerance, wins -- HITRAN lists doublets at one position that only the
        lower-state energy separates. A line HITRAN does not have keeps AER's
        value. Returns the shifts and the mask of lines that were replaced.
        """

        isotope = np.asarray(isotope)
        nu = np.asarray(wavenumber_cm1, dtype=float)
        elower = np.asarray(elower_cm1, dtype=float)
        result = np.array(delta_air, dtype=float, copy=True)
        best = np.full(nu.size, np.inf)
        source = np.full(nu.size, -1)
        start = np.searchsorted(self.wavenumber_cm1, nu - MATCH_TOLERANCE_CM1, side="left")
        stop = np.searchsorted(self.wavenumber_cm1, nu + MATCH_TOLERANCE_CM1, side="right")
        # Water has at most a handful of lines inside the tolerance; walk them
        # as a vector rather than line by line.
        for step in range(int(np.max(stop - start, initial=0))):
            candidate = start + step
            inside = candidate < stop
            index = np.where(inside, candidate, 0)
            energy = np.abs(self.elower_cm1[index] - elower)
            distance = (np.abs(self.wavenumber_cm1[index] - nu) / MATCH_TOLERANCE_CM1
                        + energy / ELOWER_TOLERANCE_CM1)
            good = (inside & (self.isotope[index] == isotope)
                    & (energy <= ELOWER_TOLERANCE_CM1) & (distance < best))
            best = np.where(good, distance, best)
            source = np.where(good, index, source)
        replaced = source >= 0
        result[replaced] = self.delta_air[source[replaced]]
        return result, replaced


def pressure_shift_provenance(source: str, table: str | Path | None) -> dict:
    """What a run record says about its H2O pressure shifts.

    ``table`` is the one the H2O lines were built with (``AERLineDatabase.
    pressure_shift_table``), None when they were not built or kept AER's. A
    record without ``h2o_pressure_shift`` was made before the choice existed and
    used AER's, which is what a rebuild of it must use.
    """

    record = {"h2o_pressure_shift": source}
    if table is not None:
        import hashlib

        loaded = PressureShiftTable.load(table)
        record.update({
            "h2o_pressure_shift_edition": loaded.provenance.get("edition"),
            "h2o_pressure_shift_fetched": loaded.provenance.get("fetched"),
            "h2o_pressure_shift_sha256": hashlib.sha256(Path(table).read_bytes()).hexdigest()})
    return record
