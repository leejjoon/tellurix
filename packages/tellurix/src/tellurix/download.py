"""Download the external AER spectroscopy data used by tellurix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen

import numpy as np


MT_CKD_URL = (
    "https://raw.githubusercontent.com/AER-RC/MT_CKD_H2O/4.3/"
    "data/absco-ref_wv-mt-ckd.nc"
)
MT_CKD_SHA256 = "69944eb8b045c268e2daeb2cddf536b99f9e9efe7e91c425067b46a5b287ddb3"
AER_LINES_URL = "https://zenodo.org/records/18881607/files/aer_v_3.9.tgz?download=1"
AER_LINES_SHA256 = "3a55d5deb9430894ce61e1a23475ad1ed7ef01f0a5c527ee6f08d9e5676a0e4e"
AER_LICENSE_URL = (
    "https://raw.githubusercontent.com/AER-RC/AER_Line_File/"
    "4de5983ac4d909f674d2ee502ac667c27f79942f/LICENSE.md"
)
AER_LICENSE_SHA256 = "eed60bbf029b81fd0abd15ef48f0fb65176906a784a34f9eb7652bbc11646ddd"
HITRAN_LBL_URL = "https://hitran.org/lbl/api"
# HITRANonline's isotopologue ids for H2O 161, 181, 171, 162, 182, 172; AER v3.9
# carries these six and not 262.
HITRAN_H2O_ISOTOPOLOGUE_IDS = (1, 2, 3, 4, 5, 6)
# What HITRANonline served when the table was introduced (2026-10). The site
# says which edition it serves; a later fetch must update this by hand.
HITRAN_EDITION = "HITRAN2024"


def default_data_directory() -> Path:
    """Return the configured per-user data directory.

    The package was called ``jax-telluric`` until 0.2, so the old environment
    variable and the old directory are still honoured -- but only when they
    point at data that is actually there. A user who downloaded gigabytes of AER
    line files under the old name should not have to fetch them again, and
    should not silently get an empty directory either.
    """

    configured = os.environ.get("TELLURIX_DATA") or os.environ.get("JAX_TELLURIC_DATA")
    if configured:
        return Path(configured).expanduser()
    xdg_data = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data).expanduser() if xdg_data else Path.home() / ".local/share"
    current = base / "tellurix"
    if not current.exists():
        legacy = base / "jax-telluric"
        if legacy.is_dir() and any(legacy.iterdir()):
            return legacy
    return current


@dataclass(frozen=True)
class DataPaths:
    """Where the AER line files and the MT_CKD coefficients are.

    Two layouts exist and neither is wrong: ``tellurix-download-data`` fills a
    per-user directory (:meth:`downloaded`), and ``scripts/bootstrap_lblrtm.sh``
    builds LBLRTM inside a checkout and leaves both files where LBLRTM reads
    them (:meth:`bootstrapped`). Code that fits takes one of these rather than a
    repository root, so it runs the same from an installed package.
    """

    line_root: Path
    mt_ckd: Path
    # HITRAN's H2O air pressure shifts (tellurix.hitran), which the fits use in
    # place of AER's unless told otherwise. None when the layout has none.
    h2o_pressure_shift: Path | None = None

    @classmethod
    def downloaded(cls, directory: str | Path | None = None) -> "DataPaths":
        root = Path(directory).expanduser() if directory else default_data_directory()
        return cls(root / "aer_v_3.9/line_files_By_Molecule",
                   root / "mt_ckd/absco-ref_wv-mt-ckd.nc",
                   root / "hitran/h2o_pressure_shift.npz")

    @classmethod
    def bootstrapped(cls, checkout: str | Path) -> "DataPaths":
        root = Path(checkout)
        return cls(root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule",
                   root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc",
                   root / "data/databases/hitran/h2o_pressure_shift.npz")

    def pressure_shift_table(self, species: str, source: str) -> Path | None:
        """The table a fit hands ``AERLineDatabase`` for this species, or None for AER's own.

        ``source`` is ``"hitran"`` or ``"aer"``; only H2O has a table.
        """

        if source not in ("hitran", "aer"):
            raise ValueError(f"unknown pressure-shift source {source!r}; use 'hitran' or 'aer'")
        if source == "aer" or species.upper() != "H2O":
            return None
        if self.h2o_pressure_shift is None or not Path(self.h2o_pressure_shift).is_file():
            raise FileNotFoundError(
                f"no HITRAN H2O pressure-shift table at {self.h2o_pressure_shift}; run "
                "`tellurix-download-data hitran-h2o` (or pass the AER shifts explicitly)")
        return Path(self.h2o_pressure_shift)

    @property
    def line_coupling(self) -> Path:
        """AER's first-order line-coupling records, beside the per-molecule files."""

        return self.line_root.parent / "lncpl_lines"

    def line_file(self, species: str, molecule_id: int) -> Path:
        """AER's per-molecule file, named ``NN_SPECIES`` inside a directory of that name."""

        name = f"{molecule_id:02d}_{species}"
        return self.line_root / name / name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        print(f"Using verified {destination}")
        return destination

    partial = destination.with_name(destination.name + ".part")
    if partial.is_file() and _sha256(partial) == expected_sha256:
        partial.replace(destination)
        print(f"Using verified {destination}")
        return destination
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "tellurix-data-downloader/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(f"Downloading {url}")
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        if not append:
            offset = 0
        content_length = response.headers.get("Content-Length")
        total = offset + int(content_length) if content_length is not None else None
        mode = "ab" if append else "wb"
        downloaded = offset
        next_report = downloaded + 32 * 1024 * 1024
        with partial.open(mode) as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    if total:
                        print(f"  {downloaded / 1024**2:.0f}/{total / 1024**2:.0f} MiB")
                    else:
                        print(f"  {downloaded / 1024**2:.0f} MiB")
                    next_report += 32 * 1024 * 1024
    if total is not None and downloaded != total:
        raise RuntimeError(
            f"incomplete download from {url}: expected {total} bytes, got {downloaded}"
        )
    actual_sha256 = _sha256(partial)
    if actual_sha256 != expected_sha256:
        partial.unlink()
        raise RuntimeError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual_sha256}"
        )
    partial.replace(destination)
    print(f"Verified {destination}")
    return destination


def download_mt_ckd(directory: str | Path | None = None) -> Path:
    """Download the official MT_CKD 4.3 H2O coefficient file."""

    root = default_data_directory() if directory is None else Path(directory).expanduser()
    destination = root / "mt_ckd" / "absco-ref_wv-mt-ckd.nc"
    return _download(MT_CKD_URL, destination, MT_CKD_SHA256)


def _extract_aer_archive(archive: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aer-lines-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != "aer_v_3.9"
                    or member.issym()
                    or member.islnk()
                    or not (member.isdir() or member.isfile())
                ):
                    raise RuntimeError(f"unsafe path or file type in AER archive: {member.name}")
                target = staging.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read AER archive member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        extracted = staging / "aer_v_3.9"
        required = extracted / "line_files_By_Molecule/01_H2O/01_H2O"
        if not required.is_file():
            raise RuntimeError("AER archive does not contain the expected H2O line file")
        extracted.replace(destination)


def download_aer_lines(directory: str | Path | None = None) -> Path:
    """Download and safely extract the official AER line file 3.9 archive."""

    root = default_data_directory() if directory is None else Path(directory).expanduser()
    destination = root / "aer_v_3.9"
    required = destination / "line_files_By_Molecule/01_H2O/01_H2O"
    if required.is_file():
        _download(AER_LICENSE_URL, destination / "LICENSE.md", AER_LICENSE_SHA256)
        print(f"Using existing {destination}")
        return destination
    if destination.exists():
        raise RuntimeError(
            f"incomplete AER line directory exists at {destination}; move or remove it and retry"
        )
    archive = _download(
        AER_LINES_URL, root / "downloads/aer_v_3.9.tgz", AER_LINES_SHA256
    )
    print(f"Extracting {archive}")
    _extract_aer_archive(archive, destination)
    _download(AER_LICENSE_URL, destination / "LICENSE.md", AER_LICENSE_SHA256)
    print(f"Installed AER line files at {destination}")
    return destination


def _fetch_hitran_par(isotopologue_ids, lower_cm1: float, upper_cm1: float) -> bytes:
    """One range of HITRANonline .par records.

    The service's Content-Length counts each record one byte longer than it
    sends (CRLF against LF), so a complete reply looks short by exactly one byte
    per record. That shortfall is accepted and any other is not.
    """

    from http.client import IncompleteRead

    ids = ",".join(str(i) for i in isotopologue_ids)
    url = f"{HITRAN_LBL_URL}?iso_ids_list={ids}&numin={lower_cm1:g}&numax={upper_cm1:g}"
    request = Request(url, headers={"User-Agent": "tellurix-data-downloader/0.1"})
    with urlopen(request, timeout=600) as response:
        expected = response.headers.get("Content-Length")
        try:
            body = response.read()
        except IncompleteRead as exc:
            body = exc.partial
    if expected is not None:
        records = body.count(b"\n")
        if len(body) not in (int(expected), int(expected) - records):
            raise RuntimeError(f"incomplete reply from {url}: {len(body)} of {expected} bytes")
    if any(len(line) != 160 for line in body.splitlines()):
        raise RuntimeError(f"{url} returned something other than 160-character .par records")
    return body


def download_hitran_h2o(directory: str | Path | None = None, *, upper_cm1: float = 42000.0,
                        step_cm1: float = 2000.0) -> Path:
    """Fetch HITRAN's H2O lines and build the pressure-shift table the fits use.

    The raw records are kept beside the table, by range, so a later field can be
    read from them without fetching again.
    """

    import datetime
    from .hitran import PressureShiftTable, parse_par

    root = default_data_directory() if directory is None else Path(directory).expanduser()
    destination = root / "hitran" / "h2o_pressure_shift.npz"
    raw_directory = root / "hitran" / "h2o_par"
    raw_directory.mkdir(parents=True, exist_ok=True)
    edges = list(np.arange(0.0, upper_cm1, step_cm1)) + [upper_cm1]
    parts = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        path = raw_directory / f"H2O_{lower:05.0f}_{upper:05.0f}.par"
        if not path.is_file():
            print(f"Fetching HITRAN H2O {lower:.0f}-{upper:.0f} cm-1")
            partial = path.with_name(path.name + ".part")
            partial.write_bytes(_fetch_hitran_par(HITRAN_H2O_ISOTOPOLOGUE_IDS, lower, upper))
            partial.replace(path)
        parts.append(path)
    lines = [line for path in parts for line in path.read_bytes().splitlines()]
    arrays = parse_par(lines, molecule_id=1)
    # Adjacent ranges share their boundary; keep one copy of a line on it.
    key = np.rec.fromarrays([arrays["isotope"], arrays["wavenumber_cm1"], arrays["elower_cm1"]])
    _, unique = np.unique(key, return_index=True)
    unique = np.sort(unique)
    table = PressureShiftTable(
        molecule_id=1, **{name: values[unique] for name, values in arrays.items()},
        provenance={"source": HITRAN_LBL_URL, "edition": HITRAN_EDITION,
                    "isotopologue_ids": list(HITRAN_H2O_ISOTOPOLOGUE_IDS),
                    "fetched": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                    "range_cm1": [0.0, upper_cm1], "lines": int(unique.size),
                    "par_sha256": {path.name: _sha256(path) for path in parts}})
    table.save(destination)
    print(f"HITRAN H2O pressure shifts: {unique.size} lines, sha256 {_sha256(destination)}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download external spectroscopy data used by tellurix"
    )
    parser.add_argument("dataset", choices=("mt-ckd", "aer-lines", "hitran-h2o", "all"))
    parser.add_argument(
        "--output",
        type=Path,
        default=default_data_directory(),
        help="data root (default: TELLURIX_DATA, the legacy JAX_TELLURIC_DATA, or the user data directory)",
    )
    args = parser.parse_args()
    if args.dataset in ("mt-ckd", "all"):
        print(f"MT_CKD: {download_mt_ckd(args.output)}")
    if args.dataset in ("aer-lines", "all"):
        print(f"AER lines: {download_aer_lines(args.output)}")
    if args.dataset in ("hitran-h2o", "all"):
        print(f"HITRAN H2O pressure shifts: {download_hitran_h2o(args.output)}")


if __name__ == "__main__":
    main()
