#!/usr/bin/env python
"""Pack, fetch and check the fitted results that live outside git.

The run records, night calibrations and species-scan cache under
``data/corrected/``, ``data/calibration/`` and ``data/scans/`` are products,
not source: they are archived on Zenodo and fetched into place, at the same
paths the scripts and docs already use. What stays in git is
``data/results_manifest.json`` -- the archive's own sha256, where to download
it, and the sha256 of every file in it -- so a fetched tree is checked file by
file, and a result that changed locally is visible against the published one.

    uv run python scripts/results_archive.py fetch            # download, verify, unpack
    uv run python scripts/results_archive.py fetch --archive tellurix-results.tar.gz
    uv run python scripts/results_archive.py check            # compare data/ with the manifest
    uv run python scripts/results_archive.py pack --output dist/tellurix-results.tar.gz

``pack`` takes the files the manifest lists, or with ``--from-git`` the files
git tracks under those directories, which is how the first archive was made.
The tarball is deterministic -- sorted entries, zero mtimes, no owners -- so
packing the same files twice gives the same sha256.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/results_manifest.json"
DIRECTORIES = ("data/corrected", "data/calibration", "data/scans")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked_files() -> list[str]:
    listed = subprocess.run(["git", "ls-files", "-z", *DIRECTORIES], cwd=ROOT,
                            capture_output=True, check=True).stdout.decode()
    return sorted(name for name in listed.split("\0") if name)


def pack(output: Path, files: list[str], url: str, record: str) -> dict:
    """Write a deterministic tar.gz of ``files`` and return its manifest."""

    for name in files:
        if not (ROOT / name).is_file():
            raise SystemExit(f"{name} is listed but not on disk")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in files:
            data = (ROOT / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 and no filename keep the gzip header itself reproducible.
    with open(output, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                                   mtime=0, compresslevel=9) as compressed:
        compressed.write(buffer.getvalue())
    return {
        "description": "Fitted results kept outside git; see scripts/results_archive.py.",
        "archive": output.name,
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "url": url,
        "zenodo_record": record,
        "directories": list(DIRECTORIES),
        "files": {name: sha256(ROOT / name) for name in files},
    }


def _safe_members(tar: tarfile.TarFile, expected: set[str]):
    for member in tar.getmembers():
        path = PurePosixPath(member.name)
        if member.name not in expected or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unexpected archive entry {member.name!r}")
        if not member.isfile():
            raise SystemExit(f"archive entry {member.name!r} is not a regular file")
        yield member


def install(archive: Path, manifest: dict, overwrite: bool) -> int:
    if sha256(archive) != manifest["sha256"]:
        raise SystemExit(f"{archive} does not match the manifest's sha256")
    expected = set(manifest["files"])
    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in _safe_members(tar, expected):
            target = ROOT / member.name
            if target.exists() and not overwrite and sha256(target) != manifest["files"][member.name]:
                raise SystemExit(f"{member.name} exists and differs from the archive; "
                                 "pass --overwrite to replace it")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
            written += 1
    return written


def check(manifest: dict) -> tuple[list[str], list[str]]:
    missing, changed = [], []
    for name, digest in manifest["files"].items():
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
        elif sha256(path) != digest:
            changed.append(name)
    return missing, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch", help="download (or take --archive), verify, unpack")
    fetch_parser.add_argument("--archive", type=Path, default=None,
                              help="a local copy of the archive instead of downloading it")
    fetch_parser.add_argument("--overwrite", action="store_true",
                              help="replace local files that differ from the archive")
    commands.add_parser("check", help="compare data/ with the manifest")
    pack_parser = commands.add_parser("pack", help="write an archive and the manifest")
    pack_parser.add_argument("--output", type=Path, required=True)
    pack_parser.add_argument("--from-git", action="store_true",
                             help="pack the files git tracks under the result directories")
    pack_parser.add_argument("--url", default="", help="where the archive will be downloaded from")
    pack_parser.add_argument("--zenodo-record", default="", help="the Zenodo record's DOI or URL")
    args = parser.parse_args()

    if args.command == "pack":
        if args.from_git:
            files = _tracked_files()
        else:
            files = sorted(json.loads(MANIFEST.read_text())["files"])
        manifest = pack(args.output, files, args.url, args.zenodo_record)
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
        print(f"wrote {args.output} ({manifest['bytes'] / 1e6:.1f} MB, {len(files)} files, "
              f"sha256 {manifest['sha256'][:16]}...) and {MANIFEST.relative_to(ROOT)}")
        return

    manifest = json.loads(MANIFEST.read_text())
    if args.command == "check":
        missing, changed = check(manifest)
        print(f"{len(manifest['files'])} files in the manifest: {len(missing)} missing, "
              f"{len(changed)} differ")
        for name in changed[:20]:
            print(f"  differs  {name}")
        for name in missing[:20]:
            print(f"  missing  {name}")
        raise SystemExit(1 if missing or changed else 0)

    archive = args.archive
    if archive is None:
        if not manifest.get("url"):
            raise SystemExit("the manifest has no download URL yet; pass --archive")
        archive = Path(tempfile.mkdtemp()) / manifest["archive"]
        request = Request(manifest["url"], headers={"User-Agent": "tellurix-results"})
        with urlopen(request) as response, open(archive, "wb") as handle:
            while block := response.read(1 << 20):
                handle.write(block)
    written = install(archive, manifest, args.overwrite)
    missing, changed = check(manifest)
    if missing or changed:
        raise SystemExit(f"after unpacking, {len(missing)} missing and {len(changed)} differ")
    print(f"unpacked {written} files; all {len(manifest['files'])} match the manifest")


if __name__ == "__main__":
    main()
