#!/usr/bin/env python
"""Fetch IGRINS A0V telluric standards from the RRISA reduced archive.

RRISA publishes two components. The *raw* one is 2D detector frames, which
would need the PLP and a whole night of calibrations, and its Box folder only
permits whole-night downloads. The *reduced* one is PLP v3 output with a direct
per-observation URL in its catalog, which is what this fetches.

``reduced_log.csv`` carries one row per reduced observation with the object,
the date, the facility, the signal to noise in each band, and ``FILE_URL``.
Its ``AM`` column is not trustworthy -- it writes ``-1`` for a missing airmass
and carries a few impossible values -- so the airmass used anywhere downstream
comes from the FITS header, not from here. This script only uses ``AM`` to
choose what to download.

    uv run python scripts/download_rrisa_standard.py --catalog        # refresh the index
    uv run python scripts/download_rrisa_standard.py --list --facility DCT --night 20181220
    uv run python scripts/download_rrisa_standard.py --night 20181220 --min-snr 200

Each tarball is 77 MB but only a few megabytes of it are needed, so the four
products per band that the fitting driver reads are extracted and the archive
is discarded unless ``--keep-archive`` says otherwise.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

CATALOG_URL = (
    "https://raw.githubusercontent.com/IGRINScontact/RRISA/HEAD/"
    "RRISA_Reduced/reduced_log.csv"
)
# What the fitting driver reads. `spec` carries the counts and the wavelength
# solution, `variance` the measured per-pixel noise, `spec_flattened` the PLP's
# own telluric model and continuum -- kept for comparison, never for the fit.
WANTED = ("spec.fits", "variance.fits", "spec_flattened.fits", "sn.fits")
FACILITIES = ("McDonald", "DCT", "Gemini South")


def load_catalog(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no catalog at {path}; run once with --catalog to fetch it")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def fetch_catalog(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {CATALOG_URL}")
    with urllib.request.urlopen(CATALOG_URL) as response:
        payload = response.read()
    path.write_bytes(payload)
    print(f"wrote {path} ({len(payload) / 1e6:.1f} MB)")


def _number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def select(rows, args) -> list[dict]:
    """Standards matching the filters, brightest first within a night."""

    chosen = []
    for row in rows:
        if row.get("OBJTYPE") != "STD":
            continue
        if args.facility and row.get("FACILITY") != args.facility:
            continue
        if args.night and row.get("CIVIL") != args.night:
            continue
        if args.object and args.object.lower() not in row.get("NAME", "").lower():
            continue
        snr = _number(row, "SNRH_res")
        if not snr >= args.min_snr:
            continue
        airmass = _number(row, "AM")
        # -1 is this catalog's missing-value sentinel, and a few rows carry
        # values no telescope can point at.
        if args.max_airmass is not None and not 1.0 <= airmass <= args.max_airmass:
            continue
        if not row.get("FILE_URL"):
            continue
        chosen.append(row)
    chosen.sort(key=lambda r: (r.get("CIVIL", ""), -_number(r, "SNRH_res")))
    return chosen[: args.limit] if args.limit else chosen


def describe(row: dict) -> str:
    return (f"{row.get('CIVIL',''):>8} {row.get('FILENUMBER',''):>5}  "
            f"{row.get('NAME','')[:22]:22s} {row.get('FACILITY',''):12s} "
            f"AM {_number(row, 'AM'):5.2f}  S/N H {_number(row, 'SNRH_res'):5.0f} "
            f"K {_number(row, 'SNRK_res'):5.0f}")


def download(row: dict, output_root: Path, keep_archive: bool) -> Path | None:
    name = f"{row['CIVIL']}_{int(row['FILENUMBER']):04d}"
    target = output_root / name
    if target.exists() and any(target.glob("*.spec.fits")):
        print(f"  {name}: already present")
        return target

    print(f"  {name}: fetching {row['FILE_URL']}")
    with urllib.request.urlopen(row["FILE_URL"]) as response:
        payload = response.read()
    print(f"  {name}: {len(payload) / 1e6:.1f} MB")

    target.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            stem = Path(member.name).name
            if not member.isfile() or not stem.endswith(WANTED):
                continue
            # Flatten: the archive's own directory layout varies across years.
            source = archive.extractfile(member)
            if source is None:
                continue
            (target / stem).write_bytes(source.read())
            extracted += 1
    if keep_archive:
        (target / f"{name}.tar.gz").write_bytes(payload)
    if not extracted:
        print(f"  {name}: nothing matched {WANTED}", file=sys.stderr)
        return None

    (target / "rrisa_row.json").write_text(json.dumps(row, indent=2))
    print(f"  {name}: extracted {extracted} files to {target}")
    return target


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", action="store_true", help="refresh reduced_log.csv and exit")
    parser.add_argument("--catalog-path", type=Path,
                        default=root / "data/igrins/reduced_log.csv")
    parser.add_argument("--output-root", type=Path, default=root / "data/igrins")
    parser.add_argument("--facility", choices=FACILITIES, default=None)
    parser.add_argument("--night", default=None, help="CIVIL date, YYYYMMDD")
    parser.add_argument("--object", default=None, help="substring of the target name")
    parser.add_argument("--min-snr", type=float, default=200.0,
                        help="signal to noise per resolution element in H")
    parser.add_argument("--max-airmass", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="show what matches and stop")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    if args.catalog:
        fetch_catalog(args.catalog_path)
        return

    rows = load_catalog(args.catalog_path)
    chosen = select(rows, args)
    if not chosen:
        raise SystemExit("nothing in the catalog matches those filters")

    print(f"{len(chosen)} standard(s):")
    for row in chosen:
        print("  " + describe(row))
    if args.list:
        return

    print()
    for row in chosen:
        download(row, args.output_root, args.keep_archive)


if __name__ == "__main__":
    main()
