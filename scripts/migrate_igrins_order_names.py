#!/usr/bin/env python
"""Rename IGRINS products from row positions to physical echelle orders.

Before ``identify_orders`` the pipeline named an order by its row in the file --
``H05``, ``order_index`` 5 -- which names a different order in a reduction that
ships a different number of rows. Every product written that way is relabelled
here, without refitting anything, from the frames the products came from:

* a run's ``record.h5``: ``order`` becomes the physical name (``H103``),
  ``order_index`` becomes ``order_number``, and ``order_source`` is added;
* its ``<frame>_<order>.npz`` caches are renamed;
* its per-frame ``*_summary.json`` rows get the physical ``order`` and ``name``;
* ``--transfer-rows`` files of ``validate_igrins_transfer.py``, whose ``order``
  is the row;
* ``--results`` reports of ``summarize_igrins_fit.py``, whose ``per_order``
  entries carry the row name.

The mapping is never assumed to be row + 98: each frame's own header and
wavelengths name its rows through :func:`identify_orders`, and every renamed npz
is checked against the order it is being given -- its cached wavenumbers must
fall on that order's centre. Everything is idempotent, and ``--dry-run`` shows
what would change. A ``order_migration.json`` log beside each run records every
rename, so it can be undone.

    uv run python scripts/migrate_igrins_order_names.py data/corrected/igrins/* \\
        --data-root /path/to/lblrtm
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

OLD_NAME = re.compile(r"^([HK])(\d{2})$")
# A band never ships more than 28 rows, and no physical order is below 71, so a
# two-digit name under this is a row. H98, H99 and K71-K96 have two digits as
# well: without the bound a second run would read them as rows and fail.
_MAX_ROW = 40


def old_row(name: str):
    """The row an old-style name like ``H05`` meant, or None for a physical name."""

    match = OLD_NAME.match(name)
    if match and int(match.group(2)) < _MAX_ROW:
        return int(match.group(2))
    return None


def frame_orders(spec_path: Path) -> tuple[str, dict[int, int], str]:
    """Band, row -> physical order, and how the orders were found."""

    from astropy.io import fits
    from tellurix import identify_orders

    with fits.open(spec_path) as handle:
        header = dict(handle[0].header)
        wavelength = np.asarray(handle[1].data, dtype=float)
    band = str(header.get("BAND", spec_path.name[3:4])).strip()
    numbers, source = identify_orders(band, wavelength, header)
    return band, {row: m for row, m in enumerate(numbers) if m is not None}, source


class Frames:
    """Row -> physical order for every frame a product names, read once each."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._cache: dict[str, tuple[str, dict[int, int], str]] = {}

    def locate(self, stem: str) -> Path:
        # SDCH_20181220_0033 lives in data/igrins/20181220_0033/.
        path = self.data_root / "data/igrins" / stem[5:] / f"{stem}.spec.fits"
        if not path.exists():
            raise FileNotFoundError(f"cannot find the frame behind {stem}: {path}")
        return path

    def __call__(self, stem: str) -> tuple[str, dict[int, int], str]:
        stem = stem.split(".")[0]
        if stem not in self._cache:
            self._cache[stem] = frame_orders(self.locate(stem))
        return self._cache[stem]

    def name(self, stem: str, row: int) -> str:
        band, rows, _ = self(stem)
        if row not in rows:
            raise ValueError(f"{stem} has no order in row {row}")
        return f"{band}{rows[row]}"


def check_npz(path: Path, band: str, number: int) -> None:
    """The cached wavenumbers must sit on the order they are being renamed to."""

    from tellurix import IGRINS_ORDER_CENTRES_UM

    centre = IGRINS_ORDER_CENTRES_UM[band].get(number)
    if centre is None:
        return
    with np.load(path) as arrays:
        nu = np.asarray(arrays["wavenumber_cm1"], dtype=float)
    low, high = 1.0e4 / nu.max(), 1.0e4 / nu.min()
    if not low < centre < high:
        raise ValueError(f"{path.name} spans {low:.4f}-{high:.4f} um, which does not hold "
                         f"{band}{number} at {centre:.4f} um; refusing to rename it")


def rewrite_pages(path: Path, pages: np.ndarray, attrs: dict) -> None:
    """Replace a record's ``pages`` table, rewriting the whole file.

    Deleting an HDF5 dataset does not return its space, and ``write_record``
    compresses the table, so the file is copied object by object into a fresh
    one rather than edited in place -- which is what the first version of this
    script did, leaving records 2.4x their size.
    """

    import os
    import h5py

    temporary = path.with_suffix(".migrating.h5")
    with h5py.File(path, "r") as source, h5py.File(temporary, "w") as target:
        for key, value in source.attrs.items():
            target.attrs[key] = value
        for name in source:
            if name != "pages":
                source.copy(source[name], target, name=name)
        dataset = target.create_dataset("pages", data=pages, compression="gzip")
        for key, value in attrs.items():
            dataset.attrs[key] = value
    os.replace(temporary, path)


def migrate_record(path: Path, frames: Frames, dry_run: bool, log: list) -> None:
    import h5py

    with h5py.File(path, "r") as handle:
        pages = handle["pages"][:]
        attrs = dict(handle["pages"].attrs)
        compressed = handle["pages"].compression is not None
    names = pages.dtype.names
    if "order_number" in names:
        if compressed:
            print(f"  {path}: already physical")
        else:
            print(f"  {path}: already physical; repacking its uncompressed table")
            if not dry_run:
                rewrite_pages(path, pages, attrs)
        return
    if "order_index" not in names:
        print(f"  {path}: not an IGRINS record, left alone")
        return
    fields = []
    for name in names:
        if name == "order_index":
            fields += [("order_number", "i4"), ("order_source", "S64")]
        else:
            fields.append((name, pages.dtype[name]))
    new = np.zeros(pages.shape, dtype=fields)
    for name in names:
        if name != "order_index":
            new[name] = pages[name]
    for i, page in enumerate(pages):
        stem = page["frame"].decode()
        band, rows, source = frames(stem)
        row = int(page["order_index"])
        old = page["order"].decode()
        if old != f"{band}{row:02d}":
            raise ValueError(f"{path}: row {i} is {old!r} but its order_index is {row}")
        new[i]["order_number"] = rows[row]
        new[i]["order_source"] = source.encode()
        new[i]["order"] = f"{band}{rows[row]}".encode()
        log.append({"file": str(path), "frame": stem, "from": old, "to": f"{band}{rows[row]}"})
    print(f"  {path}: {len(pages)} rows, e.g. {pages[0]['order'].decode()} -> "
          f"{new[0]['order'].decode()}")
    if dry_run:
        return
    rewrite_pages(path, new, attrs)


def migrate_caches(run_dir: Path, frames: Frames, dry_run: bool, log: list) -> None:
    renamed = 0
    for path in sorted(run_dir.glob("*.npz")):
        stem, _, order = path.stem.rpartition("_")
        row = old_row(order)
        if not stem.startswith("SDC") or row is None:
            continue
        band, rows, _ = frames(stem)
        number = rows[row]
        target = path.with_name(f"{stem}_{band}{number}.npz")
        check_npz(path, band, number)
        if target.exists():
            raise FileExistsError(f"{target} already exists; not overwriting it with {path}")
        log.append({"file": str(path), "to": str(target)})
        renamed += 1
        if not dry_run:
            path.rename(target)
    for path in sorted(run_dir.glob("*_summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("order_names") == "physical" or "results" not in summary:
            continue
        stem = Path(summary["observation"]["path"]).name
        band, rows, source = frames(stem)
        for row in summary["results"]:
            number = rows[int(row["order"])]
            if row.get("name") not in (None, f"{band}{int(row['order']):02d}"):
                raise ValueError(f"{path}: row {row['order']} is named {row['name']!r}")
            row["order"], row["name"] = number, f"{band}{number}"
        for failure in summary.get("failures", []):
            failure["order"] = rows.get(int(failure["order"]), failure["order"])
        summary["observation"]["orders"] = sorted(rows.values())
        summary["observation"]["order_source"] = source
        summary["order_names"] = "physical"
        log.append({"file": str(path), "summary": True})
        if not dry_run:
            path.write_text(json.dumps(summary, indent=2))
    print(f"  {run_dir}: {renamed} npz renamed")


def migrate_transfer_rows(path: Path, frames: Frames, dry_run: bool, log: list) -> None:
    rows = json.loads(path.read_text())
    if all(r.get("order_name") for r in rows):
        print(f"  {path}: already physical")
        return
    for row in rows:
        band, table, _ = frames(row["frame"])
        number = table[int(row["order"])]
        row["order"], row["order_name"] = number, f"{band}{number}"
    log.append({"file": str(path), "rows": len(rows)})
    print(f"  {path}: {len(rows)} rows")
    if not dry_run:
        path.write_text("[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
                        + "\n]\n")


def migrate_results(path: Path, frames: Frames, dry_run: bool, log: list) -> None:
    report = json.loads(path.read_text())
    if report.get("order_names") == "physical":
        print(f"  {path}: already physical")
        return
    stem = Path(report["observation"]["path"]).name
    band, rows, source = frames(stem)
    for entry in report["per_order"]:
        entry["name"] = f"{band}{rows[old_row(entry['name'])]}"
    report["order_names"] = "physical"
    log.append({"file": str(path), "per_order": len(report["per_order"])})
    print(f"  {path}: {len(report['per_order'])} orders, {source}")
    if not dry_run:
        path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="*")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="where data/igrins/<date_frame>/ holds the frames")
    parser.add_argument("--records", action=argparse.BooleanOptionalAction, default=True,
                        help="migrate each run's record.h5; --no-records for a checkout whose "
                             "records are tracked and will arrive migrated")
    parser.add_argument("--caches", action=argparse.BooleanOptionalAction, default=True,
                        help="rename npz caches and rewrite per-frame summaries")
    parser.add_argument("--transfer-rows", type=Path, nargs="*", default=[])
    parser.add_argument("--results", type=Path, nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    frames = Frames(args.data_root)
    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            continue
        log: list = []
        record = run_dir / "record.h5"
        if args.records and record.exists():
            migrate_record(record, frames, args.dry_run, log)
        if args.caches:
            migrate_caches(run_dir, frames, args.dry_run, log)
        if log and not args.dry_run:
            path = run_dir / "order_migration.json"
            previous = json.loads(path.read_text()) if path.exists() else []
            path.write_text(json.dumps(previous + log, indent=1))
    for path in args.transfer_rows:
        migrate_transfer_rows(path, frames, args.dry_run, [])
    for path in args.results:
        migrate_results(path, frames, args.dry_run, [])


if __name__ == "__main__":
    main()
