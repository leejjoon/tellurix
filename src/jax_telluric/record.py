"""A self-describing record of a fitting run, from which its arrays rebuild.

The arrays a run writes -- transmission, model, corrected spectrum -- are all
pure functions of the observed spectrum, the stellar model and about thirty
fitted numbers. Keeping the arrays as the product makes a run awkward to ship,
impossible to diff, and silent about how it was made: nothing in an ``.npz``
says which line list produced it, what the mask rule was, or which transmission
was divided out.

This module stores the parameters instead, in one HDF5 file per run, with the
configuration and the identity of every input beside them. The arrays become a
cache that can be thrown away and rebuilt.

Two things are deliberately *not* stored. The instrument kernel is not an array
here: a boxcar FTS profile is fixed by its maximum optical path difference and
the fitted residual width, so what is kept is a short sampled fingerprint used
to detect a future change to the profile code silently reinterpreting an old
record. And the covariance is kept as a standard deviation plus a correlation
matrix rather than a full matrix, which is the same information in a third of
the space -- ``covariance = correlation * outer(sigma, sigma)``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

FORMAT_VERSION = 1
_STRING = "S64"
_LONG_STRING = "S256"
_ILS_SAMPLES = 65
_ILS_HALF_WIDTH_RESOLUTION_ELEMENTS = 8.0


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hex digest of a file, read in chunks so a line list does not land in RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def ils_fingerprint(
    instrument, lsf_sigma_kms: float, velocity_step_kms: float, samples: int = _ILS_SAMPLES
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the instrument profile a run actually applied.

    Returns the velocity offsets and the normalized profile, obtained by
    pushing a unit impulse through the real ``convolve``. That keeps the
    fingerprint honest: it records what the code did, not what a second
    implementation here believes it should have done.

    ``instrument=None`` is the model's own built-in Gaussian, which is the whole
    line spread function of a grating spectrograph. It goes through
    ``model._gaussian_convolve``'s function for the same reason.
    """

    if samples < 9 or samples % 2 == 0:
        raise ValueError("use an odd number of samples, at least nine")
    from .types import TelluricParameters

    reach = (float(lsf_sigma_kms) if instrument is None
             else float(getattr(instrument, "first_zero_kms", velocity_step_kms)))
    width = _ILS_HALF_WIDTH_RESOLUTION_ELEMENTS * max(reach, velocity_step_kms)
    count = 2 * int(np.ceil(width / velocity_step_kms)) + 1
    impulse = np.zeros(count)
    impulse[count // 2] = 1.0
    parameters = TelluricParameters(
        log_column_scales={}, velocity_kms=0.0, wavelength_stretch=0.0,
        lsf_sigma_kms=float(lsf_sigma_kms), continuum_coeffs=np.zeros(1), log_jitter=0.0,
    )
    if instrument is None:
        from .model import _gaussian_convolve

        half = count // 2
        response = np.asarray(
            _gaussian_convolve(impulse, float(lsf_sigma_kms), velocity_step_kms, half),
            dtype=float)
    else:
        response = np.asarray(
            instrument.convolve(impulse, parameters, velocity_step_kms), dtype=float)
    offsets = (np.arange(count) - count // 2) * velocity_step_kms
    wanted = np.linspace(-width, width, samples)
    return wanted, np.interp(wanted, offsets, response)


@dataclass(frozen=True)
class Record:
    """One run: its configuration, its inputs, and a row per page-epoch."""

    format_version: int
    run: dict
    config: dict
    physics: dict
    inputs: dict
    parameter_names: tuple[str, ...]
    key_fields: tuple[str, ...]
    pages: np.ndarray
    sigma: np.ndarray
    correlation: np.ndarray
    ils_velocity_kms: np.ndarray
    ils_profile: np.ndarray

    def row(self, *key: str) -> np.void:
        """The one row with this key, or a clear failure.

        The key is whatever :func:`write_record` was told identifies a row --
        ``("page", "epoch")`` for the Arcturus atlas, ``("frame", "order")``
        for an IGRINS night.
        """

        if len(key) != len(self.key_fields):
            raise KeyError(
                f"this record is keyed on {', '.join(self.key_fields)}; got {len(key)} value(s)")
        match = np.ones(len(self.pages), dtype=bool)
        for field, value in zip(self.key_fields, key):
            match &= self.pages[field] == str(value).encode()
        found = np.flatnonzero(match)
        if found.size != 1:
            raise KeyError(f"{' '.join(str(k) for k in key)} is not in this record")
        return self.pages[int(found[0])]

    def key(self, index: int) -> tuple[str, ...]:
        """This row's key, decoded."""
        return tuple(text(self.pages[index][field]) for field in self.key_fields)

    def covariance(self, index: int) -> np.ndarray:
        """Rebuild the formal covariance from the stored correlation and sigma."""
        deviation = self.sigma[index]
        return self.correlation[index].astype(float) * np.outer(deviation, deviation)


def _dtype(
    columns: Sequence[tuple[str, str]],
    species: Sequence[str],
    continuum: int,
    key_fields: Sequence[str] = ("page", "epoch"),
    extra: Sequence[tuple[str, str]] = (),
) -> np.dtype:
    fields = [
        *((name, _STRING) for name in key_fields),
        # The hash of the observed data this row was fitted to: an atlas page
        # for Arcturus, a PLP spec file for IGRINS. The name is historical.
        ("page_sha256", _STRING),
        *((f"log_column_{name}", "f8") for name in species),
        ("continuum_coeffs", "f8", (continuum,)),
        *columns,
        *extra,
        ("free_species", _LONG_STRING), ("at_bound", _LONG_STRING),
    ]
    return np.dtype(fields)


_SCALARS = (
    ("v1", "f8"), ("v2", "f8"), ("mopd_cm", "f8"), ("pixels", "i4"), ("grid_points", "i4"),
    ("reliable", "i4"), ("velocity_kms", "f8"), ("stellar_velocity_kms", "f8"),
    ("wavelength_stretch", "f8"), ("lsf_sigma_kms", "f8"), ("log_jitter", "f8"),
    ("pixel_sigma", "f8"), ("residual_rms", "f8"), ("residual_rms_over_noise", "f8"),
    ("reduced_chi2", "f8"), ("median_transmission", "f8"), ("continuum_level", "f8"),
    ("continuum_level_pixels", "i4"), ("condition_number", "f8"),
    ("all_stages_converged", "?"), ("negligible_telluric", "?"),
)

_UNITS = {
    "v1": "cm-1", "v2": "cm-1", "mopd_cm": "cm", "velocity_kms": "km/s",
    "stellar_velocity_kms": "km/s", "lsf_sigma_kms": "km/s", "pixel_sigma": "normalized flux",
    "residual_rms": "normalized flux", "median_transmission": "fraction",
    "continuum_level": "ratio", "condition_number": "dimensionless",
}

_DESCRIPTIONS = {
    "continuum_level": (
        "Median of the corrected spectrum over the stellar model where the star "
        "itself is unabsorbed. 1.0 means the correction landed on the model's "
        "continuum; a page with no telluric-free pixel cannot separate continuum "
        "from column and drifts."
    ),
    "condition_number": (
        "Of the Hessian submatrix inverted for the covariance. Large means the "
        "data cannot separate some pair of parameters."
    ),
    "at_bound": "Free parameters resting on a bound, excluded from the covariance.",
    "log_jitter": "Log of the extra per-pixel noise the likelihood fitted.",
}


def write_record(
    path: str | Path,
    *,
    run: Mapping,
    config: Mapping,
    physics: Mapping,
    inputs: Mapping,
    parameter_names: Sequence[str],
    species: Sequence[str],
    pages: Sequence[Mapping],
    continuum_degree: int,
    key_fields: Sequence[str] = ("page", "epoch"),
    extra_columns: Sequence[tuple] = (),
) -> Path:
    """Write one run to HDF5. Overwrites; a run is written once, whole.

    ``key_fields`` names the columns that identify a row. The Arcturus atlas
    keys on ``("page", "epoch")``; an IGRINS night keys on ``("frame",
    "order")``. They are stored so a reader does not have to guess, and a file
    written before this was configurable is read as ``("page", "epoch")``.

    ``extra_columns`` are ``(name, dtype)`` pairs appended to the row, for
    whatever a pipeline records that the shared schema does not -- the airmass
    and the surface conditions, say. A string dtype is filled from ``str()``.
    """

    import h5py

    if not pages:
        raise ValueError("a record needs at least one page")
    names = tuple(parameter_names)
    species = tuple(species)
    key_fields = tuple(key_fields)
    extra_columns = tuple(tuple(column) for column in extra_columns)
    table = np.zeros(
        len(pages),
        dtype=_dtype(_SCALARS, species, continuum_degree + 1, key_fields, extra_columns))
    # Which columns take text rather than a number, derived from the dtype so
    # adding a string column cannot forget to update a hand-written list.
    text_fields = {
        field for field in table.dtype.names
        if table.dtype[field].kind in ("S", "U")
    }
    sigma = np.zeros((len(pages), len(names)))
    correlation = np.zeros((len(pages), len(names), len(names)), dtype=np.float32)
    ils = np.zeros((len(pages), _ILS_SAMPLES), dtype=np.float32)
    velocity = None

    for index, page in enumerate(pages):
        for field in table.dtype.names:
            if field in text_fields:
                table[field][index] = str(page.get(field, "")).encode()[:255]
            elif field == "continuum_coeffs":
                table[field][index] = np.asarray(page["continuum_coeffs"], dtype=float)
            elif field.startswith("log_column_"):
                table[field][index] = float(page["log_column_scales"].get(field[11:], 0.0))
            else:
                value = page.get(field)
                table[field][index] = 0 if value is None else value
        sigma[index] = page.get("sigma", np.zeros(len(names)))
        correlation[index] = page.get("correlation", np.zeros((len(names), len(names))))
        if page.get("ils_profile") is not None:
            ils[index] = page["ils_profile"]
            velocity = page["ils_velocity_kms"]

    path = Path(path)
    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["about"] = (
            "Fitted parameters for one telluric run. Every array a run writes is "
            "a pure function of these plus the inputs named in /inputs. Note the "
            "correction divides by the convolved effective transmission, "
            "model_flux / stellar_only, not by the unconvolved transmission."
        )
        for key, value in run.items():
            handle.attrs[key] = "" if value is None else value
        for name, block in (("config", config), ("physics", physics), ("inputs", inputs)):
            group = handle.create_group(name)
            for key, value in block.items():
                group.attrs[key] = "" if value is None else value
        handle.attrs["key_fields"] = [f.encode() for f in key_fields]
        handle.create_dataset("parameter_names", data=[n.encode() for n in names])
        handle.create_dataset("species", data=[s.encode() for s in species])
        rows = handle.create_dataset("pages", data=table, compression="gzip")
        for field in table.dtype.names:
            if field in _UNITS:
                rows.attrs[f"{field}.units"] = _UNITS[field]
            if field in _DESCRIPTIONS:
                rows.attrs[f"{field}.description"] = _DESCRIPTIONS[field]
        deviation = handle.create_dataset("sigma", data=sigma, compression="gzip")
        deviation.attrs["description"] = (
            "Formal standard deviation from the inverse Hessian, indexed by "
            "parameter_names. It assumes independent Gaussian pixel errors. On "
            "this atlas the residual is dominated by correlated stellar model "
            "error, and the formal value comes out several times too small: "
            "measured on ab5000_, 0.76% on the water column against 6.5-9% of "
            "actual sub-window scatter. Treat it as a lower bound."
        )
        shape = handle.create_dataset("correlation", data=correlation, compression="gzip")
        shape.attrs["description"] = (
            "Parameter correlation, which describes the shape of the likelihood "
            "rather than its scale and so survives a wrong noise model. This is "
            "where degeneracies show. covariance = correlation * outer(sigma, sigma)."
        )
        profile = handle.create_dataset("ils_profile", data=ils, compression="gzip")
        profile.attrs["description"] = (
            "The instrument profile a unit impulse actually produced, sampled on "
            "the velocities below. A fingerprint for detecting a changed profile, "
            "not the operational kernel -- that is rebuilt from mopd_cm and "
            "lsf_sigma_kms."
        )
        handle.create_dataset(
            "ils_velocity_kms", data=np.zeros(_ILS_SAMPLES) if velocity is None else velocity
        )
    return path


def read_record(path: str | Path) -> Record:
    """Read a record, refusing a format this code does not know."""

    import h5py

    with h5py.File(Path(path), "r") as handle:
        stored = int(handle.attrs["format_version"])
        if stored != FORMAT_VERSION:
            raise ValueError(
                f"record format {stored}, but this code writes and reads {FORMAT_VERSION}"
            )
        decode = lambda group: {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in dict(group.attrs).items()
        }
        # A file written before the key was configurable is keyed the way the
        # Arcturus atlas was, which is the only thing it could have been.
        key_fields = tuple(
            text(f) for f in handle.attrs.get("key_fields", [b"page", b"epoch"]))
        return Record(
            format_version=stored,
            key_fields=key_fields,
            run={k: v for k, v in decode(handle).items()
                 if k not in ("format_version", "key_fields")},
            config=decode(handle["config"]),
            physics=decode(handle["physics"]),
            inputs=decode(handle["inputs"]),
            parameter_names=tuple(n.decode() for n in handle["parameter_names"][:]),
            pages=handle["pages"][:],
            sigma=handle["sigma"][:],
            correlation=handle["correlation"][:],
            ils_velocity_kms=handle["ils_velocity_kms"][:],
            ils_profile=handle["ils_profile"][:],
        )


def _same(left, right) -> bool:
    """Compare two attribute values, which h5py may hand back as arrays."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left, right = np.asarray(left), np.asarray(right)
        return left.shape == right.shape and bool(np.all(left == right))
    return bool(left == right)


def merge_records(paths: Sequence[str | Path], output: str | Path) -> Path:
    """Combine the records of a sharded run into one.

    A run is normally split across devices, each writing its own record for the
    pages it took. The configuration and the inputs are the same for all of
    them -- a shard that disagrees is a mistake worth stopping for, not
    something to silently take the first of.
    """

    import h5py

    records = [read_record(path) for path in paths]
    if not records:
        raise ValueError("nothing to merge")
    first = records[0]
    for other in records[1:]:
        if other.format_version != first.format_version:
            raise ValueError("shards were written by different record formats")
        if other.parameter_names != first.parameter_names:
            raise ValueError("shards disagree about the parameter vector")
        for name, a, b in (("config", first.config, other.config),
                           ("physics", first.physics, other.physics),
                           ("inputs", first.inputs, other.inputs)):
            differing = sorted(k for k in set(a) | set(b) if not _same(a.get(k), b.get(k)))
            if differing:
                raise ValueError(f"shards disagree about {name}: {', '.join(differing)}")

    keys = {r.key_fields for r in records}
    if len(keys) != 1:
        raise ValueError(f"shards disagree about how a row is keyed: {sorted(keys)}")
    key_fields = keys.pop()
    pages = np.concatenate([r.pages for r in records])
    order = np.argsort(
        ["\x00".join(text(row[field]) for field in key_fields) for row in pages])
    sigma = np.concatenate([r.sigma for r in records])[order]
    correlation = np.concatenate([r.correlation for r in records])[order]
    profile = np.concatenate([r.ils_profile for r in records])[order]

    with h5py.File(Path(paths[0]), "r") as source, h5py.File(Path(output), "w") as handle:
        for key, value in source.attrs.items():
            handle.attrs[key] = value
        handle.attrs["shards"] = [str(Path(p).name) for p in paths]
        for name in ("config", "physics", "inputs"):
            group = handle.create_group(name)
            for key, value in source[name].attrs.items():
                group.attrs[key] = value
        for name in ("parameter_names", "species", "ils_velocity_kms"):
            handle.create_dataset(name, data=source[name][:])
        for name, data in (("pages", pages[order]), ("sigma", sigma),
                           ("correlation", correlation), ("ils_profile", profile)):
            dataset = handle.create_dataset(name, data=data, compression="gzip")
            for key, value in source[name].attrs.items():
                dataset.attrs[key] = value
    return Path(output)


def text(value) -> str:
    """Decode a fixed-width HDF5 string field."""
    return value.decode() if isinstance(value, bytes) else str(value)


def parameters_from_row(row, species: Sequence[str]):
    """Rebuild the fitted parameters of one page-epoch."""

    from .types import TelluricParameters

    return TelluricParameters(
        log_column_scales={name: float(row[f"log_column_{name}"]) for name in species},
        velocity_kms=float(row["velocity_kms"]),
        wavelength_stretch=float(row["wavelength_stretch"]),
        lsf_sigma_kms=float(row["lsf_sigma_kms"]),
        continuum_coeffs=np.asarray(row["continuum_coeffs"], dtype=float),
        log_jitter=float(row["log_jitter"]),
        stellar_velocity_kms=float(row["stellar_velocity_kms"]),
    )


def _jsonable(value):
    """Coerce an HDF5 attribute into something ``json.dumps`` accepts.

    h5py hands a list-valued attribute back as a numpy array of bytes, which
    is neither a list nor a string as far as the encoder is concerned.
    """

    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def to_json(record: Record) -> dict:
    """A readable projection of a record, for the committed report."""

    rows = []
    for index, row in enumerate(record.pages):
        entry = {field: text(row[field]) for field in record.key_fields}
        for field in record.pages.dtype.names:
            if field in record.key_fields:
                continue
            value = row[field]
            if isinstance(value, np.ndarray):
                entry[field] = [float(v) for v in value]
            elif isinstance(value, bytes):
                entry[field] = text(value)
            elif isinstance(value, np.bool_):
                entry[field] = bool(value)
            elif isinstance(value, np.integer):
                entry[field] = int(value)
            else:
                entry[field] = float(f"{float(value):.10g}")
        entry["sigma"] = [float(f"{v:.6g}") for v in record.sigma[index]]
        rows.append(entry)
    return {
        **_jsonable(record.run),
        "format_version": record.format_version,
        "key_fields": list(record.key_fields),
        "config": _jsonable(record.config),
        "physics": _jsonable(record.physics),
        "inputs": _jsonable(record.inputs),
        "parameter_names": list(record.parameter_names),
        "results": rows,
    }


__all__ = [
    "FORMAT_VERSION", "Record", "file_sha256", "ils_fingerprint",
    "merge_records", "parameters_from_row", "read_record", "text", "to_json",
    "write_record",
]
