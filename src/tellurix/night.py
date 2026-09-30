"""A night's telluric calibration, measured on its standards, for its other frames.

A science target has no telluric standard of its own. What a night's A0V
standards measure is carried to it here, per physical echelle order:

* the **dry columns** and the **LSF** -- the median over the standards. The
  model's columns are vertical and the zenith angle supplies the slant path, so
  a well-mixed gas needs no airmass interpolation;
* the **velocity zero point** -- the median;
* **water** -- as a time series, interpolated linearly to the frame's time and
  held flat outside the standards. Per order, because orders disagree about the
  water column by 10-20% within a frame but each order's offset repeats to 1-2%
  across frames. Interpolation alone is not enough -- water moves on timescales
  the standards do not sample, 4-8% rms -- so a science fit refits one water
  scale for the whole frame (see ``docs/igrins_transfer.md``);
* the **instrument response pattern** -- the median over every standard of its
  fractional residual, smoothed. A science frame is none of the standards, so
  no leave-one-out is needed; the smoothing is still required, for the reason
  :func:`~tellurix.igrins.leave_one_out_patterns` gives.

Build one with :meth:`NightCalibration.from_run` from a ``fit_igrins_standard.py``
run -- its record and its npz cache -- and keep it as one HDF5 file.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .igrins import smoothed_frame_median
from .record import file_sha256, text

FORMAT_VERSION = 1


@dataclass(frozen=True)
class OrderCalibration:
    """One physical echelle order's calibration."""

    band: str
    number: int
    columns: Mapping[str, float]
    velocity_kms: float
    lsf_sigma_kms: float
    water_mjd: np.ndarray
    water_log_column: np.ndarray
    pattern_wavenumber_cm1: np.ndarray | None = None
    pattern: np.ndarray | None = None
    frames: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mjd = np.asarray(self.water_mjd, dtype=float)
        water = np.asarray(self.water_log_column, dtype=float)
        if mjd.ndim != 1 or mjd.shape != water.shape or mjd.size == 0:
            raise ValueError("water needs one log column per standard's time")
        order = np.argsort(mjd)
        object.__setattr__(self, "water_mjd", mjd[order])
        object.__setattr__(self, "water_log_column", water[order])
        if "H2O" in self.columns:
            raise ValueError("water is a time series here, not a fixed column")
        if (self.pattern is None) != (self.pattern_wavenumber_cm1 is None):
            raise ValueError("a pattern needs its wavenumbers, and only then")
        if self.pattern is not None:
            nu = np.asarray(self.pattern_wavenumber_cm1, dtype=float)
            values = np.asarray(self.pattern, dtype=float)
            if nu.shape != values.shape or np.any(np.diff(nu) <= 0):
                raise ValueError("the pattern must be on strictly increasing wavenumbers")
            object.__setattr__(self, "pattern_wavenumber_cm1", nu)
            object.__setattr__(self, "pattern", values)

    @property
    def name(self) -> str:
        return f"{self.band}{self.number}"

    def water_at(self, mjd: float) -> float:
        # np.interp holds the end values flat outside the standards, which is
        # the honest extrapolation: a trend in water is not a law.
        return float(np.interp(float(mjd), self.water_mjd, self.water_log_column))

    def log_columns_at(self, mjd: float) -> dict[str, float]:
        return {**self.columns, "H2O": self.water_at(mjd)}

    def pattern_on(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """The fractional response on another frame's pixels; zero if unmeasured."""

        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        if self.pattern is None:
            return np.zeros_like(wavelength_nm)
        # Within a night every frame shares one wavelength solution, so this
        # lands on the same pixels; across nights it is an honest resampling.
        return np.interp(1.0e7 / wavelength_nm, self.pattern_wavenumber_cm1, self.pattern,
                         left=0.0, right=0.0)


@dataclass(frozen=True)
class NightCalibration:
    """Every order's calibration for one band of one night."""

    band: str
    orders: Mapping[int, OrderCalibration]
    source: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for number, order in self.orders.items():
            if order.number != number or order.band != self.band:
                raise ValueError(f"order {order.name} filed as {self.band}{number}")

    def order(self, number: int) -> OrderCalibration:
        if int(number) not in self.orders:
            raise KeyError(f"no calibration for {self.band}{int(number)}; this night has "
                           f"{sorted(self.orders)}")
        return self.orders[int(number)]

    @classmethod
    def from_run(
        cls,
        record_path: str | Path,
        cache_dir: str | Path | None = None,
        *,
        exclude: Sequence[str] = (),
        master: "MasterPattern | None" = None,
        smooth_pixels: int = 51,
        minimum_frames: int = 3,
    ) -> "NightCalibration":
        """Measure the calibration from a ``fit_igrins_standard.py`` run.

        ``cache_dir`` holds the run's ``<frame>_<order>.npz`` arrays, which the
        response pattern needs; without it there is no pattern. ``master``
        supplies the pattern instead, for a night with too few standards to
        measure its own -- the cache is then not read at all. ``exclude``
        leaves frames out -- for testing the calibration on a standard it did
        not see.
        """

        import h5py

        record_path = Path(record_path)
        with h5py.File(record_path, "r") as handle:
            pages = handle["pages"][:]
            species = [text(s) for s in handle["species"][:]]
            config = {k: (v.item() if isinstance(v, np.generic) else v)
                      for k, v in handle["config"].attrs.items()}
            inputs = {k: (v.item() if isinstance(v, np.generic) else v)
                      for k, v in handle["inputs"].attrs.items()}
        if "order_number" not in pages.dtype.names:
            raise ValueError(f"{record_path} names orders by row; run "
                             "scripts/migrate_igrins_order_names.py on it first")
        excluded = {str(f) for f in exclude}
        pages = pages[[text(f) not in excluded for f in pages["frame"]]]
        bands = {text(b) for b in pages["band"]}
        if len(bands) != 1:
            raise ValueError(f"a calibration is one band; {record_path} holds {sorted(bands)}")
        band = bands.pop()

        orders = {}
        for number in sorted({int(n) for n in pages["order_number"]}):
            rows = pages[pages["order_number"] == number]
            if len(rows) < minimum_frames:
                continue
            frames = tuple(text(f) for f in rows["frame"])
            columns = {s: float(np.median(rows[f"log_column_{s}"])) for s in species
                       if s != "H2O"}
            pattern_nu = pattern = None
            if master is not None:
                if number in master.orders:
                    pattern_nu, pattern = master.orders[number]
            elif cache_dir is not None:
                pattern_nu, pattern = _order_pattern(
                    Path(cache_dir), frames, f"{band}{number}", smooth_pixels, minimum_frames)
            orders[number] = OrderCalibration(
                band=band, number=number, columns=columns,
                velocity_kms=float(np.median(rows["velocity_kms"])),
                lsf_sigma_kms=float(np.median(rows["lsf_sigma_kms"])),
                water_mjd=np.asarray(rows["mjd"], dtype=float),
                water_log_column=np.asarray(rows["log_column_H2O"], dtype=float),
                pattern_wavenumber_cm1=pattern_nu, pattern=pattern, frames=frames)
        source = {
            "record": str(record_path), "record_sha256": file_sha256(record_path),
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "pattern": ("master" if master is not None
                        else "night" if cache_dir is not None else "none"),
            "master_nights": list(master.nights) if master is not None else [],
            "excluded": sorted(excluded), "smooth_pixels": smooth_pixels,
            "minimum_frames": minimum_frames, "species": species,
            "config": config, "inputs": inputs,
        }
        return cls(band=band, orders=orders, source=source)

    def save(self, path: str | Path) -> Path:
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.attrs["format_version"] = FORMAT_VERSION
            handle.attrs["band"] = self.band
            handle.attrs["source"] = json.dumps(self.source, default=_jsonable)
            for number, order in sorted(self.orders.items()):
                group = handle.create_group(order.name)
                group.attrs["number"] = number
                group.attrs["columns"] = json.dumps(dict(order.columns))
                group.attrs["velocity_kms"] = order.velocity_kms
                group.attrs["lsf_sigma_kms"] = order.lsf_sigma_kms
                group.attrs["frames"] = json.dumps(list(order.frames))
                group["water_mjd"] = order.water_mjd
                group["water_log_column"] = order.water_log_column
                if order.pattern is not None:
                    _store(group, "pattern_wavenumber_cm1", order.pattern_wavenumber_cm1)
                    _store(group, "pattern", order.pattern)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NightCalibration":
        import h5py

        with h5py.File(Path(path), "r") as handle:
            stored = int(handle.attrs["format_version"])
            if stored != FORMAT_VERSION:
                raise ValueError(f"calibration format {stored}, this code reads {FORMAT_VERSION}")
            band = text(handle.attrs["band"])
            orders = {}
            for name, group in handle.items():
                number = int(group.attrs["number"])
                orders[number] = OrderCalibration(
                    band=band, number=number,
                    columns=json.loads(text(group.attrs["columns"])),
                    velocity_kms=float(group.attrs["velocity_kms"]),
                    lsf_sigma_kms=float(group.attrs["lsf_sigma_kms"]),
                    water_mjd=group["water_mjd"][:],
                    water_log_column=group["water_log_column"][:],
                    pattern_wavenumber_cm1=(group["pattern_wavenumber_cm1"][:].astype(float)
                                            if "pattern" in group else None),
                    pattern=group["pattern"][:].astype(float) if "pattern" in group else None,
                    frames=tuple(json.loads(text(group.attrs["frames"]))))
            source = json.loads(text(handle.attrs["source"]))
        return cls(band=band, orders=orders, source=source)


@dataclass(frozen=True)
class MasterPattern:
    """One band's instrument response, the median over several nights.

    The response belongs to the spectrograph, not the night: four nights across
    three telescopes and five years correlate at median r = +0.89 to +0.95, and
    their median captures 74-88% of each night's own pattern. A night with too
    few standards to measure its own -- 46% of the archive has fewer than five --
    uses this instead. Per physical order, on the wavenumbers of the first night
    given; within a night's 0.16 cm-1 of wavelength-solution drift that lands on
    the same pixels as any other night's.
    """

    band: str
    orders: Mapping[int, tuple[np.ndarray, np.ndarray]]
    nights: tuple[str, ...] = ()

    @classmethod
    def from_calibrations(cls, calibrations: Sequence["NightCalibration"],
                          minimum_nights: int = 2) -> "MasterPattern":
        bands = {c.band for c in calibrations}
        if len(bands) != 1:
            raise ValueError(f"a master pattern is one band; got {sorted(bands)}")
        orders = {}
        numbers = sorted({n for c in calibrations for n in c.orders})
        for number in numbers:
            measured = [c.orders[number] for c in calibrations
                        if number in c.orders and c.orders[number].pattern is not None]
            if len(measured) < minimum_nights:
                continue
            grid = measured[0].pattern_wavenumber_cm1
            stack = []
            for order in measured:
                values = np.interp(grid, order.pattern_wavenumber_cm1, order.pattern,
                                   left=np.nan, right=np.nan)
                # A calibration writes exactly zero where too few standards
                # measured a pixel; that is "unknown", not "no response".
                stack.append(np.where(values == 0.0, np.nan, values))
            with warnings.catch_warnings():
                # A pixel no night measured is an all-NaN slice; it becomes 0.
                warnings.simplefilter("ignore", RuntimeWarning)
                pattern = np.nanmedian(np.asarray(stack), axis=0)
            orders[number] = (grid, np.nan_to_num(pattern, nan=0.0))
        nights = tuple(str(c.source.get("record", "")) for c in calibrations)
        return cls(band=bands.pop(), orders=orders, nights=nights)

    def save(self, path: str | Path) -> Path:
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.attrs["format_version"] = FORMAT_VERSION
            handle.attrs["kind"] = "master_pattern"
            handle.attrs["band"] = self.band
            handle.attrs["nights"] = json.dumps(list(self.nights))
            for number, (grid, pattern) in sorted(self.orders.items()):
                group = handle.create_group(f"{self.band}{number}")
                group.attrs["number"] = number
                _store(group, "pattern_wavenumber_cm1", grid)
                _store(group, "pattern", pattern)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MasterPattern":
        import h5py

        with h5py.File(Path(path), "r") as handle:
            if text(handle.attrs.get("kind", "")) != "master_pattern":
                raise ValueError(f"{path} is not a master pattern")
            band = text(handle.attrs["band"])
            orders = {int(g.attrs["number"]): (g["pattern_wavenumber_cm1"][:].astype(float),
                                               g["pattern"][:].astype(float))
                      for g in handle.values()}
            nights = tuple(json.loads(text(handle.attrs["nights"])))
        return cls(band=band, orders=orders, nights=nights)

    def pattern_on(self, number: int, wavelength_nm: np.ndarray) -> np.ndarray:
        """This order's response on another frame's pixels; zero if unmeasured."""

        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        if int(number) not in self.orders:
            return np.zeros_like(wavelength_nm)
        grid, pattern = self.orders[int(number)]
        return np.interp(1.0e7 / wavelength_nm, grid, pattern, left=0.0, right=0.0)


def _store(group, name, values):
    """A pattern array in float32: the pattern is smooth to 51 pixels, and at
    6000 cm-1 float32 resolves 6e-4 cm-1 against a 0.04 cm-1 pixel, so double
    precision would only double a committed file's size."""

    group.create_dataset(name, data=np.asarray(values, dtype=np.float32),
                         compression="gzip", shuffle=True)


def _order_pattern(cache_dir: Path, frames, name, smooth_pixels, minimum_frames):
    """The median fractional response of one order over a night's standards.

    Each cached ``observed`` is the flux *after* the run divided out its own
    leave-one-out pattern, and ``model_flux`` was fitted to it, so the data
    before that division over the model is the response plus noise. Only where
    the model is well above zero: in a saturated core the ratio is meaningless
    and would dominate a median -- the same guard the fitting driver applies.
    """

    stack, grid = [], None
    for frame in frames:
        path = cache_dir / f"{frame}_{name}.npz"
        if not path.exists():
            continue
        with np.load(path) as arrays:
            nu = np.asarray(arrays["wavenumber_cm1"], dtype=float)
            observed = np.asarray(arrays["observed"], dtype=float)
            model = np.asarray(arrays["model_flux"], dtype=float)
            continuum = np.asarray(arrays["continuum"], dtype=float)
            mask = np.asarray(arrays["mask"], dtype=bool)
            applied = (np.asarray(arrays["response_pattern"], dtype=float)
                       if "response_pattern" in arrays.files else np.zeros_like(nu))
        raw = observed * (1.0 + np.clip(applied, -0.8, 5.0))
        deep = model > 0.2 * continuum
        fractional = np.where(mask & deep & np.isfinite(raw),
                              raw / np.maximum(model, 1e-12) - 1.0, np.nan)
        if grid is None:
            grid = nu
        elif nu.shape != grid.shape or not np.allclose(nu, grid, rtol=0, atol=1e-6):
            fractional = np.interp(grid, nu, fractional, left=np.nan, right=np.nan)
        stack.append(fractional)
    if len(stack) < minimum_frames:
        return None, None
    return grid, smoothed_frame_median(np.asarray(stack), minimum_frames, smooth_pixels)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    return str(value)
