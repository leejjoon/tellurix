"""A fixed stellar spectrum used as the source behind the atmosphere.

Where the spectrum comes from is deliberately not this module's concern: it
accepts an array or an ``npz``, so a Payne-Zero synthesis, a PHOENIX or MARCS
grid, or any other model can supply it. What this module owns is getting it
onto the forward model's grid at the right resolution, broadened by the star's
own velocity fields, and normalized so it does not fight the fitted continuum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np


_C_KMS = 299792.458


@dataclass(frozen=True)
class StellarSpectrum:
    """A stellar model in the stellar rest frame, on vacuum wavenumbers."""

    wavenumber_cm1: np.ndarray
    flux: np.ndarray
    meta: Mapping[str, str] = field(default_factory=dict)
    continuum: np.ndarray | None = None
    """The model's own continuum, in its own units, on the same grid.

    ``flux`` is the spectrum already divided by this, so the continuum is not
    needed to use the source and is carried only so a consumer can put the
    stellar continuum back, or test it. It is the physical continuum and it can
    have real structure the fitted Chebyshev cannot represent -- the hydrogen
    bound-free edges are steps, and the Brackett edge at 1458.8 nm is a 0.19%
    discontinuity in the Arcturus model, 300x any other single-sample step.
    """

    def __post_init__(self) -> None:
        nu = np.asarray(self.wavenumber_cm1, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        if nu.ndim != 1 or nu.size < 8:
            raise ValueError("a stellar spectrum needs at least eight samples")
        if flux.shape != nu.shape:
            raise ValueError("stellar flux must match its wavenumber grid")
        if np.any(~np.isfinite(nu)) or np.any(np.diff(nu) <= 0.0):
            raise ValueError("stellar wavenumbers must be finite and strictly increasing")
        if np.any(~np.isfinite(flux)) or np.any(flux <= 0.0):
            raise ValueError("stellar flux must be finite and positive")
        if self.continuum is not None:
            continuum = np.asarray(self.continuum, dtype=float)
            if continuum.shape != nu.shape:
                raise ValueError("the stellar continuum must match its wavenumber grid")
            if np.any(~np.isfinite(continuum)) or np.any(continuum <= 0.0):
                raise ValueError("the stellar continuum must be finite and positive")
            object.__setattr__(self, "continuum", continuum)
        object.__setattr__(self, "wavenumber_cm1", nu)
        object.__setattr__(self, "flux", flux)

    @classmethod
    def flat(cls, wavenumber_cm1: np.ndarray) -> "StellarSpectrum":
        """A featureless unit source, for telluric-only fits and tests."""

        nu = np.asarray(wavenumber_cm1, dtype=float)
        return cls(nu, np.ones_like(nu), {"source": "flat"})

    @classmethod
    def from_wavelength_nm(
        cls, wavelength_vacuum_nm: np.ndarray, flux: np.ndarray, **meta: str
    ) -> "StellarSpectrum":
        wavelength = np.asarray(wavelength_vacuum_nm, dtype=float)
        order = np.argsort(1.0e7 / wavelength)
        return cls((1.0e7 / wavelength)[order], np.asarray(flux, dtype=float)[order], dict(meta))

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        wavenumber_key: str = "wavenumber_cm1",
        flux_key: str = "flux",
    ) -> "StellarSpectrum":
        with np.load(path) as values:
            if wavenumber_key in values:
                nu = values[wavenumber_key]
            elif "wavelength_vacuum_nm" in values:
                return cls.from_wavelength_nm(
                    values["wavelength_vacuum_nm"], values[flux_key], source=str(path)
                )
            else:
                raise ValueError(f"{path} has neither {wavenumber_key} nor wavelength_vacuum_nm")
            flux = values[flux_key]
            return cls(nu, flux, {"source": str(path)},
                       continuum=_oriented_continuum(values, flux, path))

    @property
    def velocity_step_kms(self) -> float:
        return float(np.median(np.diff(np.log(self.wavenumber_cm1))) * _C_KMS)


def _oriented_continuum(values, flux, path) -> np.ndarray | None:
    """``flux_continuum`` from an npz, put in the same order as ``flux``.

    Payne Zero's writers reordered ``flux`` into ascending wavenumber and left
    ``flux_total`` and ``flux_continuum`` in the synthesizer's own order, so in
    the two committed models those two arrays are back to front. Rather than
    hard-code a reversal, which would then corrupt a file written correctly,
    this uses the identity the three arrays satisfy -- ``flux`` is
    ``flux_total / flux_continuum`` -- and keeps whichever orientation
    reproduces it. A file that satisfies neither gets no continuum rather than
    a wrong one.
    """

    if "flux_continuum" not in values or "flux_total" not in values:
        return None
    flux = np.asarray(flux, dtype=float)
    continuum = np.asarray(values["flux_continuum"], dtype=float)
    total = np.asarray(values["flux_total"], dtype=float)
    if continuum.shape != flux.shape or total.shape != flux.shape:
        return None
    best, error = None, np.inf
    for candidate_continuum, candidate_total in ((continuum, total),
                                                 (continuum[::-1], total[::-1])):
        if np.any(candidate_continuum <= 0.0) or np.any(~np.isfinite(candidate_continuum)):
            continue
        residual = float(np.max(np.abs(flux - candidate_total / candidate_continuum)))
        if residual < error:
            best, error = candidate_continuum, residual
    # A real match is at rounding level. Anything else means these arrays are
    # not this spectrum's continuum and guessing would be worse than nothing.
    if best is None or error > 1.0e-6:
        return None
    return np.ascontiguousarray(best)


def resample_stellar_source(spectrum: StellarSpectrum, wavenumber_cm1: np.ndarray) -> np.ndarray:
    """Interpolate a stellar model onto the forward model's grid.

    Both bounds are enforced rather than clamped. Extrapolating past the end of
    the model would silently flatten the padded wings that the line spread
    function reaches into, and a source coarser than the model grid would be
    upsampled into a spectrum that looks higher resolution than it is.
    """

    target = np.asarray(wavenumber_cm1, dtype=float)
    source = spectrum.wavenumber_cm1
    if source[0] > target[0] or source[-1] < target[-1]:
        raise ValueError(
            f"the stellar model covers {source[0]:.3f}-{source[-1]:.3f} cm-1 but the model grid "
            f"needs {target[0]:.3f}-{target[-1]:.3f} cm-1"
        )
    # Compare in velocity, over the target's own range. Both grids are uniform
    # in log wavenumber, where a constant step is a constant velocity, and a
    # raw wavenumber spacing is not comparable between them: this source spans
    # 900-5400 nm, so its median spacing is the value near 4536 cm-1, six times
    # its spacing at the red end. Tested against the model's local spacing that
    # rejected every page below 2016 cm-1 even though the source is finer than
    # the model everywhere -- R = 600,000 against 400,000.
    lower, upper = np.searchsorted(source, [target[0], target[-1]])
    window = source[max(int(lower) - 1, 0) : int(upper) + 1]
    source_step = float(np.median(np.diff(np.log(window))) * _C_KMS)
    target_step = float(np.median(np.diff(np.log(target))) * _C_KMS)
    if source_step > 1.5 * target_step:
        raise ValueError(
            f"the stellar model is sampled at {source_step:.4f} km/s over this window, coarser "
            f"than the model grid's {target_step:.4f} km/s; synthesize it at higher resolution"
        )
    return np.interp(target, source, spectrum.flux)


def resample_stellar_continuum(
    spectrum: StellarSpectrum, wavenumber_cm1: np.ndarray
) -> np.ndarray | None:
    """The model's own continuum on an arbitrary grid, or None if it has none.

    Plain interpolation, with no broadening and no instrument profile: a
    continuum is smooth on those scales by construction. The one exception is a
    bound-free edge, which is a genuine step -- 0.19% at the Brackett limit for
    the Arcturus model and 5.6% for the A0V one -- and interpolating a step onto
    a coarser grid spreads it over a sample. That is a faithful description of
    what a pixel sees; it is not a reason to treat the edge as resolved.
    """

    if spectrum.continuum is None:
        return None
    target = np.asarray(wavenumber_cm1, dtype=float)
    return np.interp(target, spectrum.wavenumber_cm1, spectrum.continuum)


def _rotation_kernel(velocity_kms: np.ndarray, vsini_kms: float, limb_darkening: float) -> np.ndarray:
    """Rigid-rotation profile with a linear limb-darkening coefficient."""

    x = np.clip(velocity_kms / vsini_kms, -1.0, 1.0)
    root = np.sqrt(np.maximum(1.0 - x**2, 0.0))
    kernel = (2.0 * (1.0 - limb_darkening) * root + 0.5 * np.pi * limb_darkening * (1.0 - x**2))
    kernel = np.where(np.abs(velocity_kms) <= vsini_kms, kernel, 0.0)
    return kernel / np.sum(kernel)


def _macroturbulence_kernel(velocity_kms: np.ndarray, zeta_kms: float) -> np.ndarray:
    """Radial-tangential macroturbulence, equal radial and tangential areas.

    This is not a Gaussian: it has a sharper core and wider wings, which is why
    a Gaussian substitute needs its width calibrated rather than set to zeta.
    """

    scaled = velocity_kms / zeta_kms
    kernel = np.exp(-(scaled**2))
    return kernel / np.sum(kernel)


def broaden_stellar_source(
    flux: np.ndarray,
    velocity_step_kms: float,
    *,
    vsini_kms: float = 0.0,
    limb_darkening: float = 0.6,
    macroturbulence_kms: float = 0.0,
) -> np.ndarray:
    """Apply rotation and macroturbulence on a log-uniform grid.

    Both are constant in velocity, which is exactly what a log-uniform
    wavenumber grid samples uniformly.
    """

    values = np.asarray(flux, dtype=float)
    if velocity_step_kms <= 0.0:
        raise ValueError("the grid's velocity step must be positive")
    if vsini_kms < 0.0 or macroturbulence_kms < 0.0:
        raise ValueError("broadening velocities must be nonnegative")
    if not 0.0 <= limb_darkening <= 1.0:
        raise ValueError("the limb-darkening coefficient must be in [0, 1]")

    for width, builder in (
        (vsini_kms, lambda v: _rotation_kernel(v, vsini_kms, limb_darkening)),
        (macroturbulence_kms, lambda v: _macroturbulence_kernel(v, macroturbulence_kms)),
    ):
        if width <= 0.0:
            continue
        half = int(np.ceil(3.0 * width / velocity_step_kms))
        if half < 1:
            continue
        offsets = np.arange(-half, half + 1) * velocity_step_kms
        kernel = builder(offsets)
        padded = np.pad(values, half, mode="edge")
        values = np.convolve(padded, kernel, mode="valid")
    return values


def prepare_stellar_source(
    spectrum: StellarSpectrum,
    model,
    *,
    vsini_kms: float = 0.0,
    limb_darkening: float = 0.6,
    macroturbulence_kms: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Resample, broaden, and normalize a stellar model for one forward model.

    The absolute flux scale is exactly degenerate with the constant term of the
    fitted log-Chebyshev continuum, so normalizing by the median keeps that
    coefficient near zero instead of carrying the star's units.
    """

    grid = np.asarray(model.wavenumber_cm1)
    values = resample_stellar_source(spectrum, grid)
    values = broaden_stellar_source(
        values,
        model.velocity_step_kms,
        vsini_kms=vsini_kms,
        limb_darkening=limb_darkening,
        macroturbulence_kms=macroturbulence_kms,
    )
    if normalize:
        median = float(np.median(values))
        if not np.isfinite(median) or median <= 0.0:
            raise ValueError("the stellar source has no positive median to normalize by")
        values = values / median
    return values
