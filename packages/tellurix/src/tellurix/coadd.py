"""A spectrum co-added from observations through different atmospheres.

An atlas averaged over days is not one atmosphere seen once. The IAG solar flux
atlas (Reiners et al. 2016) co-adds 1190 scans from nine days, March to July,
each 10-scan package shifted into the solar frame before averaging: its telluric
lines are the scan-weighted mean of several transmissions, each at its own air
mass, temperature, water column and velocity -- April's and July's sit 0.6 km/s
apart, about a line's own Doppler width. One transmission at one velocity cannot
represent that, and a single-atmosphere fit of the atlas is three times worse
than the same fit of a single spectrum (docs/solar_fit_plan.md, IAG).

:class:`CoaddedTelluricModel` is that mean: components, each a
:class:`TelluricModel` on its own profile (one per observing day), and in each
the times it was observed (:class:`CoaddNode`: a weight, the air mass and where
the telluric lines sit in the co-add's frame). The mean is taken before the
source and the instrument, because the atlas was averaged before any of ours
reached it. Every component shares the fitted parameters, so a column scale
multiplies each day's own profile column and the days keep their ERA5 ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from .model import TelluricModel
from .types import TelluricParameters

_C_KMS = 299792.458


@dataclass(frozen=True)
class CoaddNode:
    """One stretch of observing time within a component.

    ``velocity_kms`` is where the telluric lines sit in the co-add's frame,
    positive to the red: an atlas shifted by -v_r into the solar frame carries
    its telluric lines at -v_r, with v_r the Sun's radial velocity from the
    observer then.
    """

    weight: float
    zenith_angle_deg: float
    velocity_kms: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.weight) and self.weight > 0.0):
            raise ValueError("a node's weight must be positive")
        if not 0.0 <= self.zenith_angle_deg < 90.0:
            raise ValueError("a node's zenith angle must lie in [0, 90) degrees")
        if not np.isfinite(self.velocity_kms):
            raise ValueError("a node's velocity must be finite")


def fourier_shift(values: jnp.ndarray, samples: float) -> jnp.ndarray:
    """``values`` moved ``samples`` toward higher index, exactly for a band-limited signal.

    A cubic interpolation (``TelluricModel._shift_log_uniform``) is fine for a
    solar line many samples wide but leaves ~0.2% of a telluric line's depth when
    its Doppler sigma is under three samples; a phase ramp does not. The straight
    line through the two ends is taken out first and shifted analytically, so the
    transform sees a periodic signal and no edge jump rings across the window.
    """

    n = values.shape[0]
    index = jnp.arange(n)
    slope = (values[-1] - values[0]) / (n - 1)
    ramp = values[0] + slope * index
    frequency = jnp.fft.rfftfreq(n)
    moved = jnp.fft.irfft(jnp.fft.rfft(values - ramp) * jnp.exp(-2j * jnp.pi * frequency * samples), n)
    return moved + values[0] + slope * (index - samples)


class CoaddedTelluricModel(TelluricModel):
    """The weighted mean transmission of several atmospheres, used like one model.

    It is a :class:`TelluricModel` to everything that holds one: ``predict``,
    :class:`tellurix.OrderObjective` and ``precompute_opacity`` work unchanged.
    The zenith angle a caller passes is **ignored** -- each node carries its own
    -- so give the order zenith 0 to keep a record honest about it.
    """

    def __init__(self, components: Sequence[tuple[TelluricModel, Sequence[CoaddNode]]]) -> None:
        components = [(model, tuple(nodes)) for model, nodes in components]
        if not components:
            raise ValueError("a co-add needs at least one component")
        first = components[0][0]
        for model, nodes in components:
            if isinstance(model, CoaddedTelluricModel):
                raise ValueError("a component must be a single-atmosphere model")
            if not nodes:
                raise ValueError("every component needs at least one node")
            if not np.array_equal(np.asarray(model.wavenumber_cm1), np.asarray(first.wavenumber_cm1)):
                raise ValueError("every component must share one wavenumber grid")
            if model.species != first.species:
                raise ValueError("every component must carry the same species")
            if model.instrument is not first.instrument or model.pixel_integration != first.pixel_integration:
                raise ValueError("every component must share one instrument")
        total = sum(node.weight for _, nodes in components for node in nodes)
        self.components = [(model, tuple(CoaddNode(node.weight / total, node.zenith_angle_deg,
                                                   node.velocity_kms) for node in nodes))
                           for model, nodes in components]
        # What predict, OrderObjective and the window fit read from a model.
        self.__dict__.update({key: value for key, value in first.__dict__.items()
                              if key not in ("components", "_cross_section_evaluator")})
        self._cross_section_evaluator = None

    @property
    def nodes(self) -> list[CoaddNode]:
        return [node for _, nodes in self.components for node in nodes]

    def precompute_opacity(self, parameters: TelluricParameters | None = None, **kwargs) -> "CoaddedTelluricModel":
        return CoaddedTelluricModel([(model.precompute_opacity(parameters, **kwargs), nodes)
                                     for model, nodes in self.components])

    def _mean(self, depth: jnp.ndarray, nodes: Sequence[CoaddNode]) -> jnp.ndarray:
        total = jnp.zeros_like(depth)
        for node in nodes:
            transmission = jnp.exp(-depth / np.cos(np.radians(node.zenith_angle_deg)))
            # Positive is to the red, toward lower wavenumber: lower index.
            samples = -np.log1p(node.velocity_kms / _C_KMS) * _C_KMS / self.velocity_step_kms
            total = total + node.weight * (transmission if samples == 0.0
                                           else fourier_shift(transmission, samples))
        return total

    def transmission(self, parameters: TelluricParameters, zenith_angle_deg=0.0) -> jnp.ndarray:
        del zenith_angle_deg  # each node carries its own
        return sum(self._mean(model.vertical_optical_depth(parameters), nodes)
                   for model, nodes in self.components)

    def vertical_optical_depth(self, parameters: TelluricParameters) -> jnp.ndarray:
        raise TypeError("a co-add has no single optical depth; use transmission()")

    def species_transmission(self, parameters: TelluricParameters, zenith_angle_deg=0.0) -> dict:
        """Each absorber's co-added transmission as if it absorbed alone.

        A mean of exponentials does not factor, so unlike a single atmosphere's
        these do not multiply back to ``transmission`` exactly; the difference is
        second order in the spread of the nodes' optical depths.
        """

        del zenith_angle_deg
        pieces = {}
        for model, nodes in self.components:
            for name, depth in model.species_vertical_optical_depth(parameters).items():
                pieces[name] = pieces.get(name, 0.0) + self._mean(depth, nodes)
        return pieces
