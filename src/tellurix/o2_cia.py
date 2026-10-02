"""O2 collision-induced absorption, as LBLRTM 12.17 computes it.

The line list carries O2's discrete transitions; collisions add broad bands
that are in no line list, and LBLRTM's continuum module (``contnm.f90``)
supplies four of them in the range this package fits. Without them the A-band
fit returned an O2 column 7% high at air mass 5.37, which is exactly what LBLRTM
predicted for a model missing this term (docs/solar_fit_plan.md §4n).

=========  ==============  ============================================
term       cm-1            source
=========  ==============  ============================================
O2INF1     7536-8500       1.27 um, Mate et al. 1999, JGR 104, 30585
O2INF2     9100-11000      1.06 um, Mlawer et al. 1998, analytic
O2INF3     12961.5-13221.5 A-band, Mlawer, from solar FTS measurements
O2_VIS     15140-29870     visible O2-O2, Greenblatt et al. 1990
=========  ==============  ============================================

LBLRTM's fifth, the 1340-1850 cm-1 fundamental, is outside every fitted range
but photatl's last 2 cm-1, and is not carried.

Each term is reproduced from the subroutine that applies it, per layer, with
LBLRTM's own layer quantities -- amagat = (P/1013)(273/T), RHOAVE =
(P/1013)(296/T), W(O2) the layer's O2 column, WTOT its total column -- its
radiation term, and its four-point XINT interpolation onto the model grid
(``tellurix.mt_ckd._cubic_stencils``). The coefficients are those tabulated in
``_o2_cia_tables``, extracted from ``contnm.f90`` by scripts/extract_o2_cia.py.

The fitted O2 column scale reaches these terms through the scaled VMR, so a
term that is quadratic in O2 (O2-O2) scales as the square and one that is
linear (O2-N2) linearly, as in the sky.
"""

# Copyright © Atmospheric and Environmental Research, Inc., 2022.
# All rights reserved. This implementation is derived from the MT_CKD module
# distributed with LBLRTM. AER grants use, copying, modification, and
# redistribution for scientific and research purposes with this notice and
# appropriate acknowledgment. It may not be incorporated into proprietary or
# commercial software without AER's express written consent. It is provided
# without express or implied warranties. General reference: Mlawer et al.
# (2012), doi:10.1098/rsta.2011.0295. Questions: aer_contnm@aer.com.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
import numpy as np

from . import _o2_cia_tables as tables
from .mt_ckd import _RADIATION_CONSTANT_K_CM, _cubic_stencils
from .types import AtmosphereProfile

_LOSCHMIDT_CM3 = 2.68675e19
# contnm.f90: P0 = 1013. mb with T0 = 296 K for RHOAVE, and 273 K for amagat.
_P0_HPA = 1013.0
_T0_K = 296.0
_T_AMAGAT_K = 273.0
# O2INF1's broadening efficiencies, relative to O2 (Mate et al. 1999).
_A_O2, _A_N2, _A_H2O = 1.0 / 0.446, 0.3 / 0.446, 1.0
# O2INF2's two damped Lorentzians.
_O2INF2 = dict(v1=9375.0, hw1=58.96, v2=9439.0, hw2=45.04, s1=1.166e-04, s2=3.086e-05)
# O2_vis's unit conversion, verbatim from contnm.f90.
_O2_VIS_FACTOR = 1.0 / ((_LOSCHMIDT_CM3 * 1.0e-20 * (55.0 * 273.0 / 296.0) ** 2) * 89.5)


def _radiation(nu: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    """LBLRTM's RADFN: v tanh(hcv / 2kT), with its small-argument branch."""

    ratio = nu * _RADIATION_CONSTANT_K_CM / temperature_k
    return np.where(ratio <= 0.01, 0.5 * ratio * nu, nu * np.tanh(0.5 * ratio))


def _o2inf2(nu: np.ndarray) -> np.ndarray:
    """O2INF2's coefficient, C0 = O2INF / v, zero outside the open (9100, 11000)."""

    p = _O2INF2
    dv1, dv2 = nu - p["v1"], nu - p["v2"]
    damp1 = np.where(dv1 < 0.0, np.exp(dv1 / 176.1), 1.0)
    damp2 = np.where(dv2 < 0.0, np.exp(dv2 / 176.1), 1.0)
    value = 0.31831 * ((p["s1"] * damp1 / p["hw1"]) / (1.0 + (dv1 / p["hw1"]) ** 2)
                       + (p["s2"] * damp2 / p["hw2"]) / (1.0 + (dv2 / p["hw2"]) ** 2)) * 1.054
    return np.where((nu > 9100.0) & (nu < 11000.0), value / nu, 0.0)


def _tabulated(grid, values, factor=1.0):
    v1, _, dv, npt = grid
    table = np.asarray(values, dtype=float)

    def coefficient(nu: np.ndarray) -> np.ndarray:
        index = np.rint((nu - v1) / dv).astype(int)
        inside = (index >= 0) & (index < npt)
        out = np.zeros_like(nu)
        out[inside] = factor * table[index[inside]] / nu[inside]
        return out

    return coefficient, (v1, dv)


# name -> (coefficient C0(v) on the term's own lattice, (lattice origin, step),
#          the range LBLRTM gates the term on)
_TERMS = {
    "O2INF1": (*_tabulated(tables.O2INF1_GRID, tables.O2INF1), (7536.0, 8500.0)),
    "O2INF2": (_o2inf2, (9100.0, 2.0), (9100.0, 11000.0)),
    "O2INF3": (*_tabulated(tables.O2INF3_GRID, tables.O2INF3), (12961.5, 13221.5)),
    "O2_VIS": (*_tabulated(tables.O2_VIS_GRID, tables.O2_VIS, _O2_VIS_FACTOR), (15000.0, 29870.0)),
}


@dataclass(frozen=True)
class O2CollisionInducedContinuum:
    """LBLRTM's O2 collision-induced continua for one spectral grid.

    Pass it to :class:`TelluricModel` through :class:`ContinuumSum` beside the
    MT_CKD water continuum. Terms whose range the grid does not reach are
    dropped at construction, so a window far from every band costs nothing.
    """

    wavenumber_cm1: np.ndarray

    def __post_init__(self) -> None:
        nu = np.asarray(self.wavenumber_cm1, dtype=float)
        if nu.ndim != 1 or nu.size < 2 or np.any(np.diff(nu) <= 0.0):
            raise ValueError("wavenumbers must be a strictly increasing 1-D grid")
        object.__setattr__(self, "wavenumber_cm1", nu)

    @property
    def terms(self) -> tuple[str, ...]:
        """The terms this grid reaches."""

        lo, hi = self.wavenumber_cm1[0], self.wavenumber_cm1[-1]
        return tuple(name for name, (_, _, (a, b)) in _TERMS.items() if hi > a and lo < b)

    def bind(self, profile: AtmosphereProfile) -> "_BoundO2CollisionInducedContinuum":
        """Interpolate each term's C0(v) RADFN(v, T) onto the grid, layer by layer."""

        if "O2" not in profile.vmr:
            raise ValueError("the O2 collision-induced continuum needs an O2 VMR profile")
        nu = self.wavenumber_cm1
        temperature = np.asarray(profile.temperature_k, dtype=float)
        basis = {}
        for name in self.terms:
            coefficient, (origin, step), _ = _TERMS[name]
            # The term's own lattice, padded so XINT's four-point stencil has
            # zeros to read beyond the table -- as LBLRTM's zero-filled C array.
            first = origin + step * (np.floor((nu[0] - origin) / step) - 3)
            last = origin + step * (np.ceil((nu[-1] - origin) / step) + 3)
            lattice = np.arange(first, last + 0.5 * step, step)
            indices, weights = _cubic_stencils(lattice, nu)
            radiated = coefficient(lattice)[None, :] * _radiation(
                lattice[None, :], temperature[:, None])
            basis[name] = np.sum(radiated[:, indices] * weights[None, :, :], axis=-1)
        edges = np.asarray(profile.pressure_edges_bar, dtype=float)
        return _BoundO2CollisionInducedContinuum(
            pressure_edges_bar=edges,
            temperature_k=temperature,
            pressure_hpa=0.5 * (edges[:-1] + edges[1:]) * 1000.0,
            air_column_cm2=np.asarray(profile.air_column_cm2, dtype=float),
            basis=basis,
        )


@dataclass(frozen=True)
class _BoundO2CollisionInducedContinuum:
    pressure_edges_bar: np.ndarray
    temperature_k: np.ndarray
    pressure_hpa: np.ndarray
    air_column_cm2: np.ndarray
    basis: dict

    def optical_depth(
        self, profile: AtmosphereProfile, scaled_vmr: Mapping[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Vertical optical depth by layer, (layer, wavenumber)."""

        if not (np.array_equal(self.pressure_edges_bar, np.asarray(profile.pressure_edges_bar))
                and np.array_equal(self.temperature_k, np.asarray(profile.temperature_k))):
            raise ValueError("prepared O2 continuum does not match the model profile")
        x_o2 = jnp.asarray(scaled_vmr["O2"])
        x_h2o = jnp.asarray(scaled_vmr["H2O"]) if "H2O" in scaled_vmr else jnp.zeros_like(x_o2)
        x_n2 = 1.0 - x_h2o - x_o2
        wtot = jnp.asarray(self.air_column_cm2)
        w_o2 = wtot * x_o2
        p, t = jnp.asarray(self.pressure_hpa), jnp.asarray(self.temperature_k)
        amagat = (p / _P0_HPA) * (_T_AMAGAT_K / t)
        rhoave = (p / _P0_HPA) * (_T0_K / t)
        factor = {
            "O2INF1": (w_o2 / _LOSCHMIDT_CM3) * amagat
                      * (_A_O2 * x_o2 + _A_N2 * x_n2 + _A_H2O * x_h2o),
            "O2INF2": (w_o2 / wtot) * (1.0 / 0.209) * (w_o2 * 1.0e-20) * rhoave,
            "O2INF3": (w_o2 / _LOSCHMIDT_CM3) * amagat,
            "O2_VIS": (w_o2 / wtot) * (w_o2 * 1.0e-20) * amagat,
        }
        tau = jnp.zeros((wtot.size, next(iter(self.basis.values())).shape[1])) if self.basis else 0.0
        for name, basis in self.basis.items():
            tau = tau + factor[name][:, None] * jnp.asarray(basis)
        return tau


@dataclass(frozen=True)
class ContinuumSum:
    """Several continuum backends as one: their optical depths add.

    :class:`TelluricModel` takes a single continuum; this is how MT_CKD water
    and the O2 collision-induced terms ride together.
    """

    members: tuple

    def bind(self, profile: AtmosphereProfile) -> "ContinuumSum":
        return ContinuumSum(tuple(
            member.bind(profile) if hasattr(member, "bind") else member
            for member in self.members))

    def optical_depth(
        self, profile: AtmosphereProfile, scaled_vmr: Mapping[str, jnp.ndarray]
    ) -> jnp.ndarray:
        total = 0.0
        for member in self.members:
            total = total + member.optical_depth(profile, scaled_vmr)
        return total
