"""Direct Voigt opacity with static selection of possible line-core pairs.

ExoJAX's vectorized ``where`` evaluates Algorithm 916 even in distant wings.
Here only pairs that can enter that branch below a configured temperature
are sent through it. No lines or wings are removed.
"""

import jax
import jax.numpy as jnp
import numpy as np

from exojax.opacity import OpaDirect
from exojax.opacity.lpf.lpf import hjert, xsvector as lpf_xsvector
from exojax.special.faddeeva import asymptotic_wofz, imwofz
from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.utils.constants import Tref_original


@jax.custom_jvp
def _wing(x, a):
    return jnp.real(asymptotic_wofz(x, a))


@_wing.defjvp
def _wing_jvp(primals, tangents):
    # Match ExoJAX's analytic Hjerting JVP, including its wing approximation.
    x, a = primals
    dx, da = tangents
    value = asymptotic_wofz(x, a)
    h, l = jnp.real(value), jnp.imag(value)
    return h, (2 * a * l - 2 * x * h) * dx + (
        2 * x * l + 2 * a * h - 2 / jnp.sqrt(jnp.pi)
    ) * da


@jax.custom_jvp
def _mixed_wing(x, a):
    """Float32 wing evaluation after forming accurate float64 coordinates."""
    return jnp.real(asymptotic_wofz(x.astype(jnp.float32), a.astype(jnp.float32))).astype(x.dtype)


@_mixed_wing.defjvp
def _mixed_wing_jvp(primals, tangents):
    # Algebraically simplify ExoJAX's derivative identity for its truncated
    # wing series. Direct evaluation of 2*x*Im(w)-2/sqrt(pi) catastrophically
    # cancels in float32. With z=x+i*a and u=1/z**2, the same identity is
    # dH/dx=Im(u+1.5*u**2+3.75*u**3)/sqrt(pi), dH/da=Re(...)/sqrt(pi).
    x, a = primals
    dx, da = tangents
    z = x.astype(jnp.float32) + 1j * a.astype(jnp.float32)
    u = 1 / (z * z)
    derivative = (u * (1 + u * (1.5 + 3.75 * u))) / jnp.sqrt(jnp.pi)
    return _mixed_wing(x, a), (
        jnp.imag(derivative).astype(x.dtype) * dx
        + jnp.real(derivative).astype(a.dtype) * da
    )


def _wing_imag(x, a):
    """Im w(x + ia) in the wing, for the line-coupling (dispersion) term.

    No float32 variant: the term is Y*P times smaller than the line, so its
    cost is not where mixed precision pays, and Im w has no cancellation."""
    return jnp.imag(asymptotic_wofz(x, a))


@jax.custom_jvp
def _core_imag(x, a):
    """Im w(x + ia) with the branch ExoJAX's hjert uses for the real part.

    Both branches are evaluated under ``where``, so each gets coordinates that
    are safe for it: the asymptotic series is singular at z = 0, which a core
    pair can reach.
    """
    small = x * x + a * a < 111.0
    series = imwofz(jnp.where(small, x, 0.0), a)
    asymptotic = jnp.imag(asymptotic_wofz(jnp.where(small, 20.0, x), a))
    return jnp.where(small, series, asymptotic)


@_core_imag.defjvp
def _core_imag_jvp(primals, tangents):
    # Differentiating Algorithm 916's series directly stores every term, which
    # under vmap over a dense (line, grid) fallback is tens of gigabytes. w is
    # analytic with w'(z) = -2 z w(z) + 2i/sqrt(pi), and dz/da = i, so both
    # partials follow from Re w and Im w alone -- as hjert's JVP does.
    x, a = primals
    dx, da = tangents
    real, imag = hjert(x, a), _core_imag(x, a)
    derivative_real = -2.0 * (x * real - a * imag)
    derivative_imag = -2.0 * (x * imag + a * real) + 2.0 / jnp.sqrt(jnp.pi)
    return imag, derivative_imag * dx + derivative_real * da


# LBLRTM cuts the coupling term off where it cuts the line (oprop.f90, CONVF4):
# at 25 cm-1, less the value that brings it to zero there. A Rosenkranz term
# decays only as 1/x, and its tails cancel across a whole band (sum S*Y = 0),
# not across a window's line list; AER's Y are made for this shape.
_COUPLING_CUTOFF_CM1 = 25.0


def _dispersion(imag, offset, gamma, scale):
    """The coupling profile in Im w units, truncated as LBLRTM truncates it.

    In the Lorentz limit Im w * scale/sqrt(pi) is offset / (pi (offset^2 +
    gamma^2)); LBLRTM's pedestal is offset / (pi (gamma^2 + B^2)), which makes
    the term vanish at |offset| = B.
    """
    pedestal = offset / (jnp.sqrt(jnp.pi) * scale * (gamma * gamma + _COUPLING_CUTOFF_CM1 ** 2))
    return jnp.where(jnp.abs(offset) <= _COUPLING_CUTOFF_CM1, imag - pedestal, 0.0)


class SparseCoreDirect(OpaDirect):
    """ExoJAX Direct with identical branch selection and a compact core list.

    Above ``maximum_temperature_k`` a full Direct calculation is used. HITRAN
    air-pressure line shifts can be enabled explicitly; the sparse core list
    then covers every shift up to ``maximum_pressure_bar`` and falls back to
    the full calculation above it. HITRAN databases are supported.
    ``mixed_precision=True`` evaluates wing values and stable wing derivative
    coefficients in float32. Coordinates, line physics, cores, accumulation,
    and the returned cross sections retain the input precision.

    ``line_coupling=True`` adds first-order (Rosenkranz) line mixing from the
    database's coupling coefficients, as LBLRTM applies AER's: each line becomes
    S (1 + G p^2) [Re w(z) + Y p Im w(z)] with p the pressure in atmospheres,
    Y and G interpolated in temperature (``AERLineDatabase.coupling``). The
    Im w term is cut at 25 cm-1 less a pedestal, as LBLRTM cuts it; the line
    itself keeps its full Voigt shape. LBLRTM approximates the dispersion
    profile as the Voigt line times its offset in Voigt widths, which is exact
    in the Lorentz limit; Im w is exact everywhere. Off by default, so earlier
    results reproduce.
    """

    def __init__(
        self,
        mdb,
        nu_grid,
        minimum_temperature_k=150.0,
        maximum_temperature_k=400.0,
        mixed_precision=False,
        pressure_shift=False,
        maximum_pressure_bar=2.0,
        line_coupling=False,
    ):
        if mdb.dbtype != "hitran":
            raise ValueError("SparseCoreDirect requires a HITRAN-style database")
        if (not np.isfinite(minimum_temperature_k) or minimum_temperature_k <= 0
                or not np.isfinite(maximum_temperature_k)
                or maximum_temperature_k < minimum_temperature_k):
            raise ValueError("temperature bounds must be finite, positive, and ordered")
        if not np.isfinite(maximum_pressure_bar) or maximum_pressure_bar <= 0:
            raise ValueError("maximum pressure must be finite and positive")
        if pressure_shift and not hasattr(mdb, "delta_air"):
            raise ValueError("pressure_shift requires HITRAN delta_air coefficients")
        if line_coupling and not hasattr(mdb, "coupling"):
            raise ValueError("line_coupling requires a database with coupling coefficients")
        super().__init__(mdb, nu_grid)
        self.minimum_temperature_k = float(minimum_temperature_k)
        self.maximum_temperature_k = float(maximum_temperature_k)
        self.maximum_pressure_bar = float(maximum_pressure_bar)
        self.mixed_precision = bool(mixed_precision)
        self.pressure_shift = bool(pressure_shift)
        # A database with no coupled line in range needs no dispersion term,
        # and leaving it out keeps the kernel exactly what it was.
        self.line_coupling = bool(line_coupling) and bool(mdb.has_line_coupling)
        self.delta_air = jnp.asarray(mdb.delta_air) if pressure_shift else None
        line_center = np.asarray(mdb.nu_lines, dtype=float)
        grid = np.asarray(nu_grid, dtype=float)
        sigma = np.asarray(doppler_sigma(mdb.nu_lines, maximum_temperature_k, mdb.molmass))
        shift_margin = (
            np.abs(np.asarray(mdb.delta_air)) * maximum_pressure_bar / 1.01325
            * Tref_original / minimum_temperature_k
            if pressure_shift else np.zeros_like(line_center)
        )
        # Per-line half width of the region that can reach Algorithm 916's
        # branch. Keeping it one-dimensional is what lets the core test be
        # recomputed inside the kernel instead of read back from a matrix.
        core_half_width = np.sqrt(222.0) * sigma + shift_margin
        # Both arrays are sorted, so each line's core region is one contiguous
        # run of grid samples. Binary search finds it in O(log N) instead of
        # scanning a dense (line, grid) boolean, which was the last place a
        # matrix of that size was built -- 2.1 s and a gigabyte on a wide page.
        # A pair right at the boundary has x*x >= 111, where ExoJAX's hjert
        # already takes its asymptotic branch, so which side it falls on cannot
        # change a value.
        lower = np.searchsorted(grid, line_center - core_half_width, side="left")
        upper = np.searchsorted(grid, line_center + core_half_width, side="right")
        counts = np.maximum(upper - lower, 0)
        total = int(counts.sum())
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        self.core_line = np.repeat(np.arange(line_center.size), counts)
        self.core_grid = np.arange(total) - starts + np.repeat(lower, counts)
        self._nu_grid = jnp.asarray(grid)
        self._nu_lines = jnp.asarray(line_center)
        self._core_half_width = jnp.asarray(core_half_width)
        self._core_grid = jnp.asarray(self.core_grid)
        self._core_line = jnp.asarray(self.core_line)
        # ExoJAX stores nu_grid[None, :] - nu_lines[:, None] as a dense
        # (line, grid) float64 matrix. At terrestrial line densities that is
        # hundreds of megabytes re-read once per layer, which dominates the
        # kernel; every use below rebuilds it from the two 1-D vectors instead.
        self.opainfo = None

    def xsvector(self, T, P, Pself=0.0):
        within_pressure_range = (
            (P >= 0.0) & (P <= self.maximum_pressure_bar)
            if self.pressure_shift else True
        )
        return jax.lax.cond(
            (T >= self.minimum_temperature_k) & (T <= self.maximum_temperature_k)
            & within_pressure_range,
            lambda: self._sparse_xsvector(T, P, Pself),
            lambda: self._full_xsvector(T, P, Pself),
        )

    def _line_shift(self, T, P):
        """Per-line pressure shift, or zero when shifts are disabled."""
        if not self.pressure_shift:
            return None
        # Match LBLRTM's RHORAT scaling: AER/HITRAN shifts are referenced to
        # one atmosphere at Tref, so their magnitude follows number density.
        return self.delta_air * ((P / 1.01325) * (Tref_original / T))

    def _dense_offsets(self, T, P):
        """Rebuild ExoJAX's (line, grid) offset matrix for the fallback path."""
        offsets = self._nu_grid[None, :] - self._nu_lines[:, None]
        shift = self._line_shift(T, P)
        return offsets if shift is None else offsets - shift[:, None]

    def _line_parameters(self, T, P, Pself):
        mdb = self.mdb
        sigma = doppler_sigma(mdb.nu_lines, T, mdb.molmass)
        gamma = gamma_hitran(P, T, Pself, mdb.n_air, mdb.gamma_air, mdb.gamma_self) + gamma_natural(mdb.A)
        strength = line_strength(T, mdb.logsij0, mdb.nu_lines, mdb.elower,
                                 mdb.qr_interp(mdb.isotope, T, Tref_original), Tref_original)
        return sigma, gamma, strength

    def _coupling(self, T, P, strength):
        """Coupling coefficient Y*p per line, and the strength with G applied."""
        mixing, factor = self.mdb.coupling(T, P)
        return mixing, strength * factor

    def _full_xsvector(self, T, P, Pself):
        # Identical to OpaDirect.xsvector, which builds the same line
        # parameters, except that the offset matrix is rebuilt here rather
        # than held on the device for a branch that is almost never taken.
        sigma, gamma, strength = self._line_parameters(T, P, Pself)
        if not self.line_coupling:
            return lpf_xsvector(self._dense_offsets(T, P), sigma, gamma, strength)
        mixing, strength = self._coupling(T, P, strength)
        scale = 1 / (jnp.sqrt(2.0) * sigma)
        offsets = self._dense_offsets(T, P)
        x = offsets * scale[:, None]
        a = jnp.broadcast_to((scale * gamma)[:, None], x.shape)
        real = jax.vmap(jax.vmap(hjert))(x, a)
        imag = _dispersion(jax.vmap(jax.vmap(_core_imag))(x, a), offsets,
                           gamma[:, None], scale[:, None])
        weights = strength * scale / jnp.sqrt(jnp.pi)
        return jnp.sum((real + mixing[:, None] * imag) * weights[:, None], axis=0)

    def _sparse_xsvector(self, T, P, Pself):
        sigma, gamma, strength = self._line_parameters(T, P, Pself)
        if self.line_coupling:
            mixing, strength = self._coupling(T, P, strength)
        scale = 1 / (jnp.sqrt(2.0) * sigma)
        a = scale * gamma
        weights = strength * scale / jnp.sqrt(jnp.pi)
        shift = self._line_shift(T, P)
        grid, centers = self._nu_grid, self._nu_lines

        # Both the offsets and the core/wing split are exact functions of two
        # 1-D vectors, so they are recomputed inside the kernel. XLA then fuses
        # the whole wing sum into one reduction that reads only those vectors,
        # instead of streaming a (line, grid) matrix per layer.
        unshifted = grid[None, :] - centers[:, None]
        wing_mask = jnp.abs(unshifted) > self._core_half_width[:, None]
        offsets = unshifted if shift is None else unshifted - shift[:, None]
        # Safe dummy coordinates prevent singular asymptotic evaluations in
        # excluded pairs and keep their reverse-mode derivatives finite.
        x = jnp.where(wing_mask, offsets * scale[:, None], 20.0)
        wing_function = _mixed_wing if self.mixed_precision else _wing
        wings = jnp.where(wing_mask, wing_function(x, a[:, None]), 0.0)
        if self.line_coupling:
            dispersion = _dispersion(_wing_imag(x, a[:, None]), offsets, gamma[:, None], scale[:, None])
            wings = wings + mixing[:, None] * jnp.where(wing_mask, dispersion, 0.0)
        spectrum = jnp.sum(wings * weights[:, None], axis=0)

        core_line, core_grid = self._core_line, self._core_grid
        core_offsets = grid[core_grid] - centers[core_line]
        if shift is not None:
            core_offsets = core_offsets - shift[core_line]
        core_x, core_a = core_offsets * scale[core_line], a[core_line]
        cores = jax.vmap(hjert)(core_x, core_a)
        if self.line_coupling:
            cores = cores + mixing[core_line] * _dispersion(
                jax.vmap(_core_imag)(core_x, core_a), core_offsets, gamma[core_line], scale[core_line])
        return spectrum.at[core_grid].add(cores * weights[core_line])
