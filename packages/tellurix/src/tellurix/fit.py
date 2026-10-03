"""Bounded MAP fitting with JAX derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .model import TelluricModel
from .types import SpectralOrder, TelluricParameters


_MAX_PROJECTED_SCALED_GRADIENT = 1.0


@dataclass(frozen=True)
class FitResult:
    """A fit and what can honestly be said about its uncertainty.

    ``covariance`` is the inverse Hessian of the objective at the solution,
    restricted to the parameters that were free and not resting on a bound, and
    padded with zeros elsewhere so its indices match ``codec.names``. That is a
    *formal* covariance: it assumes the residuals are independent and Gaussian
    with the stated uncertainties. On this atlas they are neither -- the
    residual is dominated by stellar line-list error, which is correlated
    between pixels -- so the formal errors come out far too small. Measured on
    ``ab5000_``: formal sigma on the water column 0.76%, 2.7% after scaling by
    the square root of reduced chi-squared, against 6.5-9% actually measured
    from sub-window to sub-window. Scaling by chi-squared undercorrects because
    it assumes the excess is white.

    ``correlation`` carries the part that survives all of that. It describes the
    shape of the likelihood rather than its scale, so a wrong noise model leaves
    it intact, and it is where the degeneracies show: a column scale strongly
    correlated with the continuum means the data cannot separate them.

    ``condition_number`` is that of the inverted submatrix, and ``at_bound``
    names the free parameters excluded for resting on a bound -- where the
    quadratic approximation does not hold and the matrix would be singular.
    """

    parameters: TelluricParameters
    covariance: np.ndarray | None
    transmission: np.ndarray
    model_flux: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str
    objective: float
    iterations: int
    correlation: np.ndarray | None = None
    condition_number: float | None = None
    at_bound: tuple[str, ...] = ()


class _ParameterCodec:
    """Ordered mapping between named parameters and the optimizer's vector.

    ``stellar_velocity_kms`` is present only when the order carries a
    model-grid source. Without one it cannot affect the prediction, so freeing
    it would contribute an identically zero gradient column and a singular
    Hessian rather than an unconstrained parameter.
    """

    def __init__(
        self,
        species: Sequence[str],
        continuum_size: int,
        include_stellar_velocity: bool = False,
    ) -> None:
        self.species = tuple(species)
        self.continuum_size = continuum_size
        self.include_stellar_velocity = include_stellar_velocity
        names = [*self.species, "velocity_kms"]
        if include_stellar_velocity:
            names.append("stellar_velocity_kms")
        names.extend(("wavelength_stretch", "lsf_sigma_kms"))
        names.extend(f"continuum_{index}" for index in range(continuum_size))
        names.append("log_jitter")
        self.names = tuple(names)
        self._offset = {name: index for index, name in enumerate(self.names)}

    @property
    def continuum_start(self) -> int:
        return self._offset["continuum_0"] if self.continuum_size else 0

    def pack(self, params: TelluricParameters) -> np.ndarray:
        values = {
            **{name: params.log_column_scales[name] for name in self.species},
            "velocity_kms": params.velocity_kms,
            "stellar_velocity_kms": params.stellar_velocity_kms,
            "wavelength_stretch": params.wavelength_stretch,
            "lsf_sigma_kms": params.lsf_sigma_kms,
            "log_jitter": params.log_jitter,
            **{
                f"continuum_{index}": value
                for index, value in enumerate(np.asarray(params.continuum_coeffs))
            },
        }
        return np.asarray([values[name] for name in self.names], dtype=float)

    def unpack(self, vector: jnp.ndarray) -> TelluricParameters:
        start = self.continuum_start
        return TelluricParameters(
            log_column_scales={name: vector[self._offset[name]] for name in self.species},
            velocity_kms=vector[self._offset["velocity_kms"]],
            wavelength_stretch=vector[self._offset["wavelength_stretch"]],
            lsf_sigma_kms=vector[self._offset["lsf_sigma_kms"]],
            continuum_coeffs=vector[start : start + self.continuum_size],
            log_jitter=vector[self._offset["log_jitter"]],
            stellar_velocity_kms=(
                vector[self._offset["stellar_velocity_kms"]]
                if self.include_stellar_velocity
                else 0.0
            ),
        )


def _parameter_scale(name: str, lower: float, upper: float) -> float:
    """Return a useful physical step represented by one optimizer unit."""
    natural = 1.0e-5 if name == "wavelength_stretch" else 0.01 if name.startswith("continuum_") else 1.0
    if np.isfinite(lower) and np.isfinite(upper):
        natural = min(natural, 0.5 * (upper - lower))
    return natural


def _projected_gradient(vector, gradient, lower, upper):
    """Gradient after removing directions forbidden by active bounds."""
    projected = np.asarray(gradient, dtype=float).copy()
    tolerance = 1.0e-10
    projected[(vector <= lower + tolerance) & (projected > 0.0)] = 0.0
    projected[(vector >= upper - tolerance) & (projected < 0.0)] = 0.0
    return projected

_AT_BOUND_TOLERANCE = 1.0e-6
# Past this the submatrix is singular to working precision and inverting it
# produces numbers rather than an error, which is worse than refusing.
_MAXIMUM_CONDITION_NUMBER = 1.0e12


def _uncertainty(objective, vector, free_indices, bound_array, names):
    """Formal covariance, correlation and conditioning at the solution.

    L-BFGS-B's ``hess_inv`` used to be reported here. It is a byproduct of the
    line search rather than a curvature estimate, and on this atlas it
    overstated the column errors by two orders of magnitude. This inverts the
    actual Hessian instead, over the parameters that were free and are not
    resting on a bound -- at a bound the quadratic approximation does not hold
    and the direction is not free, so including it makes the matrix singular.
    """

    at_bound = tuple(
        names[index] for index in free_indices
        if min(abs(vector[index] - bound_array[index, 0]),
               abs(vector[index] - bound_array[index, 1])) < _AT_BOUND_TOLERANCE
    )
    usable = np.asarray(
        [index for index in free_indices if names[index] not in at_bound], dtype=int
    )
    if usable.size == 0:
        return None, None, None, at_bound

    hessian = objective.hessian(vector)[np.ix_(usable, usable)]
    if not np.all(np.isfinite(hessian)):
        return None, None, None, at_bound
    condition = float(np.linalg.cond(hessian))
    eigenvalues = np.linalg.eigvalsh(hessian)
    # A minimum has positive curvature in every direction; anything else means
    # the solver stopped somewhere an error bar has no meaning.
    if condition > _MAXIMUM_CONDITION_NUMBER or eigenvalues.min() <= 0.0:
        return None, None, condition, at_bound

    inverse = np.linalg.inv(hessian)
    covariance = np.zeros((len(names), len(names)))
    covariance[np.ix_(usable, usable)] = inverse
    sigma = np.sqrt(np.diag(inverse))
    correlation = np.zeros((len(names), len(names)))
    correlation[np.ix_(usable, usable)] = inverse / np.outer(sigma, sigma)
    return covariance, correlation, condition, at_bound


class OrderObjective:
    """Compiled value and gradient for one model and one order, reusable.

    XLA compilation of a telluric forward model costs seconds while one
    evaluation costs milliseconds, so what a staged fit spends is dominated by
    how many times it compiles. ``fit_order`` builds a fresh ``jax.jit``
    closure per call, and a new Python function object is a new cache entry, so
    four stages over one page compile the same graph four times.

    This compiles over the *full* parameter vector, leaving the bounds, the
    free set, and the optimizer's rescaling outside. Those are exactly what
    changes between stages, so one instance passed to every stage of a page
    replaces N compilations with one.
    """

    def __init__(
        self,
        model: TelluricModel,
        order: SpectralOrder,
        continuum_size: int,
    ) -> None:
        self.model = model
        self.order = order
        self.codec = _ParameterCodec(
            model.species,
            int(continuum_size),
            include_stellar_velocity=order.source_flux_model_grid is not None,
        )
        self._bind(order)

        # The per-observation data are *operands*, not captured constants.
        # Closing over them would bake their values into the jaxpr, so a second
        # observation on the same grid -- the ordinary case for an echelle
        # order across a night -- would recompile an identical graph. Measured
        # on an IGRINS order: 4.9 s to compile the gradient and 8.2 s the
        # Hessian, against 2.6 ms and 3.1 ms to run them.
        def objective(
            vector: jnp.ndarray,
            flux: jnp.ndarray,
            mask: jnp.ndarray,
            uncertainty: jnp.ndarray,
            zenith_angle_deg: jnp.ndarray,
        ) -> jnp.ndarray:
            parameters = self.codec.unpack(vector)
            prediction = model.predict(order, parameters, zenith_angle_deg=zenith_angle_deg)
            variance = uncertainty**2 + jnp.exp(2.0 * jnp.asarray(parameters.log_jitter))
            # Dividing inside the logarithm removes a parameter-independent
            # constant. L-BFGS-B uses relative objective reduction to stop, and
            # the large negative normalization term otherwise causes premature
            # convergence for high-S/N orders.
            terms = (jnp.where(mask, flux, 0.0) - prediction) ** 2 / variance + jnp.log(
                variance / uncertainty**2
            )
            return 0.5 * jnp.sum(jnp.where(mask, terms, 0.0))

        self._value_and_grad = jax.jit(jax.value_and_grad(objective))
        # Over the full parameter vector, so one compilation serves every stage
        # and every choice of free set; the caller slices out what it needs.
        self._hessian = jax.jit(jax.hessian(objective))

    def _bind(self, order: SpectralOrder) -> None:
        self.order = order
        self._data = (
            jnp.asarray(order.flux),
            jnp.asarray(order.mask),
            jnp.asarray(order.uncertainty),
            jnp.asarray(float(order.zenith_angle_deg)),
        )

    def rebind(self, order: SpectralOrder) -> "OrderObjective":
        """Point this compiled objective at another observation of one order.

        The saving is the whole compilation, which is most of what fitting an
        IGRINS order costs. It is only sound when the new order samples exactly
        the same wavelengths and carries the same source, so that is checked
        rather than assumed -- everything else about the graph is shape-only.
        The zenith angle is free to differ, which is the point.
        """

        current = self.order
        if not np.array_equal(
            np.asarray(order.wavelength_vacuum_nm), np.asarray(current.wavelength_vacuum_nm)
        ):
            raise ValueError("rebinding needs the identical wavelength grid")
        if (order.source_flux_model_grid is None) != (current.source_flux_model_grid is None):
            raise ValueError("rebinding cannot change how the source is supplied")
        for name in ("source_flux_model_grid", "source_flux"):
            new, old = getattr(order, name), getattr(current, name)
            if new is not None and not np.array_equal(np.asarray(new), np.asarray(old)):
                raise ValueError(f"rebinding needs the identical {name}")
        self._bind(order)
        return self

    def __call__(self, vector: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = self._value_and_grad(jnp.asarray(vector), *self._data)
        return float(value), np.asarray(gradient, dtype=float)

    def hessian(self, vector: np.ndarray) -> np.ndarray:
        """Second derivatives of the objective at one full parameter vector."""
        return np.asarray(self._hessian(jnp.asarray(vector), *self._data), dtype=float)


def fit_order(
    model: TelluricModel,
    order: SpectralOrder,
    initial: TelluricParameters,
    bounds: Mapping[str, tuple[float, float]],
    objective: OrderObjective | None = None,
    covariance: bool = True,
) -> FitResult:
    """Fit one spectral order using a heteroscedastic Gaussian likelihood.

    Bound keys are molecule names plus ``velocity_kms``,
    ``wavelength_stretch``, ``lsf_sigma_kms``, ``continuum_0`` ... and
    ``log_jitter``, and additionally ``stellar_velocity_kms`` when the order
    carries ``source_flux_model_grid``. Setting a bound's lower and upper
    values equal pins that parameter.

    Pass ``objective`` -- an :class:`OrderObjective` built once for this model
    and order -- to reuse one compilation across the stages of a staged fit.

    ``covariance=False`` skips :func:`_uncertainty`, and with it the only call
    that ever compiles the Hessian. That compilation costs 8.2 s on an IGRINS
    order against 3.1 ms to run, so a staged fit should ask for it on its final
    stage only -- the intermediate covariances are thrown away.
    """

    if objective is None:
        objective = OrderObjective(model, order, len(np.asarray(initial.continuum_coeffs)))
    elif objective.model is not model or objective.order is not order:
        raise ValueError("the compiled objective was built for a different model or order")
    codec = objective.codec
    if codec.continuum_size != len(np.asarray(initial.continuum_coeffs)):
        raise ValueError("the compiled objective was built for a different continuum degree")
    names = list(codec.names)
    missing = [name for name in names if name not in bounds]
    if missing:
        raise ValueError(f"missing parameter bounds: {', '.join(missing)}")

    full_initial = codec.pack(initial)
    bound_array = np.asarray([bounds[name] for name in names], dtype=float)
    if np.any(np.isnan(bound_array)) or np.any(bound_array[:, 0] > bound_array[:, 1]):
        raise ValueError("parameter bounds must be ordered and not NaN")
    if np.any(~np.isfinite(full_initial)):
        raise ValueError("initial parameters must be finite")
    full_initial = np.clip(full_initial, bound_array[:, 0], bound_array[:, 1])
    free = bound_array[:, 0] < bound_array[:, 1]
    fixed = ~free
    full_initial[fixed] = bound_array[fixed, 0]
    free_indices = np.flatnonzero(free)
    scales = np.asarray(
        [_parameter_scale(names[index], *bound_array[index]) for index in free_indices]
    )
    center = full_initial.copy()
    initial_scaled = np.zeros(len(free_indices))
    scaled_bounds = [
        ((bound_array[index, 0] - center[index]) / scale,
         (bound_array[index, 1] - center[index]) / scale)
        for index, scale in zip(free_indices, scales)
    ]

    def expand(scaled_vector: np.ndarray) -> np.ndarray:
        full = center.copy()
        full[free_indices] = center[free_indices] + scales * np.asarray(scaled_vector)
        return full

    def scipy_objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        # The rescaling is affine, so the chain rule is one multiplication and
        # the compiled function never has to know which parameters are free.
        value, gradient = objective(expand(vector))
        return value, gradient[free_indices] * scales

    if free_indices.size:
        result = minimize(
            scipy_objective,
            initial_scaled,
            method="L-BFGS-B",
            jac=True,
            bounds=scaled_bounds,
            options={"ftol": 1.0e-12, "gtol": 1.0e-6, "maxiter": 1000, "maxls": 50},
        )
        final_scaled = np.asarray(result.x)
        _, final_gradient = scipy_objective(final_scaled)
        scaled_lower = np.asarray([value[0] for value in scaled_bounds])
        scaled_upper = np.asarray([value[1] for value in scaled_bounds])
        gradient_norm = float(np.max(np.abs(_projected_gradient(
            final_scaled, final_gradient, scaled_lower, scaled_upper
        ))))
        # Float32 wing values have a small quantization floor even though
        # their custom derivative remains smooth. One optimizer unit is a
        # physically useful parameter step, so this still rejects the former
        # false convergence by more than three orders of magnitude.
        converged = bool(result.success) and gradient_norm <= _MAX_PROJECTED_SCALED_GRADIENT
        message = str(result.message)
        if result.success and not converged:
            message = (
                f"{message}; projected scaled gradient {gradient_norm:.3g} "
                f"exceeds {_MAX_PROJECTED_SCALED_GRADIENT:g}"
            )
        final_vector = center.copy()
        final_vector[free_indices] += scales * final_scaled
    else:
        final_scaled = initial_scaled
        final_vector = center
        converged = True
        message = "all parameters fixed"
        gradient_norm = 0.0

        class FixedResult:
            fun = float(objective(expand(initial_scaled))[0])
            nit = 0
            hess_inv = None

        result = FixedResult()

    parameters = codec.unpack(jnp.asarray(final_vector))
    model_flux = np.asarray(model.predict(order, parameters))
    residuals = np.asarray(order.flux) - model_flux
    if covariance:
        covariance_matrix, correlation, condition, at_bound = _uncertainty(
            objective, final_vector, free_indices, bound_array, names
        )
    else:
        covariance_matrix, correlation, condition, at_bound = None, None, None, ()
    normalization = 0.5 * np.sum(np.where(np.asarray(order.mask), np.log(np.asarray(order.uncertainty) ** 2), 0.0))
    return FitResult(
        parameters=parameters,
        covariance=covariance_matrix,
        correlation=correlation,
        condition_number=condition,
        at_bound=at_bound,
        transmission=np.asarray(model.transmission(parameters, order.zenith_angle_deg)),
        model_flux=model_flux,
        residuals=residuals,
        success=converged,
        message=message,
        objective=float(result.fun + normalization),
        iterations=int(result.nit),
    )
