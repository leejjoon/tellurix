"""Compare optimized opacity with the actual ExoJAX kernel and its JVP."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tellurix  # enable float64 before importing ExoJAX
from exojax.opacity import OpaDirect
from tellurix.direct import SparseCoreDirect, _wing, _mixed_wing
from tellurix import ExoJAXOpacityBackend


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_sparse_core_matches_exojax_values_and_derivatives(mixed_precision):
    database = SimpleNamespace(
        dbtype="hitran", isotope=1, molmass=18.0,
        nu_lines=np.array([5000.0, 5000.1, 5001.0]),
        logsij0=jnp.log(jnp.array([1e-22, 2e-23, 1e-24])),
        elower=np.array([100., 300., 50.]),
        n_air=np.array([0.7, 0.6, 0.8]),
        gamma_air=np.array([0.07, 0.08, 0.09]),
        gamma_self=np.array([0.3, 0.4, 0.2]),
        A=np.array([0.1, 0.2, 0.3]),
        qr_interp=lambda isotope, temperature, reference: (temperature / reference)**1.5,
    )
    grid = np.linspace(4998., 5003., 801)
    original = OpaDirect(database, grid)
    optimized = SparseCoreDirect(database, grid, mixed_precision=mixed_precision)
    value_rtol = 1e-6 if mixed_precision else 2e-12
    # Mixed precision targets 1e-5 relative gradient accuracy; a few ppm
    # difference is expected when summing signed pressure derivatives.
    gradient_rtol = 1e-5 if mixed_precision else 1e-9
    weights = jnp.linspace(0.3, 1.2, len(grid))
    for temperature in (180., 275., 400., 401., 650.):
        for pressure in (0.001, 0.7):
            point = jnp.array([temperature, pressure, pressure * 0.02])
            reference = original.xsvector(*point)
            actual = optimized.xsvector(*point)
            np.testing.assert_allclose(actual, reference, rtol=value_rtol, atol=1e-35)
            def statistic(calculator, parameters):
                return jnp.sum(weights * calculator.xsvector(*parameters)) * 1e22
            old_grad = jax.grad(lambda p: statistic(original, p))(point)
            new_grad = jax.grad(lambda p: statistic(optimized, p))(point)
            np.testing.assert_allclose(new_grad, old_grad, rtol=gradient_rtol, atol=1e-9)

    # The public batched adapter must preserve each layer's self pressure.
    adapter = ExoJAXOpacityBackend({"H2O": optimized}, vectorize_layers=True)
    temperatures = jnp.array([200., 275., 450.])
    pressures = jnp.array([0.01, 0.5, 0.9])
    self_pressures = pressures * jnp.array([0.001, 0.02, 0.1])
    expected = jnp.stack([original.xsvector(t, p, s) for t, p, s in
                          zip(temperatures, pressures, self_pressures)])
    actual = adapter.cross_sections(temperatures, pressures, {"H2O": self_pressures})["H2O"]
    np.testing.assert_allclose(actual, expected, rtol=value_rtol, atol=1e-35)


def test_mixed_wing_derivative_survives_far_wing_cancellation():
    points = jnp.array([[11., 0.001], [100., 1.], [10000., 0.1], [10000., 100.]])
    original = jax.vmap(jax.grad(lambda p: _wing(*p)))(points)
    mixed = jax.vmap(jax.grad(lambda p: _mixed_wing(*p)))(points)
    np.testing.assert_allclose(mixed, original, rtol=2e-6, atol=1e-15)
    # A naive float32 subtraction loses the positive damping derivative here.
    assert mixed[2, 1] > 5e-9


def test_sparse_direct_applies_hitran_air_pressure_shift():
    database = SimpleNamespace(
        dbtype="hitran", isotope=1, molmass=18.0,
        nu_lines=np.array([5000.0]),
        logsij0=jnp.log(jnp.array([1e-22])),
        elower=np.array([100.]),
        n_air=np.array([0.7]),
        gamma_air=np.array([0.01]),
        gamma_self=np.array([0.01]),
        delta_air=np.array([0.1]),
        A=np.array([0.1]),
        qr_interp=lambda isotope, temperature, reference: (temperature / reference)**1.5,
    )
    grid = np.linspace(4999.7, 5000.3, 1201)
    shifted = SparseCoreDirect(database, grid, pressure_shift=True)
    at_zero_pressure = shifted.xsvector(296.0, 0.0)
    at_one_atmosphere = shifted.xsvector(296.0, 1.01325)

    zero_peak = grid[int(jnp.argmax(at_zero_pressure))]
    pressure_peak = grid[int(jnp.argmax(at_one_atmosphere))]
    assert zero_peak == pytest.approx(5000.0, abs=0.0005)
    assert pressure_peak == pytest.approx(5000.1, abs=0.0005)
    cold_half_atmosphere = shifted.xsvector(148.0, 0.506625)
    cold_peak = grid[int(jnp.argmax(cold_half_atmosphere))]
    assert cold_peak == pytest.approx(5000.1, abs=0.0005)
    gradient = jax.grad(lambda pressure: jnp.sum(
        jnp.asarray(grid) * shifted.xsvector(296.0, pressure)
    ))(0.7)
    assert jnp.isfinite(gradient)


def test_sparse_direct_holds_no_line_by_grid_matrix():
    """The dense offset matrix is what puts a fine grid out of memory.

    ExoJAX's ``OpaDirect`` keeps ``nu_grid[None, :] - nu_lines[:, None]`` on the
    device. At terrestrial line densities that is hundreds of megabytes, re-read
    once per layer. Everything here is rebuilt from the two 1-D vectors instead,
    so nothing of that shape may survive construction.
    """

    database = SimpleNamespace(
        dbtype="hitran", isotope=1, molmass=18.0,
        nu_lines=np.linspace(4999.0, 5002.0, 400),
        logsij0=jnp.log(jnp.full(400, 1e-23)),
        elower=np.full(400, 200.0),
        n_air=np.full(400, 0.7), gamma_air=np.full(400, 0.07),
        gamma_self=np.full(400, 0.3), A=np.full(400, 0.1),
        delta_air=np.full(400, -0.01),
        qr_interp=lambda isotope, temperature, reference: (temperature / reference) ** 1.5,
    )
    grid = np.linspace(4998.0, 5003.0, 2000)
    calculator = SparseCoreDirect(database, grid, pressure_shift=True)
    assert calculator.opainfo is None
    stored = [
        value for value in vars(calculator).values()
        if hasattr(value, "ndim") and hasattr(value, "shape") and value.ndim == 2
    ]
    assert stored == [], f"a 2-D array survived: {[v.shape for v in stored]}"
    # The core list is the one thing allowed to scale with line by grid, and
    # only because it is sparse. This fixture crowds 400 lines into 3 cm-1, so
    # its 6% is far denser than a real order; the measured IGRINS figure is
    # 0.14%.
    assert calculator.core_line.size < 0.1 * database.nu_lines.size * grid.size
    # Rebuilding the offsets must not change the answer. OpaDirect applies no
    # pressure shift, so the comparison uses the unshifted calculator.
    np.testing.assert_allclose(
        SparseCoreDirect(database, grid).xsvector(275.0, 0.7, 0.01),
        OpaDirect(database, grid).xsvector(275.0, 0.7, 0.01), rtol=2e-12, atol=1e-35,
    )


def test_core_list_matches_the_dense_selection_it_replaced():
    """Binary search must reproduce the dense scan exactly, order included.

    The core and wing branches are evaluated by different code, and the scatter
    add sums in list order, so a different core list is a different answer.
    """

    from exojax.database.core.broadening import doppler_sigma
    from exojax.utils.constants import Tref_original

    rng = np.random.default_rng(3)
    count = 300
    database = SimpleNamespace(
        dbtype="hitran", isotope=1, molmass=18.0,
        nu_lines=np.sort(rng.uniform(4999.0, 5002.0, count)),
        logsij0=jnp.log(jnp.full(count, 1e-23)), elower=np.full(count, 200.0),
        n_air=np.full(count, 0.7), gamma_air=np.full(count, 0.07),
        gamma_self=np.full(count, 0.3), A=np.full(count, 0.1),
        delta_air=rng.uniform(-0.02, 0.0, count),
        qr_interp=lambda isotope, temperature, reference: (temperature / reference) ** 1.5,
    )
    grid = np.linspace(4998.0, 5003.0, 1500)
    for pressure_shift in (False, True):
        calculator = SparseCoreDirect(
            database, grid, minimum_temperature_k=200.0, maximum_temperature_k=320.0,
            pressure_shift=pressure_shift, maximum_pressure_bar=1.1,
        )
        sigma = np.asarray(doppler_sigma(database.nu_lines, 320.0, database.molmass))
        margin = (np.abs(database.delta_air) * 1.1 / 1.01325 * Tref_original / 200.0
                  if pressure_shift else 0.0)
        half_width = np.sqrt(222.0) * sigma + margin
        offsets = grid[None, :] - database.nu_lines[:, None]
        line, column = np.nonzero(np.abs(offsets) <= half_width[:, None])
        np.testing.assert_array_equal(calculator.core_line, line)
        np.testing.assert_array_equal(calculator.core_grid, column)


def _coupled_database(y=0.02, g=0.0):
    """One line with Rosenkranz coefficients, as AERLineDatabase.coupling gives them."""
    return SimpleNamespace(
        dbtype="hitran", isotope=1, molmass=44.0,
        nu_lines=np.array([5000.0]),
        logsij0=jnp.log(jnp.array([1e-22])),
        elower=np.array([100.]), n_air=np.array([0.7]),
        gamma_air=np.array([0.07]), gamma_self=np.array([0.09]),
        delta_air=np.array([0.0]), A=np.array([0.0]),
        qr_interp=lambda isotope, temperature, reference: (temperature / reference) ** 1.5,
        has_line_coupling=True,
        coupling=lambda temperature, pressure: (jnp.array([y]) * pressure / 1.01325,
                                                1.0 + jnp.array([g]) * (pressure / 1.01325) ** 2),
    )


def test_line_coupling_adds_an_odd_term_that_ends_at_25_cm1():
    grid = np.linspace(4960.0, 5040.0, 8001)
    plain = SparseCoreDirect(_coupled_database(), grid).xsvector(280.0, 0.8, 0.0)
    coupled = SparseCoreDirect(_coupled_database(), grid, line_coupling=True).xsvector(280.0, 0.8, 0.0)
    dispersion = np.asarray(coupled - plain)

    centre = np.abs(grid - 5000.0) < 1e-9
    np.testing.assert_allclose(dispersion, -dispersion[::-1], atol=1e-12 * float(np.max(plain)))
    assert dispersion[centre] == pytest.approx(0.0, abs=1e-30)
    # Positive Y moves absorption to the high-wavenumber side.
    assert dispersion[np.argmin(np.abs(grid - 5000.1))] > 0.0
    np.testing.assert_array_equal(dispersion[np.abs(grid - 5000.0) > 25.0], 0.0)
    # Near the line it is Y p times the Lorentz dispersion profile.
    near = np.argmin(np.abs(grid - 5000.5))
    gamma = 0.07 * (296.0 / 280.0) ** 0.7 * 0.8
    expected = 0.02 * 0.8 / 1.01325 * 0.5 / (np.pi * (0.25 + gamma**2)) * float(
        plain[near] / (gamma / (np.pi * (0.25 + gamma**2))))
    assert dispersion[near] == pytest.approx(expected, rel=0.02)


def test_coupling_g_scales_the_line_and_off_means_off():
    grid = np.linspace(4995.0, 5005.0, 2001)
    plain = SparseCoreDirect(_coupled_database(y=0.0, g=0.5), grid).xsvector(280.0, 0.8, 0.0)
    scaled = SparseCoreDirect(_coupled_database(y=0.0, g=0.5), grid,
                              line_coupling=True).xsvector(280.0, 0.8, 0.0)
    np.testing.assert_allclose(scaled, plain * (1.0 + 0.5 * (0.8 / 1.01325) ** 2), rtol=1e-12)


def test_coupled_sparse_kernel_matches_its_dense_fallback():
    grid = np.linspace(4990.0, 5010.0, 4001)
    calculator = SparseCoreDirect(_coupled_database(g=0.1), grid, line_coupling=True)
    np.testing.assert_allclose(calculator._sparse_xsvector(250.0, 0.5, 0.01),
                               calculator._full_xsvector(250.0, 0.5, 0.01), rtol=1e-6, atol=1e-30)


def test_imaginary_faddeeva_derivative_is_analytic_and_continuous():
    from tellurix.direct import _core_imag
    from scipy.special import wofz

    points = np.array([[0.3, 0.2], [2.0, 5.0], [10.4, 1.5], [10.6, 1.5], [3.0, 12.0]])
    values = jax.vmap(lambda p: _core_imag(p[0], p[1]))(jnp.asarray(points))
    np.testing.assert_allclose(values, wofz(points[:, 0] + 1j * points[:, 1]).imag, rtol=1e-5)
    gradients = jax.vmap(jax.grad(lambda p: _core_imag(p[0], p[1])))(jnp.asarray(points))
    step = 1e-6
    for (x, a), gradient in zip(points, np.asarray(gradients)):
        numerical = [(wofz(x + step + 1j * a) - wofz(x - step + 1j * a)).imag / (2 * step),
                     (wofz(x + 1j * (a + step)) - wofz(x + 1j * (a - step))).imag / (2 * step)]
        # ExoJAX's asymptotic branch (|z|^2 > 111) is good to ~1e-6, and the
        # analytic derivative is built from it.
        np.testing.assert_allclose(gradient, numerical, rtol=1e-4, atol=1e-6)


def _water_database():
    return SimpleNamespace(
        dbtype="hitran", isotope=1, molmass=18.0,
        nu_lines=np.array([5000.0, 5000.1, 5001.0]),
        logsij0=jnp.log(jnp.array([1e-22, 2e-23, 1e-24])),
        elower=np.array([100., 300., 50.]), n_air=np.array([0.7, 0.6, 0.8]),
        gamma_air=np.array([0.07, 0.08, 0.09]), gamma_self=np.array([0.3, 0.4, 0.2]),
        A=np.array([0.1, 0.2, 0.3]), delta_air=np.array([-0.01, 0.005, -0.02]),
        qr_interp=lambda isotope, temperature, reference: (temperature / reference) ** 1.5,
    )


def _closed_form_sdv(database, grid, temperature, pressure, self_pressure, ratio, shift=False):
    """The quadratic SDV in numpy and scipy's wofz, line by line (Tran et al. 2013)."""
    from scipy.special import wofz
    from exojax.database.core.broadening import gamma_natural
    from exojax.database.core.line_strength import line_strength

    nu_d = np.sqrt(2.0) * 3.0415595e-07 * np.sqrt(temperature / database.molmass) * database.nu_lines
    pressure_width = (296.0 / temperature) ** database.n_air * (
        database.gamma_air * (pressure - self_pressure) + database.gamma_self * self_pressure) / 1.01325
    gamma0 = pressure_width + np.asarray(gamma_natural(database.A))
    gamma2 = ratio * pressure_width
    strength = np.asarray(line_strength(temperature, database.logsij0, database.nu_lines, database.elower,
                                        database.qr_interp(1, temperature, 296.0), 296.0))
    total = np.zeros_like(grid)
    for line in range(database.nu_lines.size):
        offset = grid - database.nu_lines[line]
        if shift:
            offset = offset - database.delta_air[line] * pressure / 1.01325 * 296.0 / temperature
        c = gamma0[line] - 1.5 * gamma2[line] - 1j * offset
        z1 = c / (np.sqrt(c * gamma2[line] + nu_d[line] ** 2 / 4) + nu_d[line] / 2)
        w = wofz(1j * z1) - (wofz(1j * (z1 + nu_d[line] / gamma2[line])) if gamma2[line] > 0 else 0.0)
        total += strength[line] * w.real / (np.sqrt(np.pi) * nu_d[line])
    return total


def test_speed_dependent_voigt_is_the_maxwell_average_of_lorentz_lines():
    """The closed form against its definition: Gamma(v) = Gamma0 (1 + r ((v/v_p)^2 - 3/2))
    over Maxwell speeds and directions, Doppler-shifted Lorentzians."""
    from scipy.integrate import dblquad
    from tellurix.direct import _speed_dependent

    nu_d, gamma0, ratio = 0.017, 0.08, 0.13
    def definition(offset):
        def integrand(cosine, speed):
            width = gamma0 * (1 + ratio * (speed * speed - 1.5))
            weight = 2 / np.sqrt(np.pi) * speed * speed * np.exp(-speed * speed)
            return weight * width / np.pi / ((offset - nu_d * speed * cosine) ** 2 + width ** 2)
        return dblquad(integrand, 0.0, 6.0, -1.0, 1.0, epsabs=1e-12)[0]
    offsets = np.array([0.0, 0.05, 0.2, 0.5])
    profile = np.asarray(_speed_dependent(jnp.asarray(offsets), nu_d, gamma0, ratio * gamma0))
    expected = [definition(offset) for offset in offsets]
    np.testing.assert_allclose(profile / (np.sqrt(np.pi) * nu_d), expected, rtol=1e-7)
    # Narrower than the Voigt at the same Gamma0: that is the whole effect.
    from exojax.opacity.lpf.lpf import hjert
    assert profile[0] > float(hjert(0.0, gamma0 / nu_d))


@pytest.mark.parametrize("temperature, pressure, self_pressure",
                         [(180.0, 0.001, 1e-5), (275.0, 0.7, 0.014), (300.0, 1.0, 0.03)])
def test_speed_dependent_kernel_matches_scipy_and_its_dense_fallback(temperature, pressure, self_pressure):
    database, grid = _water_database(), np.linspace(4990.0, 5010.0, 8001)
    for ratio in (0.0, 0.13, 0.3):
        calculator = SparseCoreDirect(database, grid, speed_dependence=ratio, pressure_shift=True)
        actual = np.asarray(calculator.xsvector(temperature, pressure, self_pressure))
        expected = _closed_form_sdv(database, grid, temperature, pressure, self_pressure, ratio, shift=True)
        # The asymptotic series is good to ~3e-8 of the line; ExoJAX's Voigt to ~4e-8.
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7 * expected.max())
        np.testing.assert_allclose(actual, calculator._full_xsvector(temperature, pressure, self_pressure),
                                   rtol=0, atol=1e-14 * expected.max())
    # Off is ExoJAX's Voigt; r = 0 is the same line through the SDV's own branches.
    plain = np.asarray(OpaDirect(database, grid).xsvector(temperature, pressure, self_pressure))
    zero = SparseCoreDirect(database, grid, speed_dependence=0.0).xsvector(temperature, pressure, self_pressure)
    np.testing.assert_allclose(zero, plain, rtol=0, atol=1e-7 * plain.max())


def test_speed_dependent_kernel_holds_outside_its_pressure_bounds():
    """Past maximum_pressure_bar the dense fallback; past maximum_self_pressure_bar,
    which is not checked, the wing series a little inside its radius."""
    database, grid = _water_database(), np.linspace(4990.0, 5010.0, 8001)
    calculator = SparseCoreDirect(database, grid, speed_dependence=0.13, maximum_pressure_bar=0.8,
                                  maximum_self_pressure_bar=0.02)
    for pressure, self_pressure in ((1.2, 0.01), (0.7, 0.05)):
        expected = _closed_form_sdv(database, grid, 275.0, pressure, self_pressure, 0.13)
        np.testing.assert_allclose(calculator.xsvector(275.0, pressure, self_pressure), expected,
                                   rtol=0, atol=1e-7 * expected.max())
    stored = [value for value in vars(calculator).values()
              if hasattr(value, "ndim") and hasattr(value, "shape") and value.ndim == 2]
    assert stored == []


def test_speed_dependent_kernel_derivatives_are_finite_differences():
    database, grid = _water_database(), np.linspace(4990.0, 5010.0, 4001)
    calculator = SparseCoreDirect(database, grid, speed_dependence=0.13)
    weights = jnp.linspace(0.3, 1.2, grid.size)
    def statistic(point):
        return jnp.sum(weights * calculator.xsvector(*point)) * 1e22
    point = jnp.array([275.0, 0.7, 0.014])
    gradient = np.asarray(jax.grad(statistic)(point))
    steps = (1e-3, 1e-6, 1e-6)
    numerical = [(statistic(point.at[i].add(h)) - statistic(point.at[i].add(-h))) / (2 * h)
                 for i, h in enumerate(steps)]
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6)


def test_speed_dependence_refuses_what_it_does_not_implement():
    database, grid = _water_database(), np.linspace(4990.0, 5010.0, 401)
    with pytest.raises(ValueError, match="2/3"):
        SparseCoreDirect(database, grid, speed_dependence=0.7)
    with pytest.raises(ValueError, match="mixed_precision"):
        SparseCoreDirect(database, grid, speed_dependence=0.1, mixed_precision=True)
    with pytest.raises(ValueError, match="no database"):
        ExoJAXOpacityBackend.prepare({"H2O": database}, grid, methods="direct_sparse",
                                     speed_dependence={"O2": 0.1})
    with pytest.raises(ValueError, match="direct_sparse"):
        ExoJAXOpacityBackend.prepare({"H2O": database}, grid, speed_dependence={"H2O": 0.1})
    # Mixed precision is the wings' and the SDV's wings are its own, so the species
    # that carries one runs in float64 and the rest as asked.
    backend = ExoJAXOpacityBackend.prepare({"H2O": database, "O2": _water_database()}, grid,
                                           methods="direct_sparse", mixed_precision=True,
                                           speed_dependence={"h2o": 0.1})
    assert backend.calculators["H2O"].speed_dependence is not None
    assert not backend.calculators["H2O"].mixed_precision
    assert backend.calculators["O2"].speed_dependence is None
    assert backend.calculators["O2"].mixed_precision
