import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tellurix import (
    ArrayOpacityBackend,
    AtmosphereProfile,
    CoaddNode,
    CoaddedTelluricModel,
    OrderObjective,
    SpectralOrder,
    TelluricModel,
    TelluricParameters,
    fourier_shift,
)

C_KMS = 299792.458


def make_model(temperatures=(240.0, 280.0), water=(2.0e-4, 5.0e-3), width_cm1=0.02, centre_cm1=13002.0):
    # Uniform in log wavenumber, as every model grid is.
    nu = np.geomspace(13000.0, 13004.0, 2048)
    profile = AtmosphereProfile(pressure_edges_bar=[0.1, 0.5, 1.0], temperature_k=list(temperatures),
                                altitude_km=[10.0, 2.0], vmr={"H2O": list(water)})
    line = np.exp(-0.5 * ((nu - centre_cm1) / width_cm1) ** 2) * 2.0e-23
    return TelluricModel(profile, nu, ArrayOpacityBackend({"H2O": np.vstack([line, 1.1 * line])})), nu


def params(scale=0.0):
    return TelluricParameters(log_column_scales={"H2O": scale}, velocity_kms=0.0, wavelength_stretch=0.0,
                              lsf_sigma_kms=0.5, continuum_coeffs=jnp.array([0.0, 0.0]),
                              log_jitter=np.log(1.0e-4))


def test_fourier_shift_moves_a_gaussian_exactly_and_keeps_a_ramp():
    index = np.arange(1024.0)
    sigma = 2.5  # a telluric line's Doppler sigma on the IAG grid, in samples
    values = 1.0 - 0.9 * np.exp(-0.5 * ((index - 500.0) / sigma) ** 2) + 1e-4 * index
    moved = np.asarray(fourier_shift(jnp.asarray(values), 3.37))
    expected = 1.0 - 0.9 * np.exp(-0.5 * ((index - 503.37) / sigma) ** 2) + 1e-4 * (index - 3.37)
    np.testing.assert_allclose(moved, expected, atol=1e-10)


def test_one_node_at_rest_is_the_single_model():
    model, _ = make_model()
    coadd = CoaddedTelluricModel([(model, [CoaddNode(1.0, 60.0, 0.0)])])
    np.testing.assert_allclose(np.asarray(coadd.transmission(params(0.2))),
                               np.asarray(model.transmission(params(0.2), 60.0)), rtol=1e-13, atol=1e-13)


def test_the_mean_is_weighted_and_shifted_per_node():
    model, nu = make_model()
    warm, _ = make_model(temperatures=(250.0, 295.0), water=(5.0e-4, 1.2e-2))
    velocity = 0.6
    coadd = CoaddedTelluricModel([(model, [CoaddNode(3.0, 45.0, 0.0)]),
                                  (warm, [CoaddNode(1.0, 30.0, velocity)])])
    cold_t = np.asarray(model.transmission(params(), 45.0))
    # Positive is to the red: the warm line's centre moves to lower wavenumber,
    # and on a grid uniform in log wavenumber its width scales with it.
    factor = 1.0 / (1.0 + velocity / C_KMS)
    moved, _ = make_model(temperatures=(250.0, 295.0), water=(5.0e-4, 1.2e-2),
                          width_cm1=0.02 * factor, centre_cm1=13002.0 * factor)
    expected = 0.75 * cold_t + 0.25 * np.asarray(moved.transmission(params(), 30.0))
    np.testing.assert_allclose(np.asarray(coadd.transmission(params())), expected, atol=1e-8)


def test_precomputed_coadd_fits_through_the_shared_objective():
    model, nu = make_model()
    warm, _ = make_model(temperatures=(250.0, 295.0), water=(5.0e-4, 1.2e-2))
    coadd = CoaddedTelluricModel([(model, [CoaddNode(1.0, 45.0, 0.3), CoaddNode(1.0, 50.0, 0.4)]),
                                  (warm, [CoaddNode(1.0, 30.0, -0.2)])]).precompute_opacity()
    assert isinstance(coadd, CoaddedTelluricModel)
    wavelength = np.linspace(1.0e7 / nu[-200], 1.0e7 / nu[200], 300)
    blank = SpectralOrder(wavelength, np.ones(300), np.full(300, 1e-3))
    truth = np.asarray(coadd.predict(blank, params(0.25)))
    objective = OrderObjective(coadd, SpectralOrder(wavelength, truth, np.full(300, 1e-3)), 2)
    vector = np.asarray(objective.codec.pack(params(0.0)))
    value, gradient = objective(vector)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    # Too little water in the model: more lowers the objective.
    assert gradient[objective.codec.names.index("H2O")] < 0.0


def test_components_must_share_grid_and_species():
    model, _ = make_model()
    other = TelluricModel(model.profile, np.geomspace(13000.0, 13004.0, 1024),
                          ArrayOpacityBackend({"H2O": np.zeros((2, 1024))}))
    with pytest.raises(ValueError, match="grid"):
        CoaddedTelluricModel([(model, [CoaddNode(1.0, 0.0, 0.0)]), (other, [CoaddNode(1.0, 0.0, 0.0)])])
    with pytest.raises(ValueError, match="node"):
        CoaddedTelluricModel([(model, [])])
