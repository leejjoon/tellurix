import numpy as np
import pytest

from jax_telluric import (
    ArrayOpacityBackend,
    AtmosphereProfile,
    StellarSpectrum,
    TelluricModel,
    broaden_stellar_source,
    prepare_stellar_source,
    resample_stellar_source,
)


def model_on(nu):
    profile = AtmosphereProfile(
        pressure_edges_bar=[0.1, 0.5, 1.0],
        temperature_k=[240.0, 280.0],
        altitude_km=[10.0, 2.0],
        vmr={"H2O": [2.0e-4, 5.0e-3]},
    )
    return TelluricModel(profile, nu, ArrayOpacityBackend({"H2O": np.zeros((2, nu.size))}))


def test_rejects_descending_or_nonpositive_input():
    nu = np.geomspace(4300.0, 4310.0, 32)
    with pytest.raises(ValueError, match="strictly increasing"):
        StellarSpectrum(nu[::-1], np.ones(32))
    with pytest.raises(ValueError, match="finite and positive"):
        StellarSpectrum(nu, np.concatenate([[0.0], np.ones(31)]))


def test_from_wavelength_nm_sorts_into_ascending_wavenumber():
    wavelength = np.linspace(2000.0, 2010.0, 32)
    flux = np.linspace(1.0, 2.0, 32)
    spectrum = StellarSpectrum.from_wavelength_nm(wavelength, flux)
    assert np.all(np.diff(spectrum.wavenumber_cm1) > 0.0)
    # Ascending wavelength is descending wavenumber, so the flux reverses.
    np.testing.assert_allclose(spectrum.flux, flux[::-1])


def test_resampling_requires_full_coverage_of_the_model_grid():
    grid = np.geomspace(4300.0, 4310.0, 256)
    narrow = StellarSpectrum(np.geomspace(4302.0, 4308.0, 256), np.ones(256))
    with pytest.raises(ValueError, match="needs"):
        resample_stellar_source(narrow, grid)


def test_resampling_refuses_a_source_coarser_than_the_model_grid():
    grid = np.geomspace(4300.0, 4310.0, 4096)
    coarse = StellarSpectrum(np.geomspace(4299.0, 4311.0, 64), np.ones(64))
    with pytest.raises(ValueError, match="coarser"):
        resample_stellar_source(coarse, grid)


def test_broadening_is_the_identity_at_zero_width():
    values = 1.0 - 0.3 * np.exp(-0.5 * ((np.arange(512) - 256) / 4.0) ** 2)
    np.testing.assert_allclose(
        broaden_stellar_source(values, 1.0, vsini_kms=0.0, macroturbulence_kms=0.0), values
    )


@pytest.mark.parametrize("vsini,zeta", [(10.0, 0.0), (0.0, 8.0), (6.0, 5.0)])
def test_broadening_conserves_equivalent_width_and_widens_the_line(vsini, zeta):
    step = 0.5
    velocity = (np.arange(2048) - 1024) * step
    values = 1.0 - 0.3 * np.exp(-0.5 * (velocity / 3.0) ** 2)
    widened = broaden_stellar_source(values, step, vsini_kms=vsini, macroturbulence_kms=zeta)

    absorbed = np.sum(1.0 - values) * step
    widened_absorbed = np.sum(1.0 - widened) * step
    np.testing.assert_allclose(widened_absorbed, absorbed, rtol=1e-6)

    def second_moment(profile):
        weight = 1.0 - profile
        return np.sum(weight * velocity**2) / np.sum(weight)

    assert second_moment(widened) > second_moment(values)


def test_prepare_normalizes_and_matches_the_model_grid():
    nu = np.geomspace(4300.0, 4310.0, 1024)
    model = model_on(nu)
    spectrum = StellarSpectrum(np.geomspace(4299.0, 4311.0, 4096), np.full(4096, 7.5))
    prepared = prepare_stellar_source(spectrum, model, vsini_kms=2.0, macroturbulence_kms=2.15)
    assert prepared.shape == nu.shape
    np.testing.assert_allclose(np.median(prepared), 1.0, rtol=1e-12)

    unnormalized = prepare_stellar_source(spectrum, model, normalize=False)
    np.testing.assert_allclose(np.median(unnormalized), 7.5, rtol=1e-12)


def test_flat_source_is_a_unit_spectrum():
    nu = np.geomspace(4300.0, 4310.0, 64)
    np.testing.assert_allclose(StellarSpectrum.flat(nu).flux, 1.0)


def test_a_finer_source_is_accepted_across_the_whole_range():
    """The spacing test must compare velocities, not raw wavenumbers.

    A log-uniform source's wavenumber spacing varies across it in proportion to
    wavenumber. Comparing its *global median* spacing against the model grid's
    *local* spacing made the test depend on where in the file the median fell:
    a source spanning 900-5400 nm has its median near 4536 cm-1, six times its
    spacing at the red end, and every window below 2016 cm-1 was rejected
    although the source resolves 1.5x finer than the model everywhere.
    """
    source_power, target_power = 600_000.0, 400_000.0
    source = np.geomspace(1850.0, 11200.0, int(np.log(11200.0 / 1850.0) * source_power) + 1)
    spectrum = StellarSpectrum(source, 1.0 - 0.3 * np.sin(source))
    for lower, upper in ((1860.0, 1890.0), (2021.0, 2026.0), (5005.0, 5025.0), (10800.0, 10850.0)):
        target = np.geomspace(lower, upper, int(np.log(upper / lower) * target_power) + 1)
        resampled = resample_stellar_source(spectrum, target)
        assert resampled.shape == target.shape
        assert np.all(np.isfinite(resampled))


def test_resampling_still_refuses_a_genuinely_coarser_source():
    """The guard must survive the fix: a real under-sampling is still an error."""
    target = np.geomspace(2000.0, 2010.0, int(np.log(2010.0 / 2000.0) * 400_000.0) + 1)
    coarse = np.geomspace(1990.0, 2020.0, int(np.log(2020.0 / 1990.0) * 100_000.0) + 1)
    spectrum = StellarSpectrum(coarse, np.ones_like(coarse))
    with pytest.raises(ValueError, match="coarser than the model grid"):
        resample_stellar_source(spectrum, target)


def test_from_npz_recovers_a_reversed_continuum(tmp_path):
    """The Payne Zero writers left flux_total/flux_continuum in the opposite
    order to flux, so the loader picks the orientation that satisfies
    flux = flux_total / flux_continuum rather than assuming either one."""
    import numpy as np

    from jax_telluric import StellarSpectrum

    nu = np.linspace(5000.0, 5100.0, 512)
    continuum = 1.0e6 * (1.0 + 0.3 * (nu - nu[0]) / (nu[-1] - nu[0]))
    # Off centre on purpose: a line at the midpoint makes flux symmetric, and
    # then both orientations satisfy the identity and the test proves nothing.
    flux = 1.0 - 0.4 * np.exp(-0.5 * ((nu - 5020.0) / 0.5) ** 2)
    total = flux * continuum

    right = tmp_path / "right.npz"
    np.savez(right, wavenumber_cm1=nu, flux=flux, flux_total=total, flux_continuum=continuum)
    assert np.allclose(StellarSpectrum.from_npz(right).continuum, continuum)

    # The shipped models look like this one.
    wrong = tmp_path / "reversed.npz"
    np.savez(wrong, wavenumber_cm1=nu, flux=flux,
             flux_total=total[::-1], flux_continuum=continuum[::-1])
    assert np.allclose(StellarSpectrum.from_npz(wrong).continuum, continuum)


def test_from_npz_refuses_a_continuum_that_is_not_one(tmp_path):
    """Neither orientation satisfying the identity means these arrays do not
    belong to this spectrum, and no continuum beats a wrong one."""
    import numpy as np

    from jax_telluric import StellarSpectrum

    nu = np.linspace(5000.0, 5100.0, 512)
    flux = 1.0 - 0.4 * np.exp(-0.5 * ((nu - 5050.0) / 0.5) ** 2)
    path = tmp_path / "mismatched.npz"
    np.savez(path, wavenumber_cm1=nu, flux=flux,
             flux_total=np.ones_like(nu), flux_continuum=np.full_like(nu, 2.0))
    assert StellarSpectrum.from_npz(path).continuum is None


def test_resample_stellar_continuum_needs_a_continuum():
    import numpy as np

    from jax_telluric import StellarSpectrum, resample_stellar_continuum

    nu = np.linspace(5000.0, 5100.0, 64)
    assert resample_stellar_continuum(StellarSpectrum.flat(nu), nu[::2]) is None

    continuum = 1.0e6 * (1.0 + 0.2 * (nu - nu[0]) / (nu[-1] - nu[0]))
    spectrum = StellarSpectrum(nu, np.ones_like(nu), {}, continuum=continuum)
    resampled = resample_stellar_continuum(spectrum, nu[::2])
    assert np.allclose(resampled, continuum[::2])
