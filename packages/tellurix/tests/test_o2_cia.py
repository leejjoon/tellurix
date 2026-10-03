import dataclasses
from pathlib import Path

import numpy as np
import pytest

from tellurix import ContinuumSum, O2CollisionInducedContinuum, load_atmosphere_csv

DATA = Path(__file__).resolve().parent / "data"
PROFILE = str(DATA / "profiles/kitt_peak_19830626_file3.csv")


@pytest.fixture(scope="module")
def profile():
    full = load_atmosphere_csv(PROFILE)
    return dataclasses.replace(full, vmr={"O2": full.vmr["O2"], "H2O": full.vmr["H2O"]})


def tau(profile, nu, o2_scale=1.0):
    bound = O2CollisionInducedContinuum(nu).bind(profile)
    vmr = {"O2": profile.vmr["O2"] * o2_scale, "H2O": profile.vmr["H2O"]}
    return np.asarray(bound.optical_depth(profile, vmr)).sum(axis=0)


def test_terms_switch_on_only_where_their_bands_are():
    assert O2CollisionInducedContinuum(np.linspace(8950.0, 9050.0, 50)).terms == ()
    assert O2CollisionInducedContinuum(np.linspace(13050.0, 13100.0, 50)).terms == ("O2INF3",)
    assert O2CollisionInducedContinuum(np.linspace(7800.0, 7900.0, 50)).terms == ("O2INF1",)
    assert O2CollisionInducedContinuum(np.linspace(15800.0, 15900.0, 50)).terms == ("O2_VIS",)


def test_the_a_band_is_linear_in_o2_and_the_1um_band_quadratic(profile):
    # O2INF3 is O2 column x air density: linear in the fitted O2 scale.
    # O2INF2 is O2 column x O2 fraction: quadratic -- an O2-O2 band.
    a_band = np.linspace(13100.0, 13150.0, 101)
    one_micron = np.linspace(9350.0, 9450.0, 101)
    np.testing.assert_allclose(tau(profile, a_band, 1.2), 1.2 * tau(profile, a_band), rtol=1e-12)
    np.testing.assert_allclose(tau(profile, one_micron, 1.2), 1.44 * tau(profile, one_micron),
                               rtol=1e-12)


def test_matches_the_lblrtm_peak_it_was_validated_against(profile):
    # docs/o2_cia_validation.json: LBLRTM 12.17's vertical A-band peak is 2.196e-2
    # on this profile, O2 only; tellurix 2.199e-2. H2O enters only O2INF1.
    nu = np.linspace(13050.0, 13200.0, 1501)
    o2_only = dataclasses.replace(profile, vmr={"O2": profile.vmr["O2"]})
    bound = O2CollisionInducedContinuum(nu).bind(o2_only)
    peak = np.asarray(bound.optical_depth(o2_only, {"O2": o2_only.vmr["O2"]})).sum(axis=0).max()
    assert peak == pytest.approx(2.196e-2, rel=3e-3)


def test_a_sum_of_continua_adds_their_optical_depths(profile):
    nu = np.linspace(13100.0, 13150.0, 101)
    one = O2CollisionInducedContinuum(nu)
    both = ContinuumSum((one, one)).bind(profile)
    single = one.bind(profile)
    vmr = {"O2": profile.vmr["O2"], "H2O": profile.vmr["H2O"]}
    np.testing.assert_allclose(np.asarray(both.optical_depth(profile, vmr)),
                               2.0 * np.asarray(single.optical_depth(profile, vmr)), rtol=1e-12)
