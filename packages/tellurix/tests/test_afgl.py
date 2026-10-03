"""The AFGL standard atmospheres shipped with the package.

They are what the species scan ranks against and what a site profile takes its
trace gases from. They exist because a hand-written species list hid OCS and O3
in the 4.9 um solar window; see docs/solar_fts_residual.md.
"""

import numpy as np
import pytest

from tellurix import AER_MOLECULE_IDS
from tellurix.site_profile import AFGL_DIRECTORY, AFGL_MODELS

AFGL = AFGL_DIRECTORY / "midlatitude_winter.csv"


def _read(path):
    rows = [line for line in open(path) if not line.startswith("#")]
    names = rows[0].strip().split(",")
    values = np.asarray([[float(v) for v in row.split(",")] for row in rows[1:]])
    return names, values


def test_afgl_profiles_cover_every_molecule_the_line_files_do():
    names, values = _read(AFGL)

    assert values.shape == (50, len(names))
    # Atomic oxygen is the one AER molecule HAPI has no partition function for,
    # so it is absent from the line-file table but present in the atmosphere.
    assert set(AER_MOLECULE_IDS) - set(names) == set()


def test_afgl_ozone_is_stratospheric():
    """The property that makes a surface mixing ratio the wrong ranking key.

    Ranking ozone by its ground abundance understated it by 17x and nearly
    left it out of the 2030-2060 fit; its column is stratospheric.
    """

    names, values = _read(AFGL)
    altitude = values[:, names.index("altitude_km")]
    ozone = values[:, names.index("O3")]

    peak = altitude[np.argmax(ozone)]
    assert 30.0 <= peak <= 45.0
    assert ozone.max() / ozone[0] > 100.0


def test_afgl_carries_the_species_that_were_missing():
    names, values = _read(AFGL)

    for species, floor in (("OCS", 1.0e-4), ("O3", 1.0e-2)):
        assert species in names
        assert values[:, names.index(species)].max() > floor


@pytest.mark.parametrize("model", AFGL_MODELS)
def test_every_extracted_model_has_the_same_shape(model):
    names, values = _read(AFGL_DIRECTORY / f"{model}.csv")

    assert values.shape == (50, 48)
    assert np.all(values[:, names.index("altitude_km")] >= 0.0)
    assert np.all(np.diff(values[:, names.index("altitude_km")]) > 0.0)
    # Abundances are ppmv and must be positive: the extractor floors the
    # 1e-14 placeholders rather than letting a log interpolation see zero.
    assert np.all(values[:, 1:] > 0.0)
