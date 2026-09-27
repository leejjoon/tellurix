"""The species scan and the AFGL profiles it ranks against.

These exist because a hand-written species list hid OCS and O3 in the 4.9 um
solar window, worth a factor of 2.5 in residual, and nine other explanations
were eliminated before anyone questioned the list. See
docs/solar_fts_residual.md.
"""

import numpy as np
import pytest

from tellurix import AER_MOLECULE_IDS


AFGL = "data/profiles/afgl/midlatitude_winter.csv"


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


@pytest.mark.parametrize("model", ["tropical", "midlatitude_summer",
                                   "midlatitude_winter", "subarctic_summer",
                                   "subarctic_winter", "us_standard_1976"])
def test_every_extracted_model_has_the_same_shape(model):
    names, values = _read(f"data/profiles/afgl/{model}.csv")

    assert values.shape == (50, 48)
    assert np.all(values[:, names.index("altitude_km")] >= 0.0)
    assert np.all(np.diff(values[:, names.index("altitude_km")]) > 0.0)
    # Abundances are ppmv and must be positive: the extractor floors the
    # 1e-14 placeholders rather than letting a log interpolation see zero.
    assert np.all(values[:, 1:] > 0.0)


def test_the_line_budget_cannot_move_a_species_across_the_threshold():
    """The scan's one load-bearing approximation.

    `select_significant_lines` drops the weakest lines whose summed bounded
    contribution stays under a budget, so the peak optical depth it reports is
    wrong by at most that budget. The scan's default budget is two orders of
    magnitude below its default threshold, which is what makes the verdict
    safe rather than merely cheap.
    """

    # Read the defaults from the source rather than importing the script,
    # which pulls in jax and the whole opacity stack for a two-line check.
    source = open("scripts/scan_window_species.py").read()
    budget = float(source.split('"--line-budget", type=float, default=')[1].split(",")[0])
    threshold = float(source.split('"--threshold", type=float, default=')[1].split(",")[0])
    assert budget <= threshold / 100.0
