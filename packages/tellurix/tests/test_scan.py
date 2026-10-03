"""The species scan and the AFGL profiles it ranks against.

These exist because a hand-written species list hid OCS and O3 in the 4.9 um
solar window, worth a factor of 2.5 in residual, and nine other explanations
were eliminated before anyone questioned the list. See
docs/solar_fts_residual.md.
"""

from pathlib import Path

import numpy as np
import pytest

from tellurix import AER_MOLECULE_IDS

# The repository root: these files are shared with the pipeline scripts.
REPO = Path(__file__).resolve().parents[3]


AFGL = str(REPO / "data/profiles/afgl/midlatitude_winter.csv")


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
    names, values = _read(str(REPO / f"data/profiles/afgl/{model}.csv"))

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


# The cache. A scan costs about six minutes and depends on the window and the
# atmosphere alone, so it is stored rather than repeated -- which is only safe
# if a stale entry cannot be mistaken for a fresh one.

def _identity(tmp_path, text="a,b\n1,2\n", window=(2030.0, 2060.0)):
    from tellurix import ScanIdentity

    profile = tmp_path / "profile.csv"
    profile.write_text(text)
    return ScanIdentity.for_profile(
        profile, window, threshold=1e-3, line_budget=1e-5,
        margin_cm1=25.0, fwhm_cm1=0.01753, samples_per_resolution=2.0)


def test_scan_identity_follows_the_profile_contents_not_its_name(tmp_path):
    """The failure this prevents: a profile rebuilt with a different AFGL model
    keeps its filename, and would otherwise reuse a ranking built for the old
    atmosphere -- exactly the kind of silent reuse that hid OCS."""
    original = _identity(tmp_path)
    rebuilt = _identity(tmp_path, text="a,b\n1,3\n")

    assert rebuilt.digest != original.digest

    moved = tmp_path / "elsewhere.csv"
    moved.write_text("a,b\n1,2\n")
    from tellurix import ScanIdentity
    renamed = ScanIdentity.for_profile(
        moved, (2030.0, 2060.0), threshold=1e-3, line_budget=1e-5,
        margin_cm1=25.0, fwhm_cm1=0.01753, samples_per_resolution=2.0)
    assert renamed.digest == original.digest


@pytest.mark.parametrize("field,value", [
    ("threshold", 1e-4),
    ("line_budget", 1e-6),
    ("margin_cm1", 30.0),
    ("fwhm_cm1", 0.02),
    ("samples_per_resolution", 4.0),
    ("line_files", "aer_v_4.0"),
])
def test_every_setting_that_can_change_the_ranking_changes_the_digest(
        tmp_path, field, value):
    import dataclasses

    identity = _identity(tmp_path)
    assert dataclasses.replace(identity, **{field: value}).digest != identity.digest


def test_a_saved_scan_reads_back(tmp_path):
    from tellurix import load_scan, save_scan

    identity = _identity(tmp_path)
    cache = tmp_path / "scans"
    assert load_scan(cache, identity) is None

    path = save_scan(cache, identity, {"species": ["H2O", "CO2"], "fit": []})
    assert path.name.startswith("scan_2030_2060_")
    assert load_scan(cache, identity)["species"] == ["H2O", "CO2"]
    assert not list(cache.glob("*.partial"))


def test_an_entry_filed_under_the_wrong_name_is_refused(tmp_path):
    from tellurix import read_scan, save_scan

    identity = _identity(tmp_path)
    path = save_scan(tmp_path / "scans", identity, {"species": []})
    copied = path.with_name("scan_2030_2060_deadbeefcafe.json")
    copied.write_text(path.read_text())

    with pytest.raises(ValueError, match="identity does not produce"):
        read_scan(copied)


def test_a_scan_written_before_identities_is_refused(tmp_path):
    import json

    from tellurix import read_scan

    path = tmp_path / "scan_2030_2060_000000000000.json"
    path.write_text(json.dumps({"species": ["H2O"]}))
    with pytest.raises(ValueError, match="predates cache identities"):
        read_scan(path)


# Re-thresholding. A scan ranks the vertical column; what a spectrum measures is
# the slant one, and Kitt Peak's two files differ by 2.4x in air mass.

def _report(tmp_path, floor=1e-4):
    identity = _identity(tmp_path)
    import dataclasses
    identity = dataclasses.replace(identity, threshold=floor)
    return {
        "identity": identity.as_dict(),
        "fit": [{"species": "H2O", "peak_optical_depth": 2.0e1},
                {"species": "CO2", "peak_optical_depth": 5.0e-3}],
        "rejected": [{"species": "OCS", "peak_optical_depth": 4.0e-4},
                     {"species": "NO2", "peak_optical_depth": 1.5e-4}],
    }


def test_a_slant_path_promotes_a_species_the_vertical_column_rejects(tmp_path):
    """The failure this prevents: file 4 sits at air mass 4.73, where a species
    at 4e-4 vertical reaches 1.9e-3 along the path and is measurable."""
    from tellurix import species_above

    report = _report(tmp_path)
    assert species_above(report, 1e-3, airmass=1.0) == ["H2O", "CO2"]
    assert species_above(report, 1e-3, airmass=4.73) == ["H2O", "CO2", "OCS"]
    # 1.5e-4 * 4.73 = 7.1e-4, so NO2 clears a looser cut at the same air mass.
    assert species_above(report, 7.0e-4, airmass=4.73) == ["H2O", "CO2", "OCS", "NO2"]


def test_species_are_returned_strongest_first(tmp_path):
    from tellurix import species_above

    depths = {"H2O": 2.0e1, "CO2": 5.0e-3, "OCS": 4.0e-4, "NO2": 1.5e-4}
    got = species_above(_report(tmp_path), 7.0e-4, airmass=4.73)
    assert got == sorted(got, key=lambda s: -depths[s])


def test_a_cut_below_what_the_scan_evaluated_is_refused(tmp_path):
    """Rejecting on the bound is the one cut a scan cannot undo, so asking for a
    threshold the scan never reached has to fail rather than quietly under-report."""
    from tellurix import species_above

    report = _report(tmp_path, floor=2e-4)
    with pytest.raises(ValueError, match="rejected on its bound"):
        species_above(report, 1e-4, airmass=1.0)
    # 1e-3 at air mass 4.73 asks about 2.11e-4 vertical, just inside the floor:
    # that is what the default 2e-4 floor is chosen to cover.
    assert species_above(report, 1e-3, airmass=4.73)
    with pytest.raises(ValueError, match="rejected on its bound"):
        species_above(report, 1e-3, airmass=10.0)


def test_a_freshly_computed_scan_is_as_self_describing_as_a_cached_one(tmp_path,
                                                                      monkeypatch):
    """The bug this caught: `cached_scan` returned the raw computation on a miss
    and the stored copy on a hit, so `species_above` worked on every cache hit
    and failed on every cold window -- which is a whole scan pass, and exactly
    the half that is not exercised while developing against a warm cache."""
    import tellurix.scan as scan

    from tellurix import cached_scan, species_above

    identity = _identity(tmp_path)
    computed = {"species": ["H2O"], "identity": identity.as_dict(),
                "fit": [{"species": "H2O", "peak_optical_depth": 1.0}], "rejected": []}
    monkeypatch.setattr(scan, "scan_window", lambda *a, **k: computed)

    cache = tmp_path / "scans"
    fresh = cached_scan(cache, identity, None, tmp_path)
    warm = cached_scan(cache, identity, None, tmp_path)
    assert species_above(fresh, 1e-3) == species_above(warm, 1e-3) == ["H2O"]
    assert fresh["identity"] == warm["identity"] == identity.as_dict()
