"""A site profile built from an epoch, and the paths a fit reads its data from."""

from pathlib import Path

import numpy as np
import pytest

from tellurix import DataPaths, load_atmosphere_csv
from tellurix.site_profile import (
    DEFAULT_EDGES_KM, EPOCH_DRY_VMR, afgl_dry_vmr, build_site_profile, write_profile_csv,
)

AVOGADRO = 6.02214076e23


def _kitt_peak(dry_vmr, precipitable_water_mm=3.11):
    return build_site_profile(789.1, 276.4, 2.096, 6.5, 216.65, 2.0, 5.0e-6,
                              precipitable_water_mm, dry_vmr)


def test_the_water_column_is_the_one_asked_for(tmp_path):
    """The column is renormalized, so the profile's shape cannot change it."""

    rows = write_profile_csv(tmp_path / "site.csv", _kitt_peak(EPOCH_DRY_VMR["1990"]), ["test"])
    profile = load_atmosphere_csv(tmp_path / "site.csv")

    assert rows == len(DEFAULT_EDGES_KM) - 1
    water = float(np.sum(np.asarray(profile.vmr["H2O"]) * np.asarray(profile.air_column_cm2)))
    assert water * 18.01528 / AVOGADRO * 10.0 == pytest.approx(3.11, rel=1e-3)


def test_the_written_file_reads_back_top_to_bottom(tmp_path):
    write_profile_csv(tmp_path / "site.csv", _kitt_peak(EPOCH_DRY_VMR["1994"]),
                      ["Kitt Peak", "two comment lines"])
    profile = load_atmosphere_csv(tmp_path / "site.csv")

    assert np.all(np.diff(np.asarray(profile.pressure_layer_bar)) > 0.0)
    assert float(np.max(profile.pressure_layer_bar)) < 0.7891
    assert profile.vmr["CO2"][-1] == pytest.approx(357.0e-6, rel=0.01)
    assert (tmp_path / "site.csv").read_text().startswith("# Kitt Peak\n# two comment lines\n")


def test_afgl_trace_gases_keep_their_shape_and_take_the_epochs_surface_value():
    centres = 2.096 + 0.5 * (np.asarray(DEFAULT_EDGES_KM[:-1]) + np.asarray(DEFAULT_EDGES_KM[1:]))
    dry = afgl_dry_vmr("midlatitude_winter", centres, "1990")

    assert "H2O" not in dry
    assert dry["CO2"][0] == pytest.approx(354.4e-6, rel=0.02)
    # Ozone rises by two orders of magnitude into the stratosphere.
    assert dry["O3"][-1] / dry["O3"][0] > 10.0


def test_a_missing_afgl_table_names_the_script_that_makes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract_afgl_profiles"):
        afgl_dry_vmr("tropical", np.array([1.0]), "2020", directory=tmp_path)


def test_the_two_data_layouts():
    checkout = DataPaths.bootstrapped("/repo")
    downloaded = DataPaths.downloaded("/data")

    assert checkout.line_file("H2O", 1) == Path(
        "/repo/data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule/01_H2O/01_H2O")
    assert checkout.mt_ckd == Path("/repo/data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc")
    assert downloaded.line_file("CO2", 2) == Path(
        "/data/aer_v_3.9/line_files_By_Molecule/02_CO2/02_CO2")
    assert downloaded.mt_ckd == Path("/data/mt_ckd/absco-ref_wv-mt-ckd.nc")
