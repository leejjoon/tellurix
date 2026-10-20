import gzip

import numpy as np
import pytest

from tellurix_fts.flux_atlas import read_flux_atlas, read_wallace2011_correction


def _wallace_lines(nu):
    # wavenumber, corrected flux, transmission, observed flux
    return "".join(f"{v:10.4f} {0.9:.5f} {0.8:.5f} {0.5 + 1e-3 * i:.5f}\n" for i, v in enumerate(nu))


def test_wallace_reads_the_observed_column_and_drops_ties(tmp_path):
    nu = [10000.0, 10000.0068, 10000.0068, 10000.0136, 10000.0204, 10000.0272, 10000.034,
          10000.0408, 10000.0476, 10000.0544, 10000.0612]
    path = tmp_path / "sptr.reg1"
    path.write_text(_wallace_lines(nu))

    spectrum = read_flux_atlas("wallace2011", path, (9999.0, 10001.0), airmass=1.5, fwhm_cm1=0.0227)

    assert spectrum.wavenumber_vacuum_cm1.size == len(nu) - 1
    assert spectrum.flux[0] == pytest.approx(0.5)
    # The tied neighbour is dropped, the first of the pair kept.
    assert spectrum.flux[2] == pytest.approx(0.503)
    assert spectrum.airmass_mean == 1.5
    assert spectrum.stated_resolution_cm1 == 0.0227


def test_wallace_regions_cache_apart(tmp_path):
    # sptr.reg1 and sptr.reg2 share the stem "sptr"; one cache file served both.
    for region, offset in ((1, 0.0), (2, 0.1)):
        (tmp_path / f"sptr.reg{region}").write_text(
            "".join(f"{10000.0 + 0.0068 * i:10.4f} 0.9 0.8 {0.5 + offset:.5f}\n" for i in range(10)))
    cache = tmp_path / "cache"
    one = read_flux_atlas("wallace2011", tmp_path / "sptr.reg1", (9999.0, 10001.0), cache_directory=cache)
    two = read_flux_atlas("wallace2011", tmp_path / "sptr.reg2", (9999.0, 10001.0), cache_directory=cache)

    assert one.flux[0] == pytest.approx(0.5)
    assert two.flux[0] == pytest.approx(0.6)


def test_wallace_correction_blanks_what_the_authors_did_not_correct(tmp_path):
    path = tmp_path / "sptr.reg1"
    path.write_text(" 10000.0000 0.99000 0.90000 0.89100\n"
                    " 10000.0068 0.00000 0.70000 0.69000\n"
                    " 10000.0136 NaN     NaN     0.10000\n"
                    " 10000.0204 0.00257 ******* 0.05317\n")

    nu, corrected, transmission = read_wallace2011_correction(path)

    assert nu.size == 4
    assert corrected[0] == pytest.approx(0.99)
    assert np.isnan(corrected[1:3]).all()
    assert transmission[1] == pytest.approx(0.7)
    assert np.isnan(transmission[3])


def test_iag_reads_gzip_and_cuts_and_caches(tmp_path):
    nu = 9387.0 + 0.004 * np.arange(40)
    path = tmp_path / "spvis.dat.gz"
    with gzip.open(path, "wt") as handle:
        handle.writelines(f"{v:.8f} {1.0 - 1e-3 * i:.6f} 0.01\n" for i, v in enumerate(nu))

    cut = (nu[5], nu[30])
    spectrum = read_flux_atlas("iag", path, cut, cache_directory=tmp_path / "cache")
    again = read_flux_atlas("iag", path, cut, cache_directory=tmp_path / "cache")

    assert (tmp_path / "cache" / f"spvis_{cut[0]:.0f}_{cut[1]:.0f}.npz").exists()
    assert spectrum.wavenumber_vacuum_cm1.size == 24
    np.testing.assert_array_equal(spectrum.flux, again.flux)
    assert spectrum.airmass_mean is None


def test_unknown_atlas_is_refused(tmp_path):
    path = tmp_path / "x"
    path.write_text("1 2 3 4\n")
    with pytest.raises(ValueError, match="unknown flux atlas"):
        read_flux_atlas("kurucz", path, (0.0, 2.0))
