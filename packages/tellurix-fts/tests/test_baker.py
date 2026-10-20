"""Reading Baker et al. 2020's telluric spectra, from a file written here."""

import numpy as np
import pytest
from astropy.io import fits

from tellurix_fts.baker import baker_observations, baker_wavenumber_grid, read_baker_telluric


def _group(directory, group=3):
    exact = 9020.00549 + 0.0150661616 * np.arange(5000)
    header = fits.Header()
    header["CD_H2O"] = 22.6
    for number, airmass, jd in (("0005", 1.94447, 2457185.78), ("0261", 3.50469, 2457186.22917)):
        header[f"AMS_{number}"] = airmass
        header[f"JD_{number}"] = jd
        header[f"TAU_{number}"] = 1.0
    table = fits.BinTableHDU.from_columns([
        fits.Column("v", "E", array=exact.astype(np.float32)),
        fits.Column("telluric_0005", "E", array=np.linspace(0.9, 1.0, exact.size)),
        fits.Column("telluric_0261", "E", array=np.ones(exact.size)),
    ])
    fits.HDUList([fits.PrimaryHDU(header=header), table]).writeto(
        directory / f"telluric_spectra_{group}.fits")
    return exact


def test_the_float32_column_is_put_back_on_its_linear_grid(tmp_path):
    exact = _group(tmp_path)
    stored = exact.astype(np.float32).astype(float)
    assert np.abs(stored - exact).max() > 1e-4
    assert np.abs(baker_wavenumber_grid(stored) - exact).max() < 1e-5  # 0.3 m/s


def test_a_grid_that_is_not_linear_is_refused():
    with pytest.raises(ValueError, match="not a linear grid"):
        baker_wavenumber_grid(9020.0 + 0.015 * np.arange(1000) ** 1.01)


def test_observations_and_spectra_come_from_the_header_and_table(tmp_path):
    exact = _group(tmp_path)
    observations = baker_observations(tmp_path, groups=[3])
    assert [o.number for o in observations] == ["0005", "0261"]
    late = observations[1]
    assert late.airmass == pytest.approx(3.50469)
    assert late.utc.isoformat()[:16] == "2015-06-12T17:30"
    assert late.era5_hour.isoformat()[:16] == "2015-06-12T18:00"

    spectrum = read_baker_telluric(tmp_path, observations[0])
    assert spectrum.airmass_mean == pytest.approx(1.94447)
    assert np.abs(spectrum.wavenumber_vacuum_cm1 - exact).max() < 1e-5  # 0.3 m/s
    assert spectrum.flux[0] == pytest.approx(0.9)
