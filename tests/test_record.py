"""The run record must survive a round trip and refuse what it cannot read."""

import json

import numpy as np
import pytest

from jax_telluric import BoxcarFTSInstrumentProfile, read_record, write_record
from jax_telluric.record import (
    FORMAT_VERSION,
    ils_fingerprint,
    file_sha256,
    parameters_from_row,
    text,
    to_json,
)

SPECIES = ("H2O", "CO2")
NAMES = ("H2O", "CO2", "velocity_kms", "stellar_velocity_kms", "wavelength_stretch",
         "lsf_sigma_kms", "continuum_0", "continuum_1", "log_jitter")


def make_pages(count=3, seed=0):
    rng = np.random.default_rng(seed)
    pages = []
    for index in range(count):
        correlation = np.eye(len(NAMES))
        correlation[0, 1] = correlation[1, 0] = 0.62
        pages.append({
            "page": f"ab{5000 + index}_", "epoch": "summer" if index % 2 else "winter",
            "page_sha256": "ab" * 32,
            "log_column_scales": {"H2O": 0.1 * index, "CO2": -0.2},
            "continuum_coeffs": np.array([0.01, -0.02, 0.003, 0.0]),
            "v1": 5000.0 + index, "v2": 5020.0 + index, "mopd_cm": 15.17,
            "pixels": 1250, "grid_points": 2794, "reliable": 1166,
            "velocity_kms": 0.28, "stellar_velocity_kms": 13.67,
            "wavelength_stretch": 0.0, "lsf_sigma_kms": 0.5, "log_jitter": -3.9,
            "pixel_sigma": 0.0059, "residual_rms": 0.021,
            "residual_rms_over_noise": 3.59, "reduced_chi2": 12.9,
            "median_transmission": 0.887, "continuum_level": 0.998,
            "continuum_level_pixels": 913, "condition_number": 1.2e4,
            "all_stages_converged": True, "negligible_telluric": False,
            "free_species": "CO2+H2O", "at_bound": "lsf_sigma_kms",
            "sigma": rng.uniform(1e-3, 1e-2, len(NAMES)),
            "correlation": correlation,
            "ils_velocity_kms": np.linspace(-20.0, 20.0, 65),
            "ils_profile": np.exp(-0.5 * (np.linspace(-4, 4, 65)) ** 2),
        })
    return pages


def write(tmp_path, pages=None, **overrides):
    pages = make_pages() if pages is None else pages
    return write_record(
        tmp_path / "run.h5",
        run={"created": "2026-09-18T00:00:00", "driver": "test", "driver_sha256": "0" * 64},
        config={"resolving_power": 100000.0, "saturation_floor": 0.02,
                "telluric_ceiling": 1.05, "column": "observed"},
        physics={"accuracy_mode": "mt_ckd", "pressure_shift": True},
        inputs={"stellar": "arcturus.npz", "stellar_sha256": "cd" * 32},
        parameter_names=NAMES, species=SPECIES, pages=pages, continuum_degree=3,
        **overrides,
    )


def test_a_record_survives_a_round_trip(tmp_path):
    pages = make_pages()
    record = read_record(write(tmp_path, pages))

    assert record.format_version == FORMAT_VERSION
    assert record.parameter_names == NAMES
    assert record.config["saturation_floor"] == 0.02
    assert record.config["column"] == "observed"
    assert record.physics["accuracy_mode"] == "mt_ckd"
    assert record.inputs["stellar_sha256"] == "cd" * 32
    assert record.run["driver"] == "test"
    assert len(record.pages) == len(pages)

    for index, page in enumerate(pages):
        row = record.pages[index]
        assert text(row["page"]) == page["page"]
        assert text(row["epoch"]) == page["epoch"]
        assert text(row["free_species"]) == page["free_species"]
        for scalar in ("v1", "v2", "mopd_cm", "velocity_kms", "continuum_level"):
            assert row[scalar] == pytest.approx(page[scalar])
        assert row["pixels"] == page["pixels"]
        assert bool(row["all_stages_converged"]) is page["all_stages_converged"]
        np.testing.assert_allclose(row["continuum_coeffs"], page["continuum_coeffs"])
        for name in SPECIES:
            assert row[f"log_column_{name}"] == pytest.approx(page["log_column_scales"][name])
        np.testing.assert_allclose(record.sigma[index], page["sigma"])


def test_parameters_rebuild_from_a_row(tmp_path):
    pages = make_pages()
    record = read_record(write(tmp_path, pages))
    parameters = parameters_from_row(record.row("ab5001_", "summer"), SPECIES)
    assert parameters.log_column_scales["H2O"] == pytest.approx(0.1)
    assert float(parameters.stellar_velocity_kms) == pytest.approx(13.67)
    np.testing.assert_allclose(
        np.asarray(parameters.continuum_coeffs), pages[1]["continuum_coeffs"]
    )


def test_covariance_is_the_correlation_times_the_deviations(tmp_path):
    """Stored split in two, because that is the same information in less space."""
    pages = make_pages()
    record = read_record(write(tmp_path, pages))
    covariance = record.covariance(0)
    sigma = pages[0]["sigma"]
    assert covariance[0, 1] == pytest.approx(0.62 * sigma[0] * sigma[1], rel=1e-6)
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), sigma, rtol=1e-6)


def test_a_missing_page_fails_loudly(tmp_path):
    record = read_record(write(tmp_path))
    with pytest.raises(KeyError, match="not in this record"):
        record.row("ab9999_", "summer")


def test_an_unknown_format_is_refused(tmp_path):
    import h5py

    path = write(tmp_path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="record format"):
        read_record(path)


def test_an_empty_run_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one page"):
        write(tmp_path, pages=[])


def test_the_json_view_is_serializable_and_keeps_the_provenance(tmp_path):
    record = read_record(write(tmp_path))
    view = to_json(record)
    encoded = json.dumps(view)
    assert '"format_version"' in encoded
    assert view["config"]["column"] == "observed"
    assert view["physics"]["accuracy_mode"] == "mt_ckd"
    assert len(view["results"]) == len(record.pages)
    assert view["results"][0]["page"] == "ab5000_"


def test_the_instrument_fingerprint_follows_the_profile_it_samples():
    """It must come from the real convolve, or it cannot detect a change."""
    step = 0.75
    narrow = BoxcarFTSInstrumentProfile(
        mopd_cm=15.17, wavenumber_center_cm1=5015.0, max_residual_sigma_kms=4.0
    )
    wide = BoxcarFTSInstrumentProfile(
        mopd_cm=7.5, wavenumber_center_cm1=5015.0, max_residual_sigma_kms=4.0
    )
    offsets, profile = ils_fingerprint(narrow, 0.5, step)
    again, repeat = ils_fingerprint(narrow, 0.5, step)
    np.testing.assert_array_equal(offsets, again)
    np.testing.assert_allclose(profile, repeat)

    assert profile.size == 65
    assert np.all(np.isfinite(profile))
    assert profile.argmax() == 32, "the profile should peak at zero offset"

    _, broader = ils_fingerprint(wide, 0.5, step)
    assert not np.allclose(profile, broader), "half the path difference must show"
    _, smoothed = ils_fingerprint(narrow, 3.0, step)
    assert smoothed.max() < profile.max(), "more residual broadening lowers the peak"


@pytest.mark.parametrize("samples", [8, 64, 4])
def test_the_fingerprint_refuses_a_sample_count_with_no_centre(samples):
    instrument = BoxcarFTSInstrumentProfile(
        mopd_cm=15.17, wavenumber_center_cm1=5015.0, max_residual_sigma_kms=4.0
    )
    with pytest.raises(ValueError, match="odd number of samples"):
        ils_fingerprint(instrument, 0.5, 0.75, samples=samples)


def test_file_hashing_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "lines.txt"
    payload = b"a line file\n" * 5000
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_shards_merge_into_one_record(tmp_path):
    """A run is split across devices; its record must come back together."""
    from jax_telluric.record import merge_records

    left = write_record(
        tmp_path / "a.h5", run={"created": "x", "driver": "t", "driver_sha256": "0" * 64},
        config={"resolving_power": 100000.0}, physics={"accuracy_mode": "mt_ckd"},
        inputs={"stellar": "s.npz"}, parameter_names=NAMES, species=SPECIES,
        pages=make_pages(2, seed=1), continuum_degree=3)
    right_pages = make_pages(2, seed=2)
    for index, page in enumerate(right_pages):
        page["page"] = f"ab{7000 + index}_"
    right = write_record(
        tmp_path / "b.h5", run={"created": "x", "driver": "t", "driver_sha256": "0" * 64},
        config={"resolving_power": 100000.0}, physics={"accuracy_mode": "mt_ckd"},
        inputs={"stellar": "s.npz"}, parameter_names=NAMES, species=SPECIES,
        pages=right_pages, continuum_degree=3)

    merged = read_record(merge_records([left, right], tmp_path / "all.h5"))
    assert len(merged.pages) == 4
    assert merged.config["resolving_power"] == 100000.0
    names = [text(r["page"]) for r in merged.pages]
    assert names == sorted(names), "rows should come back in a stable order"
    assert merged.sigma.shape == (4, len(NAMES))
    assert merged.correlation.shape == (4, len(NAMES), len(NAMES))
    np.testing.assert_allclose(merged.row("ab7001_", "summer")["v1"], 5001.0)


def test_merging_refuses_shards_that_disagree(tmp_path):
    from jax_telluric.record import merge_records

    common = dict(run={"created": "x", "driver": "t", "driver_sha256": "0" * 64},
                  physics={"accuracy_mode": "mt_ckd"}, inputs={"stellar": "s.npz"},
                  parameter_names=NAMES, species=SPECIES, continuum_degree=3)
    left = write_record(tmp_path / "a.h5", config={"resolving_power": 100000.0},
                        pages=make_pages(1, seed=1), **common)
    right = write_record(tmp_path / "b.h5", config={"resolving_power": 45000.0},
                         pages=make_pages(1, seed=2), **common)
    with pytest.raises(ValueError, match="disagree about config: resolving_power"):
        merge_records([left, right], tmp_path / "all.h5")


def test_merging_compares_list_valued_settings_without_tripping(tmp_path):
    """h5py hands a list-valued attribute back as an array; == is ambiguous there."""
    from jax_telluric.record import merge_records

    common = dict(run={"created": "x", "driver": "t", "driver_sha256": "0" * 64},
                  config={"stages": ["continuum", "velocity", "columns"]},
                  physics={"accuracy_mode": "mt_ckd"}, inputs={"stellar": "s.npz"},
                  parameter_names=NAMES, species=SPECIES, continuum_degree=3)
    left = write_record(tmp_path / "a.h5", pages=make_pages(1, seed=1), **common)
    other = make_pages(1, seed=2)
    other[0]["page"] = "ab8000_"
    right = write_record(tmp_path / "b.h5", pages=other, **common)
    merged = read_record(merge_records([left, right], tmp_path / "all.h5"))
    assert len(merged.pages) == 2
    assert [text(s) for s in merged.config["stages"]] == ["continuum", "velocity", "columns"]

    conflicting = {**common, "config": {"stages": ["continuum"]}}
    odd = write_record(tmp_path / "c.h5", pages=make_pages(1, seed=3), **conflicting)
    with pytest.raises(ValueError, match="disagree about config: stages"):
        merge_records([left, odd], tmp_path / "bad.h5")


def igrins_pages(count=3):
    """Rows keyed the way an IGRINS night is: one frame, many echelle orders."""
    pages = []
    for index in range(count):
        page = make_pages(1, seed=index)[0]
        page.pop("page"), page.pop("epoch")
        page.update({"frame": "SDCH_20181220_0100", "order": f"H{index:02d}",
                     "airmass": 1.07 + 0.4 * index, "zenith_angle_deg": 20.0 + index,
                     "telescope": "Discovery Channel"})
        pages.append(page)
    return pages


def write_igrins(tmp_path, pages=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return write_record(
        tmp_path / "night.h5",
        run={"created": "2026-09-19T00:00:00", "driver": "igrins", "driver_sha256": "0" * 64},
        config={"resolving_power": 45000.0}, physics={"accuracy_mode": "mt_ckd"},
        inputs={"spec": "SDCH.fits"}, parameter_names=NAMES, species=SPECIES,
        pages=igrins_pages() if pages is None else pages, continuum_degree=3,
        key_fields=("frame", "order"),
        extra_columns=(("airmass", "f8"), ("zenith_angle_deg", "f8"),
                       ("telescope", "S256")),
    )


def test_a_record_can_be_keyed_on_something_other_than_a_page(tmp_path):
    record = read_record(write_igrins(tmp_path))

    assert record.key_fields == ("frame", "order")
    assert record.key(0) == ("SDCH_20181220_0100", "H00")
    row = record.row("SDCH_20181220_0100", "H01")
    assert text(row["order"]) == "H01"
    assert row["airmass"] == pytest.approx(1.47)
    assert text(row["telescope"]) == "Discovery Channel"
    # The shared schema is still all there.
    assert row["lsf_sigma_kms"] == pytest.approx(0.5)
    assert row["log_column_H2O"] == pytest.approx(0.0)


def test_the_wrong_number_of_key_values_is_refused(tmp_path):
    record = read_record(write_igrins(tmp_path))
    with pytest.raises(KeyError, match="keyed on frame, order"):
        record.row("SDCH_20181220_0100")


def test_a_json_view_follows_whatever_the_key_is(tmp_path):
    view = to_json(read_record(write_igrins(tmp_path)))
    assert view["results"][0]["frame"] == "SDCH_20181220_0100"
    assert view["results"][0]["order"] == "H00"
    assert "page" not in view["results"][0]
    assert view["results"][0]["airmass"] == pytest.approx(1.07)
    json.dumps(view)


def test_merging_keeps_the_key_and_sorts_on_it(tmp_path):
    from jax_telluric.record import merge_records

    left = write_igrins(tmp_path / "a")
    right_pages = igrins_pages()
    for index, page in enumerate(right_pages):
        page["frame"] = "SDCH_20181220_0033"
        page["order"] = f"H{index + 10:02d}"
    right = write_igrins(tmp_path / "b", pages=right_pages)

    merged = read_record(merge_records([left, right], tmp_path / "all.h5"))
    assert merged.key_fields == ("frame", "order")
    assert len(merged.pages) == 6
    keys = [merged.key(i) for i in range(len(merged.pages))]
    assert keys == sorted(keys), "rows should come back in a stable order"
    assert merged.row("SDCH_20181220_0033", "H12")["airmass"] == pytest.approx(1.87)


def test_the_arcturus_default_is_unchanged(tmp_path):
    """The atlas record predates the configurable key and must still read."""
    record = read_record(write(tmp_path))
    assert record.key_fields == ("page", "epoch")
    assert record.row("ab5001_", "summer") is not None
