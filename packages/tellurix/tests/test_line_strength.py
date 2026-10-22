import hashlib
import json

import numpy as np
import pytest

from tellurix import AERLineDatabase, LineStrengthTable, line_strength_provenance, recorded_line_strength_table


def _record(nu, strength, elower, isotope=1):
    return (f"{5:2d}{isotope:1d}{nu:12.6f}{strength:10.3E}{2.0e-5:10.3E}"
            f"{'.0700'}{'.0800'}{elower:10.4f}{0.7:4.2f}{0.0:8.6f}")


def _table(**provenance):
    # Two lines share 5000.1 and differ in isotopologue and lower-state energy, as
    # HITRAN and AER list doublets; a third is another molecule at the same place.
    return LineStrengthTable(
        molecule_id=np.array([5, 5, 5, 1]), isotope=np.array([1, 1, 2, 1]),
        wavenumber_cm1=np.array([5000.0, 5000.1, 5000.1, 5000.2]),
        elower_cm1=np.array([100.0, 200.0, 300.0, 100.0]),
        factor=np.array([1.05, 0.9, 1.2, 2.0]), provenance=dict(provenance))


def test_factors_match_molecule_isotopologue_position_and_energy():
    factor, found = _table().factors(5, [1, 1, 2, 1, 1], [5000.0, 5000.1, 5000.1, 5000.2, 5000.0],
                                     [100.0, 200.0, 300.0, 100.0, 150.0])
    np.testing.assert_allclose(factor, [1.05, 0.9, 1.2, 1.0, 1.0])
    np.testing.assert_array_equal(found, [True, True, True, False, False])


def test_table_round_trips_and_refuses_bad_input(tmp_path):
    path = _table(built_from={"fits": ["wallace"]}).save(tmp_path / "strengths")
    assert path.name == "strengths"
    loaded = LineStrengthTable.load(path)
    np.testing.assert_array_equal(loaded.factor, [1.05, 0.9, 1.2, 2.0])
    assert loaded.provenance == {"built_from": {"fits": ["wallace"]}}
    with pytest.raises(ValueError, match="sorted"):
        LineStrengthTable(np.array([5, 5]), np.array([1, 1]), np.array([2.0, 1.0]), np.zeros(2), np.ones(2), {})
    with pytest.raises(ValueError, match="positive"):
        LineStrengthTable(np.array([5]), np.array([1]), np.array([1.0]), np.zeros(1), np.zeros(1), {})


def test_database_scales_only_the_lines_the_table_lists(tmp_path):
    lines = tmp_path / "05_CO"
    lines.write_text("\n".join([_record(5000.0, 1e-22, 100.0), _record(5000.1, 2e-22, 200.0),
                                _record(5000.1, 3e-22, 300.0, isotope=2), _record(5000.3, 4e-22, 100.0)]) + "\n")
    table = _table(built_from={"fits": ["wallace"]}).save(tmp_path / "strengths.npz")
    plain = AERLineDatabase(lines, "CO", (4999.0, 5001.0), margin_cm1=0.0)
    scaled = AERLineDatabase(lines, "CO", (4999.0, 5001.0), margin_cm1=0.0, line_strength_table=table)
    np.testing.assert_allclose(scaled.line_strength_ref_original / plain.line_strength_ref_original,
                               [1.05, 0.9, 1.2, 1.0])
    np.testing.assert_allclose(np.exp(scaled.logsij0), scaled.line_strength_ref_original)
    assert scaled.strengths_scaled == 3 and plain.strengths_scaled == 0
    # Every per-line array is restricted together, the scaled strengths with them.
    kept = scaled.restrict(np.array([False, True, False, True]))
    np.testing.assert_allclose(kept.line_strength_ref_original, [1.8e-22, 4e-22])

    # No key at all for AER's strengths: a batch record's physics are HDF5 attributes.
    assert line_strength_provenance(None) == {}
    record = line_strength_provenance(table)
    assert record["line_strength_table"] == "strengths.npz"
    assert record["line_strength_table_sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    assert json.loads(record["line_strength_built_from"]) == {"fits": ["wallace"]}

    assert recorded_line_strength_table({}, tmp_path) is None
    assert recorded_line_strength_table(record, tmp_path) == table
    with pytest.raises(ValueError, match="not the table"):
        recorded_line_strength_table(dict(record, line_strength_table_sha256="0" * 64), tmp_path)
    with pytest.raises(FileNotFoundError):
        recorded_line_strength_table(record, tmp_path / "elsewhere")
