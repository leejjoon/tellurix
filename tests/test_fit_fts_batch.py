"""The solar batch driver's run record.

This imports ``scripts/fit_fts_batch.py`` -- the repository's script, not a
package -- which is why it lives here.
"""

import importlib.util
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[1]


def _driver():
    spec = importlib.util.spec_from_file_location("fit_fts_batch", REPO / "scripts/fit_fts_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pins_are_recorded_as_one_hdf5_attribute(tmp_path):
    # A record's physics become HDF5 attributes, which hold no dict: the first
    # pinned run (Wallace 2011 region 3) fitted every window and then failed here.
    physics = _driver().physics_record(REPO, 0.0578, pin=["velocity_kms=-0.36", "H2O=0.035"])

    assert physics["pin"] == "H2O=0.035,velocity_kms=-0.36"
    with h5py.File(tmp_path / "record.h5", "w") as handle:
        for key, value in physics.items():
            handle.attrs[key] = "" if value is None else value


def test_a_run_without_pins_records_none():
    assert "pin" not in _driver().physics_record(REPO, 0.0578)
