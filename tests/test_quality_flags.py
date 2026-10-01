import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from quality_flags import species_at_upper_bound  # noqa: E402


def test_only_the_upper_bound_flags_a_window():
    # Measured across the solar runs: a column driven to the lower bound leaves
    # at most ~1% of absorption to be wrong by; one at the upper bound put up to
    # 13% of absorption into the model that is not in the sky.
    columns = {"H2O": 2.0, "CO2": -2.0, "CH4": 0.3}
    assert species_at_upper_bound(["H2O", "CO2", "lsf_sigma_kms"], columns) == ["H2O"]
    assert species_at_upper_bound(["CO2"], columns) == []


def test_reads_the_records_joined_string_and_ignores_non_columns():
    columns = {"H2O": 1.99, "O2": 0.1}
    assert species_at_upper_bound(b"H2O+lsf_sigma_kms", columns) == ["H2O"]
    assert species_at_upper_bound("lsf_sigma_kms+continuum_1", columns) == []
    assert species_at_upper_bound("", columns) == []
