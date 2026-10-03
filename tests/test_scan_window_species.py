"""The species-scan script's defaults.

This reads ``scripts/scan_window_species.py`` -- the repository's script, not the
package -- which is why it lives here.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


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
    source = (REPO / "scripts/scan_window_species.py").read_text()
    budget = float(source.split('"--line-budget", type=float, default=')[1].split(",")[0])
    threshold = float(source.split('"--threshold", type=float, default=')[1].split(",")[0])
    assert budget <= threshold / 100.0
