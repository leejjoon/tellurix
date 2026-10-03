"""Which fitted windows a product must not offer as a correction.

One rule, measured across every solar run (docs/solar_fit_plan.md §4m): a
species whose column rails at the **upper** bound is the fit using that column
to absorb something else -- usually the solar model's error -- and it puts
absorption into the model that is not in the sky. Over 81 railed species the
ten at the upper bound carried a median 7.9% of absorption, up to 12.9%; the 71
at the lower bound carried a median 0.17% and never more than 5%, because a
column the fit drives *down* leaves little absorption behind to be wrong by.
So the lower bound is harmless to a correction and the upper bound is not, and
only the upper bound flags a window.

What this does not catch: a column far too high but short of the bound (one of
the 1983 edge windows sits at 5.9x). That is stated, not thresholded.
"""

from __future__ import annotations

import numpy as np

UPPER_BOUND_NOTE = (
    "a fitted column at its upper bound: the fit used that species to absorb something "
    "else (usually the solar model's error), so the window's correction carries absorption "
    "that is not in the sky")


def species_at_upper_bound(at_bound, log_columns: dict) -> list[str]:
    """The species of one window that rail at the upper column bound.

    ``at_bound`` is the fit's list of bound parameters, or the record's
    '+'-joined string of them; ``log_columns`` maps species to fitted log scale.
    A species is at the upper bound when it is at a bound and its log scale is
    positive -- the bounds are symmetric in log, so the sign says which.
    """

    if isinstance(at_bound, bytes):
        at_bound = at_bound.decode()
    if isinstance(at_bound, str):
        at_bound = [name for name in at_bound.split("+") if name]
    return sorted(name for name, value in log_columns.items()
                  if name in at_bound and float(value) > 0.0)


def write_upper_bound_flags(handle, rows, species) -> int:
    """Per-row datasets marking a window whose correction must not be used.

    ``handle`` is an open h5py group; ``rows`` carry ``at_bound`` and a
    ``log_column_<species>`` field per species. Existing flags are replaced, so
    a file written before the flag existed can be annotated in place and ends up
    holding exactly what a fresh export would. Returns how many rows are flagged.
    """

    names = [species_at_upper_bound(row["at_bound"],
                                    {s: row[f"log_column_{s}"] for s in species})
             for row in rows]
    for key in ("column_at_upper_bound", "species_at_upper_bound"):
        if key in handle:
            del handle[key]
    handle.create_dataset("column_at_upper_bound", data=np.array([bool(n) for n in names]))
    handle["column_at_upper_bound"].attrs["description"] = (
        "True: do not use this row. " + UPPER_BOUND_NOTE + ". A species at the LOWER bound "
        "is harmless and not flagged (docs/solar_fit_plan.md §4m).")
    handle.create_dataset("species_at_upper_bound", data=["+".join(n).encode() for n in names])
    return int(sum(bool(n) for n in names))
