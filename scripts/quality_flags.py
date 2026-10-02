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
