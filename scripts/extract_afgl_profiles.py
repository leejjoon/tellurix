#!/usr/bin/env python
"""Extract the AFGL standard atmospheres from LBLRTM's own source.

`make_site_profile.py` can only offer the gases it has abundances for, and a
species-scan can only rank what the profile carries -- so a hand-written list
of five trace gases silently ranks everything else at zero. That is how OCS and
O3 stayed out of the solar fits until the residual was traced to them by hand
(docs/solar_fts_residual.md).

LBLRTM carries the AFGL profiles as Fortran DATA statements in
`src/lblatm.f90`: a 50-level altitude grid, molecules 1-8 for each of six model
atmospheres, and a single trace-gas set for molecules 8-47 in HITRAN order.
This reads them out and writes them as CSV, so the abundances in this package
come from the same source LBLRTM uses rather than from memory.

    UV_CACHE_DIR=.uv-cache uv run python scripts/extract_afgl_profiles.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

LEVELS = 50
MODELS = {1: "tropical", 2: "midlatitude_summer", 3: "midlatitude_winter",
          4: "subarctic_summer", 5: "subarctic_winter", 6: "us_standard_1976"}
# The COMMON /TRAC/ declaration order is HITRAN 8 through 47, and the Fortran
# names differ from the HITRAN ones where the first character would be numeric
# or clash with an intrinsic.
TRACE = [
    (8, "ANO"), (9, "SO2"), (10, "ANO2"), (11, "ANH3"), (12, "HNO3"), (13, "OH"),
    (14, "HF"), (15, "HCL"), (16, "HBR"), (17, "HI"), (18, "CLO"), (19, "OCS"),
    (20, "H2CO"), (21, "HOCL"), (22, "AN2"), (23, "HCN"), (24, "CH3CL"), (25, "H2O2"),
    (26, "C2H2"), (27, "C2H6"), (28, "PH3"), (29, "COF2"), (30, "SF6"), (31, "H2S"),
    (32, "HCOOH"), (33, "HO2"), (34, "O"), (35, "CLONO2"), (36, "NOPLUS"), (37, "HOBr"),
    (38, "C2H4"), (39, "CH3OH"), (40, "CH3BR"), (41, "CH3CN"), (42, "CF4"),
    (43, "C4H2"), (44, "HC3N"), (45, "H2"), (46, "CS"), (47, "SO3"),
]
HITRAN_NAME = {
    1: "H2O", 2: "CO2", 3: "O3", 4: "N2O", 5: "CO", 6: "CH4", 7: "O2", 8: "NO",
    9: "SO2", 10: "NO2", 11: "NH3", 12: "HNO3", 13: "OH", 14: "HF", 15: "HCL",
    16: "HBR", 17: "HI", 18: "CLO", 19: "OCS", 20: "H2CO", 21: "HOCL", 22: "N2",
    23: "HCN", 24: "CH3CL", 25: "H2O2", 26: "C2H2", 27: "C2H6", 28: "PH3",
    29: "COF2", 30: "SF6", 31: "H2S", 32: "HCOOH", 33: "HO2", 34: "O",
    35: "CLONO2", 36: "NO+", 37: "HOBR", 38: "C2H4", 39: "CH3OH", 40: "CH3BR",
    41: "CH3CN", 42: "CF4", 43: "C4H2", 44: "HC3N", 45: "H2", 46: "CS", 47: "SO3",
}


def read_data_array(source: str, name: str) -> np.ndarray:
    """Values of one Fortran ``DATA name / ... /`` block, to the repeat count.

    The arrays are declared at the maximum level count and padded with a
    ``MXZ50*0.0`` repeat, so parsing stops at the first token carrying a ``*``
    rather than trying to expand it. The match is case-insensitive because
    Fortran is: the COMMON block declares ``HOBr`` and the DATA statement
    spells it ``HOBR``.
    """

    match = re.search(rf"^\s*DATA\s+{re.escape(name)}\s*/(.*?)/\s*$",
                      source, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if match is None:
        raise ValueError(f"no DATA block for {name}")
    values = []
    for token in match.group(1).replace("&", " ").replace("\n", " ").split(","):
        token = token.strip()
        if not token or "*" in token:
            break
        values.append(float(token))
    if len(values) < LEVELS:
        raise ValueError(f"{name}: {len(values)} values, expected at least {LEVELS}")
    return np.asarray(values[:LEVELS], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path,
                        default=Path("data/lblrtm/LBLRTM/src/lblatm.f90"))
    parser.add_argument("--output", type=Path, default=Path("data/profiles/afgl"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = (root / args.source).read_text(encoding="latin-1")
    altitude = read_data_array(source, "ALT")

    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    trace = {number: read_data_array(source, fortran) for number, fortran in TRACE}

    for model, label in MODELS.items():
        columns = {"altitude_km": altitude}
        for molecule in range(1, 9):
            columns[HITRAN_NAME[molecule]] = read_data_array(source, f"AMOL{model}{molecule}")
        # Molecules 8-47 are one AFGL trace set shared by every model, so the
        # model-specific AMOL{m}8 (NO) is overwritten here by the trace value
        # for consistency with what LBLRTM itself uses above molecule 7.
        for number, values in trace.items():
            columns[HITRAN_NAME[number]] = values
        names = list(columns)
        lines = [
            f"# AFGL {label.replace('_', ' ')}, extracted from LBLRTM 12.17 src/lblatm.f90",
            "# by scripts/extract_afgl_profiles.py. Abundances are ppmv of dry air;",
            "# altitude is km above sea level. Molecules 8-47 are the single AFGL trace",
            "# set, which LBLRTM shares across all six model atmospheres.",
            ",".join(names),
        ]
        for index in range(LEVELS):
            lines.append(",".join(f"{columns[n][index]:.6E}" for n in names))
        (out / f"{label}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  {label}: {len(names) - 1} molecules, {LEVELS} levels")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
