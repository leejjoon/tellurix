"""The core must not depend on an instrument, nor one instrument on another.

``tellurix`` is the forward model and the fitter; ``tellurix_fts`` and
``tellurix_igrins`` read particular data into it. The split only stays a split
if nothing imports back across it, and an import buried in a function body is
as much a dependency as one at the top of a module.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN = {
    "tellurix": {"tellurix_fts", "tellurix_igrins"},
    "tellurix_fts": {"tellurix_igrins"},
    "tellurix_igrins": {"tellurix_fts"},
}


def imported_packages(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_imports_only_what_it_may(package):
    offending = sorted(
        f"{path.name} imports {name}"
        for path in (SRC / package).glob("*.py")
        for name in imported_packages(path)
        if name in FORBIDDEN[package]
    )
    assert not offending
