"""Every scipy name the scene detector calls must be imported.

tools/Progressive-Scene-Detection.py is 4,400 lines and takes a full probe
pass to reach its frame-selection block, so a missing import there does not
show up until a real run is minutes in -- and then it is a NameError, not a
message. The import line was commented out while three live calls kept using
it, which broke every scene longer than one frame.

The file cannot be imported in a test: it parses its own required arguments
at module scope. So this reads the source instead.
"""
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The three tools of the Progression-Boost family share this frame-selection
# code, so they share the rule.
SOURCES = [
    REPO / "tools" / "Progressive-Scene-Detection.py",
    REPO / "tools" / "Progression-Boost-Basic-SSIMU2-anime.py",
    REPO / "tools" / "Progression-Boost-Basic-SSIMU2-liveaction.py",
]

SCIPY_NAMES = {"fftpack", "interpolate", "signal", "stats"}


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8", errors="replace"), str(path))


def _bound_names(tree):
    """Every name the module binds at any scope: imports, defs, assignments."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return bound


def _used_scipy_names(tree):
    """{name: first line} for every scipy module the source actually reads."""
    used = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in SCIPY_NAMES:
            used.setdefault(node.value.id, node.value.lineno)
    return used


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_scipy_module_it_calls_is_imported(path):
    tree = _tree(path)
    bound = _bound_names(tree)
    used = _used_scipy_names(tree)
    assert used, f"{path.name} calls no scipy module; drop it from SOURCES"
    unbound = {n: line for n, line in used.items() if n not in bound}
    assert not unbound, (
        f"{path.name} calls scipy modules it never imports: "
        + ", ".join(f"{n} at line {line}" for n, line in sorted(unbound.items()))
    )
