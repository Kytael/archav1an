"""uv must be bootstrapped by every setup target that uses it.

ensure_uv lived in common.sh but was called from exactly one place,
install_python_libs. Every other uv call site assumed uv was already there.
That assumption holds on a host being installed from scratch, because
python_libs runs first -- and fails on a host that already has a venv, because
then python_libs is skipped and nothing bootstraps uv. `./setup.sh --update`
takes exactly that path: it rebuilds vapoursynth and runs update_python_libs
without reinstalling python_libs.

Proven on gpu3 2026-08-14: --update died in the rebuild, and uv had to be
installed by hand before it would run.
"""
import re
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "setup"

# common.sh defines ensure_uv and guards its own single call with
# `command -v uv`, so it is not a caller that needs the bootstrap.
EXEMPT = {"common.sh"}

# A function whose uv call is already guarded by an explicit availability test
# on the same line does not need the bootstrap.
GUARDED = re.compile(r"command -v uv")


def _shell_functions(text):
    """(name, body) for each top-level `name() {` in a setup script."""
    starts = [(m.group(1), m.start())
              for m in re.finditer(r"^([a-z_][a-z0-9_]*)\(\) \{", text, re.M)]
    out = []
    for i, (name, pos) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        out.append((name, text[pos:end]))
    return out


def _first_uv_line(body):
    """Index of the first unguarded uv invocation, or None."""
    for i, line in enumerate(body.splitlines()):
        if GUARDED.search(line):
            continue
        if re.search(r"(^|[;&|(\s])uv\s+(pip|venv|python|tool)\b", line):
            return i
    return None


def _ensure_uv_line(body):
    for i, line in enumerate(body.splitlines()):
        if "ensure_uv" in line:
            return i
    return None


def _uses_uv(body):
    return _first_uv_line(body) is not None


def test_every_function_that_runs_uv_also_bootstraps_it():
    offenders = []
    for path in sorted(SETUP.glob("*.sh")):
        if path.name in EXEMPT:
            continue
        text = path.read_text()
        for name, body in _shell_functions(text):
            uv_at = _first_uv_line(body)
            if uv_at is None:
                continue
            bootstrap_at = _ensure_uv_line(body)
            # Position matters, not just presence: a bootstrap after the call
            # it is meant to protect protects nothing.
            if bootstrap_at is None or bootstrap_at > uv_at:
                offenders.append(f"{path.name}:{name}")
    assert not offenders, (
        "these setup functions run uv without calling ensure_uv first, so they "
        f"fail on a host that has a venv but no uv: {offenders}")


def test_ensure_uv_is_still_defined_in_common():
    assert "ensure_uv()" in (SETUP / "common.sh").read_text()


def test_the_detector_would_have_caught_the_original_bug():
    """A test that cannot fail proves nothing -- check it flags the old code."""
    old = (
        "update_python_libs() {\n"
        '    VIRTUAL_ENV="$VENV_DIR" uv pip install -U numpy\n'
        "}\n"
    )
    name, body = _shell_functions(old)[0]
    assert name == "update_python_libs"
    assert _uses_uv(body) and "ensure_uv" not in body

    fixed = (
        "update_python_libs() {\n"
        "    ensure_uv || return 1\n"
        '    VIRTUAL_ENV="$VENV_DIR" uv pip install -U numpy\n'
        "}\n"
    )
    _n, fixed_body = _shell_functions(fixed)[0]
    assert _ensure_uv_line(fixed_body) < _first_uv_line(fixed_body)

    # And that a bootstrap placed after the call it guards is still caught.
    too_late = (
        "update_python_libs() {\n"
        '    VIRTUAL_ENV="$VENV_DIR" uv pip install -U numpy\n'
        "    ensure_uv || return 1\n"
        "}\n"
    )
    _n, late_body = _shell_functions(too_late)[0]
    assert _ensure_uv_line(late_body) > _first_uv_line(late_body)
