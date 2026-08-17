import importlib.util
from pathlib import Path

BATCH_PY = Path(__file__).resolve().parent.parent / "tools" / "archive-batch.py"
_spec = importlib.util.spec_from_file_location("archive_batch_main", BATCH_PY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
log_tail = _mod.log_tail


def test_progress_lines_do_not_bury_the_failure(tmp_path):
    """vspipe -p writes a Frame: line a second, so an unfiltered tail is
    guaranteed to be four progress lines and never the error."""
    body = ["Script evaluation done in 0.07 seconds"]
    body += [f"Frame: {i}/6726" for i in range(1, 400)]
    body += ["Error: operator std.LoadPlugin failed", "at frame 401"]
    body += [f"Frame: {i}/6726" for i in range(401, 405)]
    (tmp_path / "MVI_1_vspipe.log").write_text("\n".join(body))
    out = log_tail(str(tmp_path), "MVI_1")
    assert "Frame:" not in out, out
    assert "at frame 401" in out


def test_a_log_that_is_only_progress_still_returns_something_harmless(tmp_path):
    (tmp_path / "MVI_2_vspipe.log").write_text(
        "\n".join(f"Frame: {i}/10" for i in range(1, 11)))
    out = log_tail(str(tmp_path), "MVI_2")
    assert "Frame:" not in out
