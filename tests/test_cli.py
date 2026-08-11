import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "archive_batch_cli", REPO / "tools" / "archive-batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_callback_ip_is_a_lan_address():
    cli = _load_cli()
    assert cli.CALLBACK_IP == "10.0.0.10"


def test_summary_reports_counts():
    cli = _load_cli()
    text = cli.format_summary(done=10, failed=2,
                              failures=[("a.MOV", "igpu", "rc=1")], elapsed_s=3600.0)
    assert "10 done" in text and "2 failed" in text and "a.MOV" in text


def test_summary_with_no_failures_omits_the_failure_block():
    cli = _load_cli()
    text = cli.format_summary(done=5, failed=0, failures=[], elapsed_s=60.0)
    assert "failed" in text and "FAILED CLIPS" not in text
