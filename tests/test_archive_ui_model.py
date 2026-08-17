import os

from tools import archive_ui


def test_run_dir_follows_the_env_var(monkeypatch):
    """The batch already reads ARCHIVE_RUN_DIR; the daemon must agree with it
    or the two look at different runs."""
    monkeypatch.setenv("ARCHIVE_RUN_DIR", "/tmp/somewhere")
    paths = archive_ui.Paths.from_env()
    assert paths.run_dir == "/tmp/somewhere"
    assert paths.state == "/tmp/somewhere/state.jsonl"
    assert paths.roster == "/tmp/somewhere/denoisers.toml"
    assert paths.manifest == "/tmp/somewhere/manifest-raw.tsv"
    assert paths.lanes == "/tmp/somewhere/lanes"
    # control is unused until part 2, which is exactly why it needs asserting:
    # a field nothing reads yet is the one a refactor drops silently.
    assert paths.control == "/tmp/somewhere/control"


def test_run_dir_defaults_into_the_repo(monkeypatch):
    monkeypatch.delenv("ARCHIVE_RUN_DIR", raising=False)
    paths = archive_ui.Paths.from_env()
    assert paths.run_dir.endswith(os.path.join("archav1an", ".archive-run"))
