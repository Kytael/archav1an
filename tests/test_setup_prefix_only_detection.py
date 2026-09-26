"""FFVship and fssimu2 count as installed only when the prefix has them.

is_installed and both installers used to accept any copy on PATH. encoder-host
has old builds in /usr/local/bin, so `--install A` skipped both and left the
prefix without them, and `--update` reported the empty prefix "up to date"
while the pins had moved on (vship v5.0.1 -> v5.1.1, fssimu2 0.1.3 -> 0.2.0).

Each case runs the real `setup.sh --update` against a throwaway prefix, with
decoy binaries first on PATH, and answers "n" at the rebuild prompt so nothing
is built.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
COMPONENTS = ("ffvship", "fssimu2")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _update_verdicts(tmp_path, with_record):
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    manifest = prefix / "share/archav1an/manifest"
    manifest.mkdir(parents=True)
    if with_record:
        for c in COMPONENTS:
            (manifest / f"{c}.src").write_text("recorded\n")

    decoys = tmp_path / "decoys"
    decoys.mkdir()
    for name in ("FFVship", "fssimu2"):
        d = decoys / name
        d.write_text("#!/bin/sh\nexit 0\n")
        d.chmod(0o755)

    env = dict(os.environ, VS_PREFIX=str(prefix),
               PATH=f"{decoys}:{os.environ['PATH']}")
    out = subprocess.run(
        ["./setup.sh", "--update", *COMPONENTS], cwd=REPO, env=env,
        input="n", capture_output=True, text=True, timeout=120).stdout
    verdicts = {}
    for line in ANSI.sub("", out).splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in COMPONENTS:
            verdicts[parts[0]] = parts[1]
    return verdicts


@pytest.mark.parametrize("with_record,expected", [
    # Never installed here: a copy on PATH must not make it look installed.
    (False, "MISSING"),
    # Installed here once, files gone since: --update has to rebuild it.
    (True, "BROKEN"),
])
def test_copy_on_path_does_not_count_as_installed(tmp_path, with_record, expected):
    verdicts = _update_verdicts(tmp_path, with_record)
    assert verdicts == {c: expected for c in COMPONENTS}
