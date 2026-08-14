import subprocess

import pytest
from tools.archive_batch import transfer
from tools.archive_batch.transfer import (TransferError, TransferFailed,
                                          TransferOutage, publish_cmd, run,
                                          stage_cmd)


class _FakeClock:
    """Advances only when the code under test sleeps, so retries cost no time."""

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


def _proc(returncode, stderr=""):
    return subprocess.CompletedProcess(args=["rsync"], returncode=returncode,
                                       stdout="", stderr=stderr)


def test_stage_pulls_from_the_source_host():
    cmd = stage_cmd("gpu1", "SetA/2003/event-b/x.MOV", "/repo/Temp/_stage/igpu")
    assert cmd[0] == "rsync"
    assert cmd[-2] == "gpu1:/mnt/media/dance/SetA/2003/event-b/x.MOV"
    assert cmd[-1] == "/repo/Temp/_stage/igpu/"


def test_publish_creates_the_remote_directory_first():
    cmd = publish_cmd("gpu1", "/repo/Temp/out/x-av1.mkv", "SetA/2003/event-b")
    joined = " ".join(cmd)
    assert "mkdir -p" in joined
    assert "/mnt/media/dance/encoded/SetA/2003/event-b" in joined
    assert cmd[-1].startswith("gpu1:")


def test_publish_target_is_the_mirrored_path():
    cmd = publish_cmd("gpu1", "/repo/Temp/out/x-av1.mkv", "SetA/2003/event-b")
    assert cmd[-1] == "gpu1:/mnt/media/dance/encoded/SetA/2003/event-b/"


def test_publish_rejects_an_absolute_relative_dir():
    with pytest.raises(TransferError, match="relative"):
        publish_cmd("gpu1", "/repo/out/x.mkv", "/SetA/2003")


def test_stage_rejects_an_absolute_source():
    with pytest.raises(TransferError, match="relative"):
        stage_cmd("gpu1", "/SetA/2003/x.MOV", "/repo/Temp/_stage/igpu")


def test_publish_quotes_a_remote_directory_holding_an_apostrophe(monkeypatch):
    monkeypatch.setattr(transfer, "ARCHIVE_ROOT", "/mnt/media/dance")
    cmd = publish_cmd("gpu1", "/repo/out/x.mkv", "SetB/person-a's routine")
    rsync_path = cmd[cmd.index("--rsync-path") + 1]
    # A bare '...' would end the quote at the apostrophe and hand the rest to
    # the remote shell as commands.
    assert rsync_path == (
        """mkdir -p '/mnt/media/dance/encoded/SetB/person-a'"'"'s routine' && rsync""")


def test_run_retries_a_failing_transfer_and_succeeds(monkeypatch):
    clock = _FakeClock()
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _proc(0) if len(calls) == 3 else _proc(10, "connection refused")

    monkeypatch.setattr(subprocess, "run", fake_run)
    proc = run(["rsync", "a", "b"], sleep=clock.sleep, clock=clock)
    assert proc.returncode == 0
    assert len(calls) == 3
    assert clock.slept == [5.0, 10.0]     # backoff doubles


def test_run_gives_up_as_an_outage_once_the_window_expires(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: _proc(10, "connection refused"))
    with pytest.raises(TransferOutage, match="gave up"):
        run(["rsync", "a", "b"], retry_window=60.0, sleep=clock.sleep, clock=clock)
    assert clock.now >= 60.0


def test_run_caps_the_backoff_delay(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _proc(10, "down"))
    with pytest.raises(TransferOutage):
        run(["rsync", "a", "b"], retry_window=3600.0, sleep=clock.sleep, clock=clock)
    assert max(clock.slept) == transfer.MAX_DELAY_S


def test_a_missing_source_fails_the_clip_without_retrying(monkeypatch):
    clock = _FakeClock()
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _proc(23, "rsync: link_stat \"...\" failed: No such file or directory")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(TransferFailed, match="No such file"):
        run(["rsync", "a", "b"], sleep=clock.sleep, clock=clock)
    assert len(calls) == 1, "a permanent failure must not be retried"
    assert clock.slept == []


def test_a_permanent_failure_is_not_an_outage(monkeypatch):
    # The distinction is the whole point: an outage winds the run down, and one
    # deleted source file must not do that to the other lanes.
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _proc(24, "vanished"))
    with pytest.raises(TransferFailed):
        run(["rsync", "a", "b"], sleep=lambda s: None, clock=lambda: 0.0)
    assert not issubclass(TransferFailed, TransferOutage)
    assert issubclass(TransferFailed, TransferError)


def test_run_treats_a_timeout_as_a_retryable_failure(monkeypatch):
    clock = _FakeClock()
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd, 3600)
        return _proc(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run(["rsync", "a", "b"], sleep=clock.sleep, clock=clock).returncode == 0
    assert len(calls) == 2
