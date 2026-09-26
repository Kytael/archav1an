"""Tests for tools.archive_batch.pidfile -- is this pid still the batch?"""
import os
import subprocess
import sys
import time

from tools.archive_batch import pidfile


def _spawn(tmp_path, name):
    """A live child that sleeps, named so its command line is the fixture.

    Waits for the command line to appear: Popen returns before the child has
    finished exec, and /proc/<pid>/cmdline is empty for that instant.
    """
    script = tmp_path / name
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if pidfile.cmdline(proc.pid):
            return proc
        time.sleep(0.01)
    proc.kill()
    proc.wait()
    raise AssertionError(f"{name} never reached exec")


def _reaped(tmp_path):
    """A pid that is certainly dead: run something trivial and reap it."""
    script = tmp_path / "die.py"
    script.write_text("pass\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)])
    proc.wait()
    return proc.pid


def test_a_matching_start_time_is_alive(tmp_path):
    child = _spawn(tmp_path, "sleep.py")
    try:
        assert pidfile.alive(child.pid, pidfile.start_time(child.pid)) is True
    finally:
        child.kill()
        child.wait()


def test_a_mismatched_start_time_is_a_recycled_pid(tmp_path):
    child = _spawn(tmp_path, "sleep.py")
    try:
        start = pidfile.start_time(child.pid)
        assert pidfile.alive(child.pid, start + 1) is False
    finally:
        child.kill()
        child.wait()


def test_a_dead_pid_is_dead(tmp_path):
    assert pidfile.alive(_reaped(tmp_path), 1) is False


def test_a_pre_upgrade_record_is_checked_by_command_line(tmp_path):
    """No pid_start means a record written before that field existed. The
    process it names is still the batch if its command line says so.

    Reading it as dead instead would call a live batch dead for the rest of a
    fifteen-day run: the dashboard greys out Stop and re-enables Start, and
    one click puts a second batch over the first.
    """
    child = _spawn(tmp_path, "archive-batch.py")
    try:
        assert pidfile.alive(child.pid, None) is True
    finally:
        child.kill()
        child.wait()


def test_a_pre_upgrade_record_rejects_a_pid_running_something_else(tmp_path):
    """The other half: a pid recycled to an unrelated process is not the
    batch. Existence alone would hold encode_batch_up at 1 for ever."""
    child = _spawn(tmp_path, "ffmpeg-ish.py")
    try:
        assert pidfile.alive(child.pid, None) is False
    finally:
        child.kill()
        child.wait()


def test_a_corrupt_pid_is_dead_without_probing_anything():
    # os.kill(-1, 0) probes every process this user owns and succeeds.
    assert pidfile.alive(-1) is False
    assert pidfile.alive(0) is False
    assert pidfile.alive("1234") is False
    assert pidfile.alive(None) is False


def test_cmdline_of_a_dead_pid_is_none(tmp_path):
    assert pidfile.cmdline(_reaped(tmp_path)) is None


def test_cmdline_holds_the_arguments(tmp_path):
    child = _spawn(tmp_path, "archive-batch.py")
    try:
        line = pidfile.cmdline(child.pid)
        assert "archive-batch.py" in line
        assert os.path.basename(sys.executable) in line
    finally:
        child.kill()
        child.wait()


def test_a_zombie_is_dead(tmp_path):
    """A child nobody has waited on stays in the process table, and
    os.kill(pid, 0) succeeds on it for as long as it does.

    The daemon is the parent of the batch it spawned, so a batch it has not
    reaped yet would read as a live run: the stale batch.json a SIGKILLed
    batch left behind would hold encode_batch_up at 1, EncodeBatchDown would
    never fire, and the page would offer Stop for a run that is over.
    """
    script = tmp_path / "archive-batch.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    child = subprocess.Popen([sys.executable, str(script)])
    start = pidfile.start_time(child.pid)
    child.kill()
    # Deliberately no wait(): that is what makes it a zombie.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        fields = pidfile._stat_after_comm(child.pid)
        if fields and fields[0] == b"Z":
            break
        time.sleep(0.01)
    assert fields and fields[0] == b"Z", "the fixture never became a zombie"
    os.kill(child.pid, 0)       # the probe the old check trusted still passes
    assert pidfile.alive(child.pid, start) is False
    assert pidfile.alive(child.pid, None) is False
    child.wait()
