"""Tests for tools.encode_dash.supervisor — own or adopt the batch process."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from tools.archive_batch import pidfile
from tools.encode_dash.supervisor import (AlreadyRunning, Supervisor,
                                          SupervisorError)


def _batch_script(tmp_path, marker, spawns=None, facts=None):
    """A batch-shaped child: record a spawn (optionally facts), then sleep."""
    body = "import os, time\n"
    if spawns is not None:
        body += f"with open({str(spawns)!r}, 'a') as fh:\n    fh.write('x\\n')\n"
    body += f"with open({str(marker)!r}, 'w') as fh:\n    fh.write('hi')\n"
    if facts:
        body += "    fh.write('\\n' + os.getcwd())\n"
        for key, value in facts.items():
            body += (f"    fh.write('\\n' + os.environ.get("
                     f"{key!r}, {value!r}))\n")
    body += "time.sleep(60)\n"
    path = tmp_path / "batch.py"
    path.write_text(body, encoding="utf-8")
    return path


def _dead_pid(tmp_path):
    """A pid that is certainly dead: reap a child that exits immediately."""
    script = tmp_path / "die.py"
    script.write_text("pass\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)])
    proc.wait()
    return proc.pid


def _await(path, timeout=10):
    """Block until a child-written file exists (the spawn is async)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return path
        time.sleep(0.01)
    raise AssertionError(f"{path} was not written by the child")


def _kill(pid):
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)


def _write_batch_json(path, pid):
    """The record shape every writer produces today: pid plus its identity.

    pid_start is what separates the named process from a later one that
    inherited its pid, so a fixture that leaves it out is not a record this
    version writes -- see the pre-upgrade tests at the end of this file for
    that case.
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"batch_pid": pid, "pid_start": pidfile.start_time(pid)}, fh)


def test_start_spawns_the_batch_and_records_its_pid(tmp_path):
    marker = tmp_path / "ran"
    script = _batch_script(tmp_path, marker)
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(tmp_path / "batch.json"))
    pid = sup.start()
    try:
        assert pid > 0
        assert _await(marker).read_text(encoding="utf-8") == "hi"
        # The child is alive until we kill it.
        os.kill(pid, 0)
        assert sup.status()[1] is True
    finally:
        _kill(pid)


def test_a_second_start_while_one_is_running_is_refused(tmp_path):
    spawns = tmp_path / "spawns"
    marker = tmp_path / "ran"
    script = _batch_script(tmp_path, marker, spawns=spawns)
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(tmp_path / "batch.json"))
    pid = sup.start()
    try:
        assert _await(spawns).read_text(encoding="utf-8") == "x\n"
        with pytest.raises(AlreadyRunning):
            sup.start()
        # One spawn only: the refusal must not have launched a second child.
        assert spawns.read_text(encoding="utf-8") == "x\n"
        os.kill(pid, 0)
    finally:
        _kill(pid)


def test_start_adopts_a_live_pid_on_file_and_refuses(tmp_path):
    # Our own pid is alive; the supervisor must adopt it and refuse to spawn.
    _write_batch_json(tmp_path / "batch.json", os.getpid())
    sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                     cwd=str(tmp_path),
                     batch_file=str(tmp_path / "batch.json"))
    with pytest.raises(AlreadyRunning):
        sup.start()


def test_adopt_claims_a_live_run_and_clears_a_stale_file(tmp_path):
    batch_json = tmp_path / "batch.json"
    _write_batch_json(batch_json, os.getpid())
    sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                     cwd=str(tmp_path), batch_file=str(batch_json))
    assert sup.adopt() == os.getpid()
    assert batch_json.exists(), "a live run must stay on file"

    dead = _dead_pid(tmp_path)
    _write_batch_json(batch_json, dead)
    assert sup.adopt() is None
    assert not batch_json.exists(), "a stale file must be cleared"


def test_start_clears_a_stale_batch_json_before_spawning(tmp_path):
    batch_json = tmp_path / "batch.json"
    _write_batch_json(batch_json, _dead_pid(tmp_path))
    marker = tmp_path / "ran"
    script = _batch_script(tmp_path, marker)
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(batch_json))
    pid = sup.start()
    try:
        # The stale dead pid must be replaced by the new spawn's record: the
        # file exists again, and it names the spawn, not the stale one.
        assert batch_json.exists(), "the spawn must record its pid"
        recorded = json.loads(batch_json.read_text(encoding="utf-8"))
        assert recorded["batch_pid"] == pid, \
            "the recorded pid must be the new spawn, not the stale one"
        assert _await(marker).read_text(encoding="utf-8") == "hi"
    finally:
        _kill(pid)


def test_cwd_and_environment_reach_the_child(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHIVE_TRACE", "1")
    facts = {"ARCHIVE_TRACE": ""}
    marker = tmp_path / "ran"
    script = _batch_script(tmp_path, marker, facts=facts)
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(tmp_path / "batch.json"))
    pid = sup.start()
    try:
        lines = _await(marker).read_text(encoding="utf-8").splitlines()
        assert lines[0] == "hi"
        assert lines[1] == str(tmp_path), "cwd must reach the child"
        assert lines[2] == "1", "the environment must reach the child"
    finally:
        _kill(pid)


def test_status_reports_a_three_tuple_with_spawned_flag(tmp_path):
    # The daemon unpacks (pid, alive, spawned); the spawned flag tells the
    # snapshot merge whether the pid is this daemon's own child.
    marker = tmp_path / "ran"
    script = _batch_script(tmp_path, marker)
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(tmp_path / "batch.json"))
    pid = sup.start()
    _await(marker)
    assert sup.status() == (pid, True, True)

    _kill(pid)
    # The very first status() call after death must be (None, False, False):
    # the pid element has to be read after _own_alive() has reaped and reset,
    # because tuple elements evaluate left-to-right.
    assert sup.status() == (None, False, False)


def test_start_after_the_child_exits_spawns_again(tmp_path):
    # poll() reaps our dead child, so a second start must not be blocked by
    # the zombie that an os.kill probe would keep reporting as alive.
    marker = tmp_path / "ran"
    script = _batch_script(tmp_path, marker)
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(tmp_path / "batch.json"))
    first = sup.start()
    _await(marker)

    _kill(first)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and sup.status()[1]:
        time.sleep(0.01)
    assert sup.status()[1] is False

    second = sup.start()
    assert second != first
    assert sup.status() == (second, True, True)
    _kill(second)


def test_start_creates_the_batch_dir(tmp_path):
    # A fresh checkout has no run dir; start must create it before writing
    # the pid record, otherwise the atomic replace fails on a missing parent.
    marker = tmp_path / "ran"
    script = _batch_script(tmp_path, marker)
    batch_file = tmp_path / "run" / "batch.json"
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(batch_file))
    pid = sup.start()
    _await(marker)
    assert batch_file.parent.is_dir()
    assert batch_file.is_file()
    assert sup.status()[1] is True
    _kill(pid)


def test_pid_record_writes_started_at_atomically(tmp_path):
    # The supervisor's pid record matches the batch's own shape (batch_pid
    # plus started_at) and is written via tmp + replace, so a status poll
    # never reads a half-written file.
    marker = tmp_path / "ran"
    script = _batch_script(tmp_path, marker)
    batch_file = tmp_path / "batch.json"
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(batch_file))
    pid = sup.start()
    _await(marker)
    record = json.loads(batch_file.read_text(encoding="utf-8"))
    assert record["batch_pid"] == pid
    assert isinstance(record["started_at"], float)
    assert list(tmp_path.glob("batch.json.tmp")) == []
    _kill(pid)


def test_adopt_ignores_a_non_dict_batch_json(tmp_path):
    # A file holding "[]" has no .get; adopt must treat it as stale rather
    # than raising AttributeError at daemon boot (review finding A).
    batch_json = tmp_path / "batch.json"
    batch_json.write_text("[]", encoding="utf-8")
    sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                     cwd=str(tmp_path), batch_file=str(batch_json))
    assert sup.adopt() is None
    assert not batch_json.exists(), "a corrupt file must be cleared"


def test_start_ignores_a_non_dict_batch_json(tmp_path):
    # Same corrupt file through start: it must clear and spawn, not crash.
    batch_json = tmp_path / "batch.json"
    batch_json.write_text("[]", encoding="utf-8")
    marker = tmp_path / "ran"
    script = _batch_script(tmp_path, marker)
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(batch_json))
    pid = sup.start()
    try:
        assert pid > 0
        assert _await(marker).read_text(encoding="utf-8") == "hi"
    finally:
        _kill(pid)


def test_adopt_rejects_a_negative_pid_as_stale(tmp_path):
    # os.kill(-1, 0) probes every process this user owns and succeeds, so a
    # corrupt "-1" pid must read as no run rather than being adopted forever
    # (review finding B).
    batch_json = tmp_path / "batch.json"
    _write_batch_json(batch_json, -1)
    sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                     cwd=str(tmp_path), batch_file=str(batch_json))
    assert sup.adopt() is None
    assert not batch_json.exists(), "a corrupt pid must be cleared"


def test_start_ignores_a_negative_pid_and_spawns(tmp_path):
    # The same "-1" pid through start: it must clear and spawn, never raise
    # AlreadyRunning against a pid that is always "alive" to an os.kill probe.
    batch_json = tmp_path / "batch.json"
    _write_batch_json(batch_json, -1)
    marker = tmp_path / "ran"
    script = _batch_script(tmp_path, marker)
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(batch_json))
    pid = sup.start()
    try:
        assert pid > 0
        assert _await(marker).read_text(encoding="utf-8") == "hi"
    finally:
        _kill(pid)


def test_record_pid_does_not_drop_a_claimed_marker(tmp_path):
    # The batch's own _mark_claimed may land before the supervisor's
    # _record_pid runs. When the file already names this spawn, _record_pid
    # must leave that claimed record alone: rewriting would drop the marker,
    # and the dashboard would then report a live run as not running.
    batch_json = tmp_path / "batch.json"
    with open(batch_json, "w", encoding="utf-8") as fh:
        json.dump({"batch_pid": os.getpid(), "claimed": True}, fh)
    sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                     cwd=str(tmp_path), batch_file=str(batch_json))
    sup._record_pid(os.getpid())
    record = json.loads(batch_json.read_text(encoding="utf-8"))
    assert record["claimed"] is True, "a claimed marker must not be dropped"
    assert record["batch_pid"] == os.getpid()


def _live_child_starttime(tmp_path, name="sleep.py"):
    """A real live child and its /proc starttime, for identity fixtures."""
    script = tmp_path / name
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)])
    return proc, pidfile.start_time(proc.pid)


def test_adopt_rejects_a_recycled_pid_as_stale(tmp_path):
    # A live pid whose recorded starttime does not match is a reused pid, not
    # our batch: adopt must clear the file instead of claiming an unrelated
    # process (review finding C1).
    batch_json = tmp_path / "batch.json"
    child, start = _live_child_starttime(tmp_path)
    try:
        with open(batch_json, "w", encoding="utf-8") as fh:
            json.dump({"batch_pid": child.pid, "pid_start": start + 1}, fh)
        sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                         cwd=str(tmp_path), batch_file=str(batch_json))
        assert sup.adopt() is None
        assert not batch_json.exists(), "a recycled pid must be cleared"
        os.kill(child.pid, 0)   # the unrelated process was never touched
    finally:
        _kill(child.pid)


def test_start_ignores_a_recycled_pid_and_spawns(tmp_path):
    # The same mismatch through start: it must spawn instead of raising
    # AlreadyRunning forever against a pid that belongs to someone else.
    batch_json = tmp_path / "batch.json"
    child, start = _live_child_starttime(tmp_path)
    try:
        with open(batch_json, "w", encoding="utf-8") as fh:
            json.dump({"batch_pid": child.pid, "pid_start": start + 1}, fh)
        marker = tmp_path / "ran"
        script = _batch_script(tmp_path, marker)
        sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                         batch_file=str(batch_json))
        pid = sup.start()
        try:
            assert pid > 0
            assert pid != child.pid
            assert _await(marker).read_text(encoding="utf-8") == "hi"
        finally:
            _kill(pid)
    finally:
        _kill(child.pid)


def test_adopt_claims_a_live_run_with_matching_identity(tmp_path):
    # The happy path of the identity check: a record naming a live pid with
    # its true starttime is adopted.
    batch_json = tmp_path / "batch.json"
    child, start = _live_child_starttime(tmp_path)
    try:
        with open(batch_json, "w", encoding="utf-8") as fh:
            json.dump({"batch_pid": child.pid, "pid_start": start}, fh)
        sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                         cwd=str(tmp_path), batch_file=str(batch_json))
        assert sup.adopt() == child.pid
        assert batch_json.exists(), "a live run must stay on file"
    finally:
        _kill(child.pid)


def test_adopt_creates_a_missing_run_dir(tmp_path):
    # adopt takes batch.json.lock now, so it must be able to create the run
    # directory itself -- a fresh checkout has none, and the daemon calls
    # adopt() unguarded at boot.
    batch_json = tmp_path / "fresh-run" / "batch.json"
    sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                     cwd=str(tmp_path), batch_file=str(batch_json))
    assert sup.adopt() is None


def _pre_upgrade_child(tmp_path, name="archive-batch.py"):
    """A live stand-in for a batch started before pid_start existed.

    Its command line is its whole identity, which is what pidfile.alive falls
    back to for a record with no pid_start. Waits for exec, because
    /proc/<pid>/cmdline is empty between fork and exec.
    """
    script = tmp_path / name
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not pidfile.cmdline(proc.pid):
        time.sleep(0.01)
    assert pidfile.cmdline(proc.pid), "the stand-in never reached exec"
    return proc


def test_adopt_claims_a_live_pre_upgrade_record(tmp_path):
    # A batch already running when the checkout was upgraded wrote batch.json
    # with no pid_start. Clearing that record and spawning over it would put
    # two batches on one manifest, which is the failure this file exists to
    # prevent -- so the identity check falls back to the command line.
    batch_json = tmp_path / "batch.json"
    child = _pre_upgrade_child(tmp_path)
    try:
        with open(batch_json, "w", encoding="utf-8") as fh:
            json.dump({"batch_pid": child.pid, "started_at": 1.0}, fh)
        sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                         cwd=str(tmp_path), batch_file=str(batch_json))
        assert sup.adopt() == child.pid
        assert batch_json.exists(), "a live run must stay on file"
        with pytest.raises(AlreadyRunning):
            sup.start()
    finally:
        _kill(child.pid)


def test_adopt_clears_a_pre_upgrade_record_whose_pid_was_recycled(tmp_path):
    # The other half: the old batch died and the kernel handed its pid to an
    # ffmpeg. Existence alone would adopt that process for ever.
    batch_json = tmp_path / "batch.json"
    child = _pre_upgrade_child(tmp_path, name="ffmpeg-ish.py")
    try:
        with open(batch_json, "w", encoding="utf-8") as fh:
            json.dump({"batch_pid": child.pid, "started_at": 1.0}, fh)
        sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                         cwd=str(tmp_path), batch_file=str(batch_json))
        assert sup.adopt() is None
        assert not batch_json.exists(), "a recycled pid must be cleared"
        os.kill(child.pid, 0)   # the unrelated process was never touched
    finally:
        _kill(child.pid)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes read-only dirs")
def test_adopt_degrades_on_a_read_only_run_dir(tmp_path):
    # os.makedirs(exist_ok=True) SUCCEEDS on an existing read-only directory;
    # the call that raises is the lock's os.open. adopt() runs unguarded at
    # daemon boot, so that failure must not take the dashboard down.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_dir.chmod(0o555)
    lines = []
    try:
        sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                         cwd=str(tmp_path),
                         batch_file=str(run_dir / "batch.json"),
                         log=lambda *a: lines.append(" ".join(str(x) for x in a)))
        assert sup.adopt() is None
        assert sup.status() == (None, False, False)
        assert any("continuing without batch supervision" in l for l in lines), \
            f"the operator gets no other notice: {lines}"
    finally:
        run_dir.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes read-only dirs")
def test_start_reports_an_unwritable_run_dir_in_words(tmp_path):
    # adopt() has already degraded quietly by the time the operator clicks
    # Start, so this is the first they hear of it. A bare OSError would reach
    # the POST handler as a 500 holding a repr.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_dir.chmod(0o555)
    try:
        sup = Supervisor([sys.executable, str(tmp_path / "unused.py")],
                         cwd=str(tmp_path),
                         batch_file=str(run_dir / "batch.json"))
        with pytest.raises(SupervisorError) as caught:
            sup.start()
        assert "not writable" in str(caught.value)
        assert str(run_dir / "batch.json") in str(caught.value)
    finally:
        run_dir.chmod(0o755)


def test_start_reports_a_child_that_dies_at_once(tmp_path):
    # The reason a batch refuses to run is printed by the batch, on its own
    # stderr, milliseconds after the spawn. Without this the POST answers 200
    # with a pid, the page says "started", and the operator sees a run that
    # never appears -- the failure this test exists to make audible.
    monkey = tmp_path / "die.py"
    monkey.write_text(
        "import sys\n"
        "sys.stderr.write('[archive-batch] Error: cannot read the "
        "manifest\\nBuild it by probing the source host.\\n')\n"
        "sys.exit(2)\n", encoding="utf-8")
    batch_file = tmp_path / "batch.json"
    sup = Supervisor([sys.executable, str(monkey)], cwd=str(tmp_path),
                     batch_file=str(batch_file))
    with pytest.raises(SupervisorError) as caught:
        sup.start()
    said = str(caught.value)
    assert "cannot read the manifest" in said, said
    assert "2" in said, said
    # Nothing claimed: a dead spawn must not leave a record for the next
    # daemon to adopt, and a second Start must not answer AlreadyRunning.
    assert not batch_file.exists()
    assert sup.status() == (None, False, False)


def test_a_surviving_childs_stderr_goes_to_a_file_not_to_this_daemon(tmp_path):
    # KillMode=process and adopt() exist so a batch outlives a daemon restart.
    # A pipe to this daemon would quietly revoke that: the read end dies with
    # the daemon and the batch, and every encoder under it, takes EPIPE on its
    # next write. It killed eleven clips in a live run on 2026-08-26.
    script = tmp_path / "noisy.py"
    marker = tmp_path / "ran"
    script.write_text(
        "import sys, time\n"
        "sys.stderr.write('[archive-batch] clip 4: boom\\n')\n"
        "sys.stderr.flush()\n"
        f"open({str(marker)!r}, 'w').write('hi')\n"
        "time.sleep(60)\n", encoding="utf-8")
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(tmp_path / "batch.json"))
    pid = sup.start()
    try:
        _await(marker)
        assert not os.readlink(f"/proc/{pid}/fd/2").startswith("pipe:")
        errors = tmp_path / "batch-stderr.log"
        assert "clip 4: boom" in errors.read_text(encoding="utf-8")
    finally:
        _kill(pid)


def test_start_watches_for_the_death_outside_the_lock(tmp_path):
    # The batch takes batch.json.lock as its own first act, before it reads
    # the manifest that decides whether it can run at all. A watch held under
    # the daemon's lock therefore sees every spawn survive: the child is
    # blocked on us for exactly as long as we are watching it.
    batch_file = tmp_path / "batch.json"
    script = tmp_path / "claiming.py"
    script.write_text(
        "import fcntl, os, sys\n"
        f"fd = os.open({str(batch_file) + '.lock'!r}, os.O_RDWR | os.O_CREAT)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "os.close(fd)\n"
        "sys.stderr.write('[archive-batch] Error: cannot read the "
        "manifest\\n')\n"
        "sys.exit(2)\n", encoding="utf-8")
    sup = Supervisor([sys.executable, str(script)], cwd=str(tmp_path),
                     batch_file=str(batch_file))
    with pytest.raises(SupervisorError) as caught:
        sup.start()
    assert "cannot read the manifest" in str(caught.value)
    assert not batch_file.exists()
