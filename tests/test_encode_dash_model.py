import json
import os
import sys

from tools import encode_dash
from tools.archive_batch import pidfile
from tools.encode_dash import Paths, model, rosterio
from tools.encode_dash.liverate import RateTracker
from tools.encode_dash.model import snapshot

# This process's real /proc identity, so live-lane fixtures pass the same
# identity check production rows are read through.
_PID_START = pidfile.start_time(os.getpid())


def test_run_dir_follows_the_env_var(monkeypatch):
    """The batch already reads ARCHIVE_RUN_DIR; the daemon must agree with it
    or the two look at different runs."""
    monkeypatch.setenv("ARCHIVE_RUN_DIR", "/tmp/somewhere")
    paths = encode_dash.Paths.from_env()
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
    paths = encode_dash.Paths.from_env()
    # Against REPO, not against a directory called "archav1an". The point of
    # this test is that the default lands in the checkout, and the checkout's
    # name is the clone's business: asserting it fails for anyone who clones
    # into a differently-named directory, which is how the public tree's own
    # copy of this suite first failed.
    assert paths.run_dir == os.path.join(encode_dash.REPO, ".archive-run")


ROSTER = """
[[denoiser]]
name    = "gpu1_4090"
host    = "gpu1"
backend = "trt"
device  = 0
tiling  = "none"
root    = "/home/user/archav1an"
enabled = true

[[denoiser]]
name    = "2070s"
host    = "local"
backend = "trt"
device  = 0
tiling  = "auto"
window  = 750
margin  = 32
enabled = false

[encode]
host     = "local"
slots    = 6
lp_level = 4
"""

MANIFEST = (
    "SetA/2001/a/one.MOV\t1000\t30,600\t20.0\n"
    "SetA/2001/a/two.MOV\t2000\t30,400\t13.3\n"
    "SetA/2001/a/three.MOV\t3000\t30,1000\t33.3\n"
)


def _run_dir(tmp_path, records=(), lanes=()):
    run = tmp_path / ".archive-run"
    (run / "lanes").mkdir(parents=True)
    (run / "denoisers.toml").write_text(ROSTER)
    (run / "manifest-raw.tsv").write_text(MANIFEST)
    (run / "state.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    for row in lanes:
        (run / "lanes" / f"{row['lane']}.json").write_text(json.dumps(row))
    return Paths(run_dir=str(run), state=str(run / "state.jsonl"),
                 roster=str(run / "denoisers.toml"),
                 manifest=str(run / "manifest-raw.tsv"),
                 lanes=str(run / "lanes"), control=str(run / "control"),
                 roster_error=str(run / "roster-error.txt"),
                 batch=str(run / "batch.json"))


def _done(src, denoiser, fps, frames_wall=1.0):
    return {"src": src, "status": "done", "denoiser": denoiser,
            "wall_s": frames_wall, "fps": fps, "out_bytes": 10, "reason": "",
            "stage_s": 0.1, "work_s": 0.8, "publish_s": 0.1}


def test_totals_count_the_manifest_against_the_state(tmp_path):
    paths = _run_dir(tmp_path, records=[_done("SetA/2001/a/one.MOV", "gpu1_4090", 15.0)])
    snap = snapshot(paths, RateTracker(), now=100.0)
    assert snap["totals"]["clips"] == 3
    assert snap["totals"]["done"] == 1
    assert snap["totals"]["queued"] == 2
    assert snap["totals"]["frames"] == 2000
    assert snap["totals"]["frames_done"] == 600


def test_every_rostered_lane_appears_even_when_disabled(tmp_path):
    """A lane you switched off must stay on the page. Dropping it would make
    'off' and 'gone' look identical."""
    snap = snapshot(_run_dir(tmp_path), RateTracker(), now=0.0)
    assert [l["name"] for l in snap["lanes"]] == ["gpu1_4090", "2070s"]
    assert snap["lanes"][1]["enabled"] is False
    assert snap["lanes"][1]["state"] == "off"


def test_a_lane_carries_every_field_the_edit_form_can_write(tmp_path):
    """The page fills the edit form from this, so it has to be the whole
    writable set and not the four columns the row happens to render."""
    from tools.encode_dash import rosterio
    snap = snapshot(_run_dir(tmp_path), RateTracker(), now=0.0)
    lane = snap["lanes"][0]
    assert set(lane["fields"]) <= set(rosterio.FIELDS)
    assert lane["fields"]["name"] == "gpu1_4090"
    assert lane["fields"]["root"] == "/home/user/archav1an"


def test_a_field_the_lane_never_set_is_absent_not_zero(tmp_path):
    """The 2070s lane sets no root and does not stage. The gpu1_4090 lane is
    untiled, so it sets no window. Reporting window 0 would put a 0 in the
    form's box, and one save later the roster would carry a window nobody
    typed -- which roster.py then refuses on an untiled lane."""
    snap = snapshot(_run_dir(tmp_path), RateTracker(), now=0.0)
    assert "window" not in snap["lanes"][0]["fields"]
    fields = snap["lanes"][1]["fields"]
    assert "root" not in fields
    assert "stage_source" not in fields
    # Set on this lane, so it is here. device 0 is a real value, not an
    # absence, so it is here too.
    assert fields["window"] == 750
    assert fields["device"] == 0


def test_a_lane_never_reports_enabled_among_its_editable_fields(tmp_path):
    """The switch owns `enabled`. If the edit form could carry it, a save
    would silently undo a toggle made while the form was open."""
    snap = snapshot(_run_dir(tmp_path), RateTracker(), now=0.0)
    assert "enabled" not in snap["lanes"][0]["fields"]


def test_a_lane_with_no_completed_clips_has_no_rate(tmp_path):
    """Never borrow a figure from another lane or from the docs: those are the
    short-run numbers docs/encode-capacity.md withdrew."""
    snap = snapshot(_run_dir(tmp_path), RateTracker(), now=0.0)
    assert snap["lanes"][0]["fps_recent"] is None
    assert snap["lanes"][0]["fps_all"] is None


def test_a_working_lane_reports_its_clip_and_elapsed(tmp_path):
    lane = {"lane": "gpu1_4090", "src": "SetA/2001/a/three.MOV",
            "frames": 1000, "state": "working", "started_at": 40.0,
            "batch_pid": os.getpid(), "attempt": 1,
            "pid_start": _PID_START,
            "temp_dir": str(tmp_path / "temp")}
    snap = snapshot(_run_dir(tmp_path, lanes=[lane]), RateTracker(), now=100.0)
    row = snap["lanes"][0]
    assert row["state"] == "working"
    assert row["current"]["src"] == "SetA/2001/a/three.MOV"
    assert row["current"]["elapsed_s"] == 60.0
    assert row["current"]["frames"] == 1000


def test_a_heartbeat_from_a_dead_process_is_unknown_not_working(tmp_path):
    """A SIGKILLed batch cannot clean up after itself, and a stale row that
    claims to be working is worse than one that admits it does not know."""
    import subprocess
    # A pid that is certainly dead: run something trivial and reap it. A large
    # constant is not safe -- pid_max is 4194304 on this kernel, so a made-up
    # number can belong to a real process and make this test flap.
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()

    lane = {"lane": "gpu1_4090", "src": "SetA/2001/a/one.MOV", "frames": 600,
            "state": "working", "started_at": 1.0, "batch_pid": dead.pid,
            "pid_start": 1, "attempt": 1, "temp_dir": str(tmp_path / "temp")}
    snap = snapshot(_run_dir(tmp_path, lanes=[lane]), RateTracker(), now=10.0)
    assert snap["lanes"][0]["state"] == "unknown"


def test_a_broken_roster_is_reported_and_does_not_raise(tmp_path):
    """scheduler.py swallows this and parks every lane silently. The page is
    where it has to become visible."""
    paths = _run_dir(tmp_path)
    with open(paths.roster, "w") as fh:
        fh.write("[[denoiser]]\nname = \n")
    snap = snapshot(paths, RateTracker(), now=0.0)
    assert snap["roster_error"]
    assert snap["lanes"] == []


def test_failures_carry_the_reason_the_batch_already_recorded(tmp_path):
    rec = {"src": "SetA/2001/a/two.MOV", "status": "failed",
           "denoiser": "2070s", "wall_s": 5.0, "fps": 0.0, "out_bytes": 0,
           "reason": "dispatch exit 1. CAUSE: CUDA failure 700"}
    snap = snapshot(_run_dir(tmp_path, records=[rec]), RateTracker(), now=0.0)
    assert snap["failures"][0]["reason"].startswith("dispatch exit 1")
    assert snap["failures"][0]["attempts"] == 1


def test_a_corrupt_manifest_is_reported_and_does_not_raise(tmp_path):
    """Letting parse_manifest's ValueError out would 500 /metrics, which stops
    the Prometheus scrape and every alert built on it -- a broken manifest
    would switch off the monitoring instead of appearing on it."""
    paths = _run_dir(tmp_path)
    with open(paths.manifest, "w") as fh:
        fh.write("SetA/2001/a/one.MOV\tnot-a-size\t30,600\t20.0\n")
    snap = snapshot(paths, RateTracker(), now=0.0)
    assert snap["manifest_error"]
    assert snap["totals"]["clips"] == 0


def test_a_missing_manifest_is_not_an_error(tmp_path):
    """No manifest yet is the normal state before the first run. A banner for
    it would be noise, and totals of zero already say it."""
    paths = _run_dir(tmp_path)
    os.remove(paths.manifest)
    snap = snapshot(paths, RateTracker(), now=0.0)
    assert snap["manifest_error"] is None
    assert snap["totals"]["clips"] == 0


def test_the_queue_keeps_the_batch_ordering(tmp_path):
    """Longest first inside a folder, as manifest.order_clips does. The page
    must not re-sort it, or 'next up' would be a lie."""
    snap = snapshot(_run_dir(tmp_path), RateTracker(), now=0.0)
    assert [q["frames"] for q in snap["queue"]][:3] == [1000, 600, 400]


def _working(paths, temp, src):
    """Rewrite the one lane's heartbeat, as the batch does between clips."""
    row = {"lane": "gpu1_4090", "src": src, "frames": 1000, "state": "working",
           "started_at": 0.0, "batch_pid": os.getpid(), "attempt": 1,
           "pid_start": _PID_START,
           "temp_dir": str(temp)}
    with open(os.path.join(paths.lanes, "gpu1_4090.json"), "w") as fh:
        json.dump(row, fh)


def test_the_live_rate_does_not_carry_over_from_the_finished_clip(tmp_path):
    """Between two clips the heartbeat stays and only the counter goes away.
    If the tracker keeps the old samples, the next clip's first count is joined
    to the last clip's by a straight line drawn through the gap, and the lane
    reports a rate it never ran at."""
    temp = tmp_path / "temp"
    temp.mkdir()
    paths = _run_dir(tmp_path)
    tracker = RateTracker()

    _working(paths, temp, "SetA/2001/a/three.MOV")
    (temp / "three_vspipe.log").write_text("Frame: 100/1000\n")
    snapshot(paths, tracker, now=0.0)
    (temp / "three_vspipe.log").write_text("Frame: 200/1000\n")
    assert snapshot(paths, tracker, now=10.0)["lanes"][0]["fps_live"] == 10.0

    # The clip finishes and the lane takes the next one. Its log is not there
    # yet, which is how the change of clip shows up in the files.
    _working(paths, temp, "SetA/2001/a/two.MOV")
    snapshot(paths, tracker, now=20.0)

    (temp / "two_vspipe.log").write_text("Frame: 300/400\n")
    assert snapshot(paths, tracker, now=100.0)["lanes"][0]["fps_live"] is None


def test_a_run_that_is_producing_nothing_totals_zero_not_unknown(tmp_path):
    """0.0 is the measurement that says every lane is stalled. None says no
    lane is reporting at all, and folding one into the other hides the state
    the page exists to catch."""
    temp = tmp_path / "temp"
    temp.mkdir()
    (temp / "three_vspipe.log").write_text("Frame: 100/1000\n")
    paths = _run_dir(tmp_path)
    _working(paths, temp, "SetA/2001/a/three.MOV")
    tracker = RateTracker()

    snapshot(paths, tracker, now=0.0)
    snap = snapshot(paths, tracker, now=40.0)
    assert snap["lanes"][0]["fps_live"] == 0.0
    assert snap["totals"]["fps_live"] == 0.0


def test_the_failure_panel_keeps_the_newest_not_the_oldest(tmp_path):
    """Sliced from the front, the panel freezes on the first failures of the
    run and silently drops every later one, including clips that go on to
    exhaust their attempts. Over a few thousand clips a 1.5% transient rate fills it."""
    from tools.encode_dash.model import FAILURE_PREVIEW
    recs = [{"src": f"SetA/2001/a/c{i}.MOV", "status": "failed",
             "denoiser": "2070s", "wall_s": 1.0, "fps": 0.0, "out_bytes": 0,
             "reason": f"reason {i}"} for i in range(FAILURE_PREVIEW + 10)]
    snap = snapshot(_run_dir(tmp_path, records=recs), RateTracker(), now=0.0)
    reasons = [f["reason"] for f in snap["failures"]]
    assert len(reasons) == FAILURE_PREVIEW
    assert reasons[-1] == f"reason {FAILURE_PREVIEW + 9}"


def test_a_heartbeat_with_no_pid_is_not_reported_as_working(tmp_path):
    """os.kill(0, 0) signals the caller's own process group and never raises,
    so _alive(0) is True and a pid-less heartbeat would render as working for
    ever."""
    lane = {"lane": "gpu1_4090", "src": "SetA/2001/a/one.MOV", "frames": 600,
            "state": "working", "started_at": 1.0, "attempt": 1,
            "temp_dir": str(tmp_path / "temp")}
    snap = snapshot(_run_dir(tmp_path, lanes=[lane]), RateTracker(), now=10.0)
    assert snap["lanes"][0]["state"] == "unknown"


def test_a_done_record_without_an_fps_field_does_not_break_the_page(tmp_path):
    """state.jsonl is append-only across versions, so it holds records written
    before a field existed. Every other read of a row uses .get."""
    rec = {"src": "SetA/2001/a/one.MOV", "status": "done",
           "denoiser": "gpu1_4090", "wall_s": 1.0, "out_bytes": 1}
    snap = snapshot(_run_dir(tmp_path, records=[rec]), RateTracker(), now=0.0)
    assert snap["lanes"][0]["clips_done"] == 1
    assert snap["lanes"][0]["fps_recent"] is None


def test_an_unreadable_state_file_does_not_take_the_page_down(tmp_path):
    """load_state catches FileNotFoundError only. A directory where the file
    should be would otherwise 500 /metrics and stop the Prometheus scrape."""
    paths = _run_dir(tmp_path)
    os.remove(paths.state)
    os.mkdir(paths.state)
    snap = snapshot(paths, RateTracker(), now=0.0)
    assert snap["totals"]["done"] == 0
    assert snap["totals"]["clips"] == 3


def test_an_exhausted_clip_is_not_counted_into_the_finish_estimate(tmp_path):
    """queued and queue both exclude a clip past MAX_ATTEMPTS. If eta_finish
    does not, the page shows a finish date built on work it has given up on."""
    recs = [{"src": "SetA/2001/a/three.MOV", "status": "failed",
             "denoiser": "2070s", "wall_s": 1.0, "fps": 0.0, "out_bytes": 0,
             "reason": "x"} for _ in range(2)]
    recs.append(_done("SetA/2001/a/one.MOV", "gpu1_4090", 10.0))
    paths = _run_dir(tmp_path, records=recs)
    snap = snapshot(paths, RateTracker(), now=0.0)
    # Only two.MOV (400 frames) is really left; three.MOV (1000) is exhausted.
    assert snap["totals"]["queued"] == 1
    assert snap["totals"]["eta_finish"] == round(400 / 10.0, 0)


def test_a_windowed_lane_is_smoothed_over_sweeps_and_a_full_frame_lane_is_not():
    """At window 750 and 5.5 fps a sweep is 136 s. A 30 s window would read 0
    for 106 s of it and then 25 fps against a true 5.5 -- the exact artefact
    the smoothing exists to remove, in the default value."""
    from tools.archive_batch.roster import Denoiser
    from tools.encode_dash.model import _smooth_for

    windowed = Denoiser(name="2070s", host="local", backend="trt", device=0,
                        tiling="auto", enabled=True, window=750, margin=32)
    full = Denoiser(name="gpu1_4090", host="gpu1", backend="trt", device=0,
                    tiling="none", enabled=True)

    assert _smooth_for(full, 15.2) is None, "a streaming lane needs no sweeps"
    assert _smooth_for(windowed, 5.5) > 136.0, "must span more than one sweep"
    # With no history yet it must guess slow, because guessing fast gives a
    # window too short to contain a sweep and brings the burst straight back.
    assert _smooth_for(windowed, None) > _smooth_for(windowed, 5.5)


def test_snapshot_carries_the_roster_revision(tmp_path):
    paths = _run_dir(tmp_path)
    snap = snapshot(paths, RateTracker(), now=1000.0)
    assert snap["roster_rev"] == rosterio.rev(paths.roster)
    # A string, not [mtime_ns, size]: a JSON number that big is rounded by the
    # browser, and a page echoing a rounded revision made every write a 409.
    assert isinstance(snap["roster_rev"], str)


def test_the_revision_is_read_before_the_roster_content(tmp_path, monkeypatch):
    """A write landing mid-snapshot must give a stale pair, never a falsely
    current one. Content from version N stamped with the revision of N+1 would
    let the page compute an edit from N and quote a revision the server
    accepts, which is the clobber the whole guard exists to stop. Reading the
    revision first can only give N+1's content under N's revision -- a 409.
    """
    paths = _run_dir(tmp_path)
    seen = []
    real_rev, real_roster = rosterio.rev, model._roster

    def spy_rev(path):
        seen.append("rev")
        return real_rev(path)

    def spy_roster(path):
        seen.append("content")
        return real_roster(path)

    monkeypatch.setattr(model.rosterio, "rev", spy_rev)
    monkeypatch.setattr(model, "_roster", spy_roster)
    snapshot(paths, RateTracker(), now=1000.0)
    assert seen == ["rev", "content"]


def test_roster_revision_changes_when_the_file_is_edited(tmp_path):
    paths = _run_dir(tmp_path)
    before = snapshot(paths, RateTracker(), now=1000.0)["roster_rev"]
    rosterio.set_enabled(paths.roster, "2070s", True)
    after = snapshot(paths, RateTracker(), now=1000.0)["roster_rev"]
    # "false" -> "true" changes the size as well as the mtime, so this holds
    # even on a filesystem whose timestamp resolution is coarse.
    assert before != after


def test_roster_revision_is_none_with_no_roster(tmp_path):
    paths = _run_dir(tmp_path)
    os.unlink(paths.roster)
    snap = snapshot(paths, RateTracker(), now=1000.0)
    assert snap["roster_rev"] is None
    assert snap["roster_error"]


def test_the_snapshot_carries_the_batchs_roster_error(tmp_path):
    """Spec 5.5. The batch is a separate process, so its parse failure reaches
    the page only through the file it writes."""
    paths = _run_dir(tmp_path)
    snap = snapshot(paths, RateTracker(), 1000.0)
    assert snap["batch_roster_error"] is None
    with open(paths.roster_error, "w", encoding="utf-8") as fh:
        fh.write("roster is not valid TOML: line 3\n")
    snap = snapshot(paths, RateTracker(), 1000.0)
    assert snap["batch_roster_error"] == "roster is not valid TOML: line 3"


def test_the_batchs_roster_error_is_separate_from_the_daemons(tmp_path):
    """They are usually the same fault seen twice, and occasionally are not.
    encode_roster_error is a live alert on the Pi with stored samples; folding
    a second source into it would widen what that alert means without anyone
    deciding to."""
    paths = _run_dir(tmp_path)
    # Break the roster so the daemon fails to parse it.
    with open(paths.roster, "w", encoding="utf-8") as fh:
        fh.write("[[denoiser]]\nname = \n")
    # Write a different message for what the batch says.
    with open(paths.roster_error, "w", encoding="utf-8") as fh:
        fh.write("the batch says line 3\n")
    snap = snapshot(paths, RateTracker(), 1000.0)
    # Both are non-None and different; the daemon's failure and the batch's.
    assert snap["roster_error"]  # daemon's parse failure
    assert snap["batch_roster_error"] == "the batch says line 3"
    assert snap["roster_error"] != snap["batch_roster_error"]


def test_the_snapshot_carries_the_recent_acks(tmp_path):
    from tools.archive_batch import control

    paths = _run_dir(tmp_path)
    assert snapshot(paths, RateTracker(), 1000.0)["acks"] == []
    control.ack(paths.control, "a1", True, "killed igpu's dispatch", 999.0)
    snap = snapshot(paths, RateTracker(), 1000.0)
    assert snap["acks"] == [{"id": "a1", "accepted": True, "at": 999.0,
                             "note": "killed igpu's dispatch"}]


def test_an_unreadable_roster_error_file_does_not_break_the_snapshot(tmp_path):
    """Every reader in this module degrades rather than raising, because a
    500 on /metrics kills the Pi's scrape and the scrape is what tells you the
    run is in trouble."""
    paths = _run_dir(tmp_path)
    os.makedirs(paths.roster_error)      # a directory where a file belongs
    snap = snapshot(paths, RateTracker(), 1000.0)
    assert snap["batch_roster_error"] is None


def test_the_roster_revision_is_still_read_before_anything_else(tmp_path):
    """A regression guard on the ordering comment in snapshot(). The revision
    must be read before the content, or the page computes an edit from version
    N and quotes the revision of N+1."""
    import inspect

    from tools.encode_dash import model

    body = inspect.getsource(model.snapshot)
    lines = [ln.strip() for ln in body.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    reads = [ln for ln in lines
             if "rosterio.rev(" in ln or "_roster(" in ln or "_clips(" in ln]
    assert reads and "rosterio.rev(" in reads[0], reads


def test_every_lane_disabled_is_not_a_roster_error(tmp_path):
    """The alert guard. encode_roster_error is a live alert on the Pi with
    stored samples, and it is driven by snap["roster_error"]. Yielding the last
    enabled lane writes exactly this roster, so if it read as a parse failure
    the operator would be paged for doing what the page offered them.

    The lanes must still be listed, or there would be no switch to turn one
    back on with.
    """
    paths = _run_dir(tmp_path)
    with open(paths.roster, "w", encoding="utf-8") as fh:
        fh.write(ROSTER.replace("enabled = true", "enabled = false"))
    snap = snapshot(paths, RateTracker(), now=0.0)
    assert snap["roster_error"] is None
    assert [l["name"] for l in snap["lanes"]] == ["gpu1_4090", "2070s"]
    assert [l["enabled"] for l in snap["lanes"]] == [False, False]


def test_a_parked_run_with_no_heartbeat_still_reads_as_running(tmp_path):
    """The whole point of batch.json. A run parked because every lane was
    yielded holds no clip, so lanes/ is empty and there is no heartbeat to
    read. Without this the daemon calls a live process dead: encode_batch_up
    goes to 0 with clips still queued, which is EncodeBatchDown after five
    minutes, and app.js disables the Stop button on the same field.
    """
    paths = _run_dir(tmp_path)
    with open(paths.batch, "w", encoding="utf-8") as fh:
        json.dump({"batch_pid": os.getpid(), "started_at": 1.0,
                   "pid_start": _PID_START, "claimed": True}, fh)
    snap = snapshot(paths, RateTracker(), now=0.0)
    assert snap["batch"] == {"running": True, "pid": os.getpid()}


def test_a_batch_file_left_by_a_killed_run_is_not_running(tmp_path):
    """SIGKILL skips the cleanup, so the file outlives the process. The pid it
    names is dead, and that is what has to be checked rather than the file
    merely existing."""
    import subprocess
    # Reaped rather than invented, for the reason spelled out in
    # test_a_lane_whose_batch_is_gone_is_unknown: pid_max is large enough that
    # a made-up number can belong to a real process and make this flap.
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()
    paths = _run_dir(tmp_path)
    with open(paths.batch, "w", encoding="utf-8") as fh:
        json.dump({"batch_pid": dead.pid, "started_at": 1.0}, fh)
    snap = snapshot(paths, RateTracker(), now=0.0)
    assert snap["batch"] == {"running": False, "pid": None}


def _pre_upgrade_batch(tmp_path, name="archive-batch.py"):
    """A live stand-in for a batch started before this version was installed.

    Named so its command line is what pidfile.alive checks, because that is
    the only identity a record with no pid_start carries. Waits for exec:
    /proc/<pid>/cmdline is empty for the instant between fork and exec.
    """
    import subprocess
    import time as _time
    script = tmp_path / name
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)])
    deadline = _time.monotonic() + 10
    while _time.monotonic() < deadline and not pidfile.cmdline(proc.pid):
        _time.sleep(0.01)
    assert pidfile.cmdline(proc.pid), "the stand-in never reached exec"
    return proc


def test_a_live_pre_upgrade_heartbeat_reads_as_running(tmp_path):
    """A batch that was already running when the checkout was upgraded keeps
    its old code, so it writes neither pid_start nor the claimed marker.

    Reading that as dead is the worse direction of this bug: encode_batch_up
    drops to 0 and fires EncodeBatchDown for a fleet that is encoding, app.js
    greys out Stop and RE-ENABLES Start, and one click puts a second batch
    over the live one -- two lanes publishing to one destination path.
    """
    child = _pre_upgrade_batch(tmp_path)
    try:
        paths = _run_dir(tmp_path, lanes=[
            {"lane": "gpu1_4090", "src": "SetA/2001/a/one.MOV", "frames": 10,
             "state": "working", "started_at": 0.0, "batch_pid": child.pid,
             "attempt": 1, "temp_dir": "/tmp/t"}])
        assert not os.path.exists(paths.batch)
        snap = snapshot(paths, RateTracker(), now=0.0)
        assert snap["batch"] == {"running": True, "pid": child.pid}
        # And the lane it holds renders, rather than every lane reading
        # "unknown" with no rate, no progress and no ETA for the rest of the run.
        assert snap["lanes"][0]["state"] == "working"
    finally:
        child.kill()
        child.wait()


def test_a_batch_file_naming_a_zombie_is_not_running(tmp_path):
    """A SIGKILLed batch the daemon spawned stays a zombie until the daemon
    waits on it, and os.kill(pid, 0) succeeds on a zombie. Read as alive, the
    stale batch.json it left behind would hold encode_batch_up at 1 for the
    rest of the daemon's life and keep offering Stop for a finished run."""
    import subprocess
    import time as _time
    script = tmp_path / "archive-batch.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    child = subprocess.Popen([sys.executable, str(script)])
    start = pidfile.start_time(child.pid)
    child.kill()                # no wait(): that is what leaves the zombie
    deadline = _time.monotonic() + 10
    while _time.monotonic() < deadline:
        fields = pidfile._stat_after_comm(child.pid)
        if fields and fields[0] == b"Z":
            break
        _time.sleep(0.01)
    assert fields and fields[0] == b"Z", "the fixture never became a zombie"
    try:
        paths = _run_dir(tmp_path)
        with open(paths.batch, "w", encoding="utf-8") as fh:
            json.dump({"batch_pid": child.pid, "pid_start": start,
                       "started_at": 1.0, "claimed": True}, fh)
        snap = snapshot(paths, RateTracker(), now=0.0)
        assert snap["batch"] == {"running": False, "pid": None}
    finally:
        child.wait()


def test_a_pre_upgrade_heartbeat_whose_pid_was_recycled_reads_as_dead(tmp_path):
    """The other half. The batch never sweeps lanes/, so a SIGKILLed old run
    leaves its heartbeats behind; once the kernel hands that pid to an ffmpeg,
    an existence-only probe would hold encode_batch_up at 1 for ever."""
    child = _pre_upgrade_batch(tmp_path, name="ffmpeg-ish.py")
    try:
        paths = _run_dir(tmp_path, lanes=[
            {"lane": "gpu1_4090", "src": "SetA/2001/a/one.MOV", "frames": 10,
             "state": "working", "started_at": 0.0, "batch_pid": child.pid,
             "attempt": 1, "temp_dir": "/tmp/t"}])
        snap = snapshot(paths, RateTracker(), now=0.0)
        assert snap["batch"] == {"running": False, "pid": None}
        assert snap["lanes"][0]["state"] == "unknown"
    finally:
        child.kill()
        child.wait()


def test_a_live_pre_upgrade_batch_json_reads_as_running(tmp_path):
    """The same run through the primary path. A pre-upgrade batch.json holds
    batch_pid and started_at only: no claimed marker, because its writer had
    none. Gating it on that marker would call the live run dead, which is the
    finding above by another route."""
    child = _pre_upgrade_batch(tmp_path)
    try:
        paths = _run_dir(tmp_path)
        with open(paths.batch, "w", encoding="utf-8") as fh:
            json.dump({"batch_pid": child.pid, "started_at": 1.0}, fh)
        snap = snapshot(paths, RateTracker(), now=0.0)
        assert snap["batch"] == {"running": True, "pid": child.pid}
    finally:
        child.kill()
        child.wait()


def test_an_unclaimed_post_upgrade_batch_json_still_reads_as_not_running(tmp_path):
    """The startup window the claimed marker exists for is unchanged: a
    record that carries pid_start comes from a writer that also writes the
    marker, so its absence means the batch has not cleared the control dir
    yet and Stop must not be offered."""
    paths = _run_dir(tmp_path)
    with open(paths.batch, "w", encoding="utf-8") as fh:
        json.dump({"batch_pid": os.getpid(), "started_at": 1.0,
                   "pid_start": _PID_START}, fh)
    snap = snapshot(paths, RateTracker(), now=0.0)
    assert snap["batch"] == {"running": False, "pid": None}


def test_a_non_dict_heartbeat_does_not_take_the_snapshot_down(tmp_path):
    """A lanes/*.json holding "[]" parses fine and has no .get. Unguarded it
    raises AttributeError out of heartbeat.read_all, which 500s the page and
    /metrics -- and an absent encode_batch_up series never fires
    EncodeBatchDown, so the alert is lost rather than raised."""
    paths = _run_dir(tmp_path)
    (tmp_path / ".archive-run" / "lanes" / "gpu1_4090.json").write_text("[]")
    snap = snapshot(paths, RateTracker(), now=0.0)
    assert snap["batch"] == {"running": False, "pid": None}
    assert snap["lanes"][0]["state"] == "idle"


# --- The --lp control is only offered where set_lp_level can succeed ---------

POOL_ROSTER = """
[[denoiser]]
name    = "igpu"
host    = "local"
backend = "migraphx"
device  = 0
tiling  = "none"
enabled = true

[[encoder]]
name      = "encoder-host"
host      = "local"
stream_ip = "10.0.0.10"
port_base = 5300
slots     = 6
lp_level  = 4
enabled   = true

[[encoder]]
name      = "gpu4"
host      = "gpu4"
root      = "/home/user/reposetc/ubuntav1an"
stream_ip = "10.0.0.14"
port_base = 5320
slots     = 3
lp_level  = 4
enabled   = true
"""


def test_a_pool_roster_reports_its_lp_level_but_not_as_editable(tmp_path):
    """rosterio can only write the legacy [encode] table, and per-encoder
    editing is a separate plan. The pool still AGREES on a level, so the
    status line may name it -- but offering the select would offer a save
    that cannot succeed."""
    paths = _run_dir(tmp_path)
    open(paths.roster, "w").write(POOL_ROSTER)
    snap = snapshot(paths, RateTracker(), now=0.0)
    assert snap["encode"]["slots"] == 9
    assert snap["encode"]["lp_level"] == 4
    assert snap["encode"]["lp_editable"] is False


def test_a_legacy_roster_still_offers_the_lp_control(tmp_path):
    snap = snapshot(_run_dir(tmp_path), RateTracker(), now=0.0)
    assert snap["encode"]["lp_editable"] is True


def test_the_page_disables_the_lp_select_on_that_field(tmp_path):
    """Asserted against app.js itself, the way test_lane_presets asserts
    LANE_FIELDS: the daemon can send lp_editable forever and the control is
    still live if renderLp never reads it."""
    source = (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tools", "encode_dash", "static", "app.js"))
    text = open(source, encoding="utf-8").read()
    body = text[text.index("function renderLp"):text.index("const LANE_FIELDS")]
    assert "lp_editable" in body, "renderLp ignores lp_editable"
    assert "title" in body, "a disabled control must say why it is disabled"


def test_a_pool_roster_lists_every_host_including_the_disabled_ones(tmp_path):
    """The switch has to be able to turn a host back ON, so the row for a
    disabled host must exist. enabled_encoders(), which the slot count uses,
    cannot supply it."""
    paths = _run_dir(tmp_path)
    open(paths.roster, "w").write(POOL_ROSTER.replace(
        '''name      = "gpu4"
host      = "gpu4"
root      = "/home/user/reposetc/ubuntav1an"
stream_ip = "10.0.0.14"
port_base = 5320
slots     = 3
lp_level  = 4
enabled   = true''',
        '''name      = "gpu4"
host      = "gpu4"
root      = "/home/user/reposetc/ubuntav1an"
stream_ip = "10.0.0.14"
port_base = 5320
slots     = 3
lp_level  = 4
enabled   = false'''))
    snap = snapshot(paths, RateTracker(), now=0.0)
    hosts = {h["name"]: h for h in snap["encode"]["hosts"]}
    assert set(hosts) == {"encoder-host", "gpu4"}
    assert hosts["gpu4"]["enabled"] is False
    assert hosts["encoder-host"]["enabled"] is True
    # And the slot count still counts only what can actually take work.
    assert snap["encode"]["slots"] == 6


def test_a_legacy_roster_sends_no_host_rows(tmp_path):
    """load_roster synthesizes an Encoder named "local" from the [encode]
    table, but there is no [[encoder]] block for rosterio to write. A row for
    it would render a switch whose POST answers 404."""
    snap = snapshot(_run_dir(tmp_path), RateTracker(), now=0.0)
    assert snap["encode"]["hosts"] == []


def test_the_page_renders_a_switch_for_each_encode_host(tmp_path):
    """Asserted against app.js itself, the way the lp_editable test is: the
    daemon can send hosts forever and there is still no switch if renderHosts
    never reads them, or posts to the lane route by mistake."""
    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "tools", "encode_dash", "static", "app.js")
    text = open(source, encoding="utf-8").read()
    body = text[text.index("function renderHosts"):text.index("function renderLp")]
    assert "encode.hosts" in body, "renderHosts ignores the hosts the daemon sends"
    assert "/api/encoder/" in body, "the switch must post to the encoder route"
    assert "/api/lane/" not in body, \
        "the encode host switch must never write the lane table"


def test_host_rows_carry_the_writable_fields_the_edit_form_needs(tmp_path):
    paths = _run_dir(tmp_path)
    open(paths.roster, "w").write(POOL_ROSTER)
    snap = snapshot(paths, RateTracker(), now=0.0)
    fields = {h["name"]: h["fields"] for h in snap["encode"]["hosts"]}
    assert fields["gpu4"]["port_base"] == 5320
    assert fields["gpu4"]["slots"] == 3
    # `enabled` never travels in fields: the row's switch owns it, and an edit
    # carrying it would let a save undo a toggle made in between.
    assert "enabled" not in fields["gpu4"]
    # A local encoder needs no checkout path, so the key is absent rather than
    # empty -- typed into the form, "" would be saved as a real empty root.
    assert "root" not in fields["encoder-host"]


def test_the_page_offers_the_allowlist_and_the_host_form(tmp_path):
    """Asserted against app.js, like the lp_editable test: the daemon can send
    hosts and allowlists forever and neither control exists if the page never
    builds them."""
    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "tools", "encode_dash", "static", "app.js")
    text = open(source, encoding="utf-8").read()

    # Routing lives in the lane form.
    lane_form = text[text.index("function buildAddForm"):
                     text.index("function renderList")]
    assert "encoders" in lane_form, "the lane form cannot set an allowlist"
    assert "buildRouting" in lane_form
    # An edit must send an EMPTY list, or clearing every tick can never put the
    # lane back on "any encoder".
    assert "body.encoders = picked" in lane_form

    # Host editing lives in its own form and posts to the encoder routes.
    host_form = text[text.index("function buildHostForm"):
                     text.index("function renderHosts")]
    assert "/api/encoder" in host_form
    assert "/api/lane/" not in host_form, \
        "the host form must never write the lane table"

    # ENCODER_FIELDS is a copy of rosterio's, the way LANE_FIELDS is. A key
    # added there and not here can never be set from the page.
    from tools.encode_dash import rosterio as rio
    for key in rio.ENCODER_FIELDS:
        assert f'["{key}"' in text or f'["{key}",' in text, \
            f"ENCODER_FIELDS in app.js is missing '{key}'"


def test_the_panel_orders_by_the_newest_failure_not_the_first(tmp_path):
    """`failed[src] = row` reassigns, and reassigning a dict key keeps its
    FIRST insertion position. So slicing the tail selected the newest first-
    failures rather than the newest failures, and a clip that failed early and
    again late sat at its early position -- outside the window, invisible."""
    from tools.encode_dash.model import FAILURE_PREVIEW
    early = {"src": "SetA/2001/a/one.MOV", "status": "failed",
             "denoiser": "2070s", "wall_s": 1.0, "fps": 0.0, "out_bytes": 0,
             "reason": "the first failure, long ago"}
    filler = [{"src": f"SetA/2001/a/c{i}.MOV", "status": "failed",
               "denoiser": "2070s", "wall_s": 1.0, "fps": 0.0, "out_bytes": 0,
               "reason": f"filler {i}"} for i in range(FAILURE_PREVIEW + 10)]
    late = dict(early, reason="the newest failure in the whole run")
    snap = snapshot(_run_dir(tmp_path, records=[early] + filler + [late]),
                    RateTracker(), now=0.0)
    reasons = [f["reason"] for f in snap["failures"]]
    assert "the newest failure in the whole run" in reasons


def test_an_exhausted_clip_is_listed_however_old_its_last_failure(tmp_path):
    """The `out of attempts only` filter exists to show the clips that stopped
    for good, and those are the ones a run leaves behind early. Capping the
    panel by recency alone dropped every one of them, so ticking the box
    emptied the panel instead of narrowing it."""
    from tools.encode_dash.model import FAILURE_PREVIEW
    dead = [{"src": "SetA/2001/a/one.MOV", "status": "failed",
             "denoiser": "2070s", "wall_s": 1.0, "fps": 0.0, "out_bytes": 0,
             "reason": "out of attempts"}] * 2
    filler = [{"src": f"SetA/2001/a/c{i}.MOV", "status": "failed",
               "denoiser": "2070s", "wall_s": 1.0, "fps": 0.0, "out_bytes": 0,
               "reason": f"filler {i}"} for i in range(FAILURE_PREVIEW + 10)]
    snap = snapshot(_run_dir(tmp_path, records=dead + filler),
                    RateTracker(), now=0.0)
    assert snap["totals"]["failed"] == 1
    shown = [f for f in snap["failures"] if f["exhausted"]]
    assert [f["src"] for f in shown] == ["SetA/2001/a/one.MOV"]
