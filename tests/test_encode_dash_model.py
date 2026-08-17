import json
import os
import sys

from tools import encode_dash
from tools.encode_dash import Paths
from tools.encode_dash.liverate import RateTracker
from tools.encode_dash.model import snapshot


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
port    = 5300
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
                 lanes=str(run / "lanes"), control=str(run / "control"))


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
            "attempt": 1, "temp_dir": str(tmp_path / "temp")}
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
                    tiling="none", enabled=True, port=5300)

    assert _smooth_for(full, 15.2) is None, "a streaming lane needs no sweeps"
    assert _smooth_for(windowed, 5.5) > 136.0, "must span more than one sweep"
    # With no history yet it must guess slow, because guessing fast gives a
    # window too short to contain a sweep and brings the burst straight back.
    assert _smooth_for(windowed, None) > _smooth_for(windowed, 5.5)
