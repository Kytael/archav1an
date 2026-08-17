import json
import os
import sys

from tools import archive_ui
from tools.archive_ui import Paths
from tools.archive_ui.liverate import RateTracker
from tools.archive_ui.model import snapshot


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
