"""Tests for tools/encode-dash.py itself -- the daemon's own call site.

Every route the daemon serves goes through the one snapshot function this
module builds, and nothing here was reachable from a test while it lived as a
closure inside main(). The first version of it shipped a tuple-unpack crash
that made /api/status, /metrics and the page answer 500 from the first poll,
and the suite passed throughout.
"""
import importlib.util
import json
import os
from pathlib import Path

from tools.encode_dash import Paths
from tools.encode_dash.liverate import RateTracker

REPO = Path(__file__).resolve().parent.parent

ROSTER = """
[[denoiser]]
name    = "gpu1_4090"
host    = "local"
backend = "trt"
device  = 0
tiling  = "none"
enabled = true

[[encoder]]
name      = "encoder-host"
host      = "local"
port_base = 5300
"""


def _load_daemon():
    spec = importlib.util.spec_from_file_location(
        "encode_dash_cli", REPO / "tools" / "encode-dash.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _paths(tmp_path):
    run = tmp_path / ".archive-run"
    (run / "lanes").mkdir(parents=True)
    (run / "denoisers.toml").write_text(ROSTER)
    (run / "manifest-raw.tsv").write_text(
        "SetA/2001/a/one.MOV\tSetA/2001/a\tone\t1\t100\n")
    (run / "state.jsonl").write_text("")
    return Paths(run_dir=str(run), state=str(run / "state.jsonl"),
                 roster=str(run / "denoisers.toml"),
                 manifest=str(run / "manifest-raw.tsv"),
                 lanes=str(run / "lanes"), control=str(run / "control"),
                 roster_error=str(run / "roster-error.txt"),
                 batch=str(run / "batch.json"))


class _FakeSupervisor:
    """Records what the snapshot function asks it, and answers a live spawn."""

    def __init__(self, pid=4242):
        self.pid = pid
        self.calls = 0

    def status(self):
        self.calls += 1
        return self.pid, True, True


def test_the_snapshot_reaps_the_spawned_batch_on_every_poll(tmp_path):
    """status() is what waits on the batch this daemon spawned. Nothing else
    does: there is no SIGCHLD handler, and Popen only reaps inside the next
    Popen. Unreaped, a SIGKILLed batch stays a zombie that os.kill(pid, 0)
    reports as alive, and the stale batch.json it left behind reads as a live
    run for ever -- encode_batch_up pinned at 1, EncodeBatchDown never fired.
    """
    daemon = _load_daemon()
    sup = _FakeSupervisor()
    fn = daemon.make_snapshot(_paths(tmp_path), RateTracker(), sup, None)
    fn()
    fn()
    assert sup.calls == 2, "each poll must give the daemon its chance to reap"


def test_the_snapshot_does_not_call_a_spawn_running_before_the_batch_claims_it(tmp_path):
    """The supervisor knows its child is alive the instant Popen returns, but
    the batch clears the control directory seconds later. Reporting the run as
    running before its claim lands would make a Stop written in that window be
    dropped as stale, with the page showing it acknowledged.
    """
    daemon = _load_daemon()
    fn = daemon.make_snapshot(_paths(tmp_path), RateTracker(),
                              _FakeSupervisor(), None)
    assert fn()["batch"] == {"running": False, "pid": None}


def test_the_snapshot_reads_a_claimed_run_as_running(tmp_path):
    daemon = _load_daemon()
    paths = _paths(tmp_path)
    from tools.archive_batch import pidfile
    with open(paths.batch, "w", encoding="utf-8") as fh:
        json.dump({"batch_pid": os.getpid(), "started_at": 1.0,
                   "pid_start": pidfile.start_time(os.getpid()),
                   "claimed": True}, fh)
    fn = daemon.make_snapshot(paths, RateTracker(), _FakeSupervisor(), None)
    assert fn()["batch"] == {"running": True, "pid": os.getpid()}


def test_the_snapshot_carries_the_grafana_url_and_the_lanes(tmp_path):
    """The whole shape the page and /metrics read, through the daemon's own
    call rather than model.snapshot directly."""
    daemon = _load_daemon()
    fn = daemon.make_snapshot(_paths(tmp_path), RateTracker(),
                              _FakeSupervisor(), "http://grafana:3030")
    snap = fn()
    assert snap["grafana_url"] == "http://grafana:3030"
    assert [l["name"] for l in snap["lanes"]] == ["gpu1_4090"]


def test_the_snapshot_works_without_a_supervisor(tmp_path):
    """A daemon constructed with no supervisor still serves: the read-only
    deployment is a supported one, and the reap call must not assume."""
    daemon = _load_daemon()
    fn = daemon.make_snapshot(_paths(tmp_path), RateTracker(), None, None)
    assert fn()["batch"] == {"running": False, "pid": None}
