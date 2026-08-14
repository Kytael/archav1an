import importlib.util
import os
import subprocess
import tempfile
import time
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


def test_make_runner_takes_the_encode_pool_rather_than_reading_it():
    """A clip in flight must survive the roster's last denoiser being disabled.

    Re-reading the roster mid-clip made load_roster raise once the user turned
    off the last device, failing a clip that was encoding fine.
    """
    import inspect
    cli = _load_cli()
    assert list(inspect.signature(cli.make_runner).parameters) == ["encode"]
    src = inspect.getsource(cli.make_runner)
    assert "_roster()" not in src, "runner must not re-read the roster per clip"


def test_sweep_clears_staged_leftovers(tmp_path, monkeypatch):
    """A killed run strands up to 3 GB per denoiser and nothing else removes it."""
    cli = _load_cli()
    stage = tmp_path / "_stage"
    (stage / "igpu").mkdir(parents=True)
    (stage / "igpu" / "MVI_0001.MOV").write_bytes(b"x" * 2048)
    (stage / "gpu1_4090").mkdir()
    (stage / "gpu1_4090" / "MVI_0002-av1.mkv").write_bytes(b"y" * 1024)
    monkeypatch.setattr(cli, "STAGE_ROOT", str(stage))
    assert cli.sweep_stage_root() == 3072
    assert cli.sweep_stage_root() == 0
    assert (stage / "igpu").is_dir()          # directories stay, files go


def test_per_denoiser_rates_are_frame_weighted(tmp_path):
    import json as _json

    cli = _load_cli()
    p = tmp_path / "state.jsonl"
    rows = [
        # 100 frames in 10s and 900 frames in 90s -> 1000 frames / 100s = 10 fps
        dict(src="a", status="done", denoiser="igpu", wall_s=10.0, fps=10.0,
             out_bytes=1, reason="", work_s=5.0),
        dict(src="b", status="done", denoiser="igpu", wall_s=90.0, fps=10.0,
             out_bytes=1, reason="", work_s=45.0),
        dict(src="c", status="done", denoiser="gpu1_4090", wall_s=10.0, fps=20.0,
             out_bytes=1, reason="", work_s=10.0),
        dict(src="d", status="failed", denoiser="igpu", wall_s=1.0, fps=0.0,
             out_bytes=0, reason="boom", work_s=0.5),
    ]
    p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    rates = cli.per_denoiser_rates(str(p))
    # 1000 frames over 100 s of wall clock, but only 50 s of dispatch: the
    # overhead-free rate is double, which is the point of recording both.
    assert rates["igpu"] == (2, 10.0, 20.0)
    # A lane with no staging cost reads the same either way.
    assert rates["gpu1_4090"] == (1, 20.0, 20.0)


def test_rates_survive_records_written_before_the_split_existed():
    """work_s defaults to 0; the work rate must read 0, not divide by zero."""
    import json as _json
    import tempfile as _tf

    cli = _load_cli()
    path = _tf.mkdtemp() + "/old.jsonl"
    with open(path, "w") as fh:
        fh.write(_json.dumps(dict(src="a", status="done", denoiser="igpu",
                                  wall_s=10.0, fps=5.0, out_bytes=1,
                                  reason="")) + "\n")
    assert cli.per_denoiser_rates(path)["igpu"] == (1, 5.0, 0.0)


def test_summary_reports_each_denoiser(tmp_path):
    import json as _json

    cli = _load_cli()
    p = tmp_path / "state.jsonl"
    p.write_text(_json.dumps(dict(src="a", status="done", denoiser="igpu",
                                  wall_s=10.0, fps=5.0, out_bytes=1,
                                  reason="", work_s=4.0)) + "\n")
    out = cli.format_summary(1, 0, [], 36.0, state_path=str(p))
    assert "igpu" in out
    # 50 frames in 10 s of wall clock and 4 s of dispatch.
    assert "5.00" in out and "12.50" in out
    assert "pool" in out


def test_an_incomplete_run_exits_nonzero(tmp_path, monkeypatch):
    """An outage stop recorded no failures, so exit 0 read as 'archive finished'."""
    cli = _load_cli()

    class _Sched:
        done, failed, failures = 9, 0, []

        def __init__(self):
            import queue as _q
            self.queue = _q.Queue()
            for i in range(10):
                self.queue.put(i)

        def run(self):
            return 0

        def stop(self):
            pass

    monkeypatch.setattr(cli, "Scheduler", lambda *a, **kw: _Sched())
    monkeypatch.setattr(cli, "sweep_stage_root", lambda: 0)
    monkeypatch.setattr(cli, "_roster", lambda: _roster_stub())
    manifest = tmp_path / "m.tsv"
    manifest.write_text("SetA/2001/f/a.MOV\t100\t30000/1001,50\t1.6\n")
    monkeypatch.setattr(cli, "MANIFEST", str(manifest))
    monkeypatch.setattr(cli, "STATE", str(tmp_path / "state.jsonl"))
    assert cli.main() == 3


def _roster_stub():
    from tools.archive_batch.roster import Denoiser, EncodePool, Roster
    return Roster(denoisers=(Denoiser(name="d", host="local", backend="trt",
                                      device=0, tiling="none", enabled=True),),
                  encode=EncodePool(host="local", slots=1, lp_level=6))


def test_dispatch_timeout_scales_with_clip_length():
    cli = _load_cli()
    short, long = cli.dispatch_timeout(1000), cli.dispatch_timeout(50000)
    assert long > short > cli.DISPATCH_FLOOR_S
    # The slowest rostered denoiser is about 4.4 fps, so the budget must stay
    # well clear of a clip that is merely slow rather than hung.
    assert short >= 1000 / 4.4 * 4
    assert long >= 50000 / 4.4 * 4


def test_a_clip_with_no_frame_count_gets_the_ceiling_not_the_floor():
    cli = _load_cli()
    # frames==0 means the manifest probe failed, not that the clip is empty.
    # Handing it the floor would kill a 30-minute clip that is encoding fine.
    assert cli.dispatch_timeout(0) == cli.DISPATCH_UNKNOWN_S
    assert cli.dispatch_timeout(0) > cli.dispatch_timeout(50000)


def test_a_hung_dispatch_is_killed_and_reported():
    cli = _load_cli()
    # sleep ignores SIGTERM's default only when trapped; a bare sleep dies on
    # SIGTERM, so this proves the group kill lands and run_dispatch returns.
    rc, timed_out = cli.run_dispatch(["sleep", "60"], dict(os.environ), timeout=1.0)
    assert timed_out is True
    assert rc != 0


def test_a_dispatch_that_finishes_in_time_is_not_killed():
    cli = _load_cli()
    rc, timed_out = cli.run_dispatch(["true"], dict(os.environ), timeout=30.0)
    assert (rc, timed_out) == (0, False)


def test_the_kill_reaches_children_not_just_the_dispatch():
    cli = _load_cli()
    # A real hang is a stuck vspipe or ssh under the dispatch, so killing only
    # the direct child leaves the lane blocked. Mark the grandchild with a
    # unique argument and check nothing carrying it survives.
    marker = tempfile.mkdtemp() + "/archive-batch-killtest"
    open(marker, "w").close()
    argv = ["sh", "-c", "tail -f %s & wait" % marker]
    _rc, timed_out = cli.run_dispatch(argv, dict(os.environ), timeout=1.0)
    assert timed_out is True
    time.sleep(0.5)
    survivors = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    assert survivors.stdout.strip() == "", f"grandchild survived: {survivors.stdout}"


def test_remote_stage_is_cleared_for_a_gpu_only_host(monkeypatch):
    cli = _load_cli()
    calls = []
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _done())
    from tools.archive_batch.manifest import Clip
    from tools.archive_batch.roster import Denoiser
    d = Denoiser(name="gpu4", host="gpu4", backend="trt", device=0,
                 tiling="none", enabled=True, stage_source=True,
                 root="/home/user/reposetc/ubuntav1an")
    clip = Clip(src="SetA/2003/event-b/MVI_0068.MOV", rel_dir="SetA/2003/event-b",
                stem="MVI_0068", size=1, frames=100)
    cli.clear_remote_stage(d, clip)
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "ssh" and "gpu4" in cmd
    # The path must be the remote's own root, not this host's, and the basename
    # must be quoted: archive folders and clips can carry spaces.
    assert "/home/user/reposetc/ubuntav1an/Temp/_remote/" in cmd[-1]
    assert "MVI_0068.MOV" in cmd[-1]


def test_a_failed_cleanup_does_not_raise(monkeypatch):
    # The cleanup runs in the runner's finally, so an exception here would turn
    # a clip that encoded and published correctly into a recorded failure.
    cli = _load_cli()

    def boom(cmd, **kw):
        raise cli.subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(cli.subprocess, "run", boom)
    from tools.archive_batch.manifest import Clip
    from tools.archive_batch.roster import Denoiser
    d = Denoiser(name="gpu2", host="gpu2", backend="trt", device=0,
                 tiling="auto", enabled=True, stage_source=True, root="/r")
    cli.clear_remote_stage(d, Clip(src="a/b.MOV", rel_dir="a", stem="b",
                                   size=1, frames=1))


def _done():
    import subprocess as _sp
    return _sp.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
