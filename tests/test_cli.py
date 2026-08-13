import importlib.util
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
             out_bytes=1, reason=""),
        dict(src="b", status="done", denoiser="igpu", wall_s=90.0, fps=10.0,
             out_bytes=1, reason=""),
        dict(src="c", status="done", denoiser="gpu1_4090", wall_s=10.0, fps=20.0,
             out_bytes=1, reason=""),
        dict(src="d", status="failed", denoiser="igpu", wall_s=1.0, fps=0.0,
             out_bytes=0, reason="boom"),
    ]
    p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    rates = cli.per_denoiser_rates(str(p))
    assert rates["igpu"] == (2, 10.0)
    assert rates["gpu1_4090"] == (1, 20.0)


def test_summary_reports_each_denoiser(tmp_path):
    import json as _json

    cli = _load_cli()
    p = tmp_path / "state.jsonl"
    p.write_text(_json.dumps(dict(src="a", status="done", denoiser="igpu",
                                  wall_s=10.0, fps=5.0, out_bytes=1,
                                  reason="")) + "\n")
    out = cli.format_summary(1, 0, [], 36.0, state_path=str(p))
    assert "igpu" in out and "5.00 fps" in out


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
