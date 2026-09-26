import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

from tools.archive_batch.manifest import Clip
from tools.archive_batch.roster import Denoiser, Encoder

BATCH_PY = Path(__file__).resolve().parent.parent / "tools" / "archive-batch.py"
_spec = importlib.util.spec_from_file_location("archive_batch_main", BATCH_PY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


@pytest.fixture(autouse=True)
def clean_registry():
    """Module state is global, so one test's leftovers are another's bug."""
    with _mod._lock:
        _mod._procs.clear()
        _mod._yielded.clear()
    yield
    with _mod._lock:
        _mod._procs.clear()
        _mod._yielded.clear()


def _sleeper():
    return [sys.executable, "-c", "import time; time.sleep(120)"]


def test_yield_lane_on_a_lane_that_is_not_running_is_refused():
    assert _mod.yield_lane("nobody") is False


def test_yield_lane_kills_the_dispatch_and_run_dispatch_returns(tmp_path):
    """The kill must reach the whole process group, because a dispatch is
    vspipe piped into an encoder with an ssh in between and killing the parent
    alone leaves the lane blocked anyway."""
    done = {}

    def dispatch():
        done["rc"] = _mod.run_dispatch(_sleeper(), None, 120, lane="gpu1_4090")

    t = threading.Thread(target=dispatch, daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with _mod._lock:
            if "gpu1_4090" in _mod._procs:
                break
        time.sleep(0.01)
    assert _mod.yield_lane("gpu1_4090") is True
    t.join(70)
    assert not t.is_alive(), "run_dispatch did not return after the kill"
    rc, timed_out = done["rc"]
    # The trap this whole task exists for: a killed group returns from wait()
    # normally. timed_out is False and the return code is an ordinary failure.
    assert timed_out is False
    assert rc != 0


def test_the_registry_is_empty_again_after_a_dispatch_returns():
    _mod.run_dispatch([sys.executable, "-c", "pass"], None, 30, lane="igpu")
    with _mod._lock:
        assert "igpu" not in _mod._procs


def test_the_registry_is_empty_again_after_a_dispatch_is_killed():
    """The finally must clear the registry on the kill path too, not only the
    clean one. Not tested with a Popen that raises: that raises before the
    registration line ever runs, so such a test passes against an
    implementation with no finally at all."""
    _mod.run_dispatch(_sleeper(), None, 0.5, lane="igpu")
    with _mod._lock:
        assert "igpu" not in _mod._procs


def test_take_yield_reports_the_flag_once_and_then_clears_it():
    with _mod._lock:
        _mod._yielded.add("igpu")
    assert _mod._take_yield("igpu") is True
    assert _mod._take_yield("igpu") is False


def test_run_dispatch_without_a_lane_registers_nothing():
    """Every existing caller passes no lane. Their behaviour is unchanged."""
    _mod.run_dispatch([sys.executable, "-c", "pass"], None, 30)
    with _mod._lock:
        assert _mod._procs == {}


def test_a_timeout_still_reports_timed_out_and_not_a_yield():
    """A hung dispatch and a yielded one both end in a killed process group.
    They must not be confused: one spends an attempt, the other must not."""
    rc, timed_out = _mod.run_dispatch(_sleeper(), None, 0.5, lane="igpu")
    assert timed_out is True
    assert _mod._take_yield("igpu") is False


def _denoiser():
    return Denoiser(name="igpu", host="local", backend="migraphx", device=0,
                    tiling="none", enabled=True)


def _clip():
    return Clip("SetA/2001/f/c1.MOV", "SetA/2001/f", "c1", 1, 100)


def _encode():
    return Encoder(name="local", host="local", slots=2, lp_level=6,
                   port_base=5300)


def test_the_yield_flag_read_moved_into_the_finally_still_raises_on_a_failing_dispatch(tmp_path, monkeypatch):
    """make_runner has no unit test at all otherwise. This covers the read-once
    rule's first direction: the flag was set (as yield_lane does before its
    kill), the dispatch it landed on reports failure, and that must come back
    as YieldRequested -- not an ordinary failure tuple, which would spend one
    of the clip's two attempts on work the operator asked to abandon."""
    monkeypatch.setattr(_mod, "STAGE_ROOT", str(tmp_path / "stage"))
    monkeypatch.setattr(_mod, "REPO", str(tmp_path))
    monkeypatch.setattr(_mod, "run", lambda cmd, *a, **k: None)
    monkeypatch.setattr(_mod, "run_dispatch",
                        lambda argv, env, budget, lane=None: (1, False))

    denoiser, clip = _denoiser(), _clip()
    with _mod._lock:
        _mod._yielded.add(denoiser.name)

    runner = _mod.make_runner()
    with pytest.raises(_mod.YieldRequested):
        runner(clip, denoiser, _encode(), 0)
    assert _mod._take_yield(denoiser.name) is False, \
        "the flag must not survive to the lane's next clip"


def test_the_yield_flag_read_moved_into_the_finally_is_discarded_on_a_successful_dispatch(tmp_path, monkeypatch):
    """The other direction of the read-once rule: the flag was set, but the
    kill lost the race with a dispatch that was already finishing. The clip
    must publish normally, exactly as if the flag had never been set, and the
    flag must be gone afterward -- not returned to the caller, not left to
    misfire against the next clip on this lane."""
    monkeypatch.setattr(_mod, "STAGE_ROOT", str(tmp_path / "stage"))
    monkeypatch.setattr(_mod, "REPO", str(tmp_path))
    monkeypatch.setattr(_mod, "run", lambda cmd, *a, **k: None)

    def fake_dispatch(argv, env, budget, lane=None):
        out = argv[argv.index("-o") + 1]
        with open(out, "wb") as fh:
            fh.write(b"x")
        return 0, False

    monkeypatch.setattr(_mod, "run_dispatch", fake_dispatch)

    denoiser, clip = _denoiser(), _clip()
    with _mod._lock:
        _mod._yielded.add(denoiser.name)

    runner = _mod.make_runner()
    ok, wall_s, fps, out_bytes, reason, phases = runner(clip, denoiser, _encode(), 0)
    assert ok is True
    assert reason == ""
    assert _mod._take_yield(denoiser.name) is False, \
        "the flag must be consumed silently, not leaked to the next clip"


def test_the_yield_flag_is_still_read_when_stop_trace_raises_above_it(tmp_path, monkeypatch):
    """The discriminating case: the flag read must be the FIRST statement in
    the finally, not merely somewhere inside it. A read placed after
    stop_trace is skipped by the very exception it exists to survive -- on
    that ordering this test fails, because the flag is still set once the
    OSError has propagated out of the runner."""
    monkeypatch.setattr(_mod, "STAGE_ROOT", str(tmp_path / "stage"))
    monkeypatch.setattr(_mod, "REPO", str(tmp_path))
    monkeypatch.setattr(_mod, "run", lambda cmd, *a, **k: None)
    monkeypatch.setattr(_mod, "run_dispatch",
                        lambda argv, env, budget, lane=None: (1, False))
    monkeypatch.setattr(_mod, "start_trace",
                        lambda denoiser, clip, budget: "tracer-handle")

    def raising_stop_trace(_tracer):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_mod, "stop_trace", raising_stop_trace)

    denoiser, clip = _denoiser(), _clip()
    with _mod._lock:
        _mod._yielded.add(denoiser.name)

    runner = _mod.make_runner(trace=True)
    with pytest.raises(OSError):
        runner(clip, denoiser, _encode(), 0)
    assert _mod._take_yield(denoiser.name) is False, \
        "the flag must be consumed even when stop_trace raises above it"
