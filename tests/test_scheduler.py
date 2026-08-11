import threading

from tools.archive_batch.manifest import Clip
from tools.archive_batch.roster import Denoiser, EncodePool, Roster
from tools.archive_batch.scheduler import Scheduler

ENCODE = EncodePool(host="local", slots=2, threads_per_slot=16)
D1 = Denoiser(name="a", host="local", backend="migraphx", device=0, tiling="none", enabled=True)
D2 = Denoiser(name="b", host="local", backend="migraphx", device=0, tiling="none", enabled=True)
D2_OFF = Denoiser(name="b", host="local", backend="migraphx", device=0,
                  tiling="none", enabled=False)


def _clips(n):
    return tuple(Clip(f"SetA/2001/f/c{i}.MOV", "SetA/2001/f", f"c{i}", 1, 100, 1.0)
                 for i in range(n))


def _roster(*denoisers):
    return Roster(denoisers=tuple(denoisers), encode=ENCODE)


def test_every_clip_is_processed_once(tmp_path):
    seen = []
    lock = threading.Lock()

    def runner(clip, denoiser):
        with lock:
            seen.append(clip.src)
        return True, 1.0, 100.0, 42

    s = Scheduler(_clips(6), lambda: _roster(D1, D2), runner,
                  state_path=tmp_path / "state.jsonl")
    s.run()
    assert sorted(seen) == sorted(c.src for c in _clips(6))


def test_results_are_written_to_state(tmp_path):
    from tools.archive_batch.state import load_state
    s = Scheduler(_clips(3), lambda: _roster(D1), lambda c, d: (True, 1.0, 9.0, 7),
                  state_path=tmp_path / "state.jsonl")
    s.run()
    assert len(load_state(tmp_path / "state.jsonl").done) == 3


def test_a_failure_is_recorded_and_does_not_stop_the_run(tmp_path):
    from tools.archive_batch.state import load_state
    def runner(clip, denoiser):
        return (clip.stem != "c1"), 1.0, 1.0, 0

    s = Scheduler(_clips(3), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s.run()
    st = load_state(tmp_path / "state.jsonl")
    assert len(st.done) == 2 and st.failures == {"SetA/2001/f/c1.MOV": 1}
    assert s.failed == 1


def test_a_denoiser_disabled_mid_run_stops_taking_work(tmp_path):
    """After the roster turns 'b' off, only 'a' may take further clips.

    The queue must still drain: disabling a denoiser stops it taking work, it
    does not abandon the run.
    """
    calls = {"n": 0}
    calls_lock = threading.Lock()

    def roster_fn():
        with calls_lock:
            calls["n"] += 1
            n = calls["n"]
        return _roster(D1, D2) if n <= 3 else _roster(D1, D2_OFF)

    used = []
    used_lock = threading.Lock()

    def runner(clip, denoiser):
        with used_lock:
            used.append(denoiser.name)
        return True, 1.0, 1.0, 1

    s = Scheduler(_clips(8), roster_fn, runner, state_path=tmp_path / "state.jsonl")
    s.run()
    assert len(used) == 8, "the queue must drain even after a denoiser is disabled"
    assert used.count("b") <= 3, "'b' must stop taking work once disabled"


def test_slots_cap_concurrency(tmp_path):
    live = {"now": 0, "max": 0}
    lock = threading.Lock()
    def runner(clip, denoiser):
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        threading.Event().wait(0.01)
        with lock:
            live["now"] -= 1
        return True, 1.0, 1.0, 1

    one_slot = Roster(denoisers=(D1, D2),
                      encode=EncodePool(host="local", slots=1, threads_per_slot=16))
    s = Scheduler(_clips(6), lambda: one_slot, runner, state_path=tmp_path / "state.jsonl")
    s.run()
    assert live["max"] == 1


def test_an_empty_queue_finishes_immediately(tmp_path):
    s = Scheduler((), lambda: _roster(D1), lambda c, d: (True, 1.0, 1.0, 1),
                  state_path=tmp_path / "state.jsonl")
    s.run()
    assert s.done == 0 and s.failed == 0
