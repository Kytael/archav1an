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
    return tuple(Clip(f"SetA/2001/f/c{i}.MOV", "SetA/2001/f", f"c{i}", 1, 100)
                 for i in range(n))


def _roster(*denoisers):
    return Roster(denoisers=tuple(denoisers), encode=ENCODE)


def test_every_clip_is_processed_once(tmp_path):
    seen = []
    lock = threading.Lock()

    def runner(clip, denoiser):
        with lock:
            seen.append(clip.src)
        return True, 1.0, 100.0, 42, ""

    s = Scheduler(_clips(6), lambda: _roster(D1, D2), runner,
                  state_path=tmp_path / "state.jsonl")
    s.run()
    assert sorted(seen) == sorted(c.src for c in _clips(6))


def test_results_are_written_to_state(tmp_path):
    from tools.archive_batch.state import load_state
    s = Scheduler(_clips(3), lambda: _roster(D1), lambda c, d: (True, 1.0, 9.0, 7, ""),
                  state_path=tmp_path / "state.jsonl")
    s.run()
    assert len(load_state(tmp_path / "state.jsonl").done) == 3


def test_a_failure_is_recorded_and_does_not_stop_the_run(tmp_path):
    from tools.archive_batch.state import load_state
    def runner(clip, denoiser):
        return (clip.stem != "c1"), 1.0, 1.0, 0, "bad clip"

    s = Scheduler(_clips(3), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s.run()
    st = load_state(tmp_path / "state.jsonl")
    # c1 is retried in-run and fails again, which is MAX_ATTEMPTS. The other
    # two still finish: one bad clip must not stop the run.
    assert len(st.done) == 2 and st.failures == {"SetA/2001/f/c1.MOV": 2}
    assert s.failed == 2


def test_a_failed_clip_is_retried_within_the_same_run(tmp_path):
    """Spec 6: waiting for the next run means waiting days on a 15-day job."""
    from tools.archive_batch.state import load_state
    tries = {"n": 0}
    lock = threading.Lock()

    def runner(clip, denoiser):
        with lock:
            tries["n"] += 1
            first = tries["n"] == 1
        return (not first), 1.0, 1.0, 1, "" if not first else "transient"

    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s.run()
    assert tries["n"] == 2
    assert len(load_state(tmp_path / "state.jsonl").done) == 1


def test_the_retry_prefers_a_denoiser_that_has_not_failed_the_clip(tmp_path):
    """Some failures are specific to the device that took the clip."""
    used = {}
    lock = threading.Lock()

    def runner(clip, denoiser):
        with lock:
            used.setdefault(clip.src, []).append(denoiser.name)
            failed_here = denoiser.name == "a" and clip.stem == "c0"
        return (not failed_here), 1.0, 1.0, 1, "device specific"

    s = Scheduler(_clips(6), lambda: _roster(D1, D2), runner,
                  state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.01
    s.run()
    attempts = used["SetA/2001/f/c0.MOV"]
    assert len(attempts) == 2, "c0 must be retried"
    assert attempts[1] == "b", f"retry went back to a failing device: {attempts}"


def test_prior_failures_count_toward_the_attempt_ceiling(tmp_path):
    """A clip that already failed once gets one more try, not two."""
    tries = {"n": 0}
    lock = threading.Lock()

    def runner(clip, denoiser):
        with lock:
            tries["n"] += 1
        return False, 1.0, 1.0, 0, "still broken"

    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl",
                  prior_failures={"SetA/2001/f/c0.MOV": 1})
    s.run()
    assert tries["n"] == 1


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
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(8), roster_fn, runner, state_path=tmp_path / "state.jsonl")
    s.run()
    assert len(used) == 8, "the queue must drain even after a denoiser is disabled"
    assert used.count("b") <= 3, "'b' must stop taking work once disabled"


def test_a_re_enabled_denoiser_resumes_taking_work(tmp_path):
    """Gate 7: disabling is a pause, not a permanent exit."""
    calls = {"n": 0}
    calls_lock = threading.Lock()

    def roster_fn():
        with calls_lock:
            calls["n"] += 1
            n = calls["n"]
        # 'b' is off in the middle of the run, then comes back.
        return _roster(D1, D2_OFF) if 3 <= n <= 6 else _roster(D1, D2)

    used = []
    used_lock = threading.Lock()

    def runner(clip, denoiser):
        with used_lock:
            used.append(denoiser.name)
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(12), roster_fn, runner, state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.01
    s.run()
    assert len(used) == 12


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
        return True, 1.0, 1.0, 1, ""

    one_slot = Roster(denoisers=(D1, D2),
                      encode=EncodePool(host="local", slots=1, threads_per_slot=16))
    s = Scheduler(_clips(6), lambda: one_slot, runner, state_path=tmp_path / "state.jsonl")
    s.run()
    assert live["max"] == 1


def test_an_empty_queue_finishes_immediately(tmp_path):
    s = Scheduler((), lambda: _roster(D1), lambda c, d: (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    s.run()
    assert s.done == 0 and s.failed == 0


def test_a_source_host_outage_requeues_instead_of_failing_the_clip(tmp_path):
    """Spec 6: a Windows-update reboot must not fail 3,000 clips."""
    from tools.archive_batch.state import load_state
    from tools.archive_batch.transfer import TransferOutage

    def runner(clip, denoiser):
        raise TransferOutage("rsync failed (10): connection refused -- gave up")

    s = Scheduler(_clips(5), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s.run()
    st = load_state(tmp_path / "state.jsonl")
    # Nothing recorded at all: no clip spends an attempt on a host that was down.
    assert st.done == set() and st.failures == {}
    assert s.done == 0 and s.failed == 0
    # Every clip is still queued for the next run.
    assert s.queue.qsize() == 5


def test_a_raising_runner_does_not_strand_its_denoiser(tmp_path):
    """A worker that dies silently takes its GPU out of the run for good."""
    from tools.archive_batch.state import load_state
    calls = []

    def runner(clip, denoiser):
        calls.append(clip.src)
        if clip.stem == "c0":
            raise RuntimeError("engine build blew up")
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(4), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s.run()
    st = load_state(tmp_path / "state.jsonl")
    # 4 clips, and c0 raises on both of its attempts.
    assert len(calls) == 5                    # the one worker kept going
    assert len(st.done) == 3
    assert st.failures == {"SetA/2001/f/c0.MOV": 2}


def test_a_denoiser_disabled_at_startup_can_still_be_enabled_later(tmp_path):
    """Roster B -> roster A mid-run must actually put the device to work."""
    seen = []
    lock = threading.Lock()
    enabled = threading.Event()
    b_ran = threading.Event()

    def runner(clip, denoiser):
        with lock:
            seen.append(denoiser.name)
        if denoiser.name == "b":
            b_ran.set()
        else:
            # a's first clip enables b, then a blocks so it cannot drain the
            # queue on its own. b must wake from parked for this to return.
            enabled.set()
            b_ran.wait(2)
        return True, 1.0, 1.0, 1, ""

    def roster_fn():
        return _roster(D1, D2 if enabled.is_set() else D2_OFF)

    s = Scheduler(_clips(8), roster_fn, runner, state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.05
    s.run()
    assert len(seen) == 8
    assert b_ran.is_set(), "b never ran, so its worker was never created"
