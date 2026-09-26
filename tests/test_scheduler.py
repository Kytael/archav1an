import threading
import time
from dataclasses import replace

import pytest

from tools.archive_batch.manifest import Clip
from tools.archive_batch.netresolve import NoTrustedAddress
from tools.archive_batch.roster import Denoiser, Encoder, Roster, RosterError
from tools.archive_batch.scheduler import Scheduler

ENCODE = Encoder(name="local", host="local", slots=2, lp_level=6,
                 port_base=5300)
D1 = Denoiser(name="a", host="local", backend="migraphx", device=0, tiling="none", enabled=True)
D2 = Denoiser(name="b", host="local", backend="migraphx", device=0, tiling="none", enabled=True)
D2_OFF = Denoiser(name="b", host="local", backend="migraphx", device=0,
                  tiling="none", enabled=False)
D1_OFF = Denoiser(name="a", host="local", backend="migraphx", device=0,
                  tiling="none", enabled=False)


def _clips(n):
    return tuple(Clip(f"SetA/2001/f/c{i}.MOV", "SetA/2001/f", f"c{i}", 1, 100)
                 for i in range(n))


def _roster(*denoisers):
    return Roster(denoisers=tuple(denoisers), encoders=(ENCODE,))


def _broken_roster(where):
    def fn():
        raise RosterError(f"roster is not valid TOML: {where}")
    return fn


def test_every_clip_is_processed_once(tmp_path):
    seen = []
    lock = threading.Lock()

    def runner(clip, denoiser, encoder, slot):
        with lock:
            seen.append(clip.src)
        return True, 1.0, 100.0, 42, ""

    s = Scheduler(_clips(6), lambda: _roster(D1, D2), runner,
                  state_path=tmp_path / "state.jsonl")
    s.run()
    assert sorted(seen) == sorted(c.src for c in _clips(6))


def test_results_are_written_to_state(tmp_path):
    from tools.archive_batch.state import load_state
    s = Scheduler(_clips(3), lambda: _roster(D1), lambda c, d, e, s: (True, 1.0, 9.0, 7, ""),
                  state_path=tmp_path / "state.jsonl")
    s.run()
    assert len(load_state(tmp_path / "state.jsonl").done) == 3


def test_a_failure_is_recorded_and_does_not_stop_the_run(tmp_path):
    from tools.archive_batch.state import load_state
    def runner(clip, denoiser, encoder, slot):
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

    def runner(clip, denoiser, encoder, slot):
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

    def runner(clip, denoiser, encoder, slot):
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

    def runner(clip, denoiser, encoder, slot):
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

    def runner(clip, denoiser, encoder, slot):
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

    def runner(clip, denoiser, encoder, slot):
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
    def runner(clip, denoiser, encoder, slot):
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        threading.Event().wait(0.01)
        with lock:
            live["now"] -= 1
        return True, 1.0, 1.0, 1, ""

    one_slot = Roster(denoisers=(D1, D2),
                      encoders=(Encoder(name="local", host="local", slots=1,
                                        lp_level=6, port_base=5300),))
    s = Scheduler(_clips(6), lambda: one_slot, runner, state_path=tmp_path / "state.jsonl")
    s.run()
    assert live["max"] == 1


def test_an_empty_queue_finishes_immediately(tmp_path):
    s = Scheduler((), lambda: _roster(D1), lambda c, d, e, s: (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    s.run()
    assert s.done == 0 and s.failed == 0


def test_a_source_host_outage_requeues_instead_of_failing_the_clip(tmp_path):
    """Spec 6: a Windows-update reboot must not fail 3,000 clips."""
    from tools.archive_batch.state import load_state
    from tools.archive_batch.transfer import TransferOutage

    def runner(clip, denoiser, encoder, slot):
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

    def runner(clip, denoiser, encoder, slot):
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

    def runner(clip, denoiser, encoder, slot):
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


def test_a_heartbeat_names_the_clip_and_clears_when_done(tmp_path):
    from tools.archive_batch import heartbeat
    seen = {}

    def runner(clip, denoiser, encoder, slot):
        seen.update(heartbeat.read_all(str(tmp_path / "lanes")))
        return True, 1.0, 100.0, 42, ""

    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl",
                  lanes_dir=str(tmp_path / "lanes"))
    s.run()
    assert seen["a"]["src"] == "SetA/2001/f/c0.MOV"
    assert seen["a"]["frames"] == 100
    assert seen["a"]["state"] == "working"
    assert heartbeat.read_all(str(tmp_path / "lanes")) == {}, "not cleared"


def test_the_heartbeat_says_waiting_while_the_slot_is_held(tmp_path):
    """A lane blocked on an encode slot must not look like one that is
    denoising: its ETA would be wrong for the whole wait."""
    from tools.archive_batch import heartbeat
    lanes = str(tmp_path / "lanes")
    started = threading.Event()
    release = threading.Event()

    def runner(clip, denoiser, encoder, slot):
        # Whichever lane gets in first holds the slot. Blocking only lane a
        # would not be deterministic: nothing orders the lanes, so b can win
        # the slot, finish its instant clip and leave nothing contended.
        started.set()
        release.wait(5)
        return True, 1.0, 1.0, 1, ""

    # One slot, two clips, two lanes: the second lane to take a clip cannot
    # start until the first one finishes.
    roster = Roster(denoisers=(D1, D2),
                    encoders=(Encoder(name="local", host="local", slots=1,
                                      lp_level=4, port_base=5300),))
    s = Scheduler(_clips(2), lambda: roster, runner,
                  state_path=tmp_path / "state.jsonl", lanes_dir=lanes)
    t = threading.Thread(target=s.run)
    t.start()
    started.wait(5)
    # started only says one lane is inside the runner; the other may not have
    # reached its own heartbeat yet. Poll rather than read once, but stay
    # inside the runner so the state we catch is a lane parked behind the other.
    deadline = time.monotonic() + 5
    while True:
        parked = heartbeat.read_all(lanes)
        waiting = [r for r in parked.values() if r["state"] == "waiting_for_slot"]
        if waiting or time.monotonic() > deadline:
            break
        time.sleep(0.01)
    release.set()
    t.join(20)
    assert waiting, f"no lane reported waiting_for_slot: {parked}"


def test_a_raising_runner_still_clears_its_heartbeat(tmp_path):
    from tools.archive_batch import heartbeat

    def runner(clip, denoiser, encoder, slot):
        raise RuntimeError("boom")

    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl",
                  lanes_dir=str(tmp_path / "lanes"))
    s.run()
    assert heartbeat.read_all(str(tmp_path / "lanes")) == {}


def test_a_denoiser_added_mid_run_gets_a_worker(tmp_path):
    """Spec 5.1. Before this, a lane added to the roster mid-run got no thread,
    so adding a host meant restarting a run that can last fifteen days."""
    lanes = {"set": (D1,)}
    working = threading.Event()
    used = []
    lock = threading.Lock()

    def roster_fn():
        return _roster(*lanes["set"])

    def runner(clip, denoiser, encoder, slot):
        with lock:
            used.append(denoiser.name)
        working.set()
        time.sleep(0.05)
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(20), roster_fn, runner,
                  state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.05
    thread = threading.Thread(target=s.run, daemon=True)
    thread.start()
    assert working.wait(5), "the run never started"
    lanes["set"] = (D1, D2)             # the operator adds a lane mid-run
    thread.join(30)
    assert not thread.is_alive(), "the run did not finish"
    assert len(used) == 20
    assert "b" in used, "a lane added mid-run must get a worker"


def test_a_lane_removed_mid_run_still_drains_the_queue(tmp_path):
    """Spec 5.1: a deleted lane needs no code. Its worker's _current() returns
    None, so it parks exactly as a disabled one does."""
    lanes = {"set": (D1, D2)}
    used = []
    lock = threading.Lock()

    def roster_fn():
        return _roster(*lanes["set"])

    def runner(clip, denoiser, encoder, slot):
        with lock:
            used.append(denoiser.name)
            if len(used) == 2:
                lanes["set"] = (D1,)    # 'b' disappears from the file
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(8), roster_fn, runner,
                  state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.05
    s.run()
    assert len(used) == 8, "the queue must drain after a lane is removed"


def test_names_is_empty_when_the_roster_will_not_parse(tmp_path):
    """A broken roster must not change the worker set, in either direction.

    Tested directly rather than through run(). A Scheduler whose roster stops
    parsing while clips are still queued parks every worker BY DESIGN --
    _current() returns None and the worker waits -- so the queue never drains
    and run() never returns. A test that called run() here would hang the whole
    suite rather than fail.
    """
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")

    def broken():
        raise RosterError("roster is not valid TOML: line 3")

    s.roster_fn = broken
    assert s._names() == ()
    assert s._park_reason({"a"}) is None


def test_run_reports_loudly_when_the_first_roster_read_fails(tmp_path, capsys):
    """Review finding on Task 6: the roster is read once more at the top of
    run(), between construction (which already read it once, for the slot
    count) and here. _names() swallows a parse error on purpose -- that is
    what a mid-run re-read needs -- but that same swallowing must not turn a
    roster broken at startup into a silent no-op indistinguishable from a run
    that finished with nothing to do."""
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(3), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")

    def broken():
        raise RosterError("roster is not valid TOML: line 3")

    s.roster_fn = broken
    s.run()
    assert s.queue.qsize() == 3, "no clip may be lost"
    out = capsys.readouterr().out
    assert "roster could not be read" in out
    assert "3" in out, "the message must name the queue depth"


def test_a_broken_roster_is_recorded_not_swallowed(tmp_path):
    """Spec 5.5. This is a live defect independent of the UI: one typo in the
    TOML parks every lane for good and says nothing anywhere.

    Tested through the private methods rather than run(). A Scheduler whose
    roster stops parsing while clips are queued parks every worker BY DESIGN,
    so the queue never drains and run() never returns -- a test that called it
    here would hang the suite rather than fail."""
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    assert s.roster_error is None

    def broken():
        raise RosterError("roster is not valid TOML: line 3")

    s.roster_fn = broken
    assert s._names() == ()
    assert s.roster_error == "roster is not valid TOML: line 3"


def test_a_roster_that_parses_again_clears_the_error(tmp_path):
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s.roster_fn = _broken_roster("line 3")
    assert s._names() == ()
    assert s.roster_error is not None
    s.roster_fn = lambda: _roster(D1)
    assert s._names() == ["a"]
    assert s.roster_error is None


def test_the_roster_error_is_printed_once_per_distinct_message(capsys, tmp_path):
    """_read_roster runs on every worker's every loop. An unconditional print
    would be tens of thousands of identical lines a day, which is the same as
    no message at all."""
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s.roster_fn = _broken_roster("line 3")
    for _ in range(5):
        s._names()
        s._park_reason({"a"})
        s._current("a")
    out = capsys.readouterr().out
    assert out.count("line 3") == 1, out
    s.roster_fn = _broken_roster("line 9")
    s._names()
    out = capsys.readouterr().out
    assert out.count("line 9") == 1, out


def test_the_roster_error_reaches_a_file_the_daemon_can_read(tmp_path):
    """Spec 5.5. The daemon is a separate process, so a field on this object
    reaches it only through a file."""
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    where = tmp_path / "roster-error.txt"
    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl",
                  roster_error_path=str(where))
    s.roster_fn = _broken_roster("line 3")
    s._names()
    assert "line 3" in where.read_text()
    s.roster_fn = lambda: _roster(D1)
    s._names()
    assert not where.exists(), "a repaired roster must take its banner down"


def test_a_stale_roster_error_from_a_previous_run_is_cleared_at_startup(tmp_path):
    """_roster_ok early-returns when roster_error and _roster_said are both
    None, which is exactly the state of a freshly constructed Scheduler -- so
    without a startup clear, a file left behind by a run that broke or was
    killed survives a restart and shows a false banner forever."""
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    where = tmp_path / "roster-error.txt"
    where.write_text("stale: from a run that no longer exists\n")
    Scheduler(_clips(1), lambda: _roster(D1), runner,
              state_path=tmp_path / "state.jsonl", roster_error_path=str(where))
    assert not where.exists()


def test_every_roster_read_still_parks_rather_than_raising(tmp_path):
    """Behaviour is unchanged by spec 5.5: only the reporting is new. All four
    readers must keep returning their old fallbacks."""
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s.roster_fn = _broken_roster("line 3")
    assert s._names() == ()
    assert s._park_reason({"a"}) is None
    assert s._other_enabled("a") is False
    assert s._current("a") is None


def test_a_repaired_roster_file_stays_gone_under_a_late_publish(tmp_path):
    """Review finding on Task 3: _roster_broke and _roster_ok each decide under
    _lock and only then touch the file, so a write from a break that was
    preempted can land after a later recovery has already removed the file --
    and the ok-path guard would then never call _clear_roster_error() again.
    _publish_roster_error re-reads the live state under _roster_file_lock, so
    the file converges to "absent" no matter how many times it runs after the
    roster is healthy again."""
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    where = tmp_path / "roster-error.txt"
    s = Scheduler(_clips(1), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl",
                  roster_error_path=str(where))
    s.roster_fn = _broken_roster("line 3")
    s._names()
    assert where.exists()
    s.roster_fn = lambda: _roster(D1)
    s._names()
    assert not where.exists()
    # Simulate the late write this finding describes: a stray publish after
    # recovery, standing in for a break's I/O that was preempted past the ok.
    s._publish_roster_error()
    assert not where.exists(), "a second publish after recovery must not resurrect the file"


def test_a_yield_requeues_the_clip_and_spends_no_attempt(tmp_path):
    """Spec 5.3. The operator asked for a GPU back, not for the clip to be
    marked bad. Nothing about this clip may change except where it is."""
    from tools.archive_batch.control import YieldRequested
    state = tmp_path / "state.jsonl"
    calls = []

    def runner(clip, denoiser, encoder, slot):
        calls.append(clip.src)
        if len(calls) == 1:
            raise YieldRequested("gpu1_4090 was yielded")
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(1), lambda: _roster(D1), runner, state_path=state)
    s._process(_clips(1)[0], D1)
    assert s.queue.qsize() == 2, "the yielded clip must go back on the queue"
    assert s._attempts == {}, "a yield spends no attempt"
    assert s.failed == 0
    assert not state.exists(), "a yield writes nothing to state.jsonl"


def test_a_yield_does_not_stop_the_run(tmp_path):
    """The single most important difference from the TransferOutage arm it is
    modelled on. A transfer outage hits every lane through the one source host;
    a yield hits the lane the operator named and no other."""
    from tools.archive_batch.control import YieldRequested

    def runner(clip, denoiser, encoder, slot):
        raise YieldRequested("igpu was yielded")

    s = Scheduler(_clips(1), lambda: _roster(D1, D2), runner,
                  state_path=tmp_path / "state.jsonl")
    s._process(_clips(1)[0], D1)
    assert not s._stop.is_set(), "a yield must never stop the whole run"


def test_a_yielded_clip_is_taken_by_another_lane(tmp_path):
    """End to end through run(): one lane yields its clip, the other finishes
    every clip including that one, and the run still terminates."""
    from tools.archive_batch.control import YieldRequested
    yielded = {"once": True}
    lock = threading.Lock()
    used = []

    def runner(clip, denoiser, encoder, slot):
        with lock:
            if denoiser.name == "a" and yielded["once"]:
                yielded["once"] = False
                raise YieldRequested("a was yielded")
            used.append((clip.src, denoiser.name))
        time.sleep(0.01)
        return True, 1.0, 1.0, 1, ""

    s = Scheduler(_clips(6), lambda: _roster(D1, D2), runner,
                  state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.05
    thread = threading.Thread(target=s.run, daemon=True)
    thread.start()
    thread.join(30)
    assert not thread.is_alive(), "the run did not finish"
    assert len({src for src, _ in used}) == 6, "every clip must still be done"
    assert s.done == 6
    assert s.failed == 0


def test_retry_puts_an_exhausted_clip_back_and_forgets_its_history(tmp_path):
    """Spec 5.4, the in-run half. The state.jsonl record makes the NEXT run
    eligible; this makes this one eligible, which is the point of retrying
    during a fifteen-day job."""
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    clip = _clips(1)[0]
    s = Scheduler((), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s._attempts[clip.src] = 2
    s._failed_on[clip.src] = {"a"}
    s._bounced.add((clip.src, "a"))
    assert s.retry(clip) is True
    assert s.queue.get_nowait() is clip
    assert clip.src not in s._attempts
    assert clip.src not in s._failed_on
    assert (clip.src, "a") not in s._bounced


def test_retry_refuses_a_clip_that_is_already_queued(tmp_path):
    """The retry button stays live for two to three seconds after a click: the
    batch polls at 1s, the record then has to land, and the page only re-polls
    every 2s. A double-click inside that window would otherwise put the same
    Clip on the queue twice, two lanes would encode it, and both would publish
    to the same destination path at the same moment."""
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    clip = _clips(1)[0]
    s = Scheduler((), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    assert s.retry(clip) is True
    assert s.retry(clip) is False
    assert s.queue.qsize() == 1


def test_retry_refuses_a_clip_a_lane_is_already_encoding(tmp_path):
    """The same double-click, landing after a lane has taken the clip. The
    queue scan cannot see it, so _in_flight is what catches this one."""
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    clip = _clips(1)[0]
    s = Scheduler((), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s._in_flight.add(clip.src)
    assert s.retry(clip) is False
    assert s.queue.qsize() == 0


def test_a_lane_re_enabled_after_a_yield_takes_the_clip_back(tmp_path):
    """Spec 9 names this outright: enable-after-yield puts the clip back in
    that lane's hands. The yield must leave no residue that makes the lane
    refuse its own clip: no spent attempt, and no _bounced pair, which is what
    _should_bounce reads as "this lane already passed that clip on"."""
    from tools.archive_batch.control import YieldRequested
    once = {"yield": True}

    def runner(clip, denoiser, encoder, slot):
        if once["yield"]:
            once["yield"] = False
            raise YieldRequested("a was yielded")
        return True, 1.0, 1.0, 1, ""

    clip = _clips(1)[0]
    s = Scheduler((), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s._process(clip, D1)
    assert s.queue.get_nowait() is clip
    assert s._attempts == {}
    assert not any(b[0] == clip.src for b in s._bounced)
    # Re-enabled, the same lane may take it -- and does.
    assert s._should_bounce(clip, "a") is False
    s._process(clip, D1)
    assert s.done == 1
    assert s.failed == 0


def test_retry_does_not_clear_another_clips_bounce(tmp_path):
    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    a, b = _clips(2)
    s = Scheduler((), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    s._bounced.update({(a.src, "a"), (b.src, "a")})
    s.retry(a)
    assert (b.src, "a") in s._bounced


def test_a_runner_that_raises_past_except_exception_releases_the_clip(tmp_path):
    """SystemExit and KeyboardInterrupt do not reach `except Exception`, so
    nothing below the try/finally runs for that clip. The finally is the last
    code it gets: if the in-flight mark stays there, the clip is in neither
    the queue nor recoverable, and retry() refuses it for the rest of the run.
    """
    def runner(clip, denoiser, encoder, slot):
        raise SystemExit("the runner called sys.exit")

    clip = _clips(1)[0]
    s = Scheduler((), lambda: _roster(D1), runner,
                  state_path=tmp_path / "state.jsonl")
    with pytest.raises(SystemExit):
        s._process(clip, D1)
    assert s._in_flight == set(), "the mark must not outlive the clip"
    assert s.queue.qsize() == 0
    assert s.retry(clip) is True, "the operator must be able to requeue it"


def test_a_clip_is_in_flight_while_its_lane_waits_for_an_encode_slot(tmp_path):
    """The window this guards is not microseconds: the slot acquire can block
    for a whole clip, and the clip is already out of the queue throughout."""
    started = threading.Event()
    release = threading.Event()

    def runner(clip, denoiser, encoder, slot):
        started.set()
        release.wait(10)
        return True, 1.0, 1.0, 1, ""

    clip = _clips(1)[0]
    one_slot = Roster(denoisers=(D1,),
                       encoders=(Encoder(name="local", host="local", slots=1,
                                         lp_level=6, port_base=5300),))
    s = Scheduler((), lambda: one_slot, runner,
                  state_path=tmp_path / "state.jsonl")
    s._pool["local"].acquire()  # the only slot is taken, so _process will block
    t = threading.Thread(target=s._process, args=(clip, D1), daemon=True)
    t.start()
    try:
        # It cannot have reached the runner, but it must already be in flight.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and clip.src not in s._in_flight:
            time.sleep(0.01)
        assert clip.src in s._in_flight, "a clip waiting for a slot must be in flight"
        assert not started.is_set()
        assert s.retry(clip) is False
        s._pool["local"].release()
        assert started.wait(10), "the lane must proceed once the slot frees up"
    finally:
        # Always released, so a failed assertion above cannot hang the suite.
        release.set()
        t.join(10)
    assert not t.is_alive()
    assert clip.src not in s._in_flight


ALL_OFF = """
[[denoiser]]
name = "a"
host = "local"
backend = "migraphx"
device = 0
tiling = "none"
enabled = false

[[denoiser]]
name = "b"
host = "local"
backend = "migraphx"
device = 0
tiling = "none"
enabled = false

[encode]
host = "local"
slots = 2
lp_level = 6
"""


def test_a_roster_with_every_lane_disabled_raises_no_banner(tmp_path):
    """The banner guard, the batch half. Driven through the real load_roster,
    because that is where the refusal used to live.

    Yielding the last enabled lane writes exactly this file. While it was
    refused, the live read turned it into a parse failure and parked every lane
    behind a banner saying the roster could not be read -- when the file was
    precisely what the operator asked for. Every name must still come back, or
    no worker would exist to resume when a lane is switched on again.
    """
    from tools.archive_batch.roster import load_roster

    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, ""

    path = tmp_path / "denoisers.toml"
    path.write_text(ALL_OFF)
    where = tmp_path / "roster-error.txt"
    s = Scheduler(_clips(1), lambda: load_roster(str(path)), runner,
                  state_path=tmp_path / "state.jsonl",
                  roster_error_path=str(where))
    assert sorted(s._names()) == ["a", "b"]
    assert not where.exists(), "a deliberate all-off roster is not a parse error"


def test_a_run_with_every_lane_disabled_parks_and_resumes(tmp_path):
    """The claim the whole change rests on, pinned rather than argued.

    Yielding the last enabled lane leaves this roster. run() must start a
    worker per rostered name and park them all, rather than finding nothing
    enabled and ending the run -- and switching one lane back on must put it
    to work. This one builds the Roster directly, so it pins run() and not
    the validator; its neighbour above drives the same state through a real
    file and load_roster. Between them both halves are covered.
    """
    seen = []
    lock = threading.Lock()
    on = threading.Event()

    def runner(clip, denoiser, encoder, slot):
        with lock:
            seen.append(denoiser.name)
        return True, 1.0, 1.0, 1, ""

    def roster_fn():
        return _roster(D1 if on.is_set() else D1_OFF, D2_OFF)

    s = Scheduler(_clips(2), roster_fn, runner, state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.05
    t = threading.Thread(target=s.run, daemon=True)
    t.start()
    try:
        time.sleep(0.3)     # six polls at POLL_SECONDS
        with lock:
            assert seen == [], "a parked run must take no clip"
        assert t.is_alive(), "the run must wait for a lane, not end"
        assert s.queue.qsize() == 2
        on.set()
        t.join(10)
    finally:
        s.stop()
    assert not t.is_alive(), "the run did not finish after a lane came back"
    assert seen == ["a", "a"]


def test_a_lane_only_takes_an_allowed_encoder(tmp_path):
    """gpu1_4090 excludes gpu2, so it must never be handed gpu2."""
    from tools.archive_batch.roster import Denoiser, Encoder, Roster
    seen = []

    gpu1 = Denoiser(name="gpu1_4090", host="gpu1", backend="trt", device=0,
                     tiling="none", enabled=True, encoders=("encoder-host",))
    roster = Roster(
        denoisers=(gpu1,),
        encoders=(Encoder(name="encoder-host", host="local", slots=1),
                  Encoder(name="gpu2", host="gpu2", slots=1,
                          stream_ip="10.0.0.17", port_base=5310)))

    def runner(clip, denoiser, encoder, slot):
        seen.append(encoder.name)
        return True, 1.0, 1.0, 1, "", {}

    sched = Scheduler(clips=_clips(4), roster_fn=lambda: roster, runner=runner,
                      state_path=str(tmp_path / "s.jsonl"))
    sched.run()
    assert seen == ["encoder-host"] * 4


def test_slots_are_counted_per_encoder(tmp_path):
    """Two encoders of one slot each let two clips run at once, not one."""
    import threading
    from tools.archive_batch.roster import Denoiser, Encoder, Roster
    concurrent, peak = [0], [0]
    lock = threading.Lock()
    gate = threading.Barrier(2, timeout=10)

    roster = Roster(
        denoisers=(Denoiser(name="a", host="local", backend="trt", device=0,
                            tiling="none", enabled=True),
                   Denoiser(name="b", host="local", backend="trt", device=0,
                            tiling="none", enabled=True)),
        encoders=(Encoder(name="e1", host="local", slots=1),
                  Encoder(name="e2", host="local", slots=1)))

    def runner(clip, denoiser, encoder, slot):
        with lock:
            concurrent[0] += 1
            peak[0] = max(peak[0], concurrent[0])
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
        with lock:
            concurrent[0] -= 1
        return True, 1.0, 1.0, 1, "", {}

    sched = Scheduler(clips=_clips(2), roster_fn=lambda: roster, runner=runner,
                      state_path=str(tmp_path / "s.jsonl"))
    sched.run()
    assert peak[0] == 2


def test_the_encode_host_reaches_the_state_record(tmp_path):
    import json
    from tools.archive_batch.roster import Denoiser, Encoder, Roster
    roster = Roster(
        denoisers=(Denoiser(name="a", host="local", backend="trt", device=0,
                            tiling="none", enabled=True),),
        encoders=(Encoder(name="e1", host="local", slots=1),))

    def runner(clip, denoiser, encoder, slot):
        return True, 1.0, 1.0, 1, "", {}

    state = tmp_path / "s.jsonl"
    sched = Scheduler(clips=_clips(1), roster_fn=lambda: roster, runner=runner,
                      state_path=str(state))
    sched.run()
    row = json.loads(state.read_text().strip())
    assert row["encode_host"] == "e1"


def test_a_lane_waiting_for_a_slot_sleeps_instead_of_spinning(tmp_path):
    """_wake is set for the rest of the run once the queue drains and nothing
    ever clears it, so waiting on it there returns instantly and the picker
    polls flat out. encoder-host denoises AND encodes, so that burnt core is stolen
    from the very encode the lane is queued behind."""
    from tools.archive_batch.roster import Denoiser, Encoder, Roster
    polls = [0]
    lock = threading.Lock()
    released = threading.Event()

    d = Denoiser(name="a", host="local", backend="trt", device=0,
                 tiling="none", enabled=True)
    roster = Roster(denoisers=(d,),
                    encoders=(Encoder(name="e1", host="local", slots=1),))

    def runner(clip, denoiser, encoder, slot):
        released.set()
        return True, 1.0, 1.0, 1, "", {}

    s = Scheduler((), lambda: roster, runner,
                  state_path=tmp_path / "state.jsonl")
    real_read = s._read_roster

    def counting_read():
        with lock:
            polls[0] += 1
        return real_read()

    s._read_roster = counting_read
    s._pool["e1"].acquire()     # the only slot is taken, so the picker parks
    s._wake.set()               # what a worker does the moment the queue drains
    t = threading.Thread(target=s._process, args=(_clips(1)[0], d), daemon=True)
    t.start()
    time.sleep(1.0)
    s._pool["e1"].release()
    assert released.wait(10), "the lane must proceed once the slot frees up"
    t.join(10)
    # Four polls fit in a second at 0.25 s. Anything near a thousand is a spin.
    assert polls[0] < 20, f"the picker polled {polls[0]} times in one second"


# --- A pre-flight encoder refusal must cost the clip nothing ----------------

GPU2 = Encoder(name="gpu2", host="gpu2", slots=1, lp_level=4,
                stream_net="10.0.0.0/24", port_base=5310)
BOX = Encoder(name="encoder-host", host="local", slots=1, lp_level=4,
              port_base=5300)


# These lanes deliberately live on a host that runs no encoder, so roster
# order alone decides and the picker still reaches gpu2 first. They stopped
# being D1/D2 on 2026-09-02, when the picker gained a preference for the
# lane's own host: as host="local" lanes they went to BOX every time and the
# refusal these tests are about never happened.
PD1 = replace(D1, host="gpu1")
PD2 = replace(D2, host="gpu1")


def _pool_roster():
    """gpu2 first in roster order, so the picker always reaches for it first."""
    return Roster(denoisers=(PD1, PD2), encoders=(GPU2, BOX))


def _refusing_runner(seen, lock):
    def runner(clip, denoiser, encoder, slot):
        with lock:
            seen.append((clip.src, encoder.name))
        if encoder.name == "gpu2":
            raise NoTrustedAddress(
                f"encoder 'gpu2' holds no address in {GPU2.stream_net}")
        return True, 1.0, 10.0, 1, ""
    return runner


def test_an_encoder_refusal_does_not_spend_an_attempt(tmp_path):
    """Take the laptop off the LAN and the whole queue used to die in seconds.

    resolve_stream_ip raises before a single frame is staged, so the refusal
    says nothing about the clip. Recorded as a clip failure it was a failure
    in 0.3 s, and a fast-failing encoder is always the free one -- so the
    picker fed it every queued clip twice, which is MAX_ATTEMPTS, and
    pending_clips then dropped them from every future run.
    """
    from tools.archive_batch.state import load_state
    seen, lock = [], threading.Lock()
    s = Scheduler(_clips(3), _pool_roster, _refusing_runner(seen, lock),
                  state_path=tmp_path / "state.jsonl")
    s.run()

    assert s.failed == 0, f"a pre-flight refusal must not fail a clip: {s.failures}"
    assert s.done == 3
    state = load_state(tmp_path / "state.jsonl")
    assert len(state.done) == 3
    assert dict(state.failures) == {}, "nothing may reach state.jsonl"
    refusals = [x for x in seen if x[1] == "gpu2"]
    assert len(refusals) == 1, f"the dead encoder must be picked once: {refusals}"


def test_a_refusing_encoder_is_quarantined_and_announced_once(tmp_path, capsys):
    """One line when it goes, one when it comes back -- never one per clip.

    Six lanes against a 3000-clip queue would otherwise bury the console in
    the same sentence, which reads the same as silence.
    """
    seen, lock = [], threading.Lock()
    clock = [1000.0]
    s = Scheduler(_clips(3), _pool_roster, _refusing_runner(seen, lock),
                  state_path=tmp_path / "state.jsonl",
                  clock=lambda: clock[0])
    s.run()

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 1, lines
    assert "gpu2" in lines[0]
    assert s._is_quarantined("gpu2")
    assert not s._is_quarantined("encoder-host")


def test_a_quarantined_encoder_comes_back_by_itself(tmp_path, capsys):
    """gpu2 is a laptop. A host that walks off the LAN has to rejoin the pool
    on its own when it walks back, which is why this is a timer and not a
    disable for the rest of the run."""
    clock = [1000.0]
    s = Scheduler(_clips(1), _pool_roster, lambda *a: (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl",
                  clock=lambda: clock[0])
    s._quarantine_encoder("gpu2", "off the trusted network")
    capsys.readouterr()

    clock[0] += Scheduler.QUARANTINE_SECONDS - 1
    s._expire_quarantine()
    assert s._is_quarantined("gpu2"), "the clock has not run out yet"
    assert capsys.readouterr().out == ""

    clock[0] += 2
    s._expire_quarantine()
    assert not s._is_quarantined("gpu2")
    assert "gpu2" in capsys.readouterr().out
    s._expire_quarantine()
    assert capsys.readouterr().out == "", "the return is announced once"


def test_a_quarantined_encoder_is_skipped_by_the_picker(tmp_path):
    """The clip goes back on the queue, so without this it comes straight back
    to the same dead encoder: a hot loop, not a retry."""
    s = Scheduler(_clips(1), _pool_roster, lambda *a: (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    s._quarantine_encoder("gpu2", "off the trusted network")
    encoder, slot = s._take_slot(D1)
    assert encoder.name == "encoder-host"


# --- Slot accounting comes from the startup roster, not the live one --------

def test_shrinking_slots_mid_run_neither_drops_a_clip_nor_leaks_a_permit(tmp_path):
    """denoisers.example.toml promises the slot count is fixed at startup. It
    was fixed for the semaphore and live for the index, so lowering `slots`
    while a clip held one admitted a worker for which no index existed.
    _slot_index raised, and it raised inside _take_slot -- before the
    try/finally in _process that owns _release_slot. The clip was then in
    neither the queue nor the runner, stuck in _in_flight so retry() refused
    it, and its permit was gone for the rest of the run.
    """
    big = Encoder(name="e1", host="local", slots=3, lp_level=4, port_base=5300)
    small = Encoder(name="e1", host="local", slots=1, lp_level=4, port_base=5300)
    live = {"enc": big}
    first_in = threading.Event()
    release = threading.Event()
    done, lock = [], threading.Lock()

    def runner(clip, denoiser, encoder, slot):
        with lock:
            done.append(clip.src)
            first = len(done) == 1
        if first:
            first_in.set()
            release.wait(10)
        return True, 1.0, 10.0, 1, ""

    s = Scheduler(_clips(3),
                  lambda: Roster(denoisers=(D1, D2), encoders=(live["enc"],)),
                  runner, state_path=tmp_path / "state.jsonl")
    t = threading.Thread(target=s.run, daemon=True)
    t.start()
    assert first_in.wait(10)
    live["enc"] = small         # the operator edits the roster mid-run
    time.sleep(1.0)             # let the other worker reach the picker
    release.set()
    t.join(30)

    assert not t.is_alive(), "run() must return"
    assert sorted(done) == sorted(c.src for c in _clips(3))
    assert s._in_flight == set()
    assert s._pool["e1"]._value == 3, "every permit must come back"


def test_a_slot_is_released_when_the_index_cannot_be_handed_out(tmp_path):
    """Belt to the braces above. The window between the acquire and the return
    is the one place a permit can be taken and never given back, because
    _process's try/finally does not start until _take_slot returns."""
    roster = Roster(denoisers=(D1,), encoders=(ENCODE,))
    s = Scheduler(_clips(1), lambda: roster,
                  lambda *a: (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    before = s._pool["local"]._value

    def boom(encoder):
        raise RuntimeError("no free slot index")

    s._slot_index = boom
    with pytest.raises(RuntimeError):
        s._take_slot(D1)
    assert s._pool["local"]._value == before


def test_the_port_block_comes_from_the_startup_roster(tmp_path):
    """port_for read the live entry while _ports indexed by name, the identical
    split. Moving port_base mid-run then paid out a port from the new block
    against an index reserved in the old one, and two clips on one host can
    collide on a port -- the second binds onto the first's listener."""
    live = {"enc": Encoder(name="e1", host="local", slots=2, lp_level=4,
                           port_base=5300)}
    s = Scheduler(_clips(1),
                  lambda: Roster(denoisers=(D1,), encoders=(live["enc"],)),
                  lambda *a: (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    live["enc"] = Encoder(name="e1", host="local", slots=2, lp_level=4,
                          port_base=5400)
    encoder, slot = s._take_slot(D1)
    assert encoder.port_for(slot) == 5300


# --- A starved lane must say so, not park in silence ------------------------

D_PICKY = Denoiser(name="a", host="local", backend="migraphx", device=0,
                   tiling="none", enabled=True, encoders=("e2",))
E1 = Encoder(name="e1", host="local", slots=1, lp_level=4, port_base=5300)
E2_OFF = Encoder(name="e2", host="local", slots=1, lp_level=4, port_base=5310,
                 enabled=False)


def test_a_lane_starved_of_encoders_says_so_instead_of_parking_silently(tmp_path,
                                                                        capsys):
    """_take_slot loops on _stop.wait(0.25) for ever when nothing in the lane's
    allowlist is enabled, and the parked banner only ever inspected denoisers,
    so run() never returned and not one line was printed. Encoders are toggled
    from the dashboard mid-run, so the load-time check in roster.py cannot be
    the only guard."""
    roster = Roster(denoisers=(D_PICKY,), encoders=(E1, E2_OFF))
    s = Scheduler(_clips(1), lambda: roster,
                  lambda *a: (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.05
    t = threading.Thread(target=s.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 10
    out = ""
    while time.monotonic() < deadline and "encoder" not in out:
        out += capsys.readouterr().out
        time.sleep(0.05)
    s.stop()
    t.join(10)
    assert not t.is_alive()
    assert "encoder" in out, "a starved lane must not park in silence"
    assert "quarantine" not in out, out


def test_a_lane_starved_only_by_quarantine_reads_as_temporary(tmp_path):
    """Every allowed encoder disabled needs the operator; every allowed encoder
    quarantined clears itself in five minutes. Telling them apart is the
    difference between going to look at the dashboard and doing nothing."""
    roster = Roster(denoisers=(D_PICKY,),
                    encoders=(E1, replace(E2_OFF, enabled=True)))
    s = Scheduler(_clips(1), lambda: roster,
                  lambda *a: (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    assert s._park_reason({"a"}) is None
    s._quarantine_encoder("e2", "off the trusted network")
    reason = s._park_reason({"a"})
    assert reason and "quarantine" in reason


def test_the_parked_banner_still_names_a_fleet_with_every_lane_off(tmp_path):
    """The denoiser half of the banner is unchanged."""
    s = Scheduler(_clips(1), lambda: _roster(D1_OFF),
                  lambda *a: (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    reason = s._park_reason({"a"})
    assert reason and "every denoiser is disabled" in reason


def test_an_idle_encoder_is_preferred_over_a_second_slot_on_a_busy_one(tmp_path):
    """A slot is not a unit of capacity. Two clips on one encode host share
    that host's cores, so the picker must spend the fleet's hosts before it
    doubles up on any one of them.

    The first-free-in-roster-order picker failed this: `big` comes first and
    has a spare slot, so both clips landed on it and `small` was never asked.
    Measured on the 2026-08-26 gate run as gpu4 at load 18 feeding gpu1 at
    10.0 fps while gpu5 sat idle."""
    import threading
    from tools.archive_batch.roster import Denoiser, Encoder, Roster
    seen, lock = [], threading.Lock()
    both_running = threading.Barrier(2, timeout=10)

    roster = Roster(
        denoisers=(Denoiser(name="a", host="local", backend="trt", device=0,
                            tiling="none", enabled=True),
                   Denoiser(name="b", host="local", backend="trt", device=0,
                            tiling="none", enabled=True)),
        encoders=(Encoder(name="big", host="local", slots=2),
                  Encoder(name="small", host="s", slots=1,
                          stream_ip="10.0.0.9", port_base=5310)))

    def runner(clip, denoiser, encoder, slot):
        with lock:
            seen.append(encoder.name)
        # Hold both clips open at once, so the second one really is choosing
        # while the first still occupies its host.
        try:
            both_running.wait()
        except threading.BrokenBarrierError:
            pass
        return True, 1.0, 1.0, 1, "", {}

    sched = Scheduler(clips=_clips(2), roster_fn=lambda: roster, runner=runner,
                      state_path=str(tmp_path / "s.jsonl"))
    sched.run()
    assert sorted(seen) == ["big", "small"], seen


def test_a_busy_encoder_is_still_used_when_every_host_is_carrying_one(tmp_path):
    """The idle preference is a preference, not a reservation. Once every
    allowed host has a clip, a spare slot on a busy one is the right answer --
    otherwise the second pass would never run and lanes would park with
    capacity free."""
    import threading
    from tools.archive_batch.roster import Denoiser, Encoder, Roster
    seen, lock = [], threading.Lock()
    all_running = threading.Barrier(3, timeout=10)

    roster = Roster(
        denoisers=tuple(Denoiser(name=n, host="local", backend="trt", device=0,
                                 tiling="none", enabled=True)
                        for n in ("a", "b", "c")),
        encoders=(Encoder(name="big", host="local", slots=2),
                  Encoder(name="small", host="s", slots=1,
                          stream_ip="10.0.0.9", port_base=5310)))

    def runner(clip, denoiser, encoder, slot):
        with lock:
            seen.append(encoder.name)
        try:
            all_running.wait()
        except threading.BrokenBarrierError:
            pass
        return True, 1.0, 1.0, 1, "", {}

    sched = Scheduler(clips=_clips(3), roster_fn=lambda: roster, runner=runner,
                      state_path=str(tmp_path / "s.jsonl"))
    sched.run()
    assert sorted(seen) == ["big", "big", "small"], seen


def test_a_lane_prefers_the_encoder_on_its_own_host(tmp_path):
    """Encoding where the clip was denoised sends the y4m to a listener on the
    lane's own address, so it never reaches the LAN. Roster order would have
    sent this clip to `box`, which is first in the file and equally idle."""
    from tools.archive_batch.roster import Denoiser, Encoder, Roster
    seen = []

    roster = Roster(
        denoisers=(Denoiser(name="gpu4", host="s1", backend="trt", device=0,
                            tiling="none", enabled=True),),
        encoders=(Encoder(name="box", host="local", slots=1),
                  Encoder(name="own", host="s1", slots=1,
                          stream_ip="10.0.0.14", port_base=5320)))

    def runner(clip, denoiser, encoder, slot):
        seen.append(encoder.name)
        return True, 1.0, 1.0, 1, "", {}

    sched = Scheduler(clips=_clips(1), roster_fn=lambda: roster, runner=runner,
                      state_path=str(tmp_path / "s.jsonl"))
    sched.run()
    assert seen == ["own"], seen


def test_own_host_never_outranks_an_idle_host(tmp_path):
    """Own-host is a tiebreak inside the idle tier, not above it. Promoting it
    would put a box's second lane on that box's own second encode slot while
    another host sat empty -- the exact contention the idle rule removed."""
    import threading
    from tools.archive_batch.roster import Denoiser, Encoder, Roster
    seen, lock = [], threading.Lock()
    both_running = threading.Barrier(2, timeout=10)

    roster = Roster(
        denoisers=tuple(Denoiser(name=n, host="s1", backend="trt", device=0,
                                 tiling="none", enabled=True)
                        for n in ("a", "b")),
        # `own` is both the lanes' own host AND first in the file, so nothing
        # but the idle rule can send the second clip to `other`.
        encoders=(Encoder(name="own", host="s1", slots=2,
                          stream_ip="10.0.0.14", port_base=5320),
                  Encoder(name="other", host="s2", slots=1,
                          stream_ip="10.0.0.16", port_base=5330)))

    def runner(clip, denoiser, encoder, slot):
        with lock:
            seen.append(encoder.name)
        try:
            both_running.wait()
        except threading.BrokenBarrierError:
            pass
        return True, 1.0, 1.0, 1, "", {}

    sched = Scheduler(clips=_clips(2), roster_fn=lambda: roster, runner=runner,
                      state_path=str(tmp_path / "s.jsonl"))
    sched.run()
    assert sorted(seen) == ["other", "own"], seen
