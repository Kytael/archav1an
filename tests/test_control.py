import importlib.util
import os
import threading
import time
from pathlib import Path

import pytest

from tools.archive_batch import control
from tools.archive_batch.roster import Encoder, Roster
from tools.encode_dash import control as dash

_ROSTER = Roster(denoisers=(), encoders=(Encoder(name="local", host="local",
                                                 slots=2, lp_level=6,
                                                 port_base=5300),))

BATCH_PY = Path(__file__).resolve().parent.parent / "tools" / "archive-batch.py"
_spec = importlib.util.spec_from_file_location("archive_batch_main", BATCH_PY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_a_submitted_request_is_taken_once_and_then_gone(tmp_path):
    d = str(tmp_path / "control")
    rid = dash.submit(d, "yield", 1786934811.2, lane="gpu1_4090")
    taken = control.take(d)
    assert len(taken) == 1
    path, req = taken[0]
    assert req["id"] == rid
    assert req["action"] == "yield"
    assert req["lane"] == "gpu1_4090"
    assert req["requested_at"] == 1786934811.2
    control.drop(path)
    assert control.take(d) == []


def test_a_request_is_written_atomically(tmp_path):
    """A half-written file must never be taken. tmp-then-replace, like the
    heartbeat: the batch polls this directory from another process entirely,
    so a partial write is a real interleaving, not a theoretical one."""
    d = str(tmp_path / "control")
    dash.submit(d, "stop", 1.0)
    names = os.listdir(d)
    assert names == [n for n in names if not n.endswith(".tmp")]
    assert len(names) == 1


def test_an_unknown_action_is_refused_at_submit(tmp_path):
    d = str(tmp_path / "control")
    with pytest.raises(control.ControlError):
        dash.submit(d, "restart", 1.0)
    assert control.take(d) == []


def test_clear_removes_requests_but_keeps_the_ack_log(tmp_path):
    """Spec 4.2. Without the clear, a yield queued against a run that has since
    died fires into the next run, possibly days later. The ack log is history
    and must survive: it is the only record of what the last run was told."""
    d = str(tmp_path / "control")
    dash.submit(d, "stop", 1.0)
    dash.submit(d, "yield", 2.0, lane="igpu")
    control.ack(d, "old", True, "from the previous run", 3.0)
    assert control.clear(d) == 2
    assert control.take(d) == []
    assert len(control.read_acks(d)) == 1


def test_clear_on_a_directory_that_does_not_exist_is_not_an_error(tmp_path):
    assert control.clear(str(tmp_path / "never-made")) == 0


def test_a_torn_request_file_costs_one_row_not_the_whole_read(tmp_path):
    d = str(tmp_path / "control")
    good = dash.submit(d, "stop", 1.0)
    with open(os.path.join(d, "torn.json"), "w", encoding="utf-8") as fh:
        fh.write('{"id": "x", "action":')
    taken = control.take(d)
    assert [r["id"] for _, r in taken] == [good]


def test_an_ack_round_trips(tmp_path):
    d = str(tmp_path / "control")
    control.ack(d, "abc", True, "killed pgid 82977; clip requeued", 1786934811.9)
    rows = control.read_acks(d)
    assert rows == [{"id": "abc", "accepted": True, "at": 1786934811.9,
                     "note": "killed pgid 82977; clip requeued"}]


def test_read_acks_returns_the_tail_not_the_head(tmp_path):
    """A fifteen-day run appends to this file the whole time. Slicing from the
    front would freeze the page on the first twenty acks of the run."""
    d = str(tmp_path / "control")
    for i in range(30):
        control.ack(d, f"id{i}", True, f"note {i}", float(i))
    rows = control.read_acks(d, limit=5)
    assert [r["id"] for r in rows] == ["id25", "id26", "id27", "id28", "id29"]


def test_read_acks_survives_invalid_utf8_instead_of_500ing_the_metrics_endpoint(tmp_path):
    """The open() was guarded with except OSError, but `for line in fh` raises
    UnicodeDecodeError on invalid bytes -- a ValueError subclass, not an
    OSError. Unguarded, that escapes snapshot() and 500s /api/status and
    /metrics, which kills the Pi's Prometheus scrape."""
    d = str(tmp_path / "control")
    os.makedirs(d)
    with open(os.path.join(d, control.ACKS), "wb") as fh:
        fh.write(b'{"id": "a", "accepted": true, "at": 1.0, "note": "ok"}\n')
        fh.write(b"\xff\xfe not valid utf-8\n")
    assert control.read_acks(d) == []


def test_serve_dispatches_a_request_deletes_it_and_acks_it(tmp_path):
    d = str(tmp_path / "control")
    seen = []
    stop = threading.Event()

    def on_stop(req):
        seen.append(req["id"])
        return True, "stopping; 12 clip(s) still queued"

    t = threading.Thread(target=control.serve, daemon=True,
                         args=(d, {"stop": on_stop}, stop), kwargs={"poll_s": 0.01})
    t.start()
    rid = dash.submit(d, "stop", 1.0)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not control.read_acks(d):
        time.sleep(0.01)
    stop.set()
    t.join(5)
    assert not t.is_alive(), "serve did not return once its stop event was set"
    assert seen == [rid]
    assert control.take(d) == []
    acks = control.read_acks(d)
    assert acks[-1]["id"] == rid
    assert acks[-1]["accepted"] is True
    assert acks[-1]["note"] == "stopping; 12 clip(s) still queued"


def test_serve_rejects_an_unknown_action_rather_than_ignoring_it(tmp_path):
    """Spec 9. A request nobody handles must not sit in the directory forever
    being re-read once a second; it is answered and removed."""
    d = str(tmp_path / "control")
    stop = threading.Event()
    t = threading.Thread(target=control.serve, daemon=True,
                         args=(d, {}, stop), kwargs={"poll_s": 0.01})
    t.start()
    rid = dash.submit(d, "stop", 1.0)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not control.read_acks(d):
        time.sleep(0.01)
    stop.set()
    t.join(5)
    acks = control.read_acks(d)
    assert acks[-1]["id"] == rid
    assert acks[-1]["accepted"] is False
    assert "stop" in acks[-1]["note"]
    assert control.take(d) == []


def test_a_handler_that_raises_becomes_a_refused_ack_and_serve_keeps_going(tmp_path):
    """A handler that throws must not kill the poll loop: the loop is the only
    way to stop the run, so it has to outlive a bad request."""
    d = str(tmp_path / "control")
    stop = threading.Event()

    def boom(_req):
        raise RuntimeError("no such lane")

    t = threading.Thread(target=control.serve, daemon=True,
                         args=(d, {"yield": boom}, stop), kwargs={"poll_s": 0.01})
    t.start()
    dash.submit(d, "yield", 1.0, lane="ghost")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not control.read_acks(d):
        time.sleep(0.01)
    assert t.is_alive(), "serve died on a handler exception"
    stop.set()
    t.join(5)
    acks = control.read_acks(d)
    assert acks[-1]["accepted"] is False
    assert "no such lane" in acks[-1]["note"]


def test_requests_are_taken_in_the_order_they_were_written(tmp_path):
    """The id starts with the timestamp so a lexical sort is a time sort. Two
    yields and a stop must not arrive out of order."""
    d = str(tmp_path / "control")
    first = dash.submit(d, "yield", 1.0, lane="a")
    second = dash.submit(d, "yield", 2.0, lane="b")
    third = dash.submit(d, "stop", 3.0)
    assert [r["id"] for _, r in control.take(d)] == [first, second, third]


def test_the_stop_handler_stops_the_scheduler_gracefully(tmp_path):
    from tools.archive_batch.scheduler import Scheduler

    s = Scheduler((), lambda: _ROSTER, lambda c, d: None,
                  state_path=tmp_path / "state.jsonl")
    handlers = _mod.control_handlers(s, {})
    accepted, note = handlers["stop"]({"id": "x", "action": "stop"})
    assert accepted is True
    assert s._stop.is_set()
    assert "queued" in note


def test_the_retry_handler_writes_a_record_and_requeues_the_clip(tmp_path):
    from tools.archive_batch.manifest import Clip
    from tools.archive_batch.scheduler import Scheduler
    from tools.archive_batch.state import load_state

    state = tmp_path / "state.jsonl"
    clip = Clip("SetA/2001/a/x.MOV", "SetA/2001/a", "x", 1, 100)
    s = Scheduler((), lambda: _ROSTER, lambda c, d: None, state_path=state)
    s._attempts[clip.src] = 2
    handlers = _mod.control_handlers(s, {clip.src: clip}, state_path=str(state))
    accepted, note = handlers["retry"]({"id": "x", "action": "retry",
                                        "src": clip.src})
    assert accepted is True
    assert s.queue.get_nowait() is clip
    assert clip.src not in s._attempts
    assert load_state(state).failures == {clip.src: 0}
    assert "x.MOV" in note


def test_the_retry_handler_writes_no_record_when_the_requeue_is_refused(tmp_path):
    """A refused retry must leave state.jsonl alone. Writing the record anyway
    would reset the failure count for the next run on the strength of a button
    press that did nothing."""
    from tools.archive_batch.manifest import Clip
    from tools.archive_batch.scheduler import Scheduler

    state = tmp_path / "state.jsonl"
    clip = Clip("SetA/2001/a/x.MOV", "SetA/2001/a", "x", 1, 100)
    s = Scheduler((), lambda: _ROSTER, lambda c, d: None, state_path=state)
    s._in_flight.add(clip.src)
    handlers = _mod.control_handlers(s, {clip.src: clip}, state_path=str(state))
    accepted, note = handlers["retry"]({"id": "x", "action": "retry",
                                        "src": clip.src})
    assert accepted is False
    assert not state.exists()


def test_the_retry_handler_refuses_a_clip_the_manifest_does_not_hold(tmp_path):
    from tools.archive_batch.scheduler import Scheduler

    s = Scheduler((), lambda: _ROSTER, lambda c, d: None,
                  state_path=tmp_path / "state.jsonl")
    handlers = _mod.control_handlers(s, {}, state_path=str(tmp_path / "s.jsonl"))
    accepted, note = handlers["retry"]({"id": "x", "action": "retry",
                                        "src": "nowhere/y.MOV"})
    assert accepted is False
    assert "nowhere/y.MOV" in note
    assert s.queue.qsize() == 0


def test_the_retry_handler_refuses_a_clip_that_already_has_a_done_record(tmp_path):
    """by_src is built from the whole manifest (spec: a clip that failed twice
    this run is not in todo, but must still be a valid retry target), so a
    done clip is still in it and nothing in scheduler.retry consults the done
    set. Left unchecked, a stale tab or a direct curl spends up to three hours
    re-encoding a finished clip and republishes over the archive."""
    from tools.archive_batch.manifest import Clip
    from tools.archive_batch.scheduler import Scheduler
    from tools.archive_batch.state import Record, append_record, load_state

    state = tmp_path / "state.jsonl"
    clip = Clip("SetA/2001/a/x.MOV", "SetA/2001/a", "x", 1, 100)
    append_record(state, Record(clip.src, "done", "igpu", 1.0, 1.0, 10))
    before = state.read_text()
    s = Scheduler((), lambda: _ROSTER, lambda c, d: None, state_path=state)
    handlers = _mod.control_handlers(s, {clip.src: clip}, state_path=str(state))
    accepted, note = handlers["retry"]({"id": "x", "action": "retry",
                                        "src": clip.src})
    assert accepted is False
    assert "already finished" in note
    assert s.queue.qsize() == 0
    assert state.read_text() == before, "a refused retry must write no record"
    assert clip.src in load_state(state).done


def test_the_yield_handler_refuses_a_lane_the_roster_does_not_hold(tmp_path):
    from tools.archive_batch.scheduler import Scheduler

    s = Scheduler((), lambda: _ROSTER, lambda c, d: None,
                  state_path=tmp_path / "state.jsonl")
    handlers = _mod.control_handlers(s, {})
    accepted, note = handlers["yield"]({"id": "x", "action": "yield",
                                        "lane": "ghost"})
    assert accepted is False
    assert "ghost" in note


def test_the_yield_handler_accepts_a_rostered_lane_that_is_between_dispatches(tmp_path):
    """yield_lane returns False for this lane because it is staging,
    publishing or waiting on an encode slot, not because it does not exist.
    The daemon already disabled it before sending this request, so it stops
    at the clip boundary regardless -- reporting a refusal here would read as
    "the yield failed" and could push the operator to re-enable the lane,
    undoing the half that did work."""
    from tools.archive_batch.roster import Denoiser
    from tools.archive_batch.scheduler import Scheduler

    roster = Roster(denoisers=(Denoiser(name="igpu", host="local",
                                        backend="migraphx", device=0,
                                        tiling="none", enabled=True),),
                    encoders=_ROSTER.encoders)
    s = Scheduler((), lambda: roster, lambda c, d: None,
                  state_path=tmp_path / "state.jsonl")
    handlers = _mod.control_handlers(s, {})
    accepted, note = handlers["yield"]({"id": "x", "action": "yield",
                                        "lane": "igpu"})
    assert accepted is True
    assert "stops after the clip it holds" in note


def test_the_yield_handler_refuses_a_request_with_no_lane(tmp_path):
    from tools.archive_batch.scheduler import Scheduler

    s = Scheduler((), lambda: _ROSTER, lambda c, d: None,
                  state_path=tmp_path / "state.jsonl")
    handlers = _mod.control_handlers(s, {})
    accepted, note = handlers["yield"]({"id": "x", "action": "yield"})
    assert accepted is False


def test_the_handler_set_covers_every_action_the_vocabulary_names(tmp_path):
    """A handler missing here is a request the batch answers 'I do not handle
    that' to, which is worse than a crash because it looks deliberate."""
    from tools.archive_batch.scheduler import Scheduler

    s = Scheduler((), lambda: _ROSTER, lambda c, d: None, state_path=tmp_path / "state.jsonl")
    assert sorted(_mod.control_handlers(s, {})) == sorted(control.ACTIONS)


def test_take_action_removes_only_the_action_it_names(tmp_path):
    """clear() drops every pending request at startup, which is right for a
    yield or a stop: both name a live lane or a live run, and a stale one would
    fire into a run it was never meant for. A retry names neither -- it is an
    instruction about a clip's failure count, and it can ONLY be sent while the
    run is stopped, because a clip out of attempts is what empties the queue."""
    d = str(tmp_path / "control")
    for action in ("retry", "yield", "stop", "retry"):
        dash.submit(d, action, time.time(), src="a/clip.MOV")
    taken = control.take_action(d, "retry")
    assert [r["action"] for r in taken] == ["retry", "retry"]
    assert control.clear(d) == 2


def test_a_retry_queued_while_stopped_survives_into_the_next_run(tmp_path):
    """The deadlock this closes: pending_clips drops a clip at MAX_ATTEMPTS, so
    a run with nothing else left prints "nothing to do" and exits; the only
    thing that resets the count is a retry record, which on_retry writes from
    inside a RUNNING daemon. No retry without a run, no run without a retry --
    and clear() deleted the queued request on the way past."""
    from tools.archive_batch.state import (MAX_ATTEMPTS, Record, append_record,
                                           load_state, pending_clips)

    class _Clip:
        def __init__(self, src):
            self.src = src

    state_path = str(tmp_path / "state.jsonl")
    control_dir = str(tmp_path / "control")
    src = "SetB/2004/Post RCS/MVI_0277.MOV"
    for _ in range(MAX_ATTEMPTS):
        append_record(state_path, Record(src, "failed", "gpu2_5070", 1.0, 0.0, 0))
    clips = (_Clip(src),)
    assert pending_clips(clips, load_state(state_path)) == ()

    dash.submit(control_dir, "retry", time.time(), src=src)
    queued = control.take_action(control_dir, "retry")
    _mod._apply_queued_retries(queued, {src: clips[0]}, state_path, control_dir)

    assert pending_clips(clips, load_state(state_path)) == clips
    assert control.read_acks(control_dir)[-1]["accepted"] is True


def test_a_queued_retry_for_a_finished_clip_is_refused(tmp_path):
    """Same guard on_retry keeps: re-encoding a done clip overwrites a good
    output with an identical one at the cost of hours."""
    from tools.archive_batch.state import Record, append_record

    state_path = str(tmp_path / "state.jsonl")
    control_dir = str(tmp_path / "control")
    src = "SetB/2004/Post RCS/MVI_0277.MOV"
    append_record(state_path, Record(src, "done", "gpu2_5070", 1.0, 1.0, 10))
    dash.submit(control_dir, "retry", time.time(), src=src)
    queued = control.take_action(control_dir, "retry")
    _mod._apply_queued_retries(queued, {src: object()}, state_path, control_dir)

    ack = control.read_acks(control_dir)[-1]
    assert ack["accepted"] is False and "already finished" in ack["note"]
    assert "retry" not in open(state_path).read()


def test_retry_all_is_an_action_the_batch_answers(tmp_path):
    d = str(tmp_path / "control")
    dash.submit(d, "retry-all", time.time())
    assert [r["action"] for r in control.take_action(d, "retry-all")] == ["retry-all"]


def test_take_action_accepts_several_actions(tmp_path):
    """The startup drain has to lift both retry shapes out before clear()."""
    d = str(tmp_path / "control")
    dash.submit(d, "retry", time.time(), src="a/clip.MOV")
    dash.submit(d, "yield", time.time(), lane="2070s")
    dash.submit(d, "retry-all", time.time())
    taken = control.take_action(d, ("retry", "retry-all"))
    assert sorted(r["action"] for r in taken) == ["retry", "retry-all"]
    assert control.clear(d) == 1


def test_a_queued_retry_all_resets_every_exhausted_clip(tmp_path):
    """The panel carries a capped preview, so a mass retry driven from the
    rendered rows would silently skip every clip that did not fit. The count
    the operator sees says 29; the button has to mean those 29."""
    from tools.archive_batch.state import (MAX_ATTEMPTS, Record, append_record,
                                           load_state, pending_clips)

    class _Clip:
        def __init__(self, src):
            self.src = src

    state_path = str(tmp_path / "state.jsonl")
    control_dir = str(tmp_path / "control")
    dead = [f"SetB/2004/a/c{i}.MOV" for i in range(120)]
    alive = "SetB/2004/a/still-going.MOV"
    finished = "SetB/2004/a/finished.MOV"
    for src in dead:
        for _ in range(MAX_ATTEMPTS):
            append_record(state_path, Record(src, "failed", "gpu2_5070", 1.0, 0.0, 0))
    append_record(state_path, Record(alive, "failed", "gpu2_5070", 1.0, 0.0, 0))
    append_record(state_path, Record(finished, "done", "gpu2_5070", 1.0, 1.0, 9))
    clips = tuple(_Clip(s) for s in dead + [alive, finished])
    # Only `alive` is pending: the dead are out of attempts and `finished` is done.
    assert len(pending_clips(clips, load_state(state_path))) == 1

    dash.submit(control_dir, "retry-all", time.time())
    queued = control.take_action(control_dir, ("retry", "retry-all"))
    n = _mod._apply_queued_retries(queued, {c.src: c for c in clips},
                                   state_path, control_dir)

    assert n == 120
    # The one that still had an attempt is untouched, and so is the finished
    # one: a retry-all that reset those would be doing something nobody asked.
    assert len(pending_clips(clips, load_state(state_path))) == 121
    assert load_state(state_path).failures[alive] == 1
    assert control.read_acks(control_dir)[-1]["accepted"] is True


def test_the_startup_drain_lifts_both_retry_shapes(tmp_path):
    """The step main() performs, not the helper underneath it. Passing only
    "retry" here left every retry-all for clear() to sweep up as stale, so the
    button wrote a request, the run dropped it, and "nothing to do" was the
    only sign anything had happened."""
    d = str(tmp_path / "control")
    dash.submit(d, "retry", time.time(), src="a/clip.MOV")
    dash.submit(d, "retry-all", time.time())
    dash.submit(d, "yield", time.time(), lane="2070s")
    dash.submit(d, "stop", time.time())

    taken = _mod._take_queued_retries(d)

    assert sorted(r["action"] for r in taken) == ["retry", "retry-all"]
    # Only the two that name a live lane or a live run are left to be dropped.
    assert control.clear(d) == 2


def test_two_queued_retry_alls_do_not_count_the_same_clip_twice(tmp_path):
    """State is read once for the whole drain, so a second retry-all would
    re-list every clip the first one reset. The extra records are harmless -- a
    retry row re-zeroes an already-zero count -- but "58 clip(s) retried" for
    29 clips makes the number worthless."""
    from tools.archive_batch.state import MAX_ATTEMPTS, Record, append_record

    class _Clip:
        def __init__(self, src):
            self.src = src

    state_path = str(tmp_path / "state.jsonl")
    control_dir = str(tmp_path / "control")
    dead = [f"a/c{i}.MOV" for i in range(4)]
    for src in dead:
        for _ in range(MAX_ATTEMPTS):
            append_record(state_path, Record(src, "failed", "l", 1.0, 0.0, 0))
    dash.submit(control_dir, "retry-all", time.time())
    dash.submit(control_dir, "retry-all", time.time())
    # A single-clip retry for one already covered by the sweep counts once too.
    dash.submit(control_dir, "retry", time.time(), src=dead[0])

    queued = _mod._take_queued_retries(control_dir)
    n = _mod._apply_queued_retries(queued, {s: _Clip(s) for s in dead},
                                   state_path, control_dir)
    assert n == 4


# --- Folders queued while the run was stopped -------------------------------

def test_the_startup_drain_lifts_a_submission(tmp_path):
    """A submission names no live lane and no live run, so clear() dropping it
    was pure loss -- and it is the one request an operator can only send while
    the run is stopped, because a drained archive refuses to start."""
    d = str(tmp_path / "control")
    dash.submit(d, "submit", time.time(), path="/mnt/media/fresh",
                host="gpu1", dest="SetA/2026/New")
    dash.submit(d, "yield", time.time(), lane="2070s")
    dash.submit(d, "stop", time.time())

    taken = _mod._take_queued_submits(d)

    assert [r["action"] for r in taken] == ["submit"]
    # Only the two that name a live lane or a live run are left to be dropped.
    assert control.clear(d) == 2


def _probe_stub(rows):
    return lambda host, path: list(rows)


def test_a_folder_queued_while_stopped_becomes_this_run_s_jobs(tmp_path, monkeypatch):
    """Queue a folder, press Start, and the clips are encoded. Before this the
    poller that answers a submission started only after the "nothing to do"
    exit, so the request was swept away and that message was the only sign
    anything had happened."""
    control_dir = str(tmp_path / "control")
    monkeypatch.setattr(_mod, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(_mod, "ENCODE_MANIFEST",
                        str(tmp_path / "manifest-encode.tsv"))
    monkeypatch.setattr(_mod, "probe_folder",
                        _probe_stub([("a.MOV", 11, "30/1,300"),
                                     ("b.MP4", 22, "30/1,600")]))
    dash.submit(control_dir, "submit", time.time(), path="/mnt/media/fresh",
                host="gpu1", dest="SetA/2026/New",
                preset="run_linux_dance_HQ_crf27.sh")

    queued = _mod._take_queued_submits(control_dir)
    clips = _mod._apply_queued_submits(queued, (), control_dir)

    assert [c.src for c in clips] == ["/mnt/media/fresh/a.MOV",
                                      "/mnt/media/fresh/b.MP4"]
    for clip in clips:
        assert clip.is_encode_job and clip.src_host == "gpu1"
        assert clip.rel_dir == "SetA/2026/New"
        assert clip.preset == "run_linux_dance_HQ_crf27.sh"
    assert clips[1].frames == 600
    assert control.read_acks(control_dir)[-1]["accepted"] is True


def test_a_queued_submission_is_written_to_the_encode_manifest(tmp_path, monkeypatch):
    """The manifest is what a RESUMED run reads. Without the append, a folder
    queued while stopped would vanish on the next restart while state.jsonl
    still held its half-finished records."""
    from tools.archive_batch.manifest import parse_encode_manifest

    control_dir = str(tmp_path / "control")
    manifest = tmp_path / "manifest-encode.tsv"
    monkeypatch.setattr(_mod, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(_mod, "ENCODE_MANIFEST", str(manifest))
    monkeypatch.setattr(_mod, "probe_folder",
                        _probe_stub([("a.MOV", 11, "30/1,300")]))
    dash.submit(control_dir, "submit", time.time(), path="/mnt/media/fresh",
                host="gpu1", dest="SetA/2026/New")

    _mod._apply_queued_submits(_mod._take_queued_submits(control_dir), (),
                               control_dir)

    rows = parse_encode_manifest(manifest.read_text())
    assert [r.src for r in rows] == ["/mnt/media/fresh/a.MOV"]
    assert rows[0].rel_dir == "SetA/2026/New"


def test_a_folder_already_in_the_manifest_is_not_queued_twice(tmp_path, monkeypatch):
    """scheduler.submit refuses a duplicate mid-run for a reason that applies
    just as much here: two copies of one job are encoded twice and published to
    one destination path at the same time."""
    control_dir = str(tmp_path / "control")
    manifest = tmp_path / "manifest-encode.tsv"
    monkeypatch.setattr(_mod, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(_mod, "ENCODE_MANIFEST", str(manifest))
    monkeypatch.setattr(_mod, "probe_folder",
                        _probe_stub([("a.MOV", 11, "30/1,300"),
                                     ("b.MP4", 22, "30/1,600")]))
    dash.submit(control_dir, "submit", time.time(), path="/mnt/media/fresh",
                host="gpu1", dest="SetA/2026/New")

    clips = _mod._apply_queued_submits(_mod._take_queued_submits(control_dir),
                                       ("/mnt/media/fresh/a.MOV",), control_dir)

    assert [c.src for c in clips] == ["/mnt/media/fresh/b.MP4"]
    assert manifest.read_text().count("/mnt/media/fresh/a.MOV") == 0
    assert "1 already in the manifest" in control.read_acks(control_dir)[-1]["note"]


def test_two_queued_submissions_of_one_folder_queue_it_once(tmp_path, monkeypatch):
    """Two presses of the button, or a stale request beside a fresh one."""
    control_dir = str(tmp_path / "control")
    monkeypatch.setattr(_mod, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(_mod, "ENCODE_MANIFEST",
                        str(tmp_path / "manifest-encode.tsv"))
    monkeypatch.setattr(_mod, "probe_folder",
                        _probe_stub([("a.MOV", 11, "30/1,300")]))
    for _ in range(2):
        dash.submit(control_dir, "submit", time.time(), path="/mnt/media/fresh",
                    host="gpu1", dest="SetA/2026/New")

    clips = _mod._apply_queued_submits(_mod._take_queued_submits(control_dir),
                                       (), control_dir)

    assert [c.src for c in clips] == ["/mnt/media/fresh/a.MOV"]


def test_a_queued_submission_that_cannot_be_probed_is_refused_not_fatal(tmp_path, monkeypatch):
    """A folder that moved while the run was stopped costs an ack, not the
    start of the run: the archive clips behind it must still go."""
    from tools.archive_batch.transfer import TransferError

    control_dir = str(tmp_path / "control")
    monkeypatch.setattr(_mod, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(_mod, "ENCODE_MANIFEST",
                        str(tmp_path / "manifest-encode.tsv"))

    def _boom(host, path):
        raise TransferError(f"{path} is not a directory on {host}")

    monkeypatch.setattr(_mod, "probe_folder", _boom)
    dash.submit(control_dir, "submit", time.time(), path="/mnt/media/gone",
                host="gpu1", dest="SetA/2026/New")

    clips = _mod._apply_queued_submits(_mod._take_queued_submits(control_dir),
                                       (), control_dir)

    assert clips == ()
    ack = control.read_acks(control_dir)[-1]
    assert ack["accepted"] is False and "not a directory" in ack["note"]


def test_a_queued_submission_with_a_bad_destination_is_refused(tmp_path, monkeypatch):
    """safe_dest refuses an absolute path rather than stripping it: stripping
    would publish into encoded/mnt/media/dance without a word."""
    control_dir = str(tmp_path / "control")
    monkeypatch.setattr(_mod, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(_mod, "ENCODE_MANIFEST",
                        str(tmp_path / "manifest-encode.tsv"))
    monkeypatch.setattr(_mod, "probe_folder",
                        _probe_stub([("a.MOV", 11, "30/1,300")]))
    dash.submit(control_dir, "submit", time.time(), path="/mnt/media/fresh",
                host="gpu1", dest="/mnt/media/dance")

    clips = _mod._apply_queued_submits(_mod._take_queued_submits(control_dir),
                                       (), control_dir)

    assert clips == ()
    assert control.read_acks(control_dir)[-1]["accepted"] is False


def test_a_queued_submission_naming_an_unknown_preset_is_refused(tmp_path, monkeypatch):
    """Refused before the probe, as on_submit refuses it before the first job:
    a name refused now costs one ack, and later costs a staged copy apiece."""
    control_dir = str(tmp_path / "control")
    monkeypatch.setattr(_mod, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(_mod, "ENCODE_MANIFEST",
                        str(tmp_path / "manifest-encode.tsv"))
    monkeypatch.setattr(_mod, "probe_folder",
                        _probe_stub([("a.MOV", 11, "30/1,300")]))
    dash.submit(control_dir, "submit", time.time(), path="/mnt/media/fresh",
                host="gpu1", dest="SetA/2026/New",
                preset="run_linux_nope.sh")

    clips = _mod._apply_queued_submits(_mod._take_queued_submits(control_dir),
                                       (), control_dir)

    assert clips == ()
    assert control.read_acks(control_dir)[-1]["accepted"] is False
