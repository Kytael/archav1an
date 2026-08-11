from tools.archive_batch.state import Record, append_record, load_state, pending_clips
from tools.archive_batch.manifest import Clip


def _clip(name, frames=100):
    return Clip(f"SetA/2001/a/{name}.MOV", "SetA/2001/a", name, 1, frames)


def test_append_then_load_roundtrip(tmp_path):
    p = tmp_path / "state.jsonl"
    append_record(p, Record("SetA/2001/a/x.MOV", "done", "igpu", 12.5, 8.0, 4242))
    st = load_state(p)
    assert st.done == {"SetA/2001/a/x.MOV"}
    assert st.failures == {}


def test_load_state_missing_file_is_empty(tmp_path):
    st = load_state(tmp_path / "nope.jsonl")
    assert st.done == set() and st.failures == {}


def test_failures_are_counted(tmp_path):
    p = tmp_path / "state.jsonl"
    append_record(p, Record("a.MOV", "failed", "igpu", 1.0, 0.0, 0))
    append_record(p, Record("a.MOV", "failed", "2070s", 1.0, 0.0, 0))
    assert load_state(p).failures == {"a.MOV": 2}


def test_pending_excludes_done(tmp_path):
    p = tmp_path / "state.jsonl"
    append_record(p, Record("SetA/2001/a/x.MOV", "done", "igpu", 1.0, 1.0, 1))
    clips = (_clip("x"), _clip("y"))
    assert [c.stem for c in pending_clips(clips, load_state(p))] == ["y"]


def test_pending_retries_a_clip_that_failed_once(tmp_path):
    p = tmp_path / "state.jsonl"
    append_record(p, Record("SetA/2001/a/x.MOV", "failed", "igpu", 1.0, 0.0, 0))
    assert [c.stem for c in pending_clips((_clip("x"),), load_state(p))] == ["x"]


def test_pending_drops_a_clip_that_failed_twice(tmp_path):
    p = tmp_path / "state.jsonl"
    for lane in ("igpu", "2070s"):
        append_record(p, Record("SetA/2001/a/x.MOV", "failed", lane, 1.0, 0.0, 0))
    assert pending_clips((_clip("x"),), load_state(p)) == ()


def test_done_wins_over_an_earlier_failure(tmp_path):
    p = tmp_path / "state.jsonl"
    append_record(p, Record("SetA/2001/a/x.MOV", "failed", "igpu", 1.0, 0.0, 0))
    append_record(p, Record("SetA/2001/a/x.MOV", "done", "2070s", 1.0, 1.0, 1))
    assert pending_clips((_clip("x"),), load_state(p)) == ()


def test_corrupt_line_is_skipped(tmp_path):
    p = tmp_path / "state.jsonl"
    p.write_text('{"src": "a", "status": "done"}\nnot json\n')
    assert load_state(p).done == {"a"}
