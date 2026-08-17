import json
import os

from tools.archive_batch import heartbeat


def test_write_then_read_round_trips(tmp_path):
    heartbeat.write(str(tmp_path), lane="gpu1_4090",
                    src="SetB/2002/x/MVI_6077.MOV", frames=50000,
                    state="waiting_for_slot", started_at=1000.0,
                    batch_pid=4242, attempt=1, temp_dir="/tmp/t/MVI_6077")
    got = heartbeat.read_all(str(tmp_path))
    assert set(got) == {"gpu1_4090"}
    assert got["gpu1_4090"]["frames"] == 50000
    assert got["gpu1_4090"]["state"] == "waiting_for_slot"
    assert got["gpu1_4090"]["temp_dir"] == "/tmp/t/MVI_6077"


def test_clear_removes_the_lane(tmp_path):
    heartbeat.write(str(tmp_path), lane="igpu", src="a.MOV", frames=10,
                    state="working", started_at=1.0, batch_pid=1, attempt=1,
                    temp_dir="/tmp/t/a")
    heartbeat.clear(str(tmp_path), "igpu")
    assert heartbeat.read_all(str(tmp_path)) == {}


def test_clear_is_quiet_when_there_is_nothing_there(tmp_path):
    """Called from a finally block, so it must never raise on a lane that
    never got as far as writing one."""
    heartbeat.clear(str(tmp_path), "never-ran")


def test_a_reader_never_sees_a_half_written_file(tmp_path):
    """Written tmp-then-rename. A .tmp left behind must not parse as a lane."""
    heartbeat.write(str(tmp_path), lane="a", src="a.MOV", frames=1,
                    state="working", started_at=1.0, batch_pid=1, attempt=1,
                    temp_dir="/tmp/a")
    (tmp_path / "b.json.tmp").write_text("{ not json")
    assert set(heartbeat.read_all(str(tmp_path))) == {"a"}


def test_unreadable_json_is_skipped_not_raised(tmp_path):
    """A torn file must cost one lane's row, not the whole page."""
    (tmp_path / "broken.json").write_text("{ truncated")
    assert heartbeat.read_all(str(tmp_path)) == {}


def test_a_lane_name_with_a_slash_is_refused(tmp_path):
    """Lane names come from a TOML the operator edits. One with a slash would
    write outside the lanes directory."""
    try:
        heartbeat.write(str(tmp_path), lane="../escape", src="a.MOV", frames=1,
                        state="working", started_at=1.0, batch_pid=1, attempt=1,
                        temp_dir="/tmp/a")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a lane name with a separator")


def test_read_all_on_a_missing_directory_is_empty(tmp_path):
    assert heartbeat.read_all(str(tmp_path / "nope")) == {}
