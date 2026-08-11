import pytest
from tools.archive_batch.transfer import TransferError, publish_cmd, stage_cmd


def test_stage_pulls_from_the_source_host():
    cmd = stage_cmd("gpu1", "SetA/2003/event-b/x.MOV", "/repo/Temp/_stage/igpu")
    assert cmd[0] == "rsync"
    assert cmd[-2] == "gpu1:/mnt/media/dance/SetA/2003/event-b/x.MOV"
    assert cmd[-1] == "/repo/Temp/_stage/igpu/"


def test_publish_creates_the_remote_directory_first():
    cmd = publish_cmd("gpu1", "/repo/Temp/out/x-av1.mkv", "SetA/2003/event-b")
    joined = " ".join(cmd)
    assert "mkdir -p" in joined
    assert "/mnt/media/dance/encoded/SetA/2003/event-b" in joined
    assert cmd[-1].startswith("gpu1:")


def test_publish_target_is_the_mirrored_path():
    cmd = publish_cmd("gpu1", "/repo/Temp/out/x-av1.mkv", "SetA/2003/event-b")
    assert cmd[-1] == "gpu1:/mnt/media/dance/encoded/SetA/2003/event-b/"


def test_publish_rejects_an_absolute_relative_dir():
    with pytest.raises(TransferError, match="relative"):
        publish_cmd("gpu1", "/repo/out/x.mkv", "/SetA/2003")


def test_stage_rejects_an_absolute_source():
    with pytest.raises(TransferError, match="relative"):
        stage_cmd("gpu1", "/SetA/2003/x.MOV", "/repo/Temp/_stage/igpu")
