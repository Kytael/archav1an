import pytest
from tools.archive_batch.roster import RosterError, load_roster

GOOD = """
[[denoiser]]
name = "igpu"
host = "local"
backend = "migraphx"
device = 0
tiling = "none"
enabled = true

[[denoiser]]
name = "gpu1_4090"
host = "gpu1"
backend = "trt"
device = 0
tiling = "none"
port = 5300
enabled = true

[encode]
host = "local"
slots = 2
threads_per_slot = 16
"""

ONE_DISABLED = """
[[denoiser]]
name = "igpu"
host = "local"
backend = "migraphx"
device = 0
tiling = "none"
enabled = false

[[denoiser]]
name = "gpu1_4090"
host = "gpu1"
backend = "trt"
device = 0
tiling = "none"
port = 5300
enabled = true

[encode]
host = "local"
slots = 2
threads_per_slot = 16
"""


def _write(tmp_path, text):
    p = tmp_path / "denoisers.toml"
    p.write_text(text)
    return p


def test_loads_denoisers_and_encode_pool(tmp_path):
    r = load_roster(_write(tmp_path, GOOD), core_count=32)
    assert [d.name for d in r.denoisers] == ["igpu", "gpu1_4090"]
    assert r.encode.slots == 2 and r.encode.threads_per_slot == 16


def test_enabled_filters_disabled_entries(tmp_path):
    r = load_roster(_write(tmp_path, ONE_DISABLED), core_count=32)
    assert [d.name for d in r.denoisers] == ["igpu", "gpu1_4090"]
    assert [d.name for d in r.enabled()] == ["gpu1_4090"]


def test_remote_denoiser_defaults_are_read(tmp_path):
    r = load_roster(_write(tmp_path, GOOD), core_count=32)
    remote = [d for d in r.denoisers if d.name == "gpu1_4090"][0]
    assert remote.is_remote and remote.port == 5300
    local = [d for d in r.denoisers if d.name == "igpu"][0]
    assert not local.is_remote


def test_rejects_thread_oversubscription(tmp_path):
    text = GOOD.replace("threads_per_slot = 16", "threads_per_slot = 32")
    with pytest.raises(RosterError, match="oversubscribe"):
        load_roster(_write(tmp_path, text), core_count=32)


def test_rejects_duplicate_names(tmp_path):
    text = GOOD.replace('name = "gpu1_4090"', 'name = "igpu"')
    with pytest.raises(RosterError, match="duplicate"):
        load_roster(_write(tmp_path, text), core_count=32)


def test_rejects_remote_denoiser_without_port(tmp_path):
    text = GOOD.replace("port = 5300\n", "")
    with pytest.raises(RosterError, match="port"):
        load_roster(_write(tmp_path, text), core_count=32)


def test_rejects_duplicate_ports(tmp_path):
    text = GOOD + """
[[denoiser]]
name = "gpu2"
host = "gpu2"
backend = "trt"
device = 0
tiling = "none"
port = 5300
enabled = true
"""
    with pytest.raises(RosterError, match="port"):
        load_roster(_write(tmp_path, text), core_count=32)


def test_rejects_roster_with_no_enabled_denoiser(tmp_path):
    text = GOOD.replace("enabled = true", "enabled = false")
    with pytest.raises(RosterError, match="no enabled"):
        load_roster(_write(tmp_path, text), core_count=32)


def test_missing_file_raises_roster_error(tmp_path):
    with pytest.raises(RosterError, match="not found"):
        load_roster(tmp_path / "absent.toml", core_count=32)
