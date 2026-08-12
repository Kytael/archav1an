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


def _toml(tiling="auto", window=750, margin=32, extra=""):
    return f"""
[[denoiser]]
name = "2070s"
host = "local"
backend = "trt"
device = 0
tiling = "{tiling}"
window = {window}
margin = {margin}
{extra}
[encode]
slots = 1
threads_per_slot = 16
"""


def test_a_tiled_denoiser_is_accepted_now_that_windowing_exists(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_toml())
    roster = load_roster(p, 32)
    assert roster.denoisers[0].tiling == "auto" and roster.denoisers[0].window == 750


def test_a_margin_below_the_models_context_is_rejected(tmp_path):
    """Gate 2 is only bit-identical because the margin exceeds the model's reach."""
    p = tmp_path / "r.toml"
    p.write_text(_toml(margin=8))
    with pytest.raises(RosterError, match="below the model"):
        load_roster(p, 32)


def test_a_tiled_denoiser_without_a_window_is_rejected(tmp_path):
    """Tiling without a window buffers the whole clip: 335 GB for the longest."""
    p = tmp_path / "r.toml"
    p.write_text(_toml(window=0))
    with pytest.raises(RosterError, match="needs a window"):
        load_roster(p, 32)


def test_a_window_without_tiling_is_rejected(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_toml(tiling="none", window=750))
    with pytest.raises(RosterError, match="without tiling"):
        load_roster(p, 32)


def test_an_unknown_tiling_mode_is_rejected(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_toml(tiling="quarters"))
    with pytest.raises(RosterError, match="expected one of"):
        load_roster(p, 32)


def test_a_disabled_entry_may_carry_windowing_keys_for_later(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text("""
[[denoiser]]
name = "igpu"
host = "local"
backend = "migraphx"

[[denoiser]]
name = "2070s"
host = "local"
backend = "trt"
device = 1
tiling = "auto"
window = 1500
enabled = false

[encode]
slots = 1
threads_per_slot = 16
""")
    roster = load_roster(p, 32)
    assert [d.name for d in roster.enabled()] == ["igpu"]


def test_the_example_roster_is_valid():
    """The shipped example must load, or it teaches the wrong schema."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "tools" / "archive_batch" / "denoisers.example.toml"
    roster = load_roster(p, 32)
    assert [d.name for d in roster.denoisers] == ["gpu1_4090", "igpu", "2070s"]
    # Roster A is the shipped default; the 2070S lane is for when the 4090 is busy.
    assert [d.name for d in roster.enabled()] == ["gpu1_4090", "igpu"]
    tiled = roster.denoisers[2]
    assert tiled.tiling == "auto" and tiled.window == 750 and tiled.margin >= 16
