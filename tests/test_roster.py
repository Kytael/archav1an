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
lp_level = 6
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
lp_level = 6
"""


def _write(tmp_path, text):
    p = tmp_path / "denoisers.toml"
    p.write_text(text)
    return p


def test_loads_denoisers_and_encode_pool(tmp_path):
    r = load_roster(_write(tmp_path, GOOD))
    assert [d.name for d in r.denoisers] == ["igpu", "gpu1_4090"]
    assert r.encode.slots == 2 and r.encode.lp_level == 6


def test_enabled_filters_disabled_entries(tmp_path):
    r = load_roster(_write(tmp_path, ONE_DISABLED))
    assert [d.name for d in r.denoisers] == ["igpu", "gpu1_4090"]
    assert [d.name for d in r.enabled()] == ["gpu1_4090"]


def test_remote_denoiser_defaults_are_read(tmp_path):
    r = load_roster(_write(tmp_path, GOOD))
    remote = [d for d in r.denoisers if d.name == "gpu1_4090"][0]
    assert remote.is_remote and remote.port == 5300
    local = [d for d in r.denoisers if d.name == "igpu"][0]
    assert not local.is_remote


def test_rejects_a_core_count_in_lp_level(tmp_path):
    # The old key held a thread count. The encoder clamps 32 to level 6 and
    # only warns, so the roster has to reject it instead.
    text = GOOD.replace("lp_level = 6", "lp_level = 32")
    with pytest.raises(RosterError, match=r"level in \[0, 6\]"):
        load_roster(_write(tmp_path, text))


def test_lp_level_defaults_to_six(tmp_path):
    text = GOOD.replace("lp_level = 6\n", "")
    r = load_roster(_write(tmp_path, text))
    assert r.encode.lp_level == 6


def test_accepts_lp_level_zero(tmp_path):
    # 0 is "choose from the core count", not "absent".
    text = GOOD.replace("lp_level = 6", "lp_level = 0")
    r = load_roster(_write(tmp_path, text))
    assert r.encode.lp_level == 0


def test_rejects_duplicate_names(tmp_path):
    text = GOOD.replace('name = "gpu1_4090"', 'name = "igpu"')
    with pytest.raises(RosterError, match="duplicate"):
        load_roster(_write(tmp_path, text))


def test_rejects_remote_denoiser_without_port(tmp_path):
    text = GOOD.replace("port = 5300\n", "")
    with pytest.raises(RosterError, match="port"):
        load_roster(_write(tmp_path, text))


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
        load_roster(_write(tmp_path, text))


def test_rejects_roster_with_no_enabled_denoiser(tmp_path):
    text = GOOD.replace("enabled = true", "enabled = false")
    with pytest.raises(RosterError, match="no enabled"):
        load_roster(_write(tmp_path, text))


def test_missing_file_raises_roster_error(tmp_path):
    with pytest.raises(RosterError, match="not found"):
        load_roster(tmp_path / "absent.toml")


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
lp_level = 6
"""


def test_a_tiled_denoiser_is_accepted_now_that_windowing_exists(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_toml())
    roster = load_roster(p)
    assert roster.denoisers[0].tiling == "auto" and roster.denoisers[0].window == 750


def test_a_margin_below_the_models_context_is_rejected(tmp_path):
    """Gate 2 is only bit-identical because the margin exceeds the model's reach."""
    p = tmp_path / "r.toml"
    p.write_text(_toml(margin=8))
    with pytest.raises(RosterError, match="below the model"):
        load_roster(p)


def test_a_tiled_denoiser_without_a_window_is_rejected(tmp_path):
    """Tiling without a window buffers the whole clip: 335 GB for the longest."""
    p = tmp_path / "r.toml"
    p.write_text(_toml(window=0))
    with pytest.raises(RosterError, match="needs a window"):
        load_roster(p)


def test_a_window_without_tiling_is_rejected(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_toml(tiling="none", window=750))
    with pytest.raises(RosterError, match="without tiling"):
        load_roster(p)


def test_an_unknown_tiling_mode_is_rejected(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_toml(tiling="quarters"))
    with pytest.raises(RosterError, match="expected one of"):
        load_roster(p)


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
lp_level = 6
""")
    roster = load_roster(p)
    assert [d.name for d in roster.enabled()] == ["igpu"]


def test_the_example_roster_is_valid():
    """The shipped example must load, or it teaches the wrong schema."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "tools" / "archive_batch" / "denoisers.example.toml"
    roster = load_roster(p)
    assert [d.name for d in roster.denoisers] == ["gpu1_4090", "igpu", "2070s"]
    # Roster A is the shipped default; the 2070S lane is for when the 4090 is busy.
    assert [d.name for d in roster.enabled()] == ["gpu1_4090", "igpu"]
    tiled = roster.denoisers[2]
    assert tiled.tiling == "auto" and tiled.window == 750 and tiled.margin >= 16


def test_stage_source_defaults_off_so_gpu1_still_reads_in_place(tmp_path):
    r = load_roster(_write(tmp_path, GOOD))
    remote = [d for d in r.denoisers if d.name == "gpu1_4090"][0]
    assert remote.stage_source is False


def test_stage_source_is_read_for_a_remote_without_the_archive(tmp_path):
    text = GOOD.replace("port = 5300\nenabled = true",
                        "port = 5300\nstage_source = true\nenabled = true")
    r = load_roster(_write(tmp_path, text))
    remote = [d for d in r.denoisers if d.name == "gpu1_4090"][0]
    assert remote.stage_source is True


def test_rejects_stage_source_on_a_local_denoiser(tmp_path):
    text = GOOD.replace('backend = "migraphx"',
                        'backend = "migraphx"\nstage_source = true')
    with pytest.raises(RosterError, match="stage_source"):
        load_roster(_write(tmp_path, text))


def test_rejects_root_on_a_local_denoiser(tmp_path):
    text = GOOD.replace('backend = "migraphx"',
                        'backend = "migraphx"\nroot = "~/elsewhere"')
    with pytest.raises(RosterError, match="root"):
        load_roster(_write(tmp_path, text))
