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
enabled = true

[encode]
host = "local"
slots = 2
lp_level = 6
"""

ONLY_ENCODE = """
[encode]
host = "local"
slots = 2
lp_level = 6
"""


def _write(tmp_path, text):
    p = tmp_path / "denoisers.toml"
    p.write_text(text)
    return p


def test_loads_denoisers_and_encoders(tmp_path):
    r = load_roster(_write(tmp_path, GOOD))
    assert [d.name for d in r.denoisers] == ["igpu", "gpu1_4090"]
    assert r.encoders[0].slots == 2 and r.encoders[0].lp_level == 6


def test_enabled_filters_disabled_entries(tmp_path):
    r = load_roster(_write(tmp_path, ONE_DISABLED))
    assert [d.name for d in r.denoisers] == ["igpu", "gpu1_4090"]
    assert [d.name for d in r.enabled()] == ["gpu1_4090"]


def test_remote_denoiser_defaults_are_read(tmp_path):
    r = load_roster(_write(tmp_path, GOOD))
    remote = [d for d in r.denoisers if d.name == "gpu1_4090"][0]
    assert remote.is_remote
    local = [d for d in r.denoisers if d.name == "igpu"][0]
    assert not local.is_remote
    assert remote.margin == 32
    assert remote.tiling == "none"
    assert remote.window == 0
    assert remote.encoders == ()
    assert remote.stage_source is False


def test_rejects_a_core_count_in_lp_level(tmp_path):
    # The old key held a thread count. The encoder clamps 32 to level 6 and
    # only warns, so the roster has to reject it instead.
    text = GOOD.replace("lp_level = 6", "lp_level = 32")
    with pytest.raises(RosterError, match=r"level in \[0, 6\]"):
        load_roster(_write(tmp_path, text))


def test_lp_level_defaults_to_four(tmp_path):
    # 4, not 6: no denoise lane supplies frames fast enough to need a wider
    # encoder, and level 6 costs 4.8 GB a slot against 2.1 GB.
    text = GOOD.replace("lp_level = 6\n", "")
    r = load_roster(_write(tmp_path, text))
    assert r.encoders[0].lp_level == 4


def test_accepts_lp_level_zero(tmp_path):
    # 0 is "choose from the core count", not "absent".
    text = GOOD.replace("lp_level = 6", "lp_level = 0")
    r = load_roster(_write(tmp_path, text))
    assert r.encoders[0].lp_level == 0


def test_rejects_duplicate_names(tmp_path):
    text = GOOD.replace('name = "gpu1_4090"', 'name = "igpu"')
    with pytest.raises(RosterError, match="duplicate"):
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
    text = GOOD.replace('backend = "trt"',
                        'backend = "trt"\nstage_source = true')
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


def test_an_explicit_tile_is_accepted(tmp_path):
    """Safe since engines are named per shape; before, tiles clobbered one cache."""
    p = tmp_path / "r.toml"
    p.write_text(_toml(tiling="1112x992"))
    assert load_roster(p).denoisers[0].tiling == "1112x992"


def test_a_bare_square_tile_is_accepted(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_toml(tiling="512"))
    assert load_roster(p).denoisers[0].tiling == "512"


def test_a_malformed_tile_is_rejected(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_toml(tiling="huge"))
    with pytest.raises(RosterError, match="tiling"):
        load_roster(p)


def test_an_explicit_tile_reaches_the_dispatch_command():
    """TILE_FOR mapped only 'auto', so any explicit tile raised KeyError."""
    from tools.archive_batch.dispatch_cmd import build_command
    from tools.archive_batch.roster import Denoiser, Encoder
    d = Denoiser(name="t", host="local", backend="trt", device=0,
                 tiling="1112x992", window=750, margin=32, enabled=True)
    argv, _ = build_command(d, Encoder(name="local", host="local", slots=1,
                                       lp_level=6, port_base=5300), 0,
                            "in.mov", "out.mkv", None, "127.0.0.1")
    assert "--bsvd-tile" in argv
    assert argv[argv.index("--bsvd-tile") + 1] == "1112x992"


def test_a_roster_with_no_enabled_denoiser_loads(tmp_path):
    """Every lane off is a state the operator asks for, not a broken file.

    Yielding the last enabled lane parks the run until a lane comes back, so
    this roster has to be legal to every reader. While it raised, the batch
    live read and the daemon snapshot both turned it into a roster error --
    and the daemon one lights encode_roster_error, a live alert on the Pi.
    """
    text = ONE_DISABLED.replace("enabled = true", "enabled = false")
    r = load_roster(_write(tmp_path, text))
    assert [d.name for d in r.denoisers] == ["igpu", "gpu1_4090"]
    assert list(r.enabled()) == []


def test_rejects_a_roster_with_no_denoiser_at_all(tmp_path):
    """Zero ENABLED lanes parks the run. Zero lanes ends it: run() starts a
    worker per rostered name, so an empty roster starts none, prints that it
    could not start a worker, and returns -- with no lane left on the page to
    switch back on. Removing the last enabled lane used to be refused, which
    incidentally put this out of reach. It is reachable now, so it is pinned
    here directly."""
    text = ONLY_ENCODE
    with pytest.raises(RosterError, match="no denoiser"):
        load_roster(_write(tmp_path, text))


ENCODERS = """
[[denoiser]]
name = "igpu"
host = "local"
backend = "migraphx"
device = 0
tiling = "none"
enabled = true

[[encoder]]
name = "encoder-host"
host = "local"
port_base = 5300
slots = 6
lp_level = 6
enabled = true

[[encoder]]
name = "gpu2"
host = "gpu2"
root = "/home/user/reposetc/archav1an"
stream_ip = "10.0.0.17"
port_base = 5310
slots = 2
lp_level = 6
enabled = true
"""


def test_encoders_parse(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ENCODERS)
    roster = load_roster(str(p))
    assert [e.name for e in roster.encoders] == ["encoder-host", "gpu2"]
    assert roster.encoders[0].host == "local"
    assert roster.encoders[0].slots == 6
    assert roster.encoders[1].slots == 2
    assert roster.encoders[1].lp_level == 6
    assert roster.encoders[1].stream_ip == "10.0.0.17"
    assert roster.encoders[1].port_base == 5310
    assert roster.encoders[1].root == "/home/user/reposetc/archav1an"


def test_encoder_is_remote(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ENCODERS)
    roster = load_roster(str(p))
    assert roster.encoders[0].is_remote is False
    assert roster.encoders[1].is_remote is True


def test_encoder_port_for_slot(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ENCODERS)
    roster = load_roster(str(p))
    gpu2 = roster.encoders[1]
    assert gpu2.port_for(0) == 5310
    assert gpu2.port_for(1) == 5311


def test_enabled_encoders_filters(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ENCODERS.replace(
        'port_base = 5310\nslots = 2\nlp_level = 6\nenabled = true',
        'port_base = 5310\nslots = 2\nlp_level = 6\nenabled = false'))
    roster = load_roster(str(p))
    assert [e.name for e in roster.enabled_encoders()] == ["encoder-host"]


LEGACY_ENCODE = """
[[denoiser]]
name = "igpu"
host = "local"
backend = "migraphx"
device = 0
tiling = "none"
enabled = true

[encode]
host = "local"
slots = 2
lp_level = 6
"""


def test_legacy_encode_table_becomes_one_local_encoder(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(LEGACY_ENCODE)
    roster = load_roster(str(p))
    assert len(roster.encoders) == 1
    enc = roster.encoders[0]
    assert enc.name == "local"
    assert enc.host == "local"
    assert enc.slots == 2
    assert enc.lp_level == 6
    # No stream_ip: dispatch resolves the callback itself, exactly as it did
    # before this feature existed. port_base keeps the 5300 every pre-pool
    # roster used, so the gpu1 split lane still streams to the right port.
    assert enc.stream_ip == ""
    assert enc.port_base == 5300


def test_denoiser_port_is_refused_with_a_migration_message(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(LEGACY_ENCODE.replace(
        'tiling = "none"', 'tiling = "none"\nport = 5300'))
    with pytest.raises(RosterError) as e:
        load_roster(str(p))
    assert "igpu" in str(e.value)
    assert "port_base" in str(e.value)


def test_roster_with_no_encoder_is_refused(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text("""
[[denoiser]]
name = "igpu"
host = "local"
backend = "migraphx"
device = 0
tiling = "none"
enabled = true
""")
    with pytest.raises(RosterError, match="no encoder in roster"):
        load_roster(str(p))


def _with_encoder(body):
    return """
[[denoiser]]
name = "igpu"
host = "local"
backend = "migraphx"
device = 0
tiling = "none"
enabled = true
""" + body


def test_duplicate_encoder_name_is_refused(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "a"
host = "local"
port_base = 5300

[[encoder]]
name = "a"
host = "gpu2"
stream_ip = "10.0.0.17"
port_base = 5310
"""))
    with pytest.raises(RosterError, match="duplicate encoder name"):
        load_roster(str(p))


def test_overlapping_port_blocks_are_refused(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "a"
host = "gpu2"
stream_ip = "10.0.0.17"
port_base = 5310
slots = 4

[[encoder]]
name = "b"
host = "gpu4"
stream_ip = "10.0.0.14"
port_base = 5313
slots = 2
"""))
    with pytest.raises(RosterError, match="port block"):
        load_roster(str(p))


def test_remote_encoder_needs_stream_ip(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "gpu2"
host = "gpu2"
port_base = 5310
"""))
    with pytest.raises(RosterError, match="stream_ip"):
        load_roster(str(p))


def test_remote_encoder_needs_port_base(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "gpu2"
host = "gpu2"
stream_ip = "10.0.0.17"
"""))
    with pytest.raises(RosterError, match="port_base"):
        load_roster(str(p))


def test_stream_ip_and_stream_net_together_are_refused(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "gpu2"
host = "gpu2"
stream_ip = "10.0.0.17"
stream_net = "10.0.0.0/24"
port_base = 5310
"""))
    with pytest.raises(RosterError, match="both stream_ip and stream_net"):
        load_roster(str(p))


def test_stream_net_alone_is_accepted(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "gpu2"
host = "gpu2"
stream_net = "10.0.0.0/24"
port_base = 5310
"""))
    roster = load_roster(str(p))
    assert roster.encoders[0].stream_net == "10.0.0.0/24"
    assert roster.encoders[0].stream_ip == ""


def test_encoder_lp_level_out_of_range_is_refused(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "a"
host = "local"
port_base = 5300
lp_level = 7
"""))
    with pytest.raises(RosterError, match=r"lp_level"):
        load_roster(str(p))


def test_encoder_needs_at_least_one_slot(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "a"
host = "local"
port_base = 5300
slots = 0
"""))
    with pytest.raises(RosterError, match="slots"):
        load_roster(str(p))


def test_root_on_a_local_encoder_is_refused(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "a"
host = "local"
port_base = 5300
root = "/somewhere"
"""))
    with pytest.raises(RosterError, match="root"):
        load_roster(str(p))


ALLOWLIST = """
[[denoiser]]
name = "gpu1_4090"
host = "gpu1"
backend = "trt"
device = 0
tiling = "none"
encoders = ["encoder-host", "gpu4"]
enabled = true

[[encoder]]
name = "encoder-host"
host = "local"
port_base = 5300

[[encoder]]
name = "gpu4"
host = "gpu4"
stream_ip = "10.0.0.14"
port_base = 5320

[[encoder]]
name = "gpu2"
host = "gpu2"
stream_ip = "10.0.0.17"
port_base = 5310
"""


def test_allowlist_parses(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ALLOWLIST)
    roster = load_roster(str(p))
    assert roster.denoisers[0].encoders == ("encoder-host", "gpu4")


def test_absent_allowlist_means_every_encoder(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ALLOWLIST.replace('encoders = ["encoder-host", "gpu4"]\n', ""))
    roster = load_roster(str(p))
    assert roster.denoisers[0].encoders == ()


def test_allowlist_naming_an_unknown_encoder_is_refused(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ALLOWLIST.replace('"gpu4"]', '"nosuch"]'))
    with pytest.raises(RosterError, match="nosuch"):
        load_roster(str(p))


def test_allows_returns_the_allowed_encoders(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ALLOWLIST)
    roster = load_roster(str(p))
    d = roster.denoisers[0]
    assert d.allows("encoder-host") is True
    assert d.allows("gpu2") is False


def test_empty_allowlist_allows_everything(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ALLOWLIST.replace('encoders = ["encoder-host", "gpu4"]\n', ""))
    roster = load_roster(str(p))
    assert roster.denoisers[0].allows("gpu2") is True


def test_a_half_migrated_roster_is_refused(tmp_path):
    """[[encoder]] added but [encode] not yet deleted must not load quietly.

    Task 13 tells the operator to make exactly this edit by hand on the live
    file, so the half-done state is one that really occurs. Loading it silently
    drops the old slots and lp_level -- the same silent-wrong-value failure this
    module refuses for a denoiser's port.
    """
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "encoder-host"
host = "local"
port_base = 5300

[encode]
host = "local"
slots = 6
lp_level = 6
"""))
    with pytest.raises(RosterError, match="both"):
        load_roster(str(p))


def test_a_malformed_stream_net_is_refused_at_load(tmp_path):
    """Not at the first clip, three days in.

    netresolve parses this per clip. A typo that only raises there kills the
    lane long after the operator has stopped watching, and passes every smoke
    test in between.
    """
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "gpu2"
host = "gpu2"
stream_net = "10.0.0.0/64"
port_base = 5310
"""))
    with pytest.raises(RosterError, match="stream_net"):
        load_roster(str(p))


def test_a_port_outside_the_valid_range_is_refused(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "a"
host = "local"
port_base = 70000
"""))
    with pytest.raises(RosterError, match="port_base"):
        load_roster(str(p))


def test_port_blocks_must_be_ten_apart(tmp_path):
    """The reservation is ten wide, not `slots` wide.

    roster.py and the example roster both promise a slot-count change needs no
    firewall edit. That is only true if the next encoder starts ten ports
    further on: an encoder at 5306 loads clean beside one at 5300 with six
    slots, and collides the first time those slots go to seven.
    """
    p = tmp_path / "r.toml"
    p.write_text(_with_encoder("""
[[encoder]]
name = "encoder-host"
host = "local"
port_base = 5300
slots = 6

[[encoder]]
name = "spark3"
host = "spark3"
stream_ip = "10.0.0.170"
port_base = 5306
slots = 4
"""))
    with pytest.raises(RosterError, match="port block"):
        load_roster(str(p))


ALLOWLIST_ALL_OFF = ENCODERS.replace(
    'tiling = "none"\nenabled = true',
    'tiling = "none"\nenabled = true\nencoders = ["gpu2"]').replace(
    'port_base = 5310\nslots = 2\nlp_level = 6\nenabled = true',
    'port_base = 5310\nslots = 2\nlp_level = 6\nenabled = false')


def test_a_lane_allowed_only_on_disabled_encoders_is_refused(tmp_path):
    """A starved allowlist parks the run for ever and prints nothing, so catch
    it at load. Reachable with the shipped roster: gpu1_4090 names three
    encoders, and turning those three off wedges that lane while the rest of
    the fleet carries on and hides it."""
    p = tmp_path / "r.toml"
    p.write_text(ALLOWLIST_ALL_OFF)
    with pytest.raises(RosterError, match="igpu"):
        load_roster(str(p))
    with pytest.raises(RosterError, match="gpu2"):
        load_roster(str(p))


def test_every_encoder_off_is_still_accepted(tmp_path):
    """Distinct from the case above and deliberately allowed: every encoder off
    parks the run, which is a state the operator asks for and recovers from by
    switching one back on. Only a lane starved while the fleet still encodes is
    a roster nobody meant to write."""
    p = tmp_path / "r.toml"
    p.write_text(ALLOWLIST_ALL_OFF.replace(
        'port_base = 5300\nslots = 6\nlp_level = 6\nenabled = true',
        'port_base = 5300\nslots = 6\nlp_level = 6\nenabled = false'))
    roster = load_roster(str(p))
    assert roster.enabled_encoders() == ()


def test_a_disabled_lane_may_name_disabled_encoders(tmp_path):
    """A lane that is off takes no clip, so it starves nothing. Refusing it
    would reject a roster whose owner has simply turned a whole host off."""
    p = tmp_path / "r.toml"
    p.write_text(ALLOWLIST_ALL_OFF.replace(
        'tiling = "none"\nenabled = true\nencoders',
        'tiling = "none"\nenabled = false\nencoders'))
    roster = load_roster(str(p))
    assert roster.enabled() == ()
