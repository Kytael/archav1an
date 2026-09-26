import json
import os

import pytest

from tools.archive_batch.roster import RosterError, load_roster
from tools.encode_dash import rosterio

ROSTER = '''\
# The fleet's denoise lanes. Order does not matter.

[[denoiser]]
name    = "gpu1_4090"
host    = "gpu1"
backend = "trt"
device  = 0
tiling  = "none"
enabled = true      # the fast one

[[denoiser]]
name    = "igpu"
host    = "local"
backend = "migraphx"
device  = 0
tiling  = "none"
enabled = false

[encode]
host     = "local"
slots    = 6
lp_level = 4
'''


@pytest.fixture
def roster(tmp_path):
    p = tmp_path / "denoisers.toml"
    p.write_text(ROSTER)
    return str(p)


def test_rev_is_a_string_so_it_survives_json(roster):
    # The type is the guarantee. st_mtime_ns is around 1.79e18 and
    # Number.MAX_SAFE_INTEGER is 9.007e15, so a JSON number here is rounded by
    # every browser that parses it and echoed back changed -- which made every
    # write from the page a 409. No JavaScript test can catch a regression
    # here, so this one holds the line.
    r = rosterio.rev(roster)
    assert isinstance(r, str)
    st = os.stat(roster)
    assert r == f"{st.st_mtime_ns}-{st.st_size}"
    # And it survives a JSON round trip byte for byte.
    assert json.loads(json.dumps({"rev": r}))["rev"] == r


def test_a_rev_is_never_a_number_a_browser_would_round(roster):
    # Python's json reads an integer exactly, so no Python test can reproduce
    # the browser's failure directly. What it can pin down is the premise: the
    # mtime really is past the range a JavaScript Number holds exactly, so
    # shipping it as a JSON number would round it -- 1787015143623006887 came
    # back from Chrome as 1787015143623007000. What rev() ships instead is a
    # string, and a string has no such range.
    max_safe_integer = 2 ** 53 - 1
    assert os.stat(roster).st_mtime_ns > max_safe_integer
    assert isinstance(rosterio.rev(roster), str)


def test_rev_is_none_when_the_file_is_absent(tmp_path):
    assert rosterio.rev(str(tmp_path / "nope.toml")) is None


def test_set_enabled_flips_one_lane(roster):
    rosterio.set_enabled(roster, "igpu", True)
    lanes = {d.name: d.enabled for d in load_roster(roster).denoisers}
    assert lanes == {"gpu1_4090": True, "igpu": True}


def test_set_enabled_touches_no_other_line(roster):
    before = open(roster).read().splitlines()
    rosterio.set_enabled(roster, "igpu", True)
    after = open(roster).read().splitlines()
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1
    assert after[differing[0]].strip() == "enabled = true"
    # The comment block and the aligned '=' of every untouched line survive.
    assert after[0] == "# The fleet's denoise lanes. Order does not matter."
    assert 'name    = "gpu1_4090"' in after


def test_set_enabled_preserves_the_trailing_comment(roster):
    # gpu1_4090's enabled line carries an operator's note. Toggling the
    # switch must not erase it -- only the boolean is this function's
    # business, not the spacing or the comment that follows it.
    rosterio.set_enabled(roster, "gpu1_4090", True)
    lines = open(roster).read().splitlines()
    assert "enabled = true      # the fast one" in lines


def test_set_enabled_inserts_the_key_when_a_lane_never_set_it(tmp_path):
    p = tmp_path / "denoisers.toml"
    p.write_text('''\
[[denoiser]]
name    = "gpu1_4090"
host    = "gpu1"
backend = "trt"
device  = 0
tiling  = "none"

[[denoiser]]
name    = "igpu"
host    = "local"
backend = "migraphx"
device  = 0
tiling  = "none"
enabled = false

[encode]
host     = "local"
slots    = 6
lp_level = 4
''')
    path = str(p)
    # gpu1_4090 never sets `enabled`, relying on roster.py's default of True.
    # Setting it explicitly must insert a new line, not edit a missing one.
    rosterio.set_enabled(path, "gpu1_4090", True)
    lanes = {d.name: d.enabled for d in load_roster(path).denoisers}
    assert lanes == {"gpu1_4090": True, "igpu": False}
    assert "enabled = true" in open(path).read().splitlines()


def test_set_enabled_preserves_the_file_mode(roster):
    os.chmod(roster, 0o644)
    rosterio.set_enabled(roster, "igpu", True)
    assert os.stat(roster).st_mode & 0o777 == 0o644


def test_set_enabled_returns_the_new_rev(roster):
    out = rosterio.set_enabled(roster, "igpu", True)
    assert out == rosterio.rev(roster)


def test_set_enabled_rejects_an_unknown_lane(roster):
    with pytest.raises(rosterio.LaneNotFound):
        rosterio.set_enabled(roster, "nosuch", True)


def test_set_enabled_refuses_a_non_boolean(roster):
    # bool("false") is True in Python, so a caller that coerced instead of
    # checking would flip a lane the client asked to turn off.
    before = open(roster).read()
    with pytest.raises(RosterError):
        rosterio.set_enabled(roster, "igpu", "false")
    assert open(roster).read() == before


def test_set_enabled_refuses_a_stale_rev(roster):
    stale = rosterio.rev(roster)
    rosterio.set_enabled(roster, "igpu", True)
    with pytest.raises(rosterio.StaleRoster):
        rosterio.set_enabled(roster, "igpu", False, expect_rev=stale)


def test_set_enabled_accepts_a_current_rev(roster):
    rosterio.set_enabled(roster, "igpu", True, expect_rev=rosterio.rev(roster))
    assert load_roster(roster).denoisers[1].enabled is True


def test_disabling_the_last_enabled_lane_is_allowed(roster):
    """Yielding the last lane writes exactly this edit, so it has to land.

    The lane must stay in the file. It is the only switch left to turn the run
    back on with, and run() gives every rostered name a worker whether or not
    it is enabled, so the parked workers are still there waiting.
    """
    rosterio.set_enabled(roster, "gpu1_4090", False)
    r = load_roster(roster)
    assert list(r.enabled()) == []
    assert [d.name for d in r.denoisers] == ["gpu1_4090", "igpu"]


def test_a_refused_edit_leaves_no_temporary_file(roster):
    # An unknown tiling mode, because disabling the last enabled lane is no
    # longer refused. The directory is .archive-run, which the operator reads
    # by hand, so a rejected edit must leave nothing behind.
    d = os.path.dirname(roster)
    with pytest.raises(RosterError):
        rosterio.add_lane(roster, {"name": "gpu3", "host": "gpu3",
                                   "backend": "trt", "device": 0,
                                   "tiling": "nonsense", "window": 750,
                                   "margin": 32,
                                   "stage_source": True, "root": ""})
    assert os.listdir(d) == ["denoisers.toml"]


def test_add_lane_appends_a_block(roster):
    rosterio.add_lane(roster, {"name": "gpu3", "host": "gpu3",
                               "backend": "trt", "device": 0, "tiling": "auto",
                               "window": 750, "margin": 32,
                               "stage_source": True,
                               "root": "reposetc/archav1an"})
    names = [d.name for d in load_roster(roster).denoisers]
    assert names == ["gpu1_4090", "igpu", "gpu3"]


def test_an_added_lane_is_disabled(roster):
    rosterio.add_lane(roster, {"name": "gpu3", "host": "gpu3",
                               "backend": "trt", "device": 0,
                               "tiling": "none"})
    added = [d for d in load_roster(roster).denoisers if d.name == "gpu3"][0]
    assert added.enabled is False


def test_add_lane_keeps_the_encode_table_last(roster):
    rosterio.add_lane(roster, {"name": "gpu3", "host": "gpu3",
                               "backend": "trt", "device": 0,
                               "tiling": "none"})
    text = open(roster).read()
    assert text.index("[encode]") > text.index('name    = "gpu3"')
    assert load_roster(roster).encoders[0].slots == 6


def test_add_lane_keeps_one_blank_line_on_each_side_of_the_block(roster):
    # This module exists to preserve a file a person reads in vim, so an
    # appended block must be spaced like the ones already there: exactly one
    # blank line above it and one below, not two above and none below.
    rosterio.add_lane(roster, {"name": "gpu3", "host": "gpu3",
                               "backend": "trt", "device": 0,
                               "tiling": "none"})
    lines = open(roster).read().splitlines()
    head = lines.index("[[denoiser]]", lines.index('name    = "igpu"'))
    assert lines[head - 1] == ""
    assert lines[head - 2] != "", "two blank lines above the new block"
    tail = lines.index("[encode]")
    assert lines[tail - 1] == ""
    assert lines[tail - 2] == "enabled = false"


def test_add_lane_spaces_the_block_off_a_comment_above_it(tmp_path):
    # denoisers.example.toml's shape: a lane's extent runs to the next table
    # header, so the paragraph of comments documenting [encode] belongs to the
    # last lane and the insertion point lands right after a comment line. The
    # block must not be glued to it.
    p = tmp_path / "denoisers.toml"
    p.write_text('[[denoiser]]\nname    = "a"\nhost    = "local"\n'
                 'backend = "trt"\nenabled = true\n\n'
                 '# --- the encode pool ---\n'
                 '# slots is not just an encoder count.\n'
                 '[encode]\nhost = "local"\nslots = 6\nlp_level = 4\n')
    rosterio.add_lane(str(p), {"name": "b", "host": "local",
                               "backend": "trt", "device": 0, "tiling": "none"})
    lines = p.read_text().splitlines()
    head = lines.index("[[denoiser]]", 1)
    assert lines[head - 1] == ""
    assert lines[head - 2] == "# slots is not just an encoder count."
    assert lines[lines.index("[encode]") - 1] == ""
    assert load_roster(str(p)).encoders[0].slots == 6


def test_add_lane_refuses_a_duplicate_name(roster):
    with pytest.raises(RosterError):
        rosterio.add_lane(roster, {"name": "igpu", "host": "local",
                                   "backend": "trt", "device": 0,
                                   "tiling": "none"})


def test_remove_lane_deletes_only_its_block(roster):
    rosterio.remove_lane(roster, "igpu")
    r = load_roster(roster)
    assert [d.name for d in r.denoisers] == ["gpu1_4090"]
    # The encode table sits after the removed block and must survive it.
    assert r.encoders[0].slots == 6 and r.encoders[0].lp_level == 4
    assert open(roster).read().startswith("# The fleet's denoise lanes.")


def test_remove_lane_rejects_an_unknown_lane(roster):
    with pytest.raises(rosterio.LaneNotFound):
        rosterio.remove_lane(roster, "nosuch")


def test_removing_the_last_lane_of_all_is_refused(roster):
    """Zero enabled lanes parks the run. Zero lanes ends it: run() would start
    no worker at all, so nothing would be left to switch back on. Removing the
    last ENABLED lane is fine now, which is why this has to empty the roster to
    reach the refusal."""
    rosterio.remove_lane(roster, "igpu")
    before = open(roster).read()
    with pytest.raises(RosterError):
        rosterio.remove_lane(roster, "gpu1_4090")
    assert open(roster).read() == before


def test_add_and_remove_honour_the_rev(roster):
    stale = rosterio.rev(roster)
    rosterio.set_enabled(roster, "igpu", True)
    with pytest.raises(rosterio.StaleRoster):
        rosterio.remove_lane(roster, "igpu", expect_rev=stale)
    with pytest.raises(rosterio.StaleRoster):
        rosterio.add_lane(roster, {"name": "p", "host": "local",
                                   "backend": "trt", "device": 0,
                                   "tiling": "none"}, expect_rev=stale)


def test_set_lp_level_edits_the_encode_table(roster):
    rosterio.set_lp_level(roster, 6)
    assert load_roster(roster).encoders[0].lp_level == 6
    assert load_roster(roster).encoders[0].slots == 6


def test_set_lp_level_touches_one_line(roster):
    before = open(roster).read().splitlines()
    rosterio.set_lp_level(roster, 2)
    after = open(roster).read().splitlines()
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1
    assert after[differing[0]].strip() == "lp_level = 2"


def test_set_lp_level_refuses_a_level_above_six(roster):
    before = open(roster).read()
    with pytest.raises(RosterError) as e:
        rosterio.set_lp_level(roster, 7)
    assert "parallelism" in str(e.value)
    assert open(roster).read() == before


def test_set_lp_level_refuses_a_non_integer(roster):
    before = open(roster).read()
    with pytest.raises(RosterError):
        rosterio.set_lp_level(roster, "four")
    assert open(roster).read() == before


def test_set_lp_level_refuses_a_boolean(roster):
    # isinstance(True, int) is True in Python, so this needs its own check --
    # a bool must not slip through as a level.
    before = open(roster).read()
    with pytest.raises(RosterError):
        rosterio.set_lp_level(roster, True)
    assert open(roster).read() == before


def test_set_lp_level_adds_the_line_when_the_roster_omits_it(tmp_path):
    p = tmp_path / "denoisers.toml"
    p.write_text('[[denoiser]]\nname = "a"\nhost = "local"\n'
                 'backend = "trt"\nenabled = true\n\n'
                 '[encode]\nhost = "local"\nslots = 6\n')
    rosterio.set_lp_level(str(p), 3)
    assert load_roster(str(p)).encoders[0].lp_level == 3


def test_set_lp_level_honours_the_rev(roster):
    stale = rosterio.rev(roster)
    rosterio.set_lp_level(roster, 5)
    with pytest.raises(rosterio.StaleRoster):
        rosterio.set_lp_level(roster, 2, expect_rev=stale)


def test_set_lp_level_preserves_the_trailing_comment(tmp_path):
    # An operator note like "13 GB across the pool" must survive moving the
    # dropdown -- only the number is this function's business, not the
    # spacing or the comment that follows it.
    p = tmp_path / "denoisers.toml"
    p.write_text('[[denoiser]]\nname    = "a"\nhost    = "local"\n'
                 'backend = "trt"\nenabled = true\n\n'
                 '[encode]\nhost     = "local"\nslots    = 6\n'
                 'lp_level = 4   # 13 GB across the pool\n')
    rosterio.set_lp_level(str(p), 6)
    lines = p.read_text().splitlines()
    assert "lp_level = 6   # 13 GB across the pool" in lines


# ---------------------------------------------------------------- update_lane
#
# The page can now edit a lane in place, which is the first write here that
# touches an arbitrary key rather than one named line. The standard is the
# same as everywhere else in this module: change what was asked for and leave
# every other byte of a file somebody reads in vim exactly as it was.

EDITABLE = '''\
# The fleet's denoise lanes.

[[denoiser]]
name    = "gpu1_4090"
host    = "gpu1"
backend = "trt"
device  = 0
tiling  = "none"
margin  = 32        # the model's own context
enabled = true

# Why this lane is shaped the way it is. This paragraph is the reason the
# whole module exists, and no edit may disturb it.
[[denoiser]]
name    = "gpu2_5070"
host    = "gpu2"
backend = "trt"
device  = 0
tiling  = "auto"
window  = 300
margin  = 32
stage_source = true
root    = "/home/user/reposetc/archav1an"
enabled = false

[encode]
host     = "local"
slots    = 6
lp_level = 4
'''


@pytest.fixture
def editable(tmp_path):
    p = tmp_path / "denoisers.toml"
    p.write_text(EDITABLE)
    return str(p)


def _lane(path, name):
    return next(d for d in load_roster(path).denoisers if d.name == name)


def test_update_lane_changes_only_the_value_it_was_given(editable):
    rosterio.update_lane(editable, "gpu2_5070", {"window": 500})
    assert _lane(editable, "gpu2_5070").window == 500
    before = EDITABLE.splitlines()
    after = open(editable).read().splitlines()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after)
    assert changed == [("window  = 300", "window  = 500")]


def test_update_lane_keeps_the_alignment_and_the_inline_comment(editable):
    """This file aligns its '=' signs by hand and a person reads it in vim.
    A rewrite that normalised `margin  = 32` to `margin = 48` would break
    that column on every line the page ever touched."""
    rosterio.update_lane(editable, "gpu1_4090", {"margin": 48})
    assert "margin  = 48        # the model's own context" in \
        open(editable).read().splitlines()


def test_update_lane_inserts_a_key_the_block_does_not_set(editable):
    rosterio.update_lane(editable, "gpu1_4090",
                         {"root": "/home/user/archav1an"})
    assert _lane(editable, "gpu1_4090").root == "/home/user/archav1an"


def test_an_inserted_key_lands_in_the_file_s_own_order(editable):
    """FIELDS order, not the order the form's loop happened to run in. The
    roster is read by eye, and a lane whose keys are shuffled reads as a
    different lane from the one above it."""
    rosterio.update_lane(editable, "gpu1_4090",
                         {"root": "/home/user/archav1an", "margin": 16,
                          "window": 0})
    block = open(editable).read().split("[[denoiser]]")[1].splitlines()
    keys = [line.split("=")[0].strip() for line in block if "=" in line]
    assert keys == ["name", "host", "backend", "device", "tiling", "margin",
                    "root", "enabled"]


def test_an_empty_value_removes_the_key(editable):
    """The only way to put a lane back on a default. An absent key and a key
    set to "" are different things to roster.py, and the page has no other
    way to say the first."""
    rosterio.update_lane(editable, "gpu2_5070", {"root": ""})
    assert _lane(editable, "gpu2_5070").root == ""
    assert "root" not in open(editable).read().split("[[denoiser]]")[2]


def test_a_false_stage_source_removes_the_key(editable):
    rosterio.update_lane(editable, "gpu2_5070", {"stage_source": False})
    assert _lane(editable, "gpu2_5070").stage_source is False
    assert "stage_source" not in open(editable).read()


def test_update_lane_leaves_every_other_block_and_comment_alone(editable):
    rosterio.update_lane(editable, "gpu2_5070", {"window": 500, "margin": 48})
    text = open(editable).read()
    assert "# Why this lane is shaped the way it is." in text
    assert "margin  = 32        # the model's own context" in text
    assert _lane(editable, "gpu1_4090").margin == 32
    assert text.count("[[denoiser]]") == 2


def test_update_lane_does_not_touch_enabled(editable):
    """`enabled` is not in FIELDS, so the edit form cannot reach it. The
    switch owns it: an edit that carried it would let a save silently undo a
    toggle made while the form was open."""
    with pytest.raises(RosterError):
        rosterio.update_lane(editable, "gpu2_5070", {"enabled": True})
    assert _lane(editable, "gpu2_5070").enabled is False


def test_update_lane_refuses_a_rename(editable):
    """The name keys the worker's heartbeat, dispatch's --temp-tag and the
    scheduler's in-flight set. A rename mid-run orphans all three while the
    lane keeps working."""
    with pytest.raises(RosterError, match="cannot be renamed"):
        rosterio.update_lane(editable, "gpu2_5070", {"name": "gpu2"})
    assert _lane(editable, "gpu2_5070").name == "gpu2_5070"


def test_update_lane_accepts_the_name_it_already_has(editable):
    """The form posts every field, name included, because an edit has to be
    able to clear one. Sending the unchanged name is not a rename."""
    rosterio.update_lane(editable, "gpu2_5070",
                         {"name": "gpu2_5070", "window": 500})
    assert _lane(editable, "gpu2_5070").window == 500


def test_update_lane_refuses_an_unknown_key(editable):
    with pytest.raises(RosterError, match="unknown denoiser key"):
        rosterio.update_lane(editable, "gpu2_5070", {"gpu": "4090"})


def test_update_lane_refuses_an_edit_that_would_not_load(editable):
    """The validator has the last word, exactly as it does for an add, and
    the file is left untouched."""
    before = open(editable).read()
    with pytest.raises(RosterError, match="needs a window"):
        rosterio.update_lane(editable, "gpu2_5070", {"window": ""})
    assert open(editable).read() == before


def test_update_lane_refuses_an_unknown_lane(editable):
    with pytest.raises(rosterio.LaneNotFound):
        rosterio.update_lane(editable, "nope", {"window": 500})


def test_update_lane_honours_the_rev(editable):
    stale = rosterio.rev(editable)
    rosterio.update_lane(editable, "gpu2_5070", {"window": 500})
    with pytest.raises(rosterio.StaleRoster):
        rosterio.update_lane(editable, "gpu2_5070", {"window": 750},
                             expect_rev=stale)


# ------------------------------------------------ the write path and the read
#
# rosterio writes the SAME file the batch reads, so a key the loader refuses is
# not a cosmetic fault: the next clip boundary parks every lane. These pin the
# two paths together rather than trusting them to stay in step.

ROSTER_WITH_ENCODER = '''\
[[denoiser]]
name    = "igpu"
host    = "local"
backend = "migraphx"
tiling  = "none"
enabled = true

[[encoder]]
name      = "encoder-host"
host      = "local"
port_base = 5300
'''


def test_added_lane_is_loadable_by_the_roster_parser(tmp_path):
    """Every writable field at once, then the real validator.

    rosterio kept offering `port` after roster.py stopped accepting it on a
    [[denoiser]], so a dashboard add produced a file the batch then refused --
    and it is the same file, not a copy. Driven off FIELDS rather than a fixed
    dict so a key added to the form has to prove it round-trips too.
    """
    values = {"name": "newlane", "host": "gpu4", "backend": "trt",
              "device": 0, "tiling": "auto", "window": 300, "margin": 32,
              "stage_source": True,
              "root": "/home/user/reposetc/ubuntav1an",
              # ROSTER_WITH_ENCODER's pool is one host named "encoder-host", and
              # roster.py checks an allowlist's names against that table -- so
              # this proves the routing key round-trips through the real
              # validator, not just through the writer.
              "encoders": ["encoder-host"]}
    missing = set(rosterio.FIELDS) - set(values)
    assert not missing, \
        f"the form offers {sorted(missing)}, which this test has no value for"
    p = tmp_path / "denoisers.toml"
    p.write_text(ROSTER_WITH_ENCODER)
    rosterio.add_lane(str(p), {k: values[k] for k in rosterio.FIELDS})
    assert "newlane" in [d.name for d in load_roster(str(p)).denoisers]


def test_a_stale_client_cannot_put_a_port_on_a_denoiser(tmp_path):
    """A cached page or a curl script can still send `port`. Refusing it by
    name is the honest answer: writing it would park the run, and dropping it
    in silence would tell the operator a value they typed had been stored."""
    p = tmp_path / "denoisers.toml"
    p.write_text(ROSTER_WITH_ENCODER)
    with pytest.raises(RosterError, match="unknown denoiser key"):
        rosterio.add_lane(str(p), {"name": "newlane", "host": "gpu4",
                                   "backend": "trt", "tiling": "none",
                                   "port": 5399})
    assert "5399" not in p.read_text()


def test_an_added_lane_leaves_the_encoder_pool_alone(tmp_path):
    """A [[denoiser]] block runs to the next table header of any kind, so the
    insertion point sits above the pool. Appending at the end of the file
    instead would produce valid TOML whose sections read out of order."""
    p = tmp_path / "denoisers.toml"
    p.write_text(ROSTER_WITH_ENCODER)
    rosterio.add_lane(str(p), {"name": "newlane", "host": "local",
                               "backend": "trt", "tiling": "none"})
    encoders = load_roster(str(p)).encoders
    assert [e.name for e in encoders] == ["encoder-host"]
    assert encoders[0].port_base == 5300
    text = p.read_text()
    assert text.index("[[encoder]]") > text.index('"newlane"')


# --- The pool roster has no [encode] table, and this module cannot write it --

POOL_ROSTER = '''\
[[denoiser]]
name    = "igpu"
host    = "local"
backend = "migraphx"
device  = 0
tiling  = "none"
enabled = true

[[encoder]]
name      = "encoder-host"
host      = "local"
stream_ip = "10.0.0.10"
port_base = 5300
slots     = 6
lp_level  = 4
enabled   = true
'''


@pytest.fixture
def pool_roster(tmp_path):
    p = tmp_path / "pool.toml"
    p.write_text(POOL_ROSTER)
    return str(p)


def test_has_encode_table_tells_the_two_formats_apart(roster, pool_roster):
    assert rosterio.has_encode_table(roster)
    assert not rosterio.has_encode_table(pool_roster)


def test_set_lp_level_refuses_a_pool_roster_and_says_why(pool_roster):
    before = open(pool_roster).read()
    with pytest.raises(RosterError) as e:
        rosterio.set_lp_level(pool_roster, 6)
    # The page repeats this to the operator, so it has to say where the level
    # DOES live, not only which table is absent. It used to say per-encoder
    # editing was not built; it is built now, and a message that still said so
    # would send the operator to hand-edit a file the page can write.
    assert "Encode hosts" in str(e.value)
    assert open(pool_roster).read() == before


# --- Encode hosts get the same switch the lanes have -----------------------

# Both tables carry a "gpu4", which is the case that makes the encoder
# switch its own function rather than a flag on set_enabled: one function
# taking a bare name has to guess which table, and guesses wrong on exactly
# the hosts that both denoise and encode.
COLLIDING_ROSTER = '''\
[[denoiser]]
name    = "gpu4"
host    = "gpu4"
backend = "trt"
device  = 0
tiling  = "none"
enabled = true

[[encoder]]
name      = "gpu4"
host      = "gpu4"
stream_ip = "10.0.0.14"
port_base = 5310
slots     = 2
lp_level  = 4
enabled   = true    # the GB10

[[encoder]]
name      = "encoder-host"
host      = "local"
stream_ip = "10.0.0.10"
port_base = 5300
slots     = 4
lp_level  = 4
enabled   = false
'''


@pytest.fixture
def colliding(tmp_path):
    p = tmp_path / "colliding.toml"
    p.write_text(COLLIDING_ROSTER)
    return str(p)


def test_set_encoder_enabled_flips_one_encoder(colliding):
    rosterio.set_encoder_enabled(colliding, "encoder-host", True)
    pool = {e.name: e.enabled for e in load_roster(colliding).encoders}
    assert pool == {"gpu4": True, "encoder-host": True}


def test_set_encoder_enabled_leaves_the_lane_of_the_same_name_alone(colliding):
    """The whole reason this is a separate function.

    Turning off the encode host gpu4 must not touch the denoise lane
    gpu4. They are the same machine but different work, and an operator
    stopping the CPU encoder does not mean to stop the GPU.
    """
    rosterio.set_encoder_enabled(colliding, "gpu4", False)
    roster = load_roster(colliding)
    assert {e.name: e.enabled for e in roster.encoders}["gpu4"] is False
    assert {d.name: d.enabled for d in roster.denoisers}["gpu4"] is True


def test_set_enabled_leaves_the_encoder_of_the_same_name_alone(colliding):
    """And the converse: the lane switch must not reach the encode pool."""
    rosterio.set_enabled(colliding, "gpu4", False)
    roster = load_roster(colliding)
    assert {d.name: d.enabled for d in roster.denoisers}["gpu4"] is False
    assert {e.name: e.enabled for e in roster.encoders}["gpu4"] is True


def test_set_encoder_enabled_touches_no_other_line(colliding):
    before = open(colliding).read().splitlines()
    rosterio.set_encoder_enabled(colliding, "encoder-host", True)
    after = open(colliding).read().splitlines()
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1
    # The block's own alignment survives: [[encoder]] pads its keys to width 9
    # and the rewritten line keeps that, so the file still reads as one a
    # person formatted rather than one a program touched.
    assert after[differing[0]] == "enabled   = true"


def test_set_encoder_enabled_preserves_the_trailing_comment(colliding):
    rosterio.set_encoder_enabled(colliding, "gpu4", False)
    assert "enabled   = false    # the GB10" in open(colliding).read().splitlines()


def test_set_encoder_enabled_inserts_the_key_when_absent(tmp_path):
    # roster.py defaults enabled to True, so a block may simply not set it.
    p = tmp_path / "r.toml"
    p.write_text('''\
[[denoiser]]
name    = "igpu"
host    = "local"
backend = "migraphx"
device  = 0
tiling  = "none"

[[encoder]]
name      = "encoder-host"
host      = "local"
port_base = 5300
''')
    rosterio.set_encoder_enabled(str(p), "encoder-host", False)
    assert {e.name: e.enabled
            for e in load_roster(str(p)).encoders} == {"encoder-host": False}


def test_set_encoder_enabled_rejects_a_non_boolean(colliding):
    # bool("false") is True, so a coercing caller would turn a host ON when
    # the client asked to turn it off. Same guard set_enabled carries.
    before = open(colliding).read()
    with pytest.raises(RosterError):
        rosterio.set_encoder_enabled(colliding, "encoder-host", "false")
    assert open(colliding).read() == before


def test_set_encoder_enabled_names_a_missing_encoder(colliding):
    with pytest.raises(rosterio.EncoderNotFound) as e:
        rosterio.set_encoder_enabled(colliding, "nope", True)
    assert "encoder" in str(e.value) and "nope" in str(e.value)


def test_set_encoder_enabled_honours_the_rev_guard(colliding):
    with pytest.raises(rosterio.StaleRoster):
        rosterio.set_encoder_enabled(colliding, "encoder-host", True,
                                     expect_rev="not-the-rev")


# --- Routing: a lane's encoders allowlist ----------------------------------

def test_a_list_value_rewrites_without_doubling_it(colliding):
    """_ANY_VALUE's last alternative is \\S+, which stops at the first space.
    Without the bracket alternative it matches only `["gpu4",` and the rest
    of the old list survives in the tail -- so the line becomes
    `encoders = ["gpu5"] "gpu5"]`, which is a silent corruption of the
    routing rather than an error."""
    rosterio.update_lane(colliding, "gpu4",
                         {"encoders": ["encoder-host", "gpu4"]})
    line = [l for l in open(colliding) if l.startswith("encoders")]
    assert line == ['encoders = ["encoder-host", "gpu4"]\n']
    # The shrink is the case that corrupts: the new literal is shorter than
    # the old one, so whatever the pattern failed to consume is still there.
    rosterio.update_lane(colliding, "gpu4", {"encoders": ["gpu4"]})
    line = [l for l in open(colliding) if l.startswith("encoders")]
    assert line == ['encoders = ["gpu4"]\n'], "the old list leaked into the tail"
    assert load_roster(colliding).denoisers[0].encoders == ("gpu4",)


def test_an_empty_allowlist_removes_the_key(colliding):
    """An absent allowlist means "any enabled encoder" to roster.py. Writing
    `encoders = []` would say the same thing in a form no other line uses, and
    clearing every tick has to be able to reach the default."""
    rosterio.update_lane(colliding, "gpu4", {"encoders": ["gpu4"]})
    assert any(l.startswith("encoders") for l in open(colliding))
    rosterio.update_lane(colliding, "gpu4", {"encoders": []})
    assert not any(l.startswith("encoders") for l in open(colliding))
    d = load_roster(colliding).denoisers[0]
    assert d.encoders == () and d.allows("encoder-host")


def test_routing_to_an_unknown_encoder_is_refused_with_the_roster_message(colliding):
    before = open(colliding).read()
    with pytest.raises(RosterError) as e:
        rosterio.update_lane(colliding, "gpu4", {"encoders": ["nosuchhost"]})
    assert "nosuchhost" in str(e.value)
    assert open(colliding).read() == before


# --- Editing an encode host -------------------------------------------------

def test_update_encoder_changes_only_that_block(colliding):
    rosterio.update_encoder(colliding, "gpu4", {"slots": 4, "lp_level": 6})
    pool = {e.name: e for e in load_roster(colliding).encoders}
    assert pool["gpu4"].slots == 4 and pool["gpu4"].lp_level == 6
    assert pool["encoder-host"].slots == 4 and pool["encoder-host"].lp_level == 4
    # And the lane of the same name is untouched.
    assert load_roster(colliding).denoisers[0].enabled is True


def test_update_encoder_keeps_alignment_and_comments(colliding):
    rosterio.update_encoder(colliding, "gpu4", {"slots": 4})
    text = open(colliding).read()
    assert "slots     = 4" in text
    assert "enabled   = true    # the GB10" in text


def test_update_encoder_refuses_a_rename_and_says_why(colliding):
    before = open(colliding).read()
    with pytest.raises(RosterError) as e:
        rosterio.update_encoder(colliding, "gpu4", {"name": "spark9"})
    # It has to name the allowlist consequence, which is the reason this
    # refusal is stricter than the lane one.
    assert "allowlist" in str(e.value)
    assert open(colliding).read() == before


def test_update_encoder_refuses_a_denoiser_key(colliding):
    with pytest.raises(RosterError, match="unknown encoder key"):
        rosterio.update_encoder(colliding, "gpu4", {"backend": "trt"})


def test_add_encoder_arrives_switched_off_and_after_the_pool(colliding):
    rosterio.add_encoder(colliding, {"name": "gpu1", "host": "gpu1",
                                     "root": "/home/user/archav1an",
                                     "stream_net": "10.0.0.0/24",
                                     "port_base": 5350, "slots": 2,
                                     "lp_level": 4})
    pool = {e.name: e for e in load_roster(colliding).encoders}
    assert "gpu1" in pool
    # §6: adding does not test it.
    assert pool["gpu1"].enabled is False
    assert pool["gpu1"].port_base == 5350
    text = open(colliding).read()
    assert text.index('"gpu1"') > text.index('"encoder-host"')


def test_add_encoder_is_loadable_with_every_writable_field(colliding):
    """The FIELDS-driven guard the lane form has, for the other table: a key
    added to ENCODER_FIELDS has to prove it round-trips through the real
    validator, not just through the writer."""
    values = {"name": "newhost", "host": "gpu5", "root": "/opt/x",
              "stream_ip": "10.0.0.9", "stream_net": "",
              "port_base": 5390, "slots": 1, "lp_level": 4}
    missing = set(rosterio.ENCODER_FIELDS) - set(values)
    assert not missing, f"the form offers {sorted(missing)} with no value here"
    rosterio.add_encoder(colliding, {k: values[k]
                                     for k in rosterio.ENCODER_FIELDS})
    assert "newhost" in [e.name for e in load_roster(colliding).encoders]


def test_remove_encoder_is_refused_while_a_lane_routes_to_it(colliding):
    """roster.py checks allowlist names against the pool, so removing a host
    a lane names fails at commit with that lane's message. Nothing is written,
    which is what makes "clear the routing first" actionable."""
    rosterio.update_lane(colliding, "gpu4", {"encoders": ["gpu4"]})
    before = open(colliding).read()
    with pytest.raises(RosterError) as e:
        rosterio.remove_encoder(colliding, "gpu4")
    assert "gpu4" in str(e.value)
    assert open(colliding).read() == before


def test_remove_encoder_works_once_nothing_routes_to_it(colliding):
    rosterio.remove_encoder(colliding, "encoder-host")
    assert [e.name for e in load_roster(colliding).encoders] == ["gpu4"]


def test_remove_encoder_names_a_missing_host(colliding):
    with pytest.raises(rosterio.EncoderNotFound):
        rosterio.remove_encoder(colliding, "nope")


def test_the_switch_cannot_strand_a_lane_its_routing_pinned(colliding):
    """Routing and the enable switch are one contract, not two.

    A lane pinned to a single host, and that host switched off, is a lane that
    waits for ever WHILE THE REST OF THE FLEET drains the queue. roster.py
    refuses it at commit, so the page answers with a message naming all three
    ways out rather than writing a roster the run cannot use.

    encoder-host is enabled first on purpose. Switching off the last enabled
    encoder stops the whole fleet, which roster.py allows deliberately -- that
    is an operator halting the pool, not one lane starving beside working
    ones -- so without another live host this asserts nothing.
    """
    rosterio.set_encoder_enabled(colliding, "encoder-host", True)
    rosterio.update_lane(colliding, "gpu4", {"encoders": ["gpu4"]})
    before = open(colliding).read()
    with pytest.raises(RosterError) as e:
        rosterio.set_encoder_enabled(colliding, "gpu4", False)
    assert "would wait for ever" in str(e.value)
    assert "widen its allowlist" in str(e.value)
    assert open(colliding).read() == before


def test_a_disabled_lane_pinned_to_a_disabled_host_is_allowed(colliding):
    """The converse, and deliberate: a lane that is off takes no clip, so it
    starves nothing. Refusing this would reject a roster whose owner has
    simply turned a box off for the night."""
    rosterio.update_lane(colliding, "gpu4", {"encoders": ["gpu4"]})
    rosterio.set_enabled(colliding, "gpu4", False)
    rosterio.set_encoder_enabled(colliding, "gpu4", False)
    assert not load_roster(colliding).enabled_encoders()
