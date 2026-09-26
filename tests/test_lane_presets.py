"""The add-lane form's preset catalogue must describe lanes that load.

static/lane-presets.json exists so that adding a benchmarked host is one
choice rather than ten typed fields. That only helps if what it fills in is
accepted: a preset that names a key the roster rejects, or a port two presets
share, turns a one-click add into a refusal the operator cannot read a cause
from. So every preset here is put through the real writer and the real
validator, singly and all together.

The catalogue is the THIRD copy of the field list -- rosterio.FIELDS is the
first and static/app.js's LANE_FIELDS is the second, for the reason each of
them records. Nothing makes them agree at runtime, so they are pinned here.
"""
import json
import re
from pathlib import Path

import pytest

from tools.archive_batch.roster import load_roster
from tools.encode_dash import rosterio

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "tools" / "encode_dash" / "static"
CATALOGUE = STATIC / "lane-presets.json"

# One local lane and an encode pool: the smallest roster load_roster accepts,
# so each preset is validated against a file that adds nothing of its own.
# Deliberately not the example roster -- that already holds gpu1_4090 on port
# 5300, and every collision it produced would be the fixture's fault.
SEED = '''\
[[denoiser]]
name    = "seed"
host    = "local"
backend = "trt"
device  = 0
tiling  = "none"
enabled = false

[encode]
host     = "local"
slots    = 6
lp_level = 4
'''

PRESETS = json.loads(CATALOGUE.read_text(encoding="utf-8"))["presets"]


def _ids(presets):
    return [p["id"] for p in presets]


@pytest.fixture
def seed(tmp_path):
    path = tmp_path / "denoisers.toml"
    path.write_text(SEED, encoding="utf-8")
    return str(path)


def test_the_catalogue_is_not_empty():
    assert PRESETS, "a catalogue with no preset leaves the picker hidden"


@pytest.mark.parametrize("preset", PRESETS, ids=_ids(PRESETS))
def test_every_preset_carries_what_the_picker_shows(preset):
    for key in ("id", "label", "note", "source"):
        assert isinstance(preset.get(key), str) and preset[key].strip(), \
            f"preset is missing a usable '{key}'"
    assert isinstance(preset.get("fields"), dict) and preset["fields"]


@pytest.mark.parametrize("preset", PRESETS, ids=_ids(PRESETS))
def test_every_preset_sets_only_keys_the_roster_writer_knows(preset):
    unknown = set(preset["fields"]) - set(rosterio.FIELDS)
    assert not unknown, (
        f"rosterio.add_lane refuses these keys outright: {sorted(unknown)}")


@pytest.mark.parametrize("preset", PRESETS, ids=_ids(PRESETS))
def test_every_preset_sets_only_keys_the_form_can_show(preset):
    """A key with no input silently never reaches the POST body.

    app.js says this out loud now rather than dropping it, but a preset that
    provokes that message is still a broken preset: the operator picked a
    known lane and got an incomplete one.
    """
    unknown = set(preset["fields"]) - _form_fields()
    assert not unknown, (
        f"static/app.js has no input for these keys: {sorted(unknown)}")


def _form_fields():
    """The keys the add-lane form can fill, read out of app.js itself."""
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const LANE_FIELDS = \[(.*?)\n\];", source, re.S)
    assert block, "LANE_FIELDS is not where this test expects it in app.js"
    keys = set(re.findall(r'\["(\w+)"', block.group(1)))
    assert keys, "LANE_FIELDS parsed as empty; the pin below would pass on nothing"
    # The checkbox is built separately from the LANE_FIELDS loop, because it
    # is the one boolean on the form.
    assert 'box.name = "stage_source"' in source
    return keys | {"stage_source"}


@pytest.mark.parametrize("preset", PRESETS, ids=_ids(PRESETS))
def test_every_preset_writes_a_lane_the_batch_will_load(preset, seed):
    """The whole point: the form's own POST path, then the real validator.

    This is what catches an incoherent preset -- a tiled lane with no window,
    a remote with no port, a root on a local host. Each of those is a rule in
    roster._validate, and none of them is restated here.
    """
    rosterio.add_lane(seed, preset["fields"])
    names = [d.name for d in load_roster(seed).denoisers]
    assert preset["fields"]["name"] in names


def test_the_whole_catalogue_can_be_added_to_one_roster(seed):
    """Names and ports have to be unique ACROSS the catalogue, not only per
    preset. An operator fielding the whole fleet adds all of them, and a
    shared port fails the roster after the second add -- at which point the
    batch cannot start until they work out which two lanes collided."""
    for preset in PRESETS:
        rosterio.add_lane(seed, preset["fields"])
    roster = load_roster(seed)
    assert len(roster.denoisers) == len(PRESETS) + 1


def test_every_preset_is_offered_under_its_own_id():
    ids = _ids(PRESETS)
    assert len(ids) == len(set(ids)), "two presets share an id; the picker " \
        "matches on it, so one of them can never be chosen"
