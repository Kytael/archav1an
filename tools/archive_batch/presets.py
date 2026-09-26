"""The run_linux_*.sh presets, read as encoder settings for an encode job.

The scripts are the source of truth rather than a catalogue file beside them.
A catalogue would be a second place to edit, and the one that goes stale is
always the copy: dispatch_cmd's own ENCODER_PARAMS was itself a hand copy of
run_linux_dance_HQ_crf27.sh:51-58, and nothing made the two agree afterwards.

Two shapes are read, because the scripts come in two:

  * a pipeline.py call, with --final-speed and --final-params (the fast-pass
    numbers are not read: an encode job has one pass, and it is the final one)
  * a direct svtav1-dispatch call, with --speed and --encoder-params

Only what dispatch accepts is taken. --workers and --autocrop belong to
pipeline.py and have no meaning here, so a preset that sets them loses them
rather than passing an argument dispatch would refuse.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLOB = "run_linux_"

_NUM = r'(?:^|\s)--{flag}\s+"?([0-9]+)"?'
_STR = r'(?:^|\s)--{flag}\s+"([^"]*)"'
# --lp is the ENCODER's, not the preset's. It is a memory-for-parallelism
# choice that belongs to the host doing the encode, and build_command already
# passes the roster's value; left in, dispatch would receive two --lp flags and
# the preset's would be the one that lost the argument order lottery.
_LP = re.compile(r"(?:^|\s)--lp\s+[0-9]+")


class PresetError(Exception):
    """A named preset does not exist, or holds nothing this can use."""


def catalogue(repo=REPO):
    """Every readable preset, as a list of dicts, sorted by id.

    A script that sets none of the four settings is skipped rather than
    offered: av1an-batch-*.sh drive av1an directly and share no flags with
    dispatch, and an empty entry in the picker is one an operator would try.
    """
    out = []
    try:
        names = sorted(n for n in os.listdir(repo)
                       if n.startswith(GLOB) and n.endswith(".sh"))
    except OSError:
        return []
    for name in names:
        try:
            with open(os.path.join(repo, name), "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        preset = _parse(name, text)
        if preset is not None:
            out.append(preset)
    return out


def load_preset(preset_id, repo=REPO):
    """One preset by id, or PresetError. The id is the script's file name."""
    for preset in catalogue(repo):
        if preset["id"] == preset_id:
            return preset
    raise PresetError(f"no preset named {preset_id!r}")


def _parse(name, text):
    quality = _find(_NUM, "quality", text)
    if quality is None:
        return None
    # --final-speed first: in a pipeline.py script --speed does not appear at
    # all, and in a dispatch script --final-speed does not. Checked in this
    # order so neither shape reads the other's flag by accident.
    speed = _find(_NUM, "final-speed", text) or _find(_NUM, "speed", text)
    params = (_find(_STR, "final-params", text)
              or _find(_STR, "encoder-params", text) or "")
    return {
        "id": name,
        "label": name[len(GLOB):-len(".sh")].replace("_", " "),
        "quality": quality,
        "photon_noise": _find(_NUM, "photon-noise", text) or "0",
        "speed": speed or "4",
        "params": _LP.sub("", params).strip(),
    }


def _find(pattern, flag, text):
    match = re.search(pattern.format(flag=re.escape(flag)), text)
    return match.group(1) if match else None
