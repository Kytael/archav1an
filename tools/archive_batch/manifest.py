"""Parse the probed manifest and decide processing order.

The manifest is produced once by walking gpu1 over ssh (see the spec, 5.3),
because probing a few thousand files across 9p is slow enough that resume must never
repeat it.
"""
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

_BARE_NUMBER = re.compile(r"^[0-9]+\.?[0-9]*$")
_YEAR = re.compile(r"^[0-9]{4}$")
_SET_RANK = {"SetA": 0, "SetB": 1}
_NO_YEAR = 9999


@dataclass(frozen=True)
class Clip:
    src: str        # path relative to ARCHIVE_ROOT, e.g. "SetA/2001/event-a/x.MOV"
    rel_dir: str    # directory part of src
    stem: str       # basename without extension
    size: int       # bytes
    frames: int     # video frame count
    duration: float # seconds


def parse_manifest(text):
    """Parse manifest TSV text into a tuple of Clip.

    Row layout is path, size, then one "rate,frames" column per video stream,
    then duration. A source with an embedded thumbnail stream has two rate
    columns, so frames comes from the first and duration from the last bare
    numeric field.
    """
    clips = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        src = fields[0]
        size = int(fields[1])
        frames = _frames_from(fields[2])
        duration = _duration_from(fields[3:])
        p = PurePosixPath(src)
        clips.append(Clip(src=src, rel_dir=str(p.parent), stem=p.stem,
                          size=size, frames=frames, duration=duration))
    return tuple(clips)


def _frames_from(field):
    parts = field.split(",")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


def _duration_from(rest):
    for value in reversed(rest):
        if _BARE_NUMBER.match(value.strip()):
            return float(value)
    return 0.0


def order_clips(clips):
    """SetA before SetB, years ascending, undated folders last, then
    longest clip first inside a folder.

    Longest-first matters because clip length spans seconds to tens of minutes: a
    shortest-first queue leaves a 30-minute clip starting on the slowest
    denoiser after every other one has drained.
    """
    return tuple(sorted(clips, key=_sort_key))


def _sort_key(clip):
    parts = PurePosixPath(clip.src).parts
    top = parts[0] if parts else ""
    second = parts[1] if len(parts) > 1 else ""
    year = int(second) if _YEAR.match(second) else _NO_YEAR
    return (_SET_RANK.get(top, 2), year, clip.rel_dir, -clip.frames, clip.src)
