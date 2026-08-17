"""What each lane is holding right now.

state.jsonl gains a record only when a clip finishes, and the longest clip in
this archive is tens of thousands of frames -- nearly three hours on the slower lanes. So a
reader that only has state.jsonl cannot answer "what is this lane doing", which
is the one question a dashboard exists to answer. This file closes that gap and
is the only thing the batch publishes for the UI.

Written tmp-then-rename so a reader never sees half an object, and removed in
the caller's finally so a raising runner does not leave a stale one behind.
"""
import json
import os

# A lane name reaches here from a TOML the operator edits by hand. Anything
# that could climb out of the lanes directory is refused rather than sanitised,
# because a silently renamed lane would be worse than a loud failure.
_BAD = (os.sep, "/", "\0")


def _path(lanes_dir, lane):
    if any(ch in lane for ch in _BAD) or lane in ("", ".", ".."):
        raise ValueError(f"unusable lane name for a heartbeat file: {lane!r}")
    return os.path.join(lanes_dir, f"{lane}.json")


def write(lanes_dir, lane, src, frames, state, started_at, batch_pid,
          attempt, temp_dir):
    """Publish this lane's claim. Call again to change `state` in place."""
    path = _path(lanes_dir, lane)
    os.makedirs(lanes_dir, exist_ok=True)
    body = json.dumps({"lane": lane, "src": src, "frames": frames,
                       "state": state, "started_at": started_at,
                       "batch_pid": batch_pid, "attempt": attempt,
                       "temp_dir": temp_dir})
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(body)
    # No fsync: unlike state.jsonl this is disposable. Losing it to a power cut
    # costs one stale row until the next clip, not a repeated clip.
    os.replace(tmp, path)


def clear(lanes_dir, lane):
    """Release the claim. Safe to call when nothing was ever written."""
    try:
        os.remove(_path(lanes_dir, lane))
    except (OSError, ValueError):
        pass


def read_all(lanes_dir):
    """{lane: record} for every readable heartbeat. Never raises."""
    out = {}
    try:
        names = os.listdir(lanes_dir)
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue        # skips the .tmp of a write in flight
        try:
            with open(os.path.join(lanes_dir, name), "r", encoding="utf-8") as fh:
                row = json.load(fh)
        except (OSError, ValueError):
            continue        # a torn file costs one row, not the page
        lane = row.get("lane")
        if lane:
            out[lane] = row
    return out
