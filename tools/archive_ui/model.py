"""One status snapshot, built from the run's files and nothing else.

Pure over its inputs: give it a directory of fixtures and it produces a
snapshot, with no batch running and no network. That is what makes the page
testable, and it is why the daemon reads files rather than talking to the batch.
"""
import json
import os

from tools.archive_batch.heartbeat import read_all as read_heartbeats
from tools.archive_batch.manifest import order_clips, parse_manifest
from tools.archive_batch.roster import RosterError, load_roster
from tools.archive_batch.state import MAX_ATTEMPTS, State, load_state

from .liverate import frames_from_log

# How many clips of the queue and how many failures the page shows. The archive
# is a few thousand clips; sending all of them every 2 seconds for a fortnight is waste,
# and nobody reads past the top of either list.
QUEUE_PREVIEW = 12
FAILURE_PREVIEW = 50

# fps_recent averages this many of a lane's completed clips. A trailing window
# rather than a whole-run mean because a lane's rate is not a constant: every
# lane re-measured on a full clip in docs/encode-capacity.md came in below its
# short-run figure except the 2070S, the only card there with its own cooling.
RECENT_CLIPS = 10


def snapshot(paths, tracker, now):
    """The whole page's data. `now` is injected so tests are deterministic."""
    roster, roster_error = _roster(paths.roster)
    clips, manifest_error = _clips(paths.manifest)
    state = _state(paths.state)
    beats = read_heartbeats(paths.lanes)
    history, failures = _history(paths.state)

    # Both counted against the manifest rather than against state.jsonl. The
    # state file can name clips a regenerated manifest no longer lists, and
    # len(state.done) would then push done + queued + failed past clips -- the
    # page contradicting itself with no way to tell which half is wrong.
    done_clips = sum(1 for c in clips if c.src in state.done)
    done_frames = sum(c.frames for c in clips if c.src in state.done)
    pending = [c for c in clips
               if c.src not in state.done
               and state.failures.get(c.src, 0) < MAX_ATTEMPTS]

    lanes = [_lane(d, beats.get(d.name), history, tracker, now)
             for d in (roster.denoisers if roster else ())]

    return {
        "batch": _batch(beats),
        "roster_error": roster_error,
        "manifest_error": manifest_error,
        "encode": ({"slots": roster.encode.slots,
                    "lp_level": roster.encode.lp_level} if roster else None),
        "totals": {
            "clips": len(clips),
            "done": done_clips,
            "failed": sum(1 for v in state.failures.values() if v >= MAX_ATTEMPTS),
            "queued": len(pending),
            "frames": sum(c.frames for c in clips),
            "frames_done": done_frames,
            "fps_live": _sum_or_none(l["fps_live"] for l in lanes),
            "eta_finish": _eta_finish(pending, lanes, now),
        },
        "lanes": lanes,
        "queue": [{"src": c.src, "frames": c.frames} for c in pending[:QUEUE_PREVIEW]],
        # The LAST 50, not the first. _history builds these in file order, so
        # slicing from the front freezes the panel on the oldest failures of the
        # run and silently drops every later one -- including clips that go on
        # to exhaust their attempts and stop for good. Over a few thousand clips and
        # fifteen days even a 1.5% transient rate fills it.
        "failures": _failures(failures, state)[-FAILURE_PREVIEW:],
    }


def _roster(path):
    try:
        return load_roster(path), None
    except RosterError as exc:
        # scheduler.py swallows this and parks every lane with no message.
        # Surfacing it here is the point: a typo in the TOML currently stops a
        # 15-day run silently.
        return None, str(exc)
    except OSError as exc:
        return None, f"cannot read the roster: {exc}"


def _clips(path):
    """(clips, error). A missing manifest is not an error; a corrupt one is.

    No manifest yet is the normal state before the first run, and a red banner
    for it would be noise. A manifest that will not parse is different:
    parse_manifest raises ValueError on a non-numeric size column, and the file
    is hand-editable. Left to propagate it would 500 the /metrics endpoint,
    which takes out the Prometheus scrape and every alert built on it -- so a
    broken manifest would disable the monitoring rather than show up on it.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return order_clips(parse_manifest(fh.read())), None
    except OSError:
        return (), None
    except ValueError as exc:
        return (), f"manifest will not parse: {exc}"


def _state(path):
    """Resume state, or an empty one. Never raises.

    load_state catches FileNotFoundError only, so a state.jsonl the daemon
    cannot read -- a PermissionError when it runs as a different user from the
    batch, or a directory where the file should be -- would propagate and 500
    the /metrics endpoint, taking the Prometheus scrape and its alerts with it.
    Every other file this snapshot touches already tolerates that.
    """
    try:
        return load_state(path)
    except OSError:
        return State()


def _batch(beats):
    """Alive if any heartbeat names a pid that still exists."""
    for row in beats.values():
        pid = row.get("batch_pid")
        if pid and _alive(pid):
            return {"running": True, "pid": pid}
    return {"running": False, "pid": None}


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True     # exists, owned by someone else
    except OSError:
        return False
    return True


def _lane(denoiser, beat, history, tracker, now):
    rows = history.get(denoiser.name, [])
    row = {
        "name": denoiser.name,
        "host": denoiser.host,
        "backend": denoiser.backend,
        "tiling": denoiser.tiling,
        "window": denoiser.window,
        "enabled": denoiser.enabled,
        "state": "idle" if denoiser.enabled else "off",
        "current": None,
        "fps_live": None,
        # .get, like every other read of a state.jsonl row. A direct subscript
        # here would take the whole snapshot down on one record written before
        # a field existed, and this file is append-only across versions.
        "fps_recent": _mean(r.get("fps") for r in rows[-RECENT_CLIPS:]),
        "fps_all": _mean(r.get("fps") for r in rows),
        "clips_done": len(rows),
        "phase_split": _phases(rows),
    }
    if not beat:
        tracker.forget(denoiser.name)
        return row

    pid = beat.get("batch_pid")
    if not pid or not _alive(pid):
        # The `not pid` half is not belt-and-braces: os.kill(0, 0) signals the
        # caller's own process group and never raises, so _alive(0) is True and
        # a heartbeat with no pid would render as working for ever. _batch
        # already guards this; this is the same guard, not a new rule.
        #
        # A SIGKILLed batch leaves its heartbeats behind. Reporting "working"
        # from one of those would be a lie that never expires.
        tracker.forget(denoiser.name)
        row["state"] = "unknown"
        return row

    row["state"] = beat.get("state", "working")
    frames_total = beat.get("frames") or 0
    elapsed = max(0.0, now - beat.get("started_at", now))
    produced = frames_from_log(beat.get("temp_dir", ""), _stem(beat.get("src", "")))
    if produced is None:
        # Nothing is being counted: the clip has changed and its log does not
        # exist yet, or the counter stopped. Either way the samples held for
        # this lane describe work that is over, and a slope drawn from the last
        # clip's count through this one's would be an invention. The heartbeat
        # stays, so this is the only place the change of clip is visible.
        tracker.forget(denoiser.name)
    else:
        row["fps_live"] = tracker.sample(denoiser.name, produced, now)
    row["current"] = {
        "src": beat.get("src", ""),
        "frames": frames_total,
        "elapsed_s": round(elapsed, 1),
        "frames_done": produced,
        "progress": (min(0.999, produced / frames_total)
                     if produced is not None and frames_total else None),
        # Clamped at zero for the same reason progress is clamped at 0.999: the
        # counted frames can pass the manifest's count, and an unclamped
        # subtraction then renders a finish time in the past.
        "eta_s": (max(0.0, round((frames_total - produced) / row["fps_live"], 0))
                  if produced is not None and frames_total and row["fps_live"]
                  else None),
    }
    return row


def _stem(src):
    return os.path.splitext(os.path.basename(src))[0]


def _history(state_path):
    """({lane: [done record, ...]}, {src: last failed record}), in file order.

    One pass for these two questions, which want the same lines. `snapshot`
    still reads the file a second time through `load_state`, and that is
    deliberate: the done set and the attempt ceiling define resume, and
    duplicating that logic here to save one read would put the rule in two
    places and let them drift.
    """
    done, failed = {}, {}
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("status") == "done" and row.get("denoiser"):
                    done.setdefault(row["denoiser"], []).append(row)
                elif row.get("status") == "failed" and row.get("src"):
                    failed[row["src"]] = row
    except OSError:
        pass
    return done, failed


def _failures(failed, state):
    return [{"src": src, "lane": row.get("denoiser", ""),
             "reason": row.get("reason", ""),
             "attempts": state.failures.get(src, 0),
             "exhausted": state.failures.get(src, 0) >= MAX_ATTEMPTS}
            for src, row in failed.items()]


def _phases(rows):
    if not rows:
        return None
    total = sum(r.get("stage_s", 0) + r.get("work_s", 0) + r.get("publish_s", 0)
                for r in rows)
    if total <= 0:
        return None
    return {p: round(sum(r.get(f"{p}_s", 0) for r in rows) / total, 3)
            for p in ("stage", "work", "publish")}


def _mean(values):
    # A recorded fps of exactly 0.0 is not a rate, it is a clip whose frame
    # count the manifest never got. Averaging it in would pull a lane's figure
    # toward zero on the strength of work that was never measured, so it is
    # dropped along with the Nones and a lane of nothing but those reads as
    # "no rate" rather than "zero".
    vals = [v for v in values if v]
    return round(sum(vals) / len(vals), 3) if vals else None


def _sum_or_none(values):
    # `is not None`, unlike _mean above: a live rate of 0.0 is a measurement,
    # not a gap. A windowed lane reads zero between sweeps, and a wedged run
    # reads zero on every lane -- which has to total 0.0, because None means
    # "nobody is reporting" and would hide exactly the state worth alerting on.
    vals = [v for v in values if v is not None]
    return round(sum(vals), 3) if vals else None


def _eta_finish(pending, lanes, now):
    """Seconds-since-epoch the run finishes, or None.

    Divides remaining frames by the enabled lanes' recent rates, skipping any
    lane with no history. None rather than infinity when nothing is enabled.

    Takes `pending` rather than every not-done clip, so it agrees with the
    `queued` total and the queue list. A clip that has exhausted its attempts is
    never going to be processed, and counting its frames here would push the
    finish date out for work the run has already given up on.
    """
    remaining = sum(c.frames for c in pending)
    supply = sum(l["fps_recent"] for l in lanes
                 if l["enabled"] and l["fps_recent"])
    if not remaining or supply <= 0:
        return None
    return round(now + remaining / supply, 0)
