"""One status snapshot, built from the run's files and nothing else.

Pure over its inputs: give it a directory of fixtures and it produces a
snapshot, with no batch running and no network. That is what makes the page
testable, and it is why the daemon reads files rather than talking to the batch.
"""
import json
import os

from tools.archive_batch.heartbeat import read_all as read_heartbeats
from tools.archive_batch.manifest import order_clips, parse_manifest
from tools.archive_batch import pidfile
from tools.archive_batch.roster import RosterError, load_roster
from tools.archive_batch.state import MAX_ATTEMPTS, State, load_state

from . import rosterio
from .control import acks as read_acks
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
    # BEFORE the content, and the order is the whole point. This snapshot takes
    # real time to build -- a a few thousand-line manifest, two passes over state.jsonl,
    # every heartbeat -- so a write can land in the middle of it. Read the
    # revision last and the page gets lane data from version N stamped with the
    # revision of version N+1: it would compute an edit from N and quote a
    # revision the server accepts, which is exactly the clobber the guard
    # exists to prevent. Read it first and the worst case is version N+1's data
    # under version N's revision, and that is a 409 the page recovers from.
    roster_rev = rosterio.rev(paths.roster)
    roster, roster_error = _roster(paths.roster)
    clips, manifest_error = _clips(paths.manifest)
    state = _state(paths.state)
    beats = read_heartbeats(paths.lanes)
    history, failures = _history(paths.state)
    batch_roster_error = _batch_roster_error(paths.roster_error)
    acks = read_acks(paths.control)

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

    # Slot workers write heartbeats into the same directory, named
    # <encoder>-<slot>. Split the two populations BY NAME rather than by
    # "is it in the roster": a slot worker's file is not a lane that has
    # vanished from the roster, and treating it as one would drop it from the
    # page entirely while the job it names is running.
    lane_names = {d.name for d in (roster.denoisers if roster else ())}
    encoder_names = {e.name for e in (roster.encoders if roster else ())}
    jobs = [_job(name, beat, history, tracker, now)
            for name, beat in sorted(beats.items())
            if name not in lane_names and _encoder_of(name) in encoder_names]

    return {
        "batch": _batch(beats, paths.batch),
        "roster_error": roster_error,
        # What the BATCH says, which is a different fault from the line above:
        # that one is this daemon failing to parse the file, this one is the
        # batch parked because it could not. Kept apart on purpose --
        # encode_roster_error is a live alert on the Pi with a year of
        # retention, and it means the first of the two.
        "batch_roster_error": batch_roster_error,
        # The revision a write must quote back. Read from the file rather than
        # from the parsed Roster on purpose: a roster that fails to parse still
        # has a revision, and the page must be able to repair it by hand and
        # write again without a daemon restart.
        "roster_rev": roster_rev,
        "manifest_error": manifest_error,
        "encode": _encode(roster, paths.roster),
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
        "jobs": jobs,
        # Keyed by encoder name, so the hosts table can show a rate beside the
        # switch without walking the jobs list itself.
        "encoder_rates": _encoder_rates(jobs, history),
        "queue": [{"src": c.src, "frames": c.frames,
                   # Which job type a queued row is. Both populations share one
                   # queue, so without this the panel shows a folder of encode
                   # jobs and a folder of archive clips as one undifferentiated
                   # list.
                   "denoise": c.denoise} for c in pending[:QUEUE_PREVIEW]],
        # The LAST 50, not the first. _history builds these in last-failure
        # order, so slicing from the front freezes the panel on the oldest
        # failures of the run and silently drops every later one. Over a few thousand
        # clips and fifteen days even a 1.5% transient rate fills it.
        "failures": _preview(_failures(failures, state)),
        "acks": acks,
    }


def _encoder_of(worker_name):
    """The encoder half of an <encoder>-<slot> worker name, or "".

    rsplit on the LAST dash and only when the tail is a slot number. Encoder
    names carry dashes -- this fleet has "encoder-host-lp6" -- so an unconditional
    split would read that as encoder "encoder-host" slot "lp6" and match nothing.
    """
    head, _, tail = worker_name.rpartition("-")
    return head if head and tail.isdigit() else ""


def _job(name, beat, history, tracker, now):
    """One slot worker's row: which host, which slot, what it holds, how fast.

    The rate comes from the same two sources a lane's does, because a slot
    worker writes the same heartbeat and dispatch writes the same log. It needs
    none of _lane's windowing arguments: an encode job runs no BSVD pass, so
    its counter climbs evenly and the tracker's default smoothing is right.

    A slot's history is keyed by <encoder>-<slot>, which is what state.jsonl
    records as the "denoiser" for an encode job. That is per SLOT and not per
    host -- see _encoder_rates for the host figure.
    """
    beat = beat or {}
    rows = history.get(name, [])
    src = beat.get("src") or ""
    started = beat.get("started_at")
    row = {
        "name": name,
        "encoder": _encoder_of(name),
        "state": beat.get("state") or "idle",
        "src": src,
        "stem": os.path.basename(src) if src else "",
        "frames": beat.get("frames"),
        "elapsed_s": (now - started) if started else None,
        "fps_live": None,
        "fps_recent": _mean(r.get("fps") for r in rows[-RECENT_CLIPS:]),
        "clips_done": len(rows),
        "frames_done": None,
        "progress": None,
        "eta_s": None,
    }
    if not src:
        # Idle: no clip, so nothing the held series describes is still running.
        tracker.forget(name)
        return row

    produced = frames_from_log(beat.get("temp_dir", ""), _stem(src))
    if produced is None:
        tracker.forget(name)
        return row
    row["fps_live"] = tracker.sample(name, produced, now)
    row["frames_done"] = produced
    total = beat.get("frames") or 0
    if total:
        row["progress"] = min(0.999, produced / total)
        if row["fps_live"]:
            row["eta_s"] = max(0.0, round((total - produced) / row["fps_live"], 0))
    return row


def _encoder_rates(jobs, history):
    """{encoder: {fps_live, fps_recent, clips_done}} across its slots.

    Per HOST, which is the figure that answers "is this box worth a slot": a
    single slot's rate says nothing about a machine running two, and the
    per-slot rows cannot be compared across hosts with different slot counts.
    fps_live sums -- two slots encoding at once really do produce both rates --
    while fps_recent averages the completed clips, because that is a per-clip
    figure and summing it would double a two-slot host's apparent speed.
    """
    out = {}
    for job in jobs:
        enc = job["encoder"]
        if not enc:
            continue
        out.setdefault(enc, {"fps_live": [], "clips": []})
        if job["fps_live"] is not None:
            out[enc]["fps_live"].append(job["fps_live"])
        out[enc]["clips"] += history.get(job["name"], [])
    return {enc: {"fps_live": (round(sum(v["fps_live"]), 2)
                              if v["fps_live"] else None),
                  "fps_recent": _mean(r.get("fps")
                                      for r in v["clips"][-RECENT_CLIPS:]),
                  "clips_done": len(v["clips"])}
            for enc, v in out.items()}


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


def _encode(roster, path):
    """The status line's one-line view of the encode pool, or None.

    Slots are summed over the ENABLED encoders, because what the line answers
    is how many clips can encode at once, and a disabled host takes none.
    lp_level is quoted only when the pool agrees on one: encoders may each set
    their own, and showing the first one's would read as the whole pool's.

    lp_editable is separate from lp_level because a pool roster has both an
    agreed level and no way to save a new one: rosterio only writes the legacy
    [encode] table, and per-encoder editing is a separate plan. Folding the two
    together and sending lp_level = null would drop the level from the status
    line as well, which is true information the page can still show.
    """
    if roster is None:
        return None
    encoders = roster.enabled_encoders()
    levels = {e.lp_level for e in encoders}
    return {"slots": sum(e.slots for e in encoders),
            "lp_level": levels.pop() if len(levels) == 1 else None,
            "lp_editable": rosterio.has_encode_table(path),
            # Every rostered encoder, disabled ones included: the page needs
            # a row to switch a host back ON, and enabled_encoders() above
            # cannot supply one -- it is the set the slot count is summed over.
            #
            # Empty on a legacy [encode] roster, which has no [[encoder]]
            # blocks at all: load_roster synthesizes one named "local" from
            # that table, and rosterio cannot write a block that is not in the
            # file. A row for it would be a switch whose POST answers 404, so
            # the page is sent nothing and hides the section instead.
            "hosts": [] if rosterio.has_encode_table(path) else [
                {"name": e.name, "host": e.host, "slots": e.slots,
                 "lp_level": e.lp_level, "enabled": e.enabled,
                 # Everything the add form can write, so Edit opens it already
                 # filled -- the same contract _lane's "fields" has, and
                 # `enabled` is absent here for the same reason: the row's
                 # switch owns it, and an edit carrying it would let a save
                 # undo a toggle made in between.
                 "fields": {k: getattr(e, k) for k in rosterio.ENCODER_FIELDS
                            if getattr(e, k)
                            or k not in rosterio.ENCODER_OMIT_WHEN_FALSY}}
                for e in roster.encoders]}


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


def _batch_roster_error(path):
    """What the batch says about the roster, or None.

    A separate fault from _roster's. This one means the batch is parked; that
    one means this daemon cannot read the file. They are usually the same
    thing, and when they are not it is worth seeing both. A whitespace-only
    file reads as no error (None).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except (OSError, ValueError):
        return None


def _batch(beats, path):
    """Alive if the batch file names a live pid, or failing that a heartbeat.

    The file comes first because it is the only signal a parked run has. A
    heartbeat exists only while a lane holds a clip, so a run parked because
    every lane was yielded has none -- and reading that as dead would report
    encode_batch_up 0 for a live process, fire EncodeBatchDown five minutes
    after the operator yielded the last lane, and grey out the Stop button at
    the moment they most want it.

    The heartbeat fallback stays so a run holding no batch.json is still read
    correctly. A file left by a killed batch names a dead pid, which
    pidfile.alive rejects.

    Both readers accept a pre-upgrade record -- one with no pid_start, from a
    batch that was already running when the checkout was upgraded. Rejecting
    those would report a live batch as dead for the rest of a fifteen-day
    run, which greys out Stop, fires EncodeBatchDown continuously and (worse)
    re-enables Start over the running batch. pidfile.alive checks their
    identity by command line instead.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            row = json.load(fh)
        # isinstance, not truthiness. A file holding "[]" or "null" has no
        # .get, a string pid raises TypeError out of os.kill, and -1 is worse
        # than either: os.kill(-1, 0) probes every process this user owns and
        # succeeds, so a corrupt file would report the batch as running. All
        # three escape (OSError, ValueError) and 500 the snapshot -- and an
        # absent encode_batch_up series does not fire EncodeBatchDown, so the
        # alert would be lost rather than raised.
        pid = row.get("batch_pid") if isinstance(row, dict) else None
        if isinstance(row, dict) and "pid_start" in row \
                and not row.get("claimed"):
            # An unclaimed file is the startup window (or a stale takeover
            # before the batch marks claimed): the batch has not yet cleared
            # the control dir, so the dashboard must not offer Stop. Report
            # not-running, never None -- a None leaks into the snapshot and
            # crashes the page and /metrics on the very first poll.
            #
            # Only for a record this version wrote. Every writer here emits
            # pid_start beside the marker, so a record without that field is
            # from a batch that predates both -- and gating it on a marker
            # its writer never wrote would call a live run dead.
            return {"running": False, "pid": None}
        # Identity-checked, not existence-checked: a recycled pid would hold
        # encode_batch_up at 1 forever and EncodeBatchDown would never fire
        # for a batch that died.
        if isinstance(pid, int) and pid > 0 \
                and pidfile.alive(pid, row.get("pid_start")):
            return {"running": True, "pid": pid}
    except (OSError, ValueError):
        pass
    for row in beats.values():
        # isinstance BEFORE the .get, or a lanes/*.json holding "[]" raises
        # AttributeError here and 500s the whole snapshot.
        if not isinstance(row, dict):
            continue
        pid = row.get("batch_pid")
        # The same isinstance guard the primary path applies: a corrupt row
        # holding a string makes os.kill raise TypeError, which pidfile.alive
        # does not catch, and the scrape would 500 -- the exact failure this
        # function exists to survive.
        #
        # Identity-checked like the primary path, pid_start or not: the batch
        # never sweeps lanes/, so a SIGKILL leaves its files behind and an
        # existence-only probe would hold encode_batch_up at 1 for ever once
        # the pid was recycled.
        if not isinstance(pid, int) or pid <= 0:
            continue
        if pidfile.alive(pid, row.get("pid_start")):
            return {"running": True, "pid": pid}
    return {"running": False, "pid": None}


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
        # Everything the add-lane form can write, so Edit can open that form
        # already filled in. The four above are here too because the row
        # renders them; this is the copy the form reads, and it is the whole
        # writable set rather than the subset the page happens to show.
        # `enabled` is deliberately absent: the switch owns it, and an edit
        # that carried it would let a save undo a toggle made in between.
        "fields": {key: getattr(denoiser, key) for key in rosterio.FIELDS
                   if getattr(denoiser, key)
                   or key not in rosterio.OMIT_WHEN_FALSY},
    }
    if not beat:
        tracker.forget(denoiser.name)
        return row

    pid = beat.get("batch_pid")
    # isinstance, not truthiness: a string pid would raise TypeError out of
    # os.kill and take the snapshot with it. Identity-checked, not just
    # alive -- the batch never sweeps lanes/, so a leftover from a SIGKILLed
    # run must not render as working once its pid is recycled. A record with
    # no pid_start is checked by command line rather than dismissed: a live
    # batch from before the upgrade would otherwise leave every lane reading
    # "unknown", with no rate, no progress and no ETA, for the rest of the run.
    if not isinstance(pid, int) or pid <= 0 \
            or not pidfile.alive(pid, beat.get("pid_start")):
        # The `pid <= 0` half is not belt-and-braces: os.kill(0, 0) signals the
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
        # A windowed lane says nothing until its series covers a whole sweep:
        # below that there is no rate to report, only the encoder draining one
        # delivered window. fps_recent carries the page in the meantime.
        row["fps_live"] = tracker.sample(denoiser.name, produced, now,
                                         _smooth_for(denoiser, row["fps_recent"]),
                                         _sweep_for(denoiser, row["fps_recent"]))
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


# How many sweeps of smoothing a windowed lane gets. With a step input any
# finite window sees N or N+1 bursts depending on phase, so the reported rate
# swings by 1/N: one sweep is a factor of two, which is the artefact itself.
# Two and a half holds it inside 50% while a lane's figure still settles in
# minutes rather than tens of minutes.
SWEEPS_SMOOTHED = 2.5

# Used only to size the smoothing window before a lane has ever finished a clip.
# Deliberately at the slow end of the fleet's measured range (2.4-6.6 fps for
# every lane except gpu1's 4090), because guessing slow makes the window too
# long and guessing fast makes it too short -- and too short reproduces the
# burst. It is never displayed and never reaches a reported figure.
_ASSUMED_SLOW_FPS = 2.0


def _sweep_for(denoiser, fps_recent):
    """Seconds one window sweep takes on this lane, or None if not windowed.

    The sweep is the natural unit for both numbers below: it is the smallest
    interval over which a windowed lane produces anything at all.
    """
    if not denoiser.window:
        return None
    return denoiser.window / (fps_recent or _ASSUMED_SLOW_FPS)


def _smooth_for(denoiser, fps_recent):
    """Seconds of smoothing this lane needs, or None for the tracker default.

    A full-frame lane produces frames evenly and wants the short default, so its
    number is current. A windowed lane steps by a whole window every
    `window / fps` seconds and needs to span several of those steps, or the rate
    reads zero for most of a sweep and then spikes -- at window 750 and 5.5 fps,
    a 30 s window reads 0 for 106 s and then 25 fps against a true 5.5.
    """
    sweep = _sweep_for(denoiser, fps_recent)
    return None if sweep is None else SWEEPS_SMOOTHED * sweep


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
                    # pop first. Reassigning an existing key keeps its FIRST
                    # insertion position, so without this the dict is ordered
                    # by the oldest failure of each clip and the caller's tail
                    # slice selects the newest FIRST-failures -- not the newest
                    # failures. A clip that failed early and again late then
                    # sits at its early position and never appears.
                    failed.pop(row["src"], None)
                    failed[row["src"]] = row
    except OSError:
        pass
    return done, failed


def _preview(rows):
    """The rows the Failed panel carries: the newest, plus every exhausted one.

    Recency alone is the wrong cap. A clip that runs out of attempts stops
    failing at that moment, so it drifts out of a newest-N window while clips
    that keep retrying stay in it -- and the exhausted ones are the only rows
    that need an operator. Capping by recency therefore emptied the panel the
    moment "out of attempts only" was ticked, which is precisely when it had
    to show something.

    Both halves are capped, so a run that abandons thousands of clips still
    sends a bounded payload, and the order of the underlying list is kept so
    the panel still reads newest-last.
    """
    keep = {id(r) for r in rows[-FAILURE_PREVIEW:]}
    keep |= {id(r) for r in [r for r in rows if r["exhausted"]][-FAILURE_PREVIEW:]}
    return [r for r in rows if id(r) in keep]


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
