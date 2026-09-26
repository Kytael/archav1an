"""The control channel: actions a TOML file cannot express.

One JSON file per request under `.archive-run/control/`. The daemon writes;
this side consumes, acts, deletes and appends an ack. See the spec, 4.2.

Files rather than a socket because they survive a daemon restart, can be read
with `cat`, and test without a protocol. The cost is the poll interval, which
is the reaction time for a yield and for a stop.
"""
import json
import os
import secrets
import time

# Every action the batch knows. A request naming anything else is answered and
# removed rather than left in the directory to be re-read once a second.
ACTIONS = ("yield", "stop", "retry", "retry-all", "submit")
ACKS = "acks.jsonl"
SUFFIX = ".json"


class YieldRequested(Exception):
    """The operator asked for this lane's GPU back, so the dispatch was killed.

    Raised by archive-batch.py's runner and caught by Scheduler._process, which
    requeues the clip without spending an attempt. Lives here rather than in
    either of those files because both import it, the same way TransferOutage
    lives in transfer.py.
    """


class ControlError(Exception):
    """A request that cannot be built or written."""


def new_id(now):
    """A request id that sorts by time.

    The timestamp leads so a lexical sort of the directory is a time sort, and
    the random tail keeps two requests inside the same millisecond apart.
    """
    return f"{now:.3f}-{secrets.token_hex(2)}"


def write_request(control_dir, request):
    """Publish one request, atomically.

    tmp-then-replace, like heartbeat.write: the batch polls this directory from
    a different process, so a half-written file is a real interleaving. No
    fsync -- a request lost to a power cut is a button the operator presses
    again, and the run it was aimed at is gone anyway.
    """
    os.makedirs(control_dir, exist_ok=True)
    path = os.path.join(control_dir, request["id"] + SUFFIX)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(request))
    os.replace(tmp, path)


def clear(control_dir):
    """Delete every pending request. Returns how many went.

    Called once at startup, before the poller begins. Without it a yield queued
    against a run that has since died fires into the next run, possibly days
    later. The ack log is deliberately not touched: it is the only record of
    what the previous run was told.
    """
    gone = 0
    for name in _request_names(control_dir):
        try:
            os.remove(os.path.join(control_dir, name))
            gone += 1
        except OSError:
            pass
    return gone


def take_action(control_dir, action):
    """Pending requests naming one action, oldest first, removed from disk.

    `action` is one name or a tuple of them, because the startup drain has to
    lift both retry shapes out in a single pass -- two passes would leave a
    window in which clear() ran between them.

    clear() below drops everything at startup, which is right for a yield and
    for a stop: each names a live lane or a live run, and a stale one fires
    into a run it was never meant for, possibly days later.

    A retry is not about a live run at all. It is an instruction about a clip's
    failure count, and it is the one request an operator can ONLY send while
    the run is stopped -- a clip out of attempts is dropped by pending_clips,
    so a run with nothing else left refuses to start, and the record that
    resets the count is written from inside a running daemon. Clearing those
    closed the loop: no retry without a run, no run without a retry. So the
    caller lifts them out before clear() sees them.
    """
    wanted = (action,) if isinstance(action, str) else tuple(action)
    out = []
    for path, request in take(control_dir):
        if request.get("action") in wanted:
            out.append(request)
            drop(path)
    return out


def take(control_dir):
    """Every readable pending request, oldest first, as (path, request).

    A file that will not parse is skipped rather than fatal: one torn request
    costs one row, not the whole poll. It stays on disk, which is deliberate --
    it is evidence, and it cannot be acted on to be cleared.
    """
    out = []
    for name in _request_names(control_dir):
        path = os.path.join(control_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                request = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(request, dict) and request.get("id"):
            out.append((path, request))
    return out


def drop(path):
    """Remove one consumed request. Safe to call twice."""
    try:
        os.remove(path)
    except OSError:
        pass


def ack(control_dir, request_id, accepted, note, now):
    """Append one outcome to the ack log.

    Appended without an fsync, unlike state.jsonl: this is what the page shows,
    not what the run resumes from.
    """
    os.makedirs(control_dir, exist_ok=True)
    row = {"id": request_id, "accepted": bool(accepted), "at": now,
           "note": note}
    with open(os.path.join(control_dir, ACKS), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def read_acks(control_dir, limit=20):
    """The last `limit` acks, oldest first.

    The tail, not the head. This file grows for the whole run, and slicing from
    the front would freeze the page on the first twenty acks of fifteen days.
    """
    rows = []
    try:
        with open(os.path.join(control_dir, ACKS), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except (OSError, ValueError):
        # ValueError also catches UnicodeDecodeError from `for line in fh` on
        # invalid bytes -- unguarded, that 500s /api/status and /metrics.
        return []
    return rows[-limit:] if limit else rows


def serve(control_dir, handlers, stop_event, poll_s=1.0):
    """Poll for requests until `stop_event` is set. Runs on its own thread.

    `handlers` maps an action to a callable taking the request dict and
    returning (accepted, note). A handler that raises becomes a refused ack:
    this loop is the only way to stop the run, so it has to outlive one bad
    request.
    """
    while not stop_event.is_set():
        for path, request in take(control_dir):
            action = request.get("action")
            handler = handlers.get(action)
            if handler is None:
                accepted, note = False, (
                    f"this run does not handle {action!r}; it handles "
                    f"{', '.join(sorted(handlers))}")
            else:
                try:
                    accepted, note = handler(request)
                except Exception as exc:
                    accepted, note = False, repr(exc)
            # Delete before acking. A crash between the two leaves an
            # unanswered request, which is quiet; the other order leaves a
            # request that fires a second time, which is not.
            drop(path)
            ack(control_dir, request.get("id"), accepted, note, time.time())
        stop_event.wait(poll_s)


def _request_names(control_dir):
    """Request filenames only, oldest first. Never the ack log or a tmp file."""
    try:
        names = os.listdir(control_dir)
    except OSError:
        return []
    return sorted(n for n in names if n.endswith(SUFFIX))
