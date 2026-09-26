#!/usr/bin/env python3
# tools/archive-batch.py
"""Run the 2001-2007 dance archive through the dance-HQ BSVD pipeline.

Sources stay on gpu1. Encoding goes to whichever rostered encoder has a free
slot, here or on another host. See
the design notes, which are not part of this tree
"""
import json
import os
import re
import posixpath
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.archive_batch import ARCHIVE_ROOT, SOURCE_HOST
from tools.archive_batch import control
from tools.archive_batch.control import YieldRequested
from tools.archive_batch.dispatch_cmd import (build_command,
                                              build_encode_command)
from tools.archive_batch.manifest import (Clip, _frames_from, order_clips,
                                          parse_encode_manifest,
                                          parse_manifest)
from tools.archive_batch.netresolve import resolve_stream_ip
from tools.archive_batch.probe import probe_folder
from tools.archive_batch import pidfile
from tools.archive_batch.presets import PresetError, catalogue, load_preset
from tools.archive_batch.roster import RosterError, load_roster
from tools.archive_batch.scheduler import Scheduler
from tools.archive_batch.state import Record, append_record
from tools.archive_batch.state import (exhausted_clips, load_state,
                                        pending_clips)
from tools.archive_batch.transfer import (TransferError, TransferOutage,
                                          publish_cmd, run, safe_dest,
                                          stage_cmd, stage_job_cmd,
                                          staged_path)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A benchmark needs its own manifest, state and roster. Overriding the whole
# directory keeps it away from the real run's files: hand-swapping the live
# manifest under a running job is what destroyed it on 2026-08-11.
RUN_DIR = os.environ.get("ARCHIVE_RUN_DIR") or os.path.join(REPO, ".archive-run")
MANIFEST = os.path.join(RUN_DIR, "manifest-raw.tsv")
# Its own file, not more rows in manifest-raw.tsv: the two are different
# shapes, and one parser answering to both would have to guess which from the
# column count.
ENCODE_MANIFEST = os.path.join(RUN_DIR, "manifest-encode.tsv")
STATE = os.path.join(RUN_DIR, "state.jsonl")
ROSTER = os.path.join(RUN_DIR, "denoisers.toml")
LANES = os.path.join(RUN_DIR, "lanes")
CONTROL = os.path.join(RUN_DIR, "control")
ROSTER_ERROR = os.path.join(RUN_DIR, "roster-error.txt")
BATCH_FILE = os.path.join(RUN_DIR, "batch.json")
STAGE_ROOT = os.path.join(REPO, "Temp", "_stage")
# encoder-host's LAN address; tailscale caps at 1.5 Gbps. Overridable because the
# remotes reach this host by different routes in different setups, and because
# 127.0.0.1 plus a reverse ssh tunnel per lane is the way to run without opening
# a firewall port -- measured at 244 MB/s to gpu4, well above what any lane
# produces.
CALLBACK_IP = os.environ.get("ARCHIVE_CALLBACK_IP") or "10.0.0.10"

# Every transfer is already bounded, so the dispatch call was the one unbounded
# wait in the run: a deadlocked vspipe or a half-open ssh holds its lane until
# somebody notices, which over 15 unattended days means it never gets noticed.
# The budget is deliberately loose. It only has to catch a hang, and a false
# kill spends one of the clip's two attempts. The slowest rostered denoiser
# measures about 4.4 fps at 1080p, so budgeting one frame per second leaves more
# than four times the margin even before the floor.
DISPATCH_FLOOR_S = 3600.0
DISPATCH_FPS_FLOOR = 1.0
# Clip length spans seconds to tens of minutes, so a clip whose frame count did not
# parse must fall back to the ceiling. The floor would kill a long clip that is
# encoding correctly.
DISPATCH_UNKNOWN_S = 86400.0

SAMPLER = os.path.join(REPO, "tools", "host-sampler.py")
# Sampling every quarter second rather than every second because the thing worth
# catching is a lane whose GPU swings between 0 and 96% inside one second. A
# one-second sample averages that away into a healthy-looking number.
TRACE_INTERVAL_MS = 250


# The first line matching one of these is the root cause; everything after a
# CUDA fault is the teardown cascade, which is what a plain tail captures.
#
# The dispatch's own fatal marker is in here because a refusal it raises itself
# matches none of the library patterns. "cannot bind ... Address already in use"
# was the live example: dispatch printed exactly what was wrong and the tail
# reported a TensorRT warning from the other host instead.
_ROOT_CAUSE = re.compile(
    r"CUDA failure|Failed to retrieve frame|RuntimeError|MyelinCheckException|"
    r"out of memory|Traceback|Segmentation fault|Killed|assert|"
    r"Error in execution|No such file|Permission denied|"
    r"\[svtav1-dispatch\] Error:", re.I)

# vspipe -p and netstream --progress both emit this, about once a second.
# Dropped before the tail is taken, for two reasons of different strength.
#
# The load-bearing one is the timeout kill: run_dispatch SIGKILLs the process
# group, so the log simply stops, and its last lines are pure counters with no
# error anywhere. Unfiltered, the reason recorded for a hung clip would be four
# frame numbers.
#
# The weaker one is an ordinary failure. vspipe stops printing progress once it
# errors, so the traceback does end up last -- but every counter still inside
# the four-line tail is a slot not spent on context.
_PROGRESS = re.compile(r"^Frame:\s*\d+")


def log_tail(temp_dir, stem, lines=4, limit=600):
    """What the dispatch logged for this clip, root cause first.

    The remote log is the useful one: a split-host failure is almost always the
    denoise half, and its stderr never reaches this process.

    A plain tail is the wrong end of the file. An illegal memory access inside
    TensorRT emits one line naming the frame that failed and then thirty lines
    of destructor errors, so the last six lines are always the cascade and never
    the cause. Observed on gpu2 2026-08-14: the record said only "Error Code 1
    ... in deallocate", and the line that mattered -- which frame, which
    failure -- had scrolled past. Lead with the first matching line, then the
    real tail for context.

    A log with a root cause beats a log that merely has content, and that
    ordering is the whole point rather than a refinement. With a remote encoder
    the encode half logs to `_encode.log` alone, while `_remote.log` always has
    content -- so returning at the first non-empty file reported the denoise
    host every time. Seen on 2026-08-26: an encode refused with "cannot bind ...
    Address already in use" was recorded as a vstrt TensorRT version warning
    from gpu1, on a clip whose denoise half had finished cleanly.

    `_encode.log` is scanned first because a refusal there precedes any denoise
    symptom: the encoder never started, so nothing downstream of it is a cause.
    That cannot mask a denoise failure, because a healthy encode log carries no
    root-cause line at all -- checked across the gate run's encode logs, none
    matched, while every one of them had content.
    """
    # Cause order and fallback order differ, and deliberately. An encode-side
    # refusal outranks anything, but with no cause anywhere the denoise half is
    # still the likely story, so the fallback keeps the original preference.
    cause_order = ("_encode.log", "_remote.log", "_vspipe.log", "_netstream.log")
    fallback_order = ("_remote.log", "_vspipe.log", "_netstream.log",
                      "_encode.log")
    rendered = {}
    for suffix in cause_order:
        path = os.path.join(temp_dir, f"{stem}{suffix}")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                body = [ln.strip() for ln in fh.read().splitlines()
                        if ln.strip() and not _PROGRESS.match(ln.strip())]
        except OSError:
            continue
        if not body:
            continue
        cause = next((ln for ln in body if _ROOT_CAUSE.search(ln)), None)
        parts = ([f"CAUSE: {cause}"] if cause else []) + body[-lines:]
        text = f"{suffix[1:]}: " + " | ".join(parts)[:limit]
        if cause:
            return text
        rendered[suffix] = text
    for suffix in fallback_order:
        if suffix in rendered:
            return rendered[suffix]
    return ""


def dispatch_timeout(frames):
    if frames <= 0:
        return DISPATCH_UNKNOWN_S
    return DISPATCH_FLOOR_S + frames / DISPATCH_FPS_FLOOR


# Which lane is running which dispatch, and which lanes were killed on purpose.
# The first shared mutable state in this file, and it needs a real lock:
# adding, looking up and deleting is three steps, and every one of the
# scheduler's worker threads calls run_dispatch at once.
_lock = threading.Lock()
_procs = {}
_yielded = set()


def _killpg(proc):
    """SIGTERM then SIGKILL the whole group, waiting 30s after each."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            return          # the group is already gone
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            continue


def yield_lane(name):
    """Kill this lane's dispatch now. True if there was one.

    Spec 5.2. The flag is set before the signal and read by the runner after
    the process dies, because a killed group returns from proc.wait() normally
    -- no TimeoutExpired, just a negative return code that looks exactly like a
    crash. Without the flag the clip would be recorded failed and spend one of
    its two attempts, which is the outcome spec 5.3 forbids.

    Up to a minute in the worst case: the kill loop budgets 30s per signal.
    """
    with _lock:
        proc = _procs.get(name)
        # poll() as well as the lookup, both under the lock. run_dispatch's
        # wait() reaps the pid a few instructions before its finally clears the
        # entry, and killpg on a reaped pid can signal whatever the kernel has
        # since put in that process group. poll() is non-blocking and takes the
        # object's own _waitpid_lock, so it is safe while the worker is in
        # wait().
        if proc is None or proc.poll() is not None:
            return False
        _yielded.add(name)
    _killpg(proc)
    return True


def _take_yield(name):
    """Was this lane yielded? Clears the flag, so it answers once.

    Read-and-clear under the one lock, not `in` then `discard`: two workers
    finishing at the same instant must not both decide the yield was theirs.
    """
    with _lock:
        if name in _yielded:
            _yielded.discard(name)
            return True
        return False


def run_dispatch(argv, env, timeout, lane=None):
    """Run one dispatch. Return (returncode, timed_out).

    start_new_session puts dispatch and everything it spawns -- vspipe, ssh, the
    encoder -- into one process group, so the kill reaches whichever child is
    actually stuck. Killing the dispatch alone would leave them running and the
    lane would stay blocked anyway. yield_lane reuses that same group kill.

    `lane` registers this process so yield_lane can find it. Left out, nothing
    is registered and the behaviour is exactly what it was.
    """
    proc = subprocess.Popen(argv, cwd=REPO, env=env, start_new_session=True)
    if lane is not None:
        with _lock:
            _procs[lane] = proc
    try:
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            pass
        _killpg(proc)
        return proc.poll(), True
    finally:
        if lane is not None:
            with _lock:
                _procs.pop(lane, None)


def clear_remote_stage(denoiser, clip):
    """Drop the copy dispatch rsynced to a GPU-only host.

    dispatch stages into Temp/_remote and never clears it, so a host that takes
    a fifth of this 2.46 TiB archive keeps about 300 GB of sources it has
    already finished with, and one that takes more can fill its disk part-way
    through a run that is meant to last weeks. gpu1 is unaffected: it holds the
    archive and reads in place, so nothing is staged there at all.

    Best effort. A clip that encoded and published correctly must not be
    recorded as failed because a cleanup ssh timed out.
    """
    root = denoiser.root or "~/archav1an"
    name = os.path.basename(clip.src)
    remote = f"{root}/Temp/_remote/{shlex.quote(name)}"
    try:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                        denoiser.host, f"rm -f {remote}"],
                       capture_output=True, timeout=60)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[archive-batch] could not clear {denoiser.host}:{remote}: {exc!r}",
              file=sys.stderr)


def start_trace(denoiser, clip, budget):
    """Sample the denoise host for the length of one dispatch. None if off.

    The rate in the state file says a lane was slow. It never says why, and by
    the time a clip finishes the evidence is gone. This writes one CSV per clip
    per lane while the work is happening, so "was the run traced?" has an answer
    that is not "no".

    The sampler runs on the host being sampled, so a remote lane gets the script
    piped over ssh rather than run from the remote checkout: the trace must not
    depend on that checkout being current, which is exactly the condition you
    are most likely to be debugging.

    Never fatal. A failed trace loses a diagnostic; a failed clip loses an hour.
    """
    trace_dir = os.path.join(RUN_DIR, "trace")
    csv = os.path.join(trace_dir, f"{denoiser.name}-{clip.stem}.csv")
    args = ["--interval-ms", str(TRACE_INTERVAL_MS),
            # Bound it by the dispatch budget as well as by the kill below. An
            # ssh that dies leaves the remote python orphaned, and an orphan
            # sampling every 250 ms for a day is worse than no sampler at all.
            "--max-seconds", str(int(budget) + 60),
            "--pid-of", "vspipe -c y4m"]
    try:
        os.makedirs(trace_dir, exist_ok=True)
        handle = open(csv, "wb")
        if denoiser.is_remote:
            with open(SAMPLER, "rb") as fh:
                script = fh.read()
            # -u matters more than it looks: stdout is a pipe, so without it the
            # remote python block-buffers and the whole trace is still sitting
            # in an 8 KB buffer when the ssh channel closes. That loses the file
            # silently -- the first version of this wrote a zero-row CSV.
            proc = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                 denoiser.host, "python3 -u - " + " ".join(shlex.quote(a) for a in args)],
                stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.DEVNULL,
                start_new_session=True)
            proc.stdin.write(script)
            proc.stdin.close()
        else:
            proc = subprocess.Popen([sys.executable, SAMPLER] + args,
                                    stdout=handle, stderr=subprocess.DEVNULL,
                                    start_new_session=True)
        return proc, handle
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[archive-batch] trace for {denoiser.name} did not start: {exc!r}",
              file=sys.stderr)
        return None


def stop_trace(trace):
    """SIGTERM the sampler so it flushes, then let the file close."""
    if trace is None:
        return
    proc, handle = trace
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    handle.close()


def make_runner(trace=False):
    def runner(clip, denoiser, encoder, slot):
        phases = dict(stage_s=0.0, work_s=0.0, publish_s=0.0)
        # Per clip, not per run: gpu2 and gpu3 move, and a lease that
        # changed mid-run must heal without a restart (spec 5.6).
        #
        # NoTrustedAddress is deliberately NOT caught here. It is raised
        # before a single frame is staged, so it is a fact about the encoder
        # and none at all about the clip. The scheduler puts the clip back
        # untouched and quarantines the encoder. Returned as a clip failure it
        # was a failure in 0.3 s, and a fast-failing encoder is always the
        # free one, so one laptop off the LAN took every queued clip to its
        # attempt ceiling within seconds and dropped it from the manifest for
        # good, while the run reported success.
        encoder = replace(encoder, stream_ip=resolve_stream_ip(encoder))
        stage_dir = os.path.join(STAGE_ROOT, denoiser.name)
        os.makedirs(stage_dir, exist_ok=True)
        # Must match --temp-tag in dispatch_cmd: 185 stems repeat across the
        # archive, so a stem-only temp dir lets one worker delete another's
        # working files mid-encode (spec 5.1 step 2).
        temp_dir = os.path.join(REPO, "Temp", denoiser.name, clip.stem)
        shutil.rmtree(temp_dir, ignore_errors=True)

        started = time.monotonic()
        staged = staged_path(stage_dir, clip.src)
        out = os.path.join(stage_dir, f"{clip.stem}-av1.mkv")
        try:
            # Resolved before staging: a preset renamed away since the folder
            # was submitted is a fact about the request and none about the
            # file, and there is no point copying gigabytes to learn it.
            preset = load_preset(clip.preset) if clip.preset else None
            _t = time.monotonic()
            if clip.is_encode_job:
                run(stage_job_cmd(clip.src_host, clip.src, stage_dir))
            else:
                run(stage_cmd(SOURCE_HOST, clip.src, stage_dir))
            phases["stage_s"] = time.monotonic() - _t
            if clip.is_encode_job:
                # No denoiser, so no BSVD flags and no remote-denoise half.
                # denoiser.name is <encoder>-<slot> here, which is what keeps
                # two slot workers on one host out of each other's temp dir.
                argv, env_overlay = build_encode_command(
                    encoder, slot, staged=staged, out=out,
                    temp_tag=denoiser.name, preset=preset)
            else:
                argv, env_overlay = build_command(
                    denoiser, encoder, slot, staged=staged, out=out,
                    # None makes dispatch rsync the clip to the remote's
                    # Temp/_remote instead of reading it in place, which is the
                    # only mode a host without the archive can use.
                    remote_src=(f"{ARCHIVE_ROOT}/{clip.src}"
                                if denoiser.is_remote and not denoiser.stage_source
                                else None),
                    callback=CALLBACK_IP)
            env = dict(os.environ)
            env.update(env_overlay)
            budget = dispatch_timeout(clip.frames)
            tracer = start_trace(denoiser, clip, budget) if trace else None
            _t = time.monotonic()
            try:
                rc, timed_out = run_dispatch(argv, env, budget,
                                             lane=denoiser.name)
            finally:
                # First in this finally, not last. Statements here run in
                # order, so a read placed below stop_trace is skipped by the
                # very exception it is meant to survive -- and the flag would
                # still be set when this lane takes its next clip, whose own
                # ordinary dispatch failure would then be misread as a yield
                # and requeued for ever with no attempt spent.
                yielded = _take_yield(denoiser.name)
                phases["work_s"] = time.monotonic() - _t
                stop_trace(tracer)
            if timed_out or rc != 0 or not os.path.exists(out):
                if yielded:
                    # Not a failure. Raised rather than returned, like
                    # TransferOutage, so the scheduler requeues the clip
                    # without spending an attempt or writing a record.
                    raise YieldRequested(
                        f"{denoiser.name} was yielded, so its dispatch was "
                        f"killed")
                if timed_out:
                    why = f"dispatch hung: killed after {budget:.0f}s"
                elif rc:
                    why = f"dispatch exit {rc}"
                else:
                    why = "dispatch exit 0 but no output file"
                # Read the log before the finally below deletes temp_dir.
                tail = log_tail(temp_dir, clip.stem)
                return False, time.monotonic() - started, 0.0, 0, \
                    f"{why}. {tail}".strip(), phases
            _t = time.monotonic()
            run(publish_cmd(SOURCE_HOST, out, clip.rel_dir))
            phases["publish_s"] = time.monotonic() - _t
            size = os.path.getsize(out)
        except TransferOutage:
            # The host is down, not the clip bad. Let the scheduler requeue it
            # rather than spend one of this clip's two attempts (spec 6).
            raise
        except PresetError as exc:
            # Failing beats quietly encoding a folder of anime at the dance
            # settings, which is the whole reason a preset is named.
            print(f"[archive-batch] {clip.src}: {exc}", file=sys.stderr)
            return False, time.monotonic() - started, 0.0, 0, str(exc), phases
        except TransferError as exc:
            print(f"[archive-batch] {clip.src}: {exc}", file=sys.stderr)
            return False, time.monotonic() - started, 0.0, 0, str(exc), phases
        finally:
            for path in (staged, out):
                if os.path.exists(path):
                    os.remove(path)
            shutil.rmtree(temp_dir, ignore_errors=True)
            # A _SlotLane carries only a name and a host, so these two
            # attributes exist on a denoise job's Denoiser and nowhere else.
            if (not clip.is_encode_job and denoiser.is_remote
                    and denoiser.stage_source):
                clear_remote_stage(denoiser, clip)

        wall = time.monotonic() - started
        return True, wall, (clip.frames / wall if wall else 0.0), size, "", phases
    return runner


def _roster():
    return load_roster(ROSTER)


def _write_batch_file():
    # Same as state.py, heartbeat.py and control.py: the run directory may not
    # exist yet on the first run of a fresh checkout.
    os.makedirs(RUN_DIR, exist_ok=True)
    tmp = BATCH_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"batch_pid": os.getpid(),
                   "pid_start": pidfile.start_time(os.getpid()),
                   "started_at": time.time()}, fh)
    os.replace(tmp, BATCH_FILE)


def _clear_batch_file():
    try:
        os.remove(BATCH_FILE)
    except OSError:
        pass


def sweep_stage_root():
    """Drop staged sources left behind by a run that was killed.

    Each interrupted clip strands up to about 3 GB. Resume only reads the state
    file, so a leftover is never picked up again -- it just sits there.
    """
    freed = 0
    for dirpath, _dirs, files in os.walk(STAGE_ROOT):
        for name in files:
            path = os.path.join(dirpath, name)
            try:
                freed += os.path.getsize(path)
                os.remove(path)
            except OSError:
                pass
    return freed


def per_denoiser_rates(state_path):
    """Per denoiser: (clips, fps including overhead, fps excluding it).

    Two rates because they answer different questions. The first divides by the
    whole clip's wall clock, so it is the rate the archive actually drains at
    and the one that predicts a finish date. The second divides by the dispatch
    alone, so it is how fast that denoiser and encoder pair really is, with the
    source copy in and the result copy out taken out. A lane that stages every
    source across the network can differ a lot between the two, and only the
    gap tells you whether to buy a faster card or a faster link.
    """
    rates = {}
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") != "done":
                    continue
                name = row.get("denoiser", "?")
                clips, frames, wall, work = rates.get(name, (0, 0.0, 0.0, 0.0))
                # fps*wall recovers the frame count the record does not store.
                rates[name] = (clips + 1,
                               frames + row.get("fps", 0.0) * row.get("wall_s", 0.0),
                               wall + row.get("wall_s", 0.0),
                               work + row.get("work_s", 0.0))
    except OSError:
        return {}
    return {name: (clips,
                   frames / wall if wall else 0.0,
                   frames / work if work else 0.0)
            for name, (clips, frames, wall, work) in rates.items()}


def format_summary(done, failed, failures, elapsed_s, state_path=None):
    lines = [
        "",
        "=" * 70,
        f"archive-batch: {done} done, {failed} failed in {elapsed_s / 3600:.2f} h",
    ]
    rates = per_denoiser_rates(state_path) if state_path else {}
    if rates:
        lines.append("")
        lines.append(f"  {'lane':<12} {'clips':>5}  {'fps':>7}  {'fps':>7}")
        lines.append(f"  {'':<12} {'':>5}  {'(total)':>7}  {'(work)':>7}")
        pool_total = pool_work = 0.0
        for name in sorted(rates):
            clips, fps, work_fps = rates[name]
            pool_total += fps
            pool_work += work_fps
            lines.append(f"  {name:<12} {clips:>5}  {fps:7.2f}  {work_fps:7.2f}")
        lines.append(f"  {'-' * 34}")
        # The pool figure is the sum of the lanes, which is what the encoder
        # actually has to absorb. It is not a measurement of the encoder.
        lines.append(f"  {'pool':<12} {done:>5}  {pool_total:7.2f}  {pool_work:7.2f}")
        lines.append("  (total) includes staging the source in and publishing the "
                     "result out; (work) is the dispatch alone.")
    if failures:
        lines.append("")
        lines.append("FAILED CLIPS")
        for src, denoiser, reason in failures:
            lines.append(f"  {src}  [{denoiser}]  {reason}")
    lines.append("=" * 70)
    return "\n".join(lines)


# Both retry shapes. A named constant because main() and _take_queued_retries
# must never disagree about it: passing only "retry" here left every retry-all
# to be swept up by clear() as stale, and the run then printed "nothing to do"
# with the request silently gone.
RETRY_ACTIONS = ("retry", "retry-all")


def _take_queued_retries(control_dir=None):
    """Lift both retry shapes out of the control directory, before clear().

    A function rather than a call inline in main() so a test covers the step
    main() actually performs. Testing take_action alone proved the helper and
    not the caller, which is exactly where the action list went wrong.
    """
    return control.take_action(control_dir or CONTROL, RETRY_ACTIONS)


def _apply_queued_retries(requests, by_src, state_path=None, control_dir=None):
    """Act on retry requests that arrived while no run was going.

    Mirrors on_retry below, minus the scheduler: there is no queue to put the
    clip back on yet, and none is needed. on_retry writes the record to make
    the NEXT run eligible -- this is that run, and it has not read the state
    file yet, so the record alone is the whole job.

    `done` is read once. Nothing appended here can add to it, and re-reading a
    15,000-line state file per request would make a bulk retry quadratic.
    """
    state_path = state_path or STATE
    control_dir = control_dir or CONTROL
    if not requests:
        return 0
    state = load_state(state_path)
    done = state.done
    applied = 0
    # Which clips this drain has already put back. `state` is read once, so a
    # second retry-all in the same batch would re-list every clip the first one
    # reset and report them twice -- two clicks, or a stale request beside a
    # fresh one, and the count stops meaning anything. The extra records are
    # harmless (a retry row re-zeroes an already-zero count), the number is not.
    seen = set()
    for request in requests:
        rid = request.get("id", "")
        if request.get("action") == "retry-all":
            # Read from the state file, not from the request: the page cannot
            # send a list of what to retry, because its failure panel is a
            # capped preview and the clips it dropped are still exhausted.
            batch = [s for s in exhausted_clips(state)
                     if s in by_src and s not in seen]
            seen.update(batch)
            for src in batch:
                append_record(state_path,
                              Record(src, "retry", "", 0.0, 0.0, 0))
            applied += len(batch)
            control.ack(control_dir, rid, True,
                        f"{len(batch)} clip(s) out of attempts are back on the "
                        f"queue with their failure counts reset", time.time())
            continue
        src = request.get("src")
        if src not in by_src:
            control.ack(control_dir, rid, False,
                        f"the manifest holds no clip {src!r}", time.time())
            continue
        if src in done:
            control.ack(control_dir, rid, False,
                        f"{os.path.basename(src)} is already finished",
                        time.time())
            continue
        if src in seen:
            # A bulk retry earlier in this same drain already put it back.
            # Appending a second retry row is harmless, but counting it again
            # is exactly the inflated number `seen` exists to prevent.
            control.ack(control_dir, rid, True,
                        f"{os.path.basename(src)} was already put back by a "
                        f"bulk retry in this batch", time.time())
            continue
        seen.add(src)
        append_record(state_path, Record(src, "retry", "", 0.0, 0.0, 0))
        applied += 1
        control.ack(control_dir, rid, True,
                    f"{os.path.basename(src)} is back on the queue and its "
                    f"failure count is reset", time.time())
    return applied


SUBMIT_ACTION = "submit"


def _take_queued_submits(control_dir=None):
    """Lift submissions out of the control directory, before clear().

    Same reason as the retries above, and a stronger one. A submission names
    no live lane and no live run, so clear() dropping it was pure loss -- and
    on a drained archive the startup is the only place it can ever be acted
    on, because a run with nothing to do exits before the control poller that
    would answer it starts. Queue a folder, press Start, and the request was
    swept away with a "nothing to do" as the only sign anything had happened.
    """
    return control.take_action(control_dir or CONTROL, SUBMIT_ACTION)


def _apply_queued_submits(requests, known_srcs=(), control_dir=None):
    """Act on submissions that arrived while no run was going.

    Returns the Clips to add to this run. Mirrors on_submit below, minus the
    scheduler: there is no queue yet, and none is needed -- a job added to the
    manifest here is picked up by pending_clips a few lines later, exactly as
    one read from the encode manifest is.

    The probe is serial and blocking, where on_submit runs it on a thread. The
    reason that thread exists is that a control handler which blocks stops the
    poller from answering a stop; nothing is polling yet here, and a run that
    starts before its own queue is known would report "nothing to do".
    """
    control_dir = control_dir or CONTROL
    if not requests:
        return ()
    out = []
    known = set(known_srcs)
    for request in requests:
        rid = request.get("id", "")
        path = (request.get("path") or "").strip()
        host = (request.get("host") or "").strip()
        preset = (request.get("preset") or "").strip()
        if not path:
            control.ack(control_dir, rid, False,
                        "a submission must name a folder", time.time())
            continue
        try:
            dest = safe_dest(request.get("dest"))
            if preset:
                load_preset(preset)
            rows = probe_folder(host, path)
        except (TransferError, PresetError) as exc:
            control.ack(control_dir, rid, False, str(exc), time.time())
            continue
        if not rows:
            control.ack(control_dir, rid, False,
                        f"no .MOV or .MP4 directly in {path}", time.time())
            continue
        # Deduplicated against what the manifest already holds, and against
        # what an earlier request in this same drain added. scheduler.submit
        # refuses a duplicate mid-run for a reason that applies just as much
        # here: two copies of one job are encoded twice and published to one
        # destination path at the same time.
        fresh = [r for r in rows if posixpath.join(path, r[0]) not in known]
        for row in fresh:
            known.add(posixpath.join(path, row[0]))
        if fresh:
            _append_encode_manifest(host, dest, preset, path, fresh)
        out.extend(
            Clip(src=posixpath.join(path, name), rel_dir=dest,
                 stem=os.path.splitext(name)[0], size=size,
                 frames=_frames_from(rate), denoise=False, src_host=host,
                 preset=preset)
            for name, size, rate in fresh)
        skipped = len(rows) - len(fresh)
        control.ack(control_dir, rid, True,
                    f"queued {len(fresh)} encode job(s) from {path} into "
                    f"encoded/{dest}"
                    + (f"; {skipped} already in the manifest" if skipped
                       else ""),
                    time.time())
    return tuple(out)


def control_handlers(scheduler, by_src, state_path=None):
    """The three actions this run answers. See the spec, 4.2.

    A function rather than three closures inside main() so a test can call them
    without starting a run, and so main() stays readable. `scheduler` is a
    local in main(), which is why it arrives as an argument -- on_signal
    reaches it the same way, by being defined where it is visible.
    """
    state_path = state_path or STATE

    def on_yield(request):
        lane = request.get("lane")
        if not lane:
            return False, "a yield must name a lane"
        if yield_lane(lane):
            return True, (f"signalled {lane}'s dispatch; if it was still "
                          f"encoding, its clip goes back on the queue with no "
                          f"attempt spent")
        # yield_lane cannot tell a lane that does not exist from a real one
        # that is merely between dispatches -- _procs holds a lane only inside
        # run_dispatch, while staging, publishing and waiting for a slot all
        # hold a clip and beat, so the page offers the button for them too.
        # The roster separates the two, and they deserve opposite answers: a
        # typo is a refusal, a real lane between dispatches is not, because the
        # daemon disabled it before sending this and it stops at the clip
        # boundary either way.
        try:
            roster = scheduler.roster_fn()
        except Exception:
            roster = None
        if roster is not None and not any(d.name == lane
                                          for d in roster.denoisers):
            return False, f"no lane named {lane} is in the roster"
        return True, (f"{lane} had no dispatch to kill; it is disabled and "
                      f"stops after the clip it holds")

    def on_stop(_request):
        scheduler.stop()
        return True, (f"stopping; the clips in flight finish first and "
                      f"{scheduler.queue.qsize()} stay queued")

    def on_retry_all(_request):
        """Every clip this run has given up on, back at once.

        Reads the state file rather than a list from the page: the failure
        panel is a capped preview, so a list built from its rows would skip
        whatever did not fit while still reporting success. scheduler.retry
        refuses a clip already queued or in flight, and the record is written
        only for the ones it accepted -- the same pairing on_retry keeps, so a
        refused clip never gets its count reset.
        """
        back = 0
        for src in exhausted_clips(load_state(state_path)):
            clip = by_src.get(src)
            if clip is None or not scheduler.retry(clip):
                continue
            append_record(state_path, Record(src, "retry", "", 0.0, 0.0, 0))
            back += 1
        if not back:
            return True, "no clip is out of attempts; nothing to put back"
        return True, (f"{back} clip(s) out of attempts are back on the queue "
                      f"with their failure counts reset")

    def on_retry(request):
        src = request.get("src")
        clip = by_src.get(src)
        if clip is None:
            return False, f"the manifest holds no clip {src!r}"
        # by_src is built from the whole manifest, not just todo, so a clip
        # that already has a "done" record is still in it -- the scheduler
        # never learns of done clips at all, so nothing below this would catch
        # one. Re-encoding it would overwrite a good output with an identical
        # one at the cost of three hours (see test_state.py's
        # test_a_retry_does_not_resurrect_a_finished_clip). Checked before the
        # requeue and before the record, so a refused retry writes nothing.
        if src in load_state(state_path).done:
            return False, f"{os.path.basename(src)} is already finished"
        # scheduler.retry makes THIS run eligible and refuses a clip already
        # queued or already being encoded. The record makes the NEXT run
        # eligible, and is written only if the requeue actually happened --
        # writing it for a refused retry would reset a count nothing asked to
        # reset.
        if not scheduler.retry(clip):
            return False, (f"{os.path.basename(src)} is already queued or "
                           f"being encoded right now")
        append_record(state_path, Record(src, "retry", "", 0.0, 0.0, 0))
        return True, (f"{os.path.basename(src)} is back on the queue and its "
                      f"failure count is reset")

    def on_submit(request):
        """Queue a folder of encode jobs. The probe runs on its own thread.

        The probe walks a network path with one ffprobe per file, and a
        control handler that blocks stops serve() from answering a stop. So
        this acks "probing" at once and the thread acks the count when it is
        done -- two acks for one request, which the ack log already allows.
        """
        path = (request.get("path") or "").strip()
        host = (request.get("host") or "").strip()
        preset = (request.get("preset") or "").strip()
        request_id = request.get("id")
        if not path:
            return False, "a submission must name a folder"
        try:
            dest = safe_dest(request.get("dest"))
        except TransferError as exc:
            return False, str(exc)
        # Checked here, not when the first job runs. The submission is the
        # moment somebody is watching; a name refused now costs one ack, and
        # the same name refused later costs a probe and a staged copy apiece.
        if preset:
            try:
                load_preset(preset)
            except PresetError as exc:
                return False, str(exc)

        def work():
            try:
                rows = probe_folder(host, path)
            except TransferError as exc:
                control.ack(CONTROL, request_id, False, str(exc), time.time())
                return
            if not rows:
                control.ack(CONTROL, request_id, False,
                            f"no .MOV or .MP4 directly in {path}", time.time())
                return
            clips = [
                Clip(src=posixpath.join(path, name), rel_dir=dest,
                     stem=os.path.splitext(name)[0], size=size,
                     frames=_frames_from(rate), denoise=False, src_host=host,
                     preset=preset)
                for name, size, rate in rows]
            _append_encode_manifest(host, dest, preset, path, rows)
            queued = scheduler.submit(clips)
            skipped = len(clips) - queued
            control.ack(CONTROL, request_id, True,
                        f"queued {queued} encode job(s) from {path} into "
                        f"encoded/{dest}"
                        + (f"; {skipped} already queued or in flight"
                           if skipped else ""),
                        time.time())

        threading.Thread(target=work, daemon=True).start()
        return True, (f"probing {path} on {host or 'this host'}; the count "
                      f"follows when the walk finishes")

    return {"yield": on_yield, "stop": on_stop, "retry": on_retry,
            "retry-all": on_retry_all,
            "submit": on_submit}


def _append_encode_manifest(host, dest, preset, folder, rows):
    """Record what was submitted, so a resumed run still knows about it.

    The encode manifest is read at startup exactly as manifest-raw.tsv is, so
    without this a submitted folder would vanish on the next restart while
    state.jsonl still held its half-finished records.
    """
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(ENCODE_MANIFEST, "a", encoding="utf-8") as fh:
        for name, size, rate in rows:
            fh.write(f"{host}\t{dest}\t{preset}\t"
                     f"{posixpath.join(folder, name)}\t"
                     f"{size}\t{rate}\t\n")


def _claim_run():
    """Own the run dir before any slow startup work.

    O_EXCL makes the claim authoritative: a second batch (started from a
    shell while this one is still starting) sees a live different pid and
    exits, so two schedulers never run over the same manifest. A batch the
    daemon spawned may find the file already present -- encode_dash writes
    it right after Popen -- with our own pid, which is our claim, not a
    competitor's.

    Liveness is decided with the recorded /proc identity, not existence
    alone: a recycled pid would otherwise block this start forever or make
    us exit for a batch that no longer exists.

    The whole claim -- fast path included -- sits under pidfile.exclusive.
    The fast path alone is not enough: between os.open(O_EXCL) and the
    json.dump the file exists but is zero-length, and a concurrent batch
    that read that window would mistake it for a stale claim and replace it
    out from under us. Under the lock the second batch waits, then reads
    the finished record; two batches that both read the same genuinely
    stale file also serialize here, so only the first takes over.
    """
    os.makedirs(RUN_DIR, exist_ok=True)
    with pidfile.exclusive(BATCH_FILE + ".lock"):
        claim = {"batch_pid": os.getpid(),
                 "pid_start": pidfile.start_time(os.getpid()),
                 "started_at": time.time()}
        try:
            fd = os.open(BATCH_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            try:
                with open(BATCH_FILE, "r", encoding="utf-8") as fh:
                    row = json.load(fh)
            except (OSError, ValueError):
                row = None
            other = row.get("batch_pid") if isinstance(row, dict) else None
            if other == os.getpid():
                return            # the daemon's spawn record is our claim
            other_start = row.get("pid_start") if isinstance(row, dict) else None
            if isinstance(other, int) and pidfile.alive(other, other_start):
                print(f"[archive-batch] another batch is already running "
                      f"(pid {other}); exiting", file=sys.stderr)
                sys.exit(3)
            # Stale or unreadable claim from a dead batch: take it over,
            # still under the lock.
            _write_batch_file()
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(claim, fh)


def _mark_claimed():
    """Mark the run claimed after the control dir is cleared.

    The dashboard (encode_dash.model) shows a batch running only once this
    marker appears, so Stop is offered only after stale control requests
    are gone and a request written after that point survives.
    """
    tmp = BATCH_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"batch_pid": os.getpid(),
                   "pid_start": pidfile.start_time(os.getpid()),
                   "started_at": time.time(), "claimed": True}, fh)
    os.replace(tmp, BATCH_FILE)


def main():
    # Own the run before any slow startup work. The daemon's start()
    # checks batch.json before spawning, so this claim stops a second
    # scheduler over the same manifest; and clearing the control dir
    # before the roster/manifest/state/sweep work means a Stop written
    # once the dashboard shows us running is never dropped as stale.
    _claim_run()
    # Before clear(), which would delete these along with the stale yields.
    # Applied further down, once the manifest is loaded and can say whether
    # each named clip exists -- but taken here, in the same breath as the
    # clear, so no window exists in which one is dropped.
    queued_retries = _take_queued_retries(CONTROL)
    queued_submits = _take_queued_submits(CONTROL)
    dropped = control.clear(CONTROL)
    if dropped:
        print(f"[archive-batch] dropped {dropped} stale control request(s)")
    _mark_claimed()
    try:
        roster = _roster()
    except RosterError as exc:
        print(f"[archive-batch] Error: {exc}", file=sys.stderr)
        _clear_batch_file()
        return 2

    try:
        with open(MANIFEST, "r", encoding="utf-8") as fh:
            clips = order_clips(parse_manifest(fh.read()))
    except OSError as exc:
        print(f"[archive-batch] Error: cannot read the manifest: {exc}\n"
              f"Build it by probing the source host, as the spec describes in 5.3.",
              file=sys.stderr)
        _clear_batch_file()
        return 2

    # Encode jobs go AFTER order_clips, in manifest order. order_clips sorts on
    # SetA/SetB and a year folder, which an arbitrary submitted path does
    # not have, so applying it here would sort them by a rule that means
    # nothing for them. A missing file is the normal state: most runs have no
    # encode jobs at all.
    try:
        with open(ENCODE_MANIFEST, "r", encoding="utf-8") as fh:
            jobs = parse_encode_manifest(fh.read())
    except OSError:
        jobs = ()
    # Folders queued from the page while no run was going. Acted on here and
    # not left to the control poller: the poller starts after the "nothing to
    # do" exit below, so a folder submitted against a drained archive would
    # never reach it.
    jobs = jobs + _apply_queued_submits(queued_submits,
                                        {j.src for j in jobs})
    if jobs:
        print(f"[archive-batch] {len(jobs)} encode job(s) in "
              f"{os.path.basename(ENCODE_MANIFEST)}")
        clips = clips + jobs
    reset = _apply_queued_retries(queued_retries, {c.src: c for c in clips})
    if reset:
        print(f"[archive-batch] {reset} clip(s) retried by request; their "
              f"failure counts are reset")
    state = load_state(STATE)
    todo = pending_clips(clips, state)

    print(f"[archive-batch] {len(clips)} clips in manifest, {len(todo)} to do")
    print(f"[archive-batch] denoisers: {[d.name for d in roster.enabled()]}, "
          f"encoders: "
          f"{[f'{e.name} x{e.slots} --lp {e.lp_level}' for e in roster.enabled_encoders()]}")
    if not todo:
        print("[archive-batch] nothing to do.")
        _clear_batch_file()
        return 0

    freed = sweep_stage_root()
    if freed:
        print(f"[archive-batch] cleared {freed / 1073741824:.2f} GiB of staged "
              f"sources left by an interrupted run")

    started = time.monotonic()
    # An env var rather than a flag, to match RUN_DIR and CALLBACK_IP: this tool
    # is always launched with an env prefix already.
    trace = os.environ.get("ARCHIVE_TRACE", "") not in ("", "0")
    if trace:
        print(f"[archive-batch] tracing to {os.path.join(RUN_DIR, 'trace')}, "
              f"one CSV per clip per lane")
    scheduler = Scheduler(todo, _roster, make_runner(trace=trace),
                          STATE, prior_failures=state.failures, lanes_dir=LANES,
                          roster_error_path=ROSTER_ERROR)

    # Says this process is alive without holding a clip. A run parked because
    # every lane was yielded writes no heartbeat, and the daemon would
    # otherwise call it dead (spec 5.2).
    control_stop = threading.Event()
    poller = threading.Thread(
        target=control.serve, daemon=True,
        args=(CONTROL, control_handlers(scheduler, {c.src: c for c in clips}),
              control_stop))
    poller.start()

    def on_signal(signum, _frame):
        # Restore the default so a second press aborts at once; the first press
        # lets the clips in flight finish and be recorded, which is what makes
        # the run resumable at any point.
        signal.signal(signum, signal.SIG_DFL)
        print(f"\n[archive-batch] caught signal {signum}: letting the clips in "
              f"flight finish, then stopping. Press again to abort now.",
              flush=True)
        scheduler.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)

    try:
        scheduler.run()
    finally:
        # The FIRST SIGINT or SIGTERM comes back through on_signal into a
        # normal return, so this runs. A second press does not: on_signal has
        # restored SIG_DFL by then, which is the point of it. Nor does SIGKILL.
        # Either way the file left behind names a pid that is now dead, and the
        # daemon checks the pid rather than the file existing.
        _clear_batch_file()
    control_stop.set()
    poller.join(2)
    print(format_summary(scheduler.done, scheduler.failed, scheduler.failures,
                         time.monotonic() - started, state_path=STATE))

    # An outage or a signal stops the run with clips still queued and nothing
    # recorded failed. Exiting 0 there would tell a supervising script the
    # archive is finished when it is not. Proven on 2026-08-11, when gpu1 took
    # a Windows-update reboot mid-run and this returned 0 with 10 clips left.
    remaining = scheduler.queue.qsize()
    if remaining:
        print(f"[archive-batch] stopped early with {remaining} clip(s) still "
              f"queued. Re-run to resume.")
        return 3
    return 1 if scheduler.failed else 0


if __name__ == "__main__":
    sys.exit(main())
