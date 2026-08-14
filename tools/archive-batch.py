#!/usr/bin/env python3
# tools/archive-batch.py
"""Run the 2001-2007 dance archive through the dance-HQ BSVD pipeline.

Sources stay on gpu1. Encoding always happens here. See
the design notes, which are not part of this tree
"""
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.archive_batch import ARCHIVE_ROOT, SOURCE_HOST
from tools.archive_batch.dispatch_cmd import build_command
from tools.archive_batch.manifest import order_clips, parse_manifest
from tools.archive_batch.roster import RosterError, load_roster
from tools.archive_batch.scheduler import Scheduler
from tools.archive_batch.state import load_state, pending_clips
from tools.archive_batch.transfer import (TransferError, TransferOutage,
                                          publish_cmd, run, stage_cmd,
                                          staged_path)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A benchmark needs its own manifest, state and roster. Overriding the whole
# directory keeps it away from the real run's files: hand-swapping the live
# manifest under a running job is what destroyed it on 2026-08-11.
RUN_DIR = os.environ.get("ARCHIVE_RUN_DIR") or os.path.join(REPO, ".archive-run")
MANIFEST = os.path.join(RUN_DIR, "manifest-raw.tsv")
STATE = os.path.join(RUN_DIR, "state.jsonl")
ROSTER = os.path.join(RUN_DIR, "denoisers.toml")
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


def log_tail(temp_dir, stem, lines=6, limit=600):
    """Last few lines of whatever the dispatch logged for this clip.

    The remote log is the useful one: a split-host failure is almost always the
    denoise half, and its stderr never reaches this process.
    """
    for suffix in ("_remote.log", "_vspipe.log", "_netstream.log"):
        path = os.path.join(temp_dir, f"{stem}{suffix}")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                tail = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
        except OSError:
            continue
        if tail:
            return f"{suffix[1:]}: " + " | ".join(tail[-lines:])[:limit]
    return ""


def dispatch_timeout(frames):
    if frames <= 0:
        return DISPATCH_UNKNOWN_S
    return DISPATCH_FLOOR_S + frames / DISPATCH_FPS_FLOOR


def run_dispatch(argv, env, timeout):
    """Run one dispatch. Return (returncode, timed_out).

    start_new_session puts dispatch and everything it spawns -- vspipe, ssh, the
    encoder -- into one process group, so the kill reaches whichever child is
    actually stuck. Killing the dispatch alone would leave them running and the
    lane would stay blocked anyway.
    """
    proc = subprocess.Popen(argv, cwd=REPO, env=env, start_new_session=True)
    try:
        return proc.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            break       # the group is already gone
        try:
            proc.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            continue
    return proc.poll(), True


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


def make_runner(encode):
    def runner(clip, denoiser):
        stage_dir = os.path.join(STAGE_ROOT, denoiser.name)
        os.makedirs(stage_dir, exist_ok=True)
        # Must match --temp-tag in dispatch_cmd: 185 stems repeat across the
        # archive, so a stem-only temp dir lets one worker delete another's
        # working files mid-encode (spec 5.1 step 2).
        temp_dir = os.path.join(REPO, "Temp", denoiser.name, clip.stem)
        shutil.rmtree(temp_dir, ignore_errors=True)

        started = time.monotonic()
        phases = dict(stage_s=0.0, work_s=0.0, publish_s=0.0)
        staged = staged_path(stage_dir, clip.src)
        out = os.path.join(stage_dir, f"{clip.stem}-av1.mkv")
        try:
            _t = time.monotonic()
            run(stage_cmd(SOURCE_HOST, clip.src, stage_dir))
            phases["stage_s"] = time.monotonic() - _t
            argv, env_overlay = build_command(
                denoiser, encode, staged=staged, out=out,
                # None makes dispatch rsync the clip to the remote's
                # Temp/_remote instead of reading it in place, which is the
                # only mode a host without the archive can use.
                remote_src=(f"{ARCHIVE_ROOT}/{clip.src}"
                            if denoiser.is_remote and not denoiser.stage_source
                            else None),
                callback=CALLBACK_IP if denoiser.is_remote else None)
            env = dict(os.environ)
            env.update(env_overlay)
            budget = dispatch_timeout(clip.frames)
            _t = time.monotonic()
            rc, timed_out = run_dispatch(argv, env, budget)
            phases["work_s"] = time.monotonic() - _t
            if timed_out or rc != 0 or not os.path.exists(out):
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
        except TransferError as exc:
            print(f"[archive-batch] {clip.src}: {exc}", file=sys.stderr)
            return False, time.monotonic() - started, 0.0, 0, str(exc), phases
        finally:
            for path in (staged, out):
                if os.path.exists(path):
                    os.remove(path)
            shutil.rmtree(temp_dir, ignore_errors=True)
            if denoiser.is_remote and denoiser.stage_source:
                clear_remote_stage(denoiser, clip)

        wall = time.monotonic() - started
        return True, wall, (clip.frames / wall if wall else 0.0), size, "", phases
    return runner


def _roster():
    return load_roster(ROSTER)


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


def main():
    try:
        roster = _roster()
    except RosterError as exc:
        print(f"[archive-batch] Error: {exc}", file=sys.stderr)
        return 2

    try:
        with open(MANIFEST, "r", encoding="utf-8") as fh:
            clips = order_clips(parse_manifest(fh.read()))
    except OSError as exc:
        print(f"[archive-batch] Error: cannot read the manifest: {exc}\n"
              f"Build it by probing the source host, as the spec describes in 5.3.",
              file=sys.stderr)
        return 2
    state = load_state(STATE)
    todo = pending_clips(clips, state)

    print(f"[archive-batch] {len(clips)} clips in manifest, {len(todo)} to do")
    print(f"[archive-batch] denoisers: {[d.name for d in roster.enabled()]}, "
          f"{roster.encode.slots} encoder slots at --lp {roster.encode.lp_level}")
    if not todo:
        print("[archive-batch] nothing to do.")
        return 0

    freed = sweep_stage_root()
    if freed:
        print(f"[archive-batch] cleared {freed / 1073741824:.2f} GiB of staged "
              f"sources left by an interrupted run")

    started = time.monotonic()
    scheduler = Scheduler(todo, _roster, make_runner(roster.encode), STATE,
                          prior_failures=state.failures)

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

    scheduler.run()
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
