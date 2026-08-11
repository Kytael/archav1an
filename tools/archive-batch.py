#!/usr/bin/env python3
# tools/archive-batch.py
"""Run the 2001-2007 dance archive through the dance-HQ BSVD pipeline.

Sources stay on gpu1. Encoding always happens here. See
the design notes, which are not part of this tree
"""
import json
import os
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
RUN_DIR = os.path.join(REPO, ".archive-run")
MANIFEST = os.path.join(RUN_DIR, "manifest-raw.tsv")
STATE = os.path.join(RUN_DIR, "state.jsonl")
ROSTER = os.path.join(RUN_DIR, "denoisers.toml")
STAGE_ROOT = os.path.join(REPO, "Temp", "_stage")
CALLBACK_IP = "10.0.0.10"     # encoder-host LAN address; tailscale caps at 1.5 Gbps


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
        staged = staged_path(stage_dir, clip.src)
        out = os.path.join(stage_dir, f"{clip.stem}-av1.mkv")
        try:
            run(stage_cmd(SOURCE_HOST, clip.src, stage_dir))
            argv, env_overlay = build_command(
                denoiser, encode, staged=staged, out=out,
                remote_src=f"{ARCHIVE_ROOT}/{clip.src}" if denoiser.is_remote else None,
                callback=CALLBACK_IP if denoiser.is_remote else None)
            env = dict(os.environ)
            env.update(env_overlay)
            proc = subprocess.run(argv, cwd=REPO, env=env)
            if proc.returncode != 0 or not os.path.exists(out):
                why = (f"dispatch exit {proc.returncode}" if proc.returncode
                       else "dispatch exit 0 but no output file")
                # Read the log before the finally below deletes temp_dir.
                tail = log_tail(temp_dir, clip.stem)
                return False, time.monotonic() - started, 0.0, 0, \
                    f"{why}. {tail}".strip()
            run(publish_cmd(SOURCE_HOST, out, clip.rel_dir))
            size = os.path.getsize(out)
        except TransferOutage:
            # The host is down, not the clip bad. Let the scheduler requeue it
            # rather than spend one of this clip's two attempts (spec 6).
            raise
        except TransferError as exc:
            print(f"[archive-batch] {clip.src}: {exc}", file=sys.stderr)
            return False, time.monotonic() - started, 0.0, 0, str(exc)
        finally:
            for path in (staged, out):
                if os.path.exists(path):
                    os.remove(path)
            shutil.rmtree(temp_dir, ignore_errors=True)

        wall = time.monotonic() - started
        return True, wall, (clip.frames / wall if wall else 0.0), size, ""
    return runner


def _roster():
    return load_roster(ROSTER, os.cpu_count() or 1)


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
    """Achieved fps and clip count per denoiser, from the run state (spec 7.2)."""
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
                clips, frames, wall = rates.get(name, (0, 0.0, 0.0))
                rates[name] = (clips + 1,
                               frames + row.get("fps", 0.0) * row.get("wall_s", 0.0),
                               wall + row.get("wall_s", 0.0))
    except OSError:
        return {}
    return {name: (clips, frames / wall if wall else 0.0)
            for name, (clips, frames, wall) in rates.items()}


def format_summary(done, failed, failures, elapsed_s, state_path=None):
    lines = [
        "",
        "=" * 70,
        f"archive-batch: {done} done, {failed} failed in {elapsed_s / 3600:.2f} h",
    ]
    rates = per_denoiser_rates(state_path) if state_path else {}
    if rates:
        lines.append("")
        for name in sorted(rates):
            clips, fps = rates[name]
            lines.append(f"  {name:<12} {clips:>5} clips  {fps:6.2f} fps")
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
          f"{roster.encode.slots} encoder slots x --lp {roster.encode.threads_per_slot}")
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
    return 1 if scheduler.failed else 0


if __name__ == "__main__":
    sys.exit(main())
