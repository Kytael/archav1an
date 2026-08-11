#!/usr/bin/env python3
# tools/archive-batch.py
"""Run the 2001-2007 dance archive through the dance-HQ BSVD pipeline.

Sources stay on gpu1. Encoding always happens here. See
the design notes, which are not part of this tree
"""
import os
import shutil
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
from tools.archive_batch.transfer import (TransferError, publish_cmd, run,
                                          stage_cmd, staged_path)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(REPO, ".archive-run")
MANIFEST = os.path.join(RUN_DIR, "manifest-raw.tsv")
STATE = os.path.join(RUN_DIR, "state.jsonl")
ROSTER = os.path.join(RUN_DIR, "denoisers.toml")
STAGE_ROOT = os.path.join(REPO, "Temp", "_stage")
CALLBACK_IP = "10.0.0.10"     # encoder-host LAN address; tailscale caps at 1.5 Gbps


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
                return False, time.monotonic() - started, 0.0, 0
            run(publish_cmd(SOURCE_HOST, out, clip.rel_dir))
            size = os.path.getsize(out)
        except TransferError as exc:
            print(f"[archive-batch] {clip.src}: {exc}", file=sys.stderr)
            return False, time.monotonic() - started, 0.0, 0
        finally:
            for path in (staged, out):
                if os.path.exists(path):
                    os.remove(path)
            shutil.rmtree(temp_dir, ignore_errors=True)

        wall = time.monotonic() - started
        return True, wall, (clip.frames / wall if wall else 0.0), size
    return runner


def _roster():
    return load_roster(ROSTER, os.cpu_count() or 1)


def format_summary(done, failed, failures, elapsed_s):
    lines = [
        "",
        "=" * 70,
        f"archive-batch: {done} done, {failed} failed in {elapsed_s / 3600:.2f} h",
    ]
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

    with open(MANIFEST, "r", encoding="utf-8") as fh:
        clips = order_clips(parse_manifest(fh.read()))
    todo = pending_clips(clips, load_state(STATE))

    print(f"[archive-batch] {len(clips)} clips in manifest, {len(todo)} to do")
    print(f"[archive-batch] denoisers: {[d.name for d in roster.enabled()]}, "
          f"{roster.encode.slots} encoder slots x --lp {roster.encode.threads_per_slot}")
    if not todo:
        print("[archive-batch] nothing to do.")
        return 0

    started = time.monotonic()
    scheduler = Scheduler(todo, _roster, make_runner(roster.encode), STATE)
    scheduler.run()
    print(format_summary(scheduler.done, scheduler.failed,
                         scheduler.failures, time.monotonic() - started))
    return 1 if scheduler.failed else 0


if __name__ == "__main__":
    sys.exit(main())
