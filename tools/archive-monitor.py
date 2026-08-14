#!/usr/bin/env python3
# tools/archive-monitor.py
"""Watch an archive-batch run: per-host GPU load and per-denoiser fps.

Read-only. It never touches the run's state file, so it is safe to start, stop
and restart at any point, and safe to run in several terminals at once.

Two things are worth watching over a multi-week run, and neither is visible
from the batch's own output until a clip finishes:

  fps   comes from the state file, so it is the achieved rate including
        staging, encoding and publishing -- not the denoiser's peak.
  GPU   comes from the hosts themselves, so a lane that has silently stopped
        doing work shows up long before its clip times out.

The two disagree in a useful way. A lane with a healthy fps and a GPU that
keeps dropping to zero is being starved by its host's CPU rather than limited
by its card, which is what the GB10 lane does.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.environ.get("ARCHIVE_RUN_DIR") or os.path.join(REPO, ".archive-run")
STATE = os.path.join(RUN_DIR, "state.jsonl")
ROSTER = os.path.join(RUN_DIR, "denoisers.toml")

GPU_QUERY = ["--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"]
# One sample says almost nothing here: BSVD swings between 0 and 96% within a
# second on some hosts, so a single reading is as likely to catch a trough as a
# peak. Averaging a short burst is the difference between "this lane is dead"
# and "this lane is bursty".
BURST = 5
BURST_GAP = 0.4


def read_roster():
    """Denoiser name -> ssh host. 'local' means this machine.

    tomllib rather than the project's roster loader: this tool must keep
    working against a roster the loader would reject, because a rejected roster
    is exactly when you want to look at the run.
    """
    try:
        import tomllib
        with open(ROSTER, "rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        print(f"[monitor] cannot read {ROSTER}: {exc}", file=sys.stderr)
        return {}
    return {d.get("name", "?"): (d.get("host", "local"), d.get("enabled", False))
            for d in data.get("denoiser", [])}


def _num(text):
    """nvidia-smi field as a float, or None when the card does not report it."""
    try:
        return float(text)
    except ValueError:
        return None


def gpu_sample(host):
    """Average a short burst of nvidia-smi readings on one host."""
    if host == "local":
        cmd = ["nvidia-smi"] + GPU_QUERY
    else:
        cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", host,
               "nvidia-smi " + " ".join(GPU_QUERY)]
    utils, mem, total, power = [], 0.0, 0.0, []
    for i in range(BURST):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=20).stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            return None
        if not out:
            return None
        # A host with two cards prints a row each. The first row is the one the
        # roster points at for every host in this fleet.
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        if len(parts) < 4:
            return None
        try:
            utils.append(float(parts[0]))
            power.append(float(parts[3]))
        except ValueError:
            return None
        # GB10 has unified memory and reports memory.used/total as [N/A]. That
        # is not a failure -- treating it as one hid the whole gpu4 lane.
        mem, total = _num(parts[1]), _num(parts[2])
        if i + 1 < BURST:
            time.sleep(BURST_GAP)
    return dict(util=sum(utils) / len(utils), peak=max(utils), low=min(utils),
                mem=mem, total=total, power=sum(power) / len(power))


def read_state(path):
    """(per-denoiser stats, done, failed, last N failures)."""
    per = defaultdict(lambda: dict(clips=0, frames=0.0, wall=0.0, work=0.0, bytes=0))
    done = failed = 0
    recent = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = row.get("denoiser", "?")
                if row.get("status") == "done":
                    done += 1
                    d = per[name]
                    d["clips"] += 1
                    d["wall"] += row.get("wall_s", 0.0)
                    # work_s is the dispatch alone, so frames/work is the lane's
                    # rate with staging and publishing taken out.
                    d["work"] += row.get("work_s", 0.0)
                    # fps*wall recovers the frame count the record does not store.
                    d["frames"] += row.get("fps", 0.0) * row.get("wall_s", 0.0)
                    d["bytes"] += row.get("out_bytes", 0)
                else:
                    failed += 1
                    recent.append((row.get("src", "?"), name,
                                   row.get("reason", "")[:70]))
    except OSError:
        return {}, 0, 0, []
    return per, done, failed, recent[-5:]


def manifest_total():
    try:
        with open(os.path.join(RUN_DIR, "manifest-raw.tsv"), "r",
                  encoding="utf-8") as fh:
            return sum(1 for ln in fh if ln.strip())
    except OSError:
        return 0


def bar(pct, width=22):
    filled = int(round(pct / 100.0 * width))
    return "#" * filled + "." * (width - filled)


def render(roster, total):
    per, done, failed, recent = read_state(STATE)
    out = [time.strftime("archive-batch monitor   %Y-%m-%d %H:%M:%S")]
    if total:
        pct = done / total * 100
        out.append(f"clips: {done}/{total} done ({pct:.1f}%), {failed} failed")
    else:
        out.append(f"clips: {done} done, {failed} failed")
    out.append("")
    out.append(f"{'lane':<12} {'host':<9} {'clips':>6} {'fps':>7} {'work':>7}   "
               f"{'GPU avg':>7} {'range':>9}  {'W':>5}  {'VRAM':>11}")
    out.append("-" * 96)

    pool_fps = pool_work = 0.0
    for name in sorted(roster):
        host, enabled = roster[name]
        st = per.get(name, dict(clips=0, frames=0.0, wall=0.0, work=0.0))
        fps = st["frames"] / st["wall"] if st["wall"] else 0.0
        work_fps = st["frames"] / st["work"] if st.get("work") else 0.0
        pool_fps += fps
        pool_work += work_fps
        g = gpu_sample(host)
        if g is None:
            gpu_txt = f"{'--':>7} {'unreachable':>9}  {'--':>5}  {'--':>11}"
        else:
            vram = ("unified" if g["mem"] is None or g["total"] is None
                    else f"{g['mem']:.0f}/{g['total']:.0f}")
            gpu_txt = (f"{g['util']:>6.0f}% {g['low']:>3.0f}-{g['peak']:>3.0f}% "
                       f"{g['power']:>5.0f}  {vram:>11}")
        flag = "" if enabled else " (off)"
        out.append(f"{name + flag:<12} {host:<9} {st['clips']:>6} {fps:>7.2f} "
                   f"{work_fps:>7.2f}   {gpu_txt}")
        if g is not None:
            out.append(f"{'':<12} {'':<9} {'':>6} {'':>7} {'':>7}   [{bar(g['util'])}]")

    out.append("-" * 96)
    out.append(f"{'pool':<12} {'':<9} {done:>6} {pool_fps:>7.2f} {pool_work:>7.2f}")
    out.append("fps includes staging and publishing; work is the dispatch alone.")
    if recent:
        out.append("")
        out.append("recent failures")
        for src, name, reason in recent:
            out.append(f"  {src}  [{name}]  {reason}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=30.0,
                    help="seconds between refreshes (default 30)")
    ap.add_argument("--once", action="store_true", help="print one sample and exit")
    args = ap.parse_args()

    roster = read_roster()
    if not roster:
        print("[monitor] no denoisers in the roster -- nothing to watch",
              file=sys.stderr)
        return 2
    total = manifest_total()

    while True:
        text = render(roster, total)
        if args.once:
            print(text)
            return 0
        # Redraw in place rather than scrolling: this is meant to be left open
        # on a second screen for days.
        sys.stdout.write("\033[2J\033[H" + text + "\n")
        sys.stdout.flush()
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            return 0


if __name__ == "__main__":
    sys.exit(main())
