#!/usr/bin/env python3
# tools/host-sampler.py
"""Sample this host's GPU and CPU while something else runs, one CSV row per tick.

Runs on any fleet host, local or remote. Stop it with SIGTERM or SIGINT; it
flushes and exits cleanly, so a run that is cut short still yields its samples.

Why one nvidia-smi rather than one per tick: forking nvidia-smi costs 50-150 ms,
which is the same order as the interval worth sampling at. A single process with
--loop-ms drives the cadence itself, and the CPU read is taken as each GPU line
arrives, so the two are aligned rather than merely close.

Why 250 ms: the denoise lanes swing between idle and full inside a second. At
1 Hz that reads as a steady mid-range number and the swing is invisible, which
is exactly the failure this tool exists to catch.

CSV columns: t,gpu_util,gpu_power_w,gpu_clock_mhz,cpu_busy_pct,cpu_top_core_pct
cpu_busy_pct is summed across cores, so 100 means one core saturated and 2000
means twenty. That distinction matters: a lane pinned at 100 of 2000 is
single-thread bound, which no per-core average would show.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

QUERY = "utilization.gpu,power.draw,clocks.sm"

_stop = False


def _on_signal(_sig, _frame):
    global _stop
    _stop = True


def read_cpu():
    """Per-core (total_jiffies, idle_jiffies)."""
    out = {}
    try:
        with open("/proc/stat", "r") as fh:
            for line in fh:
                if line.startswith("cpu") and len(line) > 3 and line[3].isdigit():
                    parts = line.split()
                    out[parts[0]] = (sum(int(x) for x in parts[1:]), int(parts[4]))
    except OSError:
        pass
    return out


def cpu_deltas(prev, cur):
    """(summed busy percent across cores, busiest single core percent)."""
    busy = []
    for k in cur:
        if k not in prev:
            continue
        dt = cur[k][0] - prev[k][0]
        di = cur[k][1] - prev[k][1]
        if dt <= 0:
            continue
        busy.append((dt - di) / dt * 100.0)
    if not busy:
        return 0.0, 0.0
    return sum(busy), max(busy)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval-ms", type=int, default=250)
    ap.add_argument("--gpu-index", type=int, default=0,
                    help="which card, for hosts with more than one")
    ap.add_argument("--out", default="-", help="CSV path, or - for stdout")
    ap.add_argument("--max-seconds", type=float, default=86400,
                    help="hard stop, so a forgotten sampler cannot outlive the run")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    cmd = ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits",
           "-lms", str(args.interval_ms), "-i", str(args.gpu_index)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, bufsize=1)
    except OSError as exc:
        print(f"host-sampler: cannot run nvidia-smi: {exc}", file=sys.stderr)
        return 2

    fh = sys.stdout if args.out == "-" else open(args.out, "w", buffering=1)
    fh.write("t,gpu_util,gpu_power_w,gpu_clock_mhz,cpu_busy_pct,cpu_top_core_pct\n")

    prev, t0 = read_cpu(), time.monotonic()
    try:
        for line in proc.stdout:
            if _stop or time.monotonic() - t0 > args.max_seconds:
                break
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            cur = read_cpu()
            total, top = cpu_deltas(prev, cur)
            prev = cur
            # A card that does not report a field prints [N/A]; keep the row and
            # let the reader decide, rather than dropping the sample.
            vals = []
            for p in parts[:3]:
                try:
                    vals.append(f"{float(p):.1f}")
                except ValueError:
                    vals.append("")
            fh.write(f"{time.monotonic() - t0:.2f},{vals[0]},{vals[1]},{vals[2]},"
                     f"{total:.1f},{top:.1f}\n")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        if fh is not sys.stdout:
            fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
