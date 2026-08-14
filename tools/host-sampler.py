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

CSV columns:
  t,gpu_util,gpu_power_w,gpu_clock_mhz,cpu_busy_pct,cpu_top_core_pct,
  proc_cpu_pct,proc_top_thread_pct,proc_threads_busy,proc_threads

cpu_busy_pct is summed across cores, so 100 means one core saturated and 2000
means twenty.

cpu_top_core_pct is the max across cores and is NOT evidence about threads. It
is a maximum over many noisy samples, so it reads high whenever work migrates
across cores even though no thread is hot. Reading it as "single-thread bound"
was wrong on gpu4: it showed 77% mean hottest core while the busiest thread
in the process was 11%. It is kept because a genuinely pinned core is still
worth seeing, but the thread columns are what settle the question.

--pid-of names a process (matched like pgrep -f); its per-thread jiffies are
read from /proc/<pid>/task/*/stat. proc_top_thread_pct is the busiest single
thread. If that approaches 100 while the GPU sits below its ceiling, the lane
is serialised on one thread. If it is low and the GPU is also low, nothing is
compute bound and the lane is waiting -- on a copy, on I/O, or on back
pressure.
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


HZ = os.sysconf("SC_CLK_TCK")


def find_pid(pattern):
    """First pid whose full command line contains `pattern`."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pattern in cmd and str(os.getpid()) != entry:
            return entry
    return None


def read_threads(pid):
    """tid -> busy jiffies, for every thread of `pid`."""
    out = {}
    if not pid:
        return out
    base = f"/proc/{pid}/task"
    try:
        tids = os.listdir(base)
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"{base}/{tid}/stat") as fh:
                # The comm field can contain spaces and parentheses, so split
                # after the last ") " rather than on whitespace from the left.
                fields = fh.read().rsplit(") ", 1)[1].split()
            out[tid] = int(fields[11]) + int(fields[12])
        except (OSError, IndexError, ValueError):
            continue
    return out


def thread_deltas(prev, cur, dt):
    """(process total pct, busiest thread pct, threads above 5%, thread count)."""
    if not cur or dt <= 0:
        return 0.0, 0.0, 0, len(cur)
    pcts = []
    for tid, jiff in cur.items():
        if tid in prev:
            pcts.append((jiff - prev[tid]) / HZ / dt * 100.0)
    if not pcts:
        return 0.0, 0.0, 0, len(cur)
    return sum(pcts), max(pcts), len([p for p in pcts if p > 5], ), len(cur)


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
    ap.add_argument("--pid-of", default="vspipe -c y4m",
                    help="substring of the target's command line; its per-thread "
                         "CPU is what shows whether a lane is serialised")
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
    fh.write("t,gpu_util,gpu_power_w,gpu_clock_mhz,cpu_busy_pct,cpu_top_core_pct,"
             "proc_cpu_pct,proc_top_thread_pct,proc_threads_busy,proc_threads\n")

    prev, t0 = read_cpu(), time.monotonic()
    pid = find_pid(args.pid_of)
    prev_thr, prev_t = read_threads(pid), t0
    try:
        for line in proc.stdout:
            now = time.monotonic()
            if _stop or now - t0 > args.max_seconds:
                break
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            cur = read_cpu()
            total, top = cpu_deltas(prev, cur)
            prev = cur
            # The target may start after the sampler; keep looking until found,
            # and look again if it goes away, so a restart is picked up too.
            if not pid or not os.path.isdir(f"/proc/{pid}"):
                pid = find_pid(args.pid_of)
                prev_thr = read_threads(pid)
            cur_thr = read_threads(pid)
            p_tot, p_top, p_busy, p_n = thread_deltas(prev_thr, cur_thr, now - prev_t)
            prev_thr, prev_t = cur_thr, now
            # A card that does not report a field prints [N/A]; keep the row and
            # let the reader decide, rather than dropping the sample.
            vals = []
            for p in parts[:3]:
                try:
                    vals.append(f"{float(p):.1f}")
                except ValueError:
                    vals.append("")
            fh.write(f"{now - t0:.2f},{vals[0]},{vals[1]},{vals[2]},"
                     f"{total:.1f},{top:.1f},"
                     f"{p_tot:.1f},{p_top:.1f},{p_busy},{p_n}\n")
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
