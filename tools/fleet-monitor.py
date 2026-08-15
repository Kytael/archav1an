#!/usr/bin/env python3
# tools/fleet-monitor.py
"""Watch several hosts at once while something is running on them.

archive-monitor.py watches an archive run and reads its state file. This
watches HOSTS, and needs no run to be in progress: it is for a benchmark, a
sweep, a migration, anything where several machines are working and the only
question is whether they are actually working.

Written after starting three benchmarks and being asked, twice, why nothing
appeared to be in use. Both times the honest answer was that nothing was
watching -- the first run had crashed on a bug in its own sampler, and the
second was fine. Polling by hand after the fact cannot tell those apart while
they are happening.

One line per host per tick, appended, so it can be tailed or read later:

  18:41:02 gpu1    gpu  97% 341W 1815MHz 62C | cpu 78% 71C load 9.2 | enc1 vs1
  18:41:02 gpu3    gpu  93%  80W  1extra   86C | cpu 91% 88C load 4.4 | enc1 vs1

A host that stops responding prints `unreachable` rather than vanishing,
because a silent gap reads like idleness and is the thing worth catching.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Kept in one string so it survives the trip through ssh and a login shell
# that may be fish. Everything here is POSIX sh and reads only /proc and
# nvidia-smi, so it is safe to run against a busy machine.
PROBE = r"""
smi=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
g=$("$smi" --query-gpu=utilization.gpu,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
[ -z "$g" ] && g="na,na,na,na"
read one five rest < /proc/loadavg
cpu=$(awk '/^cpu /{i=$5+$6; t=$2+$3+$4+$5+$6+$7+$8; print t" "i}' /proc/stat)
tmax=0
for z in /sys/class/thermal/thermal_zone*/temp; do
  [ -r "$z" ] || continue
  v=$(cat "$z" 2>/dev/null)
  [ -n "$v" ] && [ "$v" -gt "$tmax" ] 2>/dev/null && tmax=$v
done
enc=$(pgrep -c SvtAv1EncApp 2>/dev/null); [ -z "$enc" ] && enc=0
vsp=$(pgrep -c vspipe 2>/dev/null); [ -z "$vsp" ] && vsp=0
echo "$g|$one|$cpu|$tmax|$enc|$vsp"
"""


def probe(host):
    cmd = (["bash", "-c", PROBE] if host == "local"
           else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                 host, "bash -s"])
    try:
        r = subprocess.run(cmd, input=None if host == "local" else PROBE,
                           capture_output=True, text=True, timeout=25)
        line = r.stdout.strip().splitlines()[-1]
        gpu, load, cputot, cpuidle, temp, enc, vsp = _split(line)
        return dict(host=host, gpu=gpu, load=load, cpu_total=cputot,
                    cpu_idle=cpuidle, temp=temp, enc=enc, vsp=vsp)
    except (subprocess.SubprocessError, OSError, IndexError, ValueError):
        return dict(host=host, error=True)


def _split(line):
    g, load, cpu, temp, enc, vsp = line.split("|")
    tot, idle = cpu.split()
    return (g.split(","), float(load), float(tot), float(idle),
            float(temp or 0) / 1000.0, int(enc), int(vsp))


def _n(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def render(row, prev):
    if row.get("error"):
        return f"{row['host']:<8} unreachable"
    g = row["gpu"]
    util, power, clock, gtemp = (_n(g[0]), _n(g[1]), _n(g[2]), _n(g[3]))
    # CPU busy needs two samples: /proc/stat is cumulative since boot, so a
    # single reading is a lifetime average and says nothing about now.
    cpu = ""
    p = prev.get(row["host"])
    if p and not p.get("error"):
        dt = row["cpu_total"] - p["cpu_total"]
        di = row["cpu_idle"] - p["cpu_idle"]
        if dt > 0:
            cpu = f"{(1 - di / dt) * 100:4.0f}%"
    parts = [f"{row['host']:<8}"]
    parts.append(f"gpu {util:3.0f}%" if util is not None else "gpu   --")
    parts.append(f"{power:4.0f}W" if power is not None else "  --W")
    parts.append(f"{clock:5.0f}MHz" if clock is not None else "   --MHz")
    parts.append(f"{gtemp:3.0f}C" if gtemp is not None else " --C")
    parts.append(f"| cpu {cpu or '  --'}")
    parts.append(f"{row['temp']:3.0f}C" if row["temp"] else " --C")
    parts.append(f"load {row['load']:5.2f}")
    parts.append(f"| enc {row['enc']} vspipe {row['vsp']}")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hosts", nargs="+", help="ssh aliases, or 'local'")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--out", default="-", help="append here instead of stdout")
    ap.add_argument("--max-seconds", type=float, default=86400)
    args = ap.parse_args()

    sink = sys.stdout if args.out == "-" else open(args.out, "a", buffering=1)
    prev, started = {}, time.monotonic()
    with ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
        while time.monotonic() - started < args.max_seconds:
            rows = list(pool.map(probe, args.hosts))
            stamp = time.strftime("%H:%M:%S")
            for row in rows:
                print(f"{stamp} {render(row, prev)}", file=sink, flush=True)
                prev[row["host"]] = row
            print("", file=sink, flush=True)
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
