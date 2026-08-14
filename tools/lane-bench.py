#!/usr/bin/env python3
# tools/lane-bench.py
"""Benchmark one denoise lane and say WHY it ran at the rate it did.

An fps number on its own cannot tell a lane that is limited by its card from
one that is starved by its host, and those want opposite fixes. This runs the
real split-host path, samples the denoise host at 4 Hz throughout, and reports
the utilisation shape next to the rate:

  - GPU idle fraction and the longest unbroken idle stretch. A lane that is
    genuinely GPU-bound sits near its ceiling; one that sawtooths spends half
    its wall clock at zero and the mean hides it.
  - Summed CPU busy percent against the host's core count. 100 of 2000 means
    one saturated thread, which is a serialised pipeline, not a busy machine.
  - Power against the card's limit, which separates a card that is working
    from one that is merely resident.

Sampling is done by tools/host-sampler.py on the denoise host, so it needs the
same checkout there -- which every rostered host already has.

  tools/lane-bench.py --host gpu4 --root /home/user/reposetc/ubuntav1an \\
      --clip Input/clip.MOV --port 5302
"""
import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISPATCH = os.path.join(REPO, "tools", "svtav1-dispatch.py")
SAMPLER = "tools/host-sampler.py"
LOCAL_PYTHON = "/opt/archav1an/venv/bin/python"
REMOTE_CSV = "/tmp/archav1an-lane-bench.csv"

BLOCKS = " .:-=+*#%@"


def sparkline(values, width=60):
    """Downsample to `width` buckets, worst-case (min) per bucket.

    Min rather than mean on purpose: this chart exists to show idle gaps, and
    averaging is what hid them in the first place.
    """
    if not values:
        return ""
    out = []
    for i in range(width):
        lo = int(i * len(values) / width)
        hi = max(lo + 1, int((i + 1) * len(values) / width))
        chunk = values[lo:hi]
        v = min(chunk) if chunk else 0.0
        out.append(BLOCKS[min(len(BLOCKS) - 1, int(v / 100.0 * (len(BLOCKS) - 1)))])
    return "".join(out)


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def longest_run(values, predicate):
    best = run = 0
    for v in values:
        run = run + 1 if predicate(v) else 0
        best = max(best, run)
    return best


def parse_csv(text):
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append(dict(t=float(parts[0]),
                             util=float(parts[1]) if parts[1] else None,
                             power=float(parts[2]) if parts[2] else None,
                             clock=float(parts[3]) if parts[3] else None,
                             cpu=float(parts[4]), top=float(parts[5])))
        except ValueError:
            continue
    return rows


def report(rows, host, cores, fps, frames, wall, interval=0.25):
    utils = [r["util"] for r in rows if r["util"] is not None]
    if not utils:
        return f"{host}: no GPU samples collected"
    su = sorted(utils)
    idle = [u for u in utils if u < 10]
    powers = [r["power"] for r in rows if r["power"] is not None]
    cpus = [r["cpu"] for r in rows]
    tops = [r["top"] for r in rows]
    gap = longest_run(utils, lambda u: u < 10) * interval

    lines = [
        "",
        f"=== {host} " + "=" * max(0, 66 - len(host)),
        f"  rate          {fps:.2f} fps   ({frames} frames in {wall:.1f} s)",
        f"  samples       {len(rows)} over {rows[-1]['t']:.0f} s at {interval * 1000:.0f} ms",
        "",
        f"  GPU util      mean {sum(utils) / len(utils):5.1f}%   "
        f"p10 {pct(su, 10):5.1f}%  p50 {pct(su, 50):5.1f}%  p90 {pct(su, 90):5.1f}%",
        f"  GPU idle      {len(idle) / len(utils) * 100:5.1f}% of samples below 10%   "
        f"longest idle stretch {gap:.1f} s",
    ]
    if powers:
        lines.append(f"  GPU power     mean {sum(powers) / len(powers):5.1f} W   "
                     f"peak {max(powers):5.1f} W")
    lines += [
        f"  CPU busy      mean {sum(cpus) / len(cpus):6.0f}% of {cores * 100}%  "
        f"({sum(cpus) / len(cpus) / 100:.1f} of {cores} cores)",
        f"  CPU hottest   mean {sum(tops) / len(tops):5.1f}%   peak {max(tops):5.1f}%  "
        f"on a single core",
        "",
        f"  GPU util over time (each cell is the WORST sample in its bucket)",
        f"  |{sparkline(utils)}|",
        f"  0s{' ' * 56}{rows[-1]['t']:.0f}s",
        "",
        "  " + verdict(utils, cpus, tops, cores),
        "",
    ]
    return "\n".join(lines)


def verdict(utils, cpus, tops, cores):
    idle_frac = len([u for u in utils if u < 10]) / len(utils)
    mean_util = sum(utils) / len(utils)
    mean_cpu = sum(cpus) / len(cpus)
    mean_top = sum(tops) / len(tops)
    if idle_frac > 0.25:
        why = (f"VERDICT: starved. The card is idle {idle_frac * 100:.0f}% of the time. "
               f"CPU is {mean_cpu / 100:.1f} of {cores} cores")
        if mean_top > 70:
            return why + (f", with one core at {mean_top:.0f}% -- a serialised "
                          f"stage between inference calls is the limit, not the GPU.")
        return why + (", and no single core is saturated either, so the stall is a "
                      "wait (I/O, transport, or a synchronous copy), not compute.")
    if mean_util > 80:
        return (f"VERDICT: GPU-bound at {mean_util:.0f}% mean utilisation. This lane is "
                f"at its card's ceiling; only a faster card or a cheaper model helps.")
    return (f"VERDICT: mixed. {mean_util:.0f}% mean utilisation with the card idle "
            f"{idle_frac * 100:.0f}% of the time. Worth a longer sample before acting.")


def host_cores(host):
    cmd = ["nproc"] if host == "local" else ["ssh", host, "nproc"]
    try:
        return int(subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=30).stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="ssh alias of the denoise host")
    ap.add_argument("--root", required=True, help="checkout path on that host")
    ap.add_argument("--clip", required=True, help="local source clip")
    ap.add_argument("--port", type=int, default=5300)
    ap.add_argument("--tile", default="")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--margin", type=int, default=32)
    ap.add_argument("--sigma", default="0.05")
    ap.add_argument("--quality", default="40")
    ap.add_argument("--speed", default="8")
    ap.add_argument("--callback", default="127.0.0.1",
                    help="127.0.0.1 uses an ssh reverse tunnel this script opens")
    ap.add_argument("--interval-ms", type=int, default=250)
    ap.add_argument("--keep", action="store_true", help="keep the output and temp dir")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.clip))[0]
    out = os.path.join(REPO, "Output", f"{stem}-lanebench.mkv")
    temp = os.path.join(REPO, "Temp", stem)
    subprocess.run(["rm", "-rf", temp, out])

    tunnel = None
    if args.callback == "127.0.0.1":
        # The remote connects back to its own loopback, so no firewall rule is
        # needed to measure a lane. Measured at 244 MB/s to gpu4, far above
        # what any lane here produces, so it does not colour the result.
        subprocess.run(["ssh", "-f", "-N", "-R",
                        f"{args.port}:127.0.0.1:{args.port}", args.host], check=True)
        tunnel = True
        check = subprocess.run(["ssh", args.host, f"ss -ltn | grep -q {args.port}"])
        if check.returncode != 0:
            print(f"lane-bench: reverse tunnel to {args.host} did not come up",
                  file=sys.stderr)
            return 2

    cores = host_cores(args.host)
    sampler = subprocess.Popen(
        ["ssh", args.host, f"cd {shlex.quote(args.root)} && "
         f"exec python3 {SAMPLER} --interval-ms {args.interval_ms} --out {REMOTE_CSV}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    argv = [LOCAL_PYTHON, DISPATCH, "-i", args.clip, "-o", out,
            "--quality", args.quality, "--speed", args.speed,
            "--denoise-bsvd", "--bsvd-sigma", args.sigma,
            "--remote-denoise", args.host, "--remote-root", args.root,
            "--remote-port", str(args.port), "--remote-callback", args.callback]
    if args.tile:
        argv += ["--bsvd-tile", args.tile, "--bsvd-margin", str(args.margin)]
        if args.window:
            argv += ["--bsvd-window", str(args.window)]

    print(f"lane-bench: {args.host}  tile={args.tile or 'none'} "
          f"window={args.window or '-'}  sampling every {args.interval_ms} ms",
          flush=True)
    t0 = time.monotonic()
    log = os.path.join(temp, f"{stem}_lanebench.log")
    os.makedirs(temp, exist_ok=True)
    with open(log, "w") as fh:
        rc = subprocess.run(argv, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = time.monotonic() - t0

    sampler.send_signal(signal.SIGTERM)
    try:
        sampler.wait(timeout=15)
    except subprocess.TimeoutExpired:
        sampler.kill()
    # The ssh that carried the sampler is gone, but the sampler itself may have
    # outlived it; SIGTERM it by name on the far side before reading the CSV.
    subprocess.run(["ssh", args.host, f"pkill -f {SAMPLER} 2>/dev/null; true"],
                   capture_output=True)

    text = subprocess.run(["ssh", args.host, f"cat {REMOTE_CSV}"],
                          capture_output=True, text=True).stdout
    rows = parse_csv(text)

    body = open(log, errors="replace").read()
    m = re.search(r"Frame count verified: (\d+) frames", body)
    frames = int(m.group(1)) if m else 0
    ok = m is not None and rc == 0

    if not ok:
        print(f"\nlane-bench: dispatch FAILED (rc={rc}). Root cause from the log:")
        for line in body.splitlines():
            if re.search(r"CUDA failure|Error:|RuntimeError|Failed to retrieve", line):
                print("   " + line.strip()[:160])
        print(f"   full log: {log}")

    fps = frames / wall if wall and frames else 0.0
    print(report(rows, args.host, cores, fps, frames, wall, args.interval_ms / 1000.0))

    if tunnel:
        subprocess.run(["pkill", "-f", f"ssh -f -N -R {args.port}:127.0.0.1:{args.port}"],
                       capture_output=True)
    if not args.keep:
        subprocess.run(["rm", "-f", out])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
