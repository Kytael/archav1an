#!/usr/bin/env python3
# tools/denoise-rate.py
"""Separate a denoise lane's sustained rate from its fixed per-clip overhead.

Runs ON the denoise host. Pipe it over ssh.

Every lane figure this project had before mixed two different things. A run
pays a fixed cost once -- ffms2 indexing, VapourSynth startup, the TensorRT or
MIGraphX engine load, and filling the first window -- and then streams at
whatever rate the GPU sustains. Divide total frames by total wall and the
answer depends on how long the clip was, which is why the same lane read 4.40,
5.24 and 6.72 fps depending on what the timer covered.

The GPU only denoises. It does nothing during the fixed part, so on a long
clip the fixed part vanishes and the sustained rate is what the lane is worth.
On a short clip it dominates. Both numbers are needed to plan a run, and only
the sustained one should be compared between hosts.

The sink is local to the denoise host on purpose: the point here is the card
and the filter graph, not the network. Transfer cost is a separate term.

  time to first frame   fixed: index + startup + engine + first window
  sustained fps         steady state, measured after the ramp
  end-to-end fps        what a clip of THIS length actually delivers

Output is one JSON object on stdout; everything else goes to stderr.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time

PREFIX = ("/opt/archav1an/bin", "/usr/local/bin", "/usr/bin")
MIGX_VENV = os.path.expanduser("~/reposetc/bsvd/migraphx-venv")
MIGX_LIBS = os.path.expanduser("~/reposetc/bsvd/migraphx-libs/lib")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class TimedSink(threading.Thread):
    """Accept the y4m stream, discard it, and timestamp every frame boundary.

    Frame arrival is what carries the timing information. A windowed lane
    emits a whole window at once, so the series is stepwise rather than
    smooth -- which is exactly why the sustained rate has to be measured over
    a span of frames rather than from a single interval.
    """

    def __init__(self, frame_bytes, port=0):
        super().__init__(daemon=True)
        self.frame_bytes = frame_bytes
        self.bytes = 0
        self.marks = []          # (frame_index, monotonic) at each frame
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self._halt = threading.Event()

    def run(self):
        self.sock.settimeout(10.0)
        conn = None
        while conn is None and not self._halt.is_set():
            try:
                conn, _ = self.sock.accept()
            except (socket.timeout, OSError):
                continue
        if conn is None:
            return
        seen = 0
        try:
            while not self._halt.is_set():
                b = conn.recv(1 << 20)
                if not b:
                    break
                self.bytes += len(b)
                n = self.bytes // self.frame_bytes
                if n > seen:
                    now = time.monotonic()
                    # One mark per frame, even when a burst delivers many:
                    # the burst boundary is real information about a windowed
                    # lane and averaging it away would hide the step.
                    for i in range(seen + 1, n + 1):
                        self.marks.append((i, now))
                    seen = n
        except OSError:
            pass
        finally:
            conn.close()

    def stop(self):
        self._halt.set()
        try:
            self.sock.close()
        except OSError:
            pass


def rates(marks, t0, skip_frac=0.2):
    """Fixed overhead, sustained rate and end-to-end rate from the marks."""
    if not marks:
        return {}
    frames = marks[-1][0]
    t_first = marks[0][1]
    t_last = marks[-1][1]
    out = {
        "frames": frames,
        "overhead_s": round(t_first - t0, 2),
        "wall_s": round(t_last - t0, 2),
        "end_to_end_fps": round(frames / (t_last - t0), 3) if t_last > t0 else None,
    }
    # Skip the leading fraction so the first window's fill does not drag the
    # sustained figure down. What remains is the lane in steady state.
    start = int(frames * skip_frac)
    tail = [m for m in marks if m[0] > start]
    if len(tail) > 1:
        span = tail[-1][1] - tail[0][1]
        got = tail[-1][0] - tail[0][0]
        out["sustained_fps"] = round(got / span, 3) if span > 0 else None
        out["sustained_over_frames"] = got
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--backend", default="trt", choices=("trt", "migraphx"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tile", default="none")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--margin", type=int, default=32)
    ap.add_argument("--sigma", default="0.05")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    frame_bytes = args.width * args.height * 3 // 2 * 2
    sink = TimedSink(frame_bytes)
    env = dict(os.environ)
    if args.backend == "migraphx":
        py = os.path.join(MIGX_VENV, "bin", "python")
        env["VSPIPE"] = os.path.join(MIGX_VENV, "bin", "vspipe")
        env["LD_LIBRARY_PATH"] = MIGX_LIBS
    else:
        py = "/opt/archav1an/venv/bin/python"
        if not os.access(py, os.X_OK):
            py = sys.executable

    argv = [py, os.path.join(args.root, "tools", "svtav1-dispatch.py"),
            "--denoise-serve", f"127.0.0.1:{sink.port}", "-i", args.clip,
            "--denoise-bsvd", "--bsvd-sigma", str(args.sigma),
            "--bsvd-device", str(args.device)]
    if args.tile and args.tile != "none":
        argv += ["--bsvd-tile", args.tile, "--bsvd-margin", str(args.margin)]
        if args.window:
            argv += ["--bsvd-window", str(args.window)]
    log(f"[denoise-rate] {' '.join(argv)}")

    sink.start()
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, cwd=args.root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    out = proc.communicate()[0].decode("utf-8", "replace")
    sink.stop()
    sink.join(timeout=10)

    result = dict(host=socket.gethostname(), label=args.label or args.backend,
                  backend=args.backend, device=args.device, tile=args.tile,
                  window=args.window, margin=args.margin, rc=proc.returncode,
                  bytes_received=sink.bytes)
    result.update(rates(sink.marks, t0))
    if proc.returncode != 0:
        result["tail"] = out.strip()[-600:]
    json.dump(result, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
