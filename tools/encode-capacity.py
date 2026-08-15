#!/usr/bin/env python3
# tools/encode-capacity.py
"""Measure what a host can encode, and what that costs its denoiser.

Runs ON the host being measured. Pipe it over ssh; it needs no checkout of its
own beyond the one it is pointed at.

Why three phases rather than one. Every machine in this fleet except the
desktop shares a thermal and power budget between its CPU and its GPU, so an
encoder benchmarked on an idle box reports a number that host can never
actually deliver while it is also denoising. gpu3 already throttles 8% with
no encoder running at all -- SW Thermal Slowdown at 86C, power cap 105 W to
90 W. The useful figure is not "how fast can this CPU encode" but "how much
encoding can this host do without spending its GPU", and only the delta
between phases shows that.

  solo-encode    encoder alone, GPU idle
  solo-denoise   denoiser alone, encoder idle
  both           the two together

Output is one JSON object on stdout. Everything else goes to stderr, so the
caller can pipe it straight into a report.
"""
import argparse
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

# The prefix build is not on a non-interactive PATH on at least one host, so
# never rely on the environment to find these.
PREFIX = ("/opt/archav1an/bin", "/usr/local/bin", "/usr/bin")

# encoder-host's iGPU lane, copied from tools/archive_batch/dispatch_cmd.py:12-13.
MIGX_VENV = os.path.expanduser("~/reposetc/bsvd/migraphx-venv")
MIGX_LIBS = os.path.expanduser("~/reposetc/bsvd/migraphx-libs/lib")

# Copied from tools/archive_batch/dispatch_cmd.py. Benchmarking a bare
# --preset/--crf encode measures an encode this fleet never runs: on gpu1 it
# read 68 fps against 26 for the real thing. Keep these in step with that file.
ENCODER_PARAMS = ("--tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.8 --sharp-tx 1 "
                  "--sharpness 1 --tf-strength 2 --variance-boost-strength 1 "
                  "--variance-octile 7 --enable-dlf 2")

# The preset matters more than every other parameter combined. The pipeline
# runs `--speed 4`, set in run_linux_dance_HQ_crf27.sh and copied verbatim into
# archive_batch/dispatch_cmd.py. Benchmarking preset 8 made gpu1 read 51.7 fps
# against encoder-host's documented 26 -- a machine with half the cores appearing
# to be twice as fast, which is the tell that the workloads differed.
#
# Every host has TWO encoders: the project's PSY v2.3.0-C build under
# $VS_PREFIX/bin, and a distro mainline v4.1.0. The pipeline gets PSY because
# dispatch calls prefer_prefix_bin() before it resolves anything; a bare shell
# gets mainline, which lacks --sharp-tx and --hbd-mds and would fail outright.
# find() below mirrors the prefix order deliberately. Record the banner with
# every result so a cross-host comparison cannot silently mix builds.


def find(name):
    for d in PREFIX:
        p = os.path.join(d, name)
        if os.access(p, os.X_OK):
            return p
    return shutil.which(name)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def gpu_sample():
    """(util%, power W, clock MHz, temp C). GB10 reports some as [N/A]."""
    smi = find("nvidia-smi") or "nvidia-smi"
    try:
        out = subprocess.run(
            [smi, "--query-gpu=utilization.gpu,power.draw,clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")]

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None
    return dict(util=num(parts[0]), power=num(parts[1]),
                clock=num(parts[2]), temp=num(parts[3]))


def _hwmon(name):
    """Path of the first hwmon whose name matches, or None."""
    base = "/sys/class/hwmon"
    try:
        for d in sorted(os.listdir(base)):
            try:
                with open(f"{base}/{d}/name") as fh:
                    if fh.read().strip() == name:
                        return f"{base}/{d}"
            except OSError:
                continue
    except OSError:
        pass
    return None


def _read_num(path, scale=1.0):
    try:
        with open(path) as fh:
            return float(fh.read().strip()) / scale
    except (OSError, ValueError):
        return None


def amd_sample():
    """(util%, power W, clock MHz, temp C) for an amdgpu card, from sysfs.

    nvidia-smi on encoder-host reports the 2070S, which sits idle while the 8060S
    iGPU denoises -- so sampling it would describe the wrong device and gate
    the cooldown on a card that never got hot. The iGPU has no SMI equivalent;
    everything here comes from sysfs.
    """
    hw = _hwmon("amdgpu")
    util = None
    for card in sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
        util = _read_num(card)
        if util is not None:
            break
    if hw is None and util is None:
        return None
    return dict(util=util,
                power=_read_num(f"{hw}/power1_input", 1e6) if hw else None,
                clock=_read_num(f"{hw}/freq1_input", 1e6) if hw else None,
                temp=_read_num(f"{hw}/temp1_input", 1e3) if hw else None)


def cpu_temp():
    """Package temperature, best effort. Names differ across these machines."""
    best = None
    base = "/sys/class/thermal"
    try:
        for z in os.listdir(base):
            if not z.startswith("thermal_zone"):
                continue
            try:
                with open(f"{base}/{z}/type") as fh:
                    kind = fh.read().strip()
                if kind not in ("x86_pkg_temp", "acpitz", "cpu-thermal",
                                "TCPU", "soc_thermal"):
                    continue
                with open(f"{base}/{z}/temp") as fh:
                    v = int(fh.read().strip()) / 1000.0
                best = v if best is None else max(best, v)
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    if best is None:
        # encoder-host exposes no thermal zone at all, only hwmon. Without this
        # the one host whose CPU and GPU share a die reports no temperature.
        hw = _hwmon("k10temp")
        if hw:
            best = _read_num(f"{hw}/temp1_input", 1e3)
    return best


class Sampler(threading.Thread):
    """Poll GPU and CPU while a phase runs. One thread, not one process."""

    def __init__(self, interval=1.0, probe=None):
        super().__init__(daemon=True)
        self.interval = interval
        self.probe = probe or gpu_sample
        self.rows = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            g = self.probe() or {}
            self.rows.append(dict(gpu_util=g.get("util"), gpu_power=g.get("power"),
                                  gpu_clock=g.get("clock"), gpu_temp=g.get("temp"),
                                  cpu_temp=cpu_temp()))
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
        return self.summary()

    def summary(self):
        out = {}
        for key in ("gpu_util", "gpu_power", "gpu_clock", "gpu_temp", "cpu_temp"):
            vals = [r[key] for r in self.rows if r.get(key) is not None]
            if vals:
                out[key + "_mean"] = round(sum(vals) / len(vals), 1)
                out[key + "_max"] = round(max(vals), 1)
        out["samples"] = len(self.rows)
        return out


def cooldown(target_c=70.0, limit_s=180.0, probe=None):
    """Wait for the GPU to drop below target before the next phase.

    Every host here except the desktop shares a power and thermal budget, so a
    phase inherits the heat of the phase before it. gpu3's solo denoise ran at
    85.5C directly after 82s of full-CPU encoding and came out 39% slower than
    the same work later in the run -- an ordering artifact that read as a real
    measurement. Cooling between phases is what makes them comparable.
    """
    probe = probe or gpu_sample
    start = time.monotonic()
    while time.monotonic() - start < limit_s:
        g = probe()
        # Gate on the hottest sensor available, not on the GPU alone. On
        # encoder-host the iGPU edge sensor read 44C straight after an encode that
        # left the package at 83C, so a GPU-only gate passed instantly and the
        # denoise phase inherited the encoder's heat anyway -- the same
        # ordering artefact this function exists to remove, arriving through a
        # different sensor. WSL2 hosts expose no CPU sensor, so they are
        # unaffected and their earlier runs stay comparable.
        t = max([v for v in ((g or {}).get("temp"), cpu_temp())
                 if v is not None] or [None])
        if t is None or t <= target_c:
            break
        time.sleep(5)
    return round(time.monotonic() - start, 1)


def encode_until(stop_event, clip, frames, preset, crf, lp, label, streams=1):
    """Encode in a loop until told to stop; return the aggregate rate.

    A single 600-frame encode finishes long before a full-clip denoise does,
    so measuring the denoiser "under load" that way measured a denoiser that
    was alone for most of the window: 88% of it on gpu3, 94% on gpu4. The
    encoder has to keep working for as long as the denoiser runs, or neither
    number describes the two of them together.
    """
    total_frames, total_wall, passes = 0, 0.0, 0
    while not stop_event.is_set():
        r = encode_phase(clip, frames, preset, crf, lp, f"{label}#{passes}",
                         streams)
        if r.get("rc") not in (0, None) or r.get("error"):
            break
        total_frames += r["frames"]
        total_wall += r["wall_s"]
        passes += 1
    return dict(frames=total_frames, wall_s=round(total_wall, 2), passes=passes,
                streams=streams,
                fps=round(total_frames / total_wall, 2) if total_wall else 0.0)


def lp_auto_level(svt):
    """What --lp 0 resolves to here, read from the encoder's own banner."""
    try:
        out = subprocess.run([svt, "--help"], capture_output=True, text=True,
                             timeout=20).stdout
        return "0-6" if "Level of parallelism" in out else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def encode_phase(clip, frames, preset, crf, lp, label, streams=1):
    """Run `streams` encoders at once; return the aggregate rate.

    One encoder at --lp 0 is not what this fleet runs. The roster asks for
    slots=5 at lp_level=6, and a single stream cannot show what a machine does
    with several: an encoder that leaves cores idle on its own may scale
    almost linearly to two, while one that already saturates gains nothing.
    Aggregate fps is total frames over the WINDOW the streams overlapped in,
    not the sum of their individual walls, or a straggler would inflate it.
    """
    if streams > 1:
        out, box = {}, [None] * streams
        t0 = time.monotonic()
        threads = []
        for k in range(streams):
            def _one(k=k):
                box[k] = _encode_one(clip, frames, preset, crf, lp,
                                     f"{label}/s{k}")
            th = threading.Thread(target=_one)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        wall = time.monotonic() - t0
        runs = [r for r in box if r]
        bad = [r for r in runs if r.get("error") or r.get("rc") not in (0, None)]
        out = dict(streams=streams, frames=sum(r.get("frames", 0) for r in runs),
                   wall_s=round(wall, 2),
                   per_stream_fps=[r.get("fps") for r in runs],
                   rc=(bad[0].get("rc") if bad else 0))
        if bad:
            out["error"] = bad[0].get("error") or f"stream rc={bad[0].get('rc')}"
        out["fps"] = round(out["frames"] / wall, 2) if wall else 0.0
        return out
    return _encode_one(clip, frames, preset, crf, lp, label)


def _encode_one(clip, frames, preset, crf, lp, label):
    """ffmpeg decode | SvtAv1EncApp, discarding output. Returns fps."""
    ff, svt = find("ffmpeg"), find("SvtAv1EncApp")
    if not ff or not svt:
        return dict(error=f"missing binary: ffmpeg={ff} SvtAv1EncApp={svt}")
    dec = [ff, "-v", "error", "-i", clip, "-frames:v", str(frames),
           "-pix_fmt", "yuv420p10le", "-strict", "-1", "-f", "yuv4mpegpipe", "-"]
    enc = ([svt, "-i", "stdin", "--preset", str(preset), "--crf", str(crf),
            "--lp", str(lp), "--input-depth", "10"]
           + ENCODER_PARAMS.split() + ["-b", os.devnull])
    log(f"[{label}] {' '.join(enc)}")
    t0 = time.monotonic()
    p1 = subprocess.Popen(dec, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(enc, stdin=p1.stdout, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE)
    p1.stdout.close()
    err = p2.communicate()[1].decode("utf-8", "replace")
    p1.wait()
    wall = time.monotonic() - t0
    rss = None
    m = re.search(r"(\d+)\s*(?:kB|KB)", err)
    if m:
        rss = int(m.group(1))
    return dict(frames=frames, wall_s=round(wall, 2),
                fps=round(frames / wall, 2) if wall else 0.0, rss_kb=rss,
                rc=p2.returncode)


class Discard(threading.Thread):
    """Accept the denoised stream and throw it away.

    The denoiser must have somewhere to send frames or it blocks, and using a
    real encoder as the sink would contaminate the very thing being measured.
    """

    def __init__(self, port=0):
        """port=0 lets the kernel pick a free one.

        A fixed port made a stale run from 40 minutes earlier block a new one
        with EADDRINUSE, which killed the whole benchmark on that host. The
        sink is local and its port is handed to the denoiser directly, so
        nothing needs to know the number in advance.
        """
        super().__init__(daemon=True)
        self.bytes = 0
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self._halt = threading.Event()

    def run(self):
        self.sock.settimeout(5.0)
        while not self._halt.is_set():
            try:
                conn, _ = self.sock.accept()
            except (socket.timeout, OSError):
                continue
            try:
                while not self._halt.is_set():
                    b = conn.recv(1 << 20)
                    if not b:
                        break
                    self.bytes += len(b)
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


def denoise_phase(root, clip, port, tile, window, margin, sigma, label,
                  backend="trt", device=0, width=1920, height=1080):
    """Run the real denoise path, streaming to a discard sink. Returns fps."""
    sink = Discard(port)
    port = sink.port
    env = dict(os.environ)
    if backend == "migraphx":
        # Same construction as tools/archive_batch/dispatch_cmd.py: BSVD's
        # MIGraphX wheels stop at cp312, so this lane needs its own
        # interpreter, its own vspipe and its own ROCm libraries.
        py = os.path.join(MIGX_VENV, "bin", "python")
        env["VSPIPE"] = os.path.join(MIGX_VENV, "bin", "vspipe")
        env["LD_LIBRARY_PATH"] = MIGX_LIBS
    else:
        py = os.path.join(root, ".venv", "bin", "python")
        for cand in ("/opt/archav1an/venv/bin/python", py, sys.executable):
            if cand and os.access(cand, os.X_OK):
                py = cand
                break
    argv = [py, os.path.join(root, "tools", "svtav1-dispatch.py"),
            "--denoise-serve", f"127.0.0.1:{port}", "-i", clip,
            "--denoise-bsvd", "--bsvd-sigma", str(sigma),
            "--bsvd-device", str(device)]
    # tools/archive_batch/dispatch_cmd.py passes no tile flags at all for a
    # full-frame lane. Passing --bsvd-tile none reaches int("none").
    if tile and tile != "none":
        argv += ["--bsvd-tile", tile, "--bsvd-overlap", "32"]
    if window:
        argv += ["--bsvd-window", str(window), "--bsvd-margin", str(margin)]
    log(f"[{label}] {' '.join(argv)}")
    sink.start()
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    out = proc.communicate()[0].decode("utf-8", "replace")
    wall = time.monotonic() - t0
    sink.stop()
    # Count frames from the bytes the sink accepted rather than by parsing
    # dispatch's output, which does not print a total. The stream is raw
    # YUV420P10 at a fixed size per frame, so this is exact -- it reproduced
    # ffprobe's 3357 for the bench clip to the frame.
    frames = 0
    if width and height:
        per_frame = width * height * 3 // 2 * 2
        frames = sink.bytes // per_frame
    return dict(wall_s=round(wall, 2), frames=frames,
                fps=round(frames / wall, 2) if (wall and frames) else None,
                bytes_received=sink.bytes, rc=proc.returncode,
                tail=out.strip()[-400:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="repo checkout on this host")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--preset", default="4",
                    help="4 is what the pipeline uses (--speed 4)")
    ap.add_argument("--crf", default="27")
    ap.add_argument("--lp", default="0", help="0 lets the encoder choose")
    ap.add_argument("--streams", type=int, default=1,
                    help="concurrent encoders; the roster runs slots=5")
    ap.add_argument("--tile", default="none")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--margin", type=int, default=32)
    ap.add_argument("--sigma", default="0.05")
    ap.add_argument("--backend", default="trt", choices=("trt", "migraphx"),
                    help="migraphx selects encoder-host's iGPU lane and its own venv")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--port", type=int, default=0,
                    help="0 lets the kernel pick, which is what you want")
    ap.add_argument("--cool-to", type=float, default=70.0,
                    help="wait for the GPU to reach this C between phases")
    ap.add_argument("--skip-denoise", action="store_true",
                    help="encode phases only, for a host with no GPU lane")
    args = ap.parse_args()

    probe = amd_sample if args.backend == "migraphx" else gpu_sample

    svt = find("SvtAv1EncApp")
    banner = ""
    if svt:
        try:
            banner = subprocess.run([svt, "--version"], capture_output=True,
                                    text=True, timeout=20).stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            banner = "unknown"
    result = dict(host=socket.gethostname(), cores=os.cpu_count(), encoder=banner,
                  clip=os.path.basename(args.clip), frames=args.frames,
                  preset=args.preset, crf=args.crf, lp=args.lp,
                  backend=args.backend, device=args.device,
                  streams=args.streams)
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    result["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    # 1. encoder alone
    result["cool_before_solo_encode"] = cooldown(args.cool_to, probe=probe)
    s = Sampler(probe=probe); s.start()
    result["solo_encode"] = encode_phase(args.clip, args.frames, args.preset,
                                         args.crf, args.lp, "solo-encode",
                                         args.streams)
    result["solo_encode"].update(s.stop())

    if not args.skip_denoise:
        # 2. denoiser alone, from the same thermal state as phase 3. Without
        #    this it starts heat-soaked from the encode above, which on gpu3
        #    made it read 39% slower than the same work later in the run.
        result["cool_before_solo_denoise"] = cooldown(args.cool_to, probe=probe)
        s = Sampler(probe=probe); s.start()
        result["solo_denoise"] = denoise_phase(args.root, args.clip, args.port,
                                               args.tile, args.window,
                                               args.margin, args.sigma,
                                               "solo-denoise", args.backend,
                                               args.device)
        result["solo_denoise"].update(s.stop())

        # 3. both. The encoder loops for as long as the denoiser runs, so the
        #    window really is concurrent throughout. One 600-frame pass left
        #    the denoiser alone for 88% of the window on gpu3 and 94% on
        #    gpu4, which is why they looked untouched by the encoder.
        result["cool_before_both"] = cooldown(args.cool_to, probe=probe)
        s = Sampler(probe=probe); s.start()
        box, done = {}, threading.Event()

        def _dn():
            try:
                box["both"] = denoise_phase(args.root, args.clip, args.port,
                                            args.tile, args.window, args.margin,
                                            args.sigma, "both-denoise",
                                            args.backend, args.device)
            finally:
                done.set()

        t = threading.Thread(target=_dn, daemon=True)
        t.start()
        time.sleep(45)      # let the engine load and the lane reach steady state
        result["both_encode"] = encode_until(done, args.clip, args.frames,
                                             args.preset, args.crf, args.lp,
                                             "both-encode", args.streams)
        t.join(timeout=7200)
        result["both_denoise"] = box.get("both", {})
        result["both_encode"].update(s.stop())

    json.dump(result, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
