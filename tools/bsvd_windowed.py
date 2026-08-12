#!/usr/bin/env python3
"""Bounded-memory BSVD for cards that cannot hold the full-frame state.

The RTX 2070 SUPER has 8 GB. BSVD-64 V2 stateful streaming needs about 7.4 GB
of temporal state for a whole 1080p frame, so this card must run TILE
SEQUENTIAL: one streamer at tile resolution, reset between tiles, walking each
tile across the clip (about 1 GB peak).

Tile-sequential completes no output frame until the last tile pass, so the
naive worker buffers the entire denoised clip. At 1080p fp16 that is 12.4 MB a
frame, which is 335 GB for the longest clip in the dance archive and cannot
process 78% of it (spec 5.5(a)).

The fix bounds the clip, not the tile:

    for each window [a, b):
        for each tile T:
            reset state; feed source frames [a-M, b+M) through T
        keep outputs for [a, b); discard both margins

Memory becomes (b - a) frames and stops depending on clip length.

Why a discarded margin works, and why nothing is blended: BSVD's temporal state
is bounded. It is all shift registers -- FIFOs of depth SKIP_LENS=[8, 8, 4] and
the double-buffered memconv state, every one of them zeroed by reset() and
overwritten as frames pass. Nothing accumulates, so old frames fall off the end
rather than decaying away. Feed a window real neighbouring frames for M frames
either side and its state on entering [a, b) approaches the state a whole-clip
run would have had, and stops depending on where the window began.

How close that is to exact is measured, not assumed. Gate 2 compares a windowed
run against a whole-clip run on the same engine; gate 3 sweeps M. The reach is
inferred from buffer shapes and names, not read out of the ONNX graph, and fp16
kernels need not be bit-reproducible across differing call sequences, so the
honest expectation is agreement at fp16 epsilon with no error concentrated at
the window boundaries. A seam would show up as error clustered at a and b; that
is the thing to look for, and the thing M exists to prevent.

Blending was rejected regardless: it softens detail and breathes at the window
period, and it would hide a seam rather than remove one. M defaults to 32,
double the inferred requirement.

The window's source frames are decoded by a separate vspipe process, not by
this graph. Reading them in process deadlocks: get_frame() called from the
selector runs on a VapourSynth worker thread and needs another worker to
produce the frame, but every other worker is already parked in the selector
waiting for the same window. Observed on the 2070S as 116 threads in
futex_wait with both the CPU and the GPU at 0%. A subprocess has its own core
and its own thread pool, so nothing this graph waits on waits on this graph.
"""
# torch, numpy, vapoursynth and the streamer are imported inside the builder,
# not here. The geometry below is plain arithmetic and decides whether output is
# exact, so it must stay unit-testable on a host with no GPU stack.
import os
import shutil
import subprocess
import tempfile

# vspipe writes RGB planes in GBR order, while frame[i] in Python is RGB:
# a BlankClip of color [0.25, 0.5, 0.75] reads back as [0.25, 0.5, 0.75] from
# frame[i] and as [0.5, 0.75, 0.25] from the raw pipe. Measured on R76, not
# assumed. RAW_PLANE_ORDER[p] is the frame plane that raw plane p holds, and
# test_the_subprocess_reader_matches_get_frame checks it against the real thing.
RAW_PLANE_ORDER = (1, 2, 0)


def tile_origins(size, tile, overlap):
    """Start offsets covering `size`, each tile contributing `tile - overlap`.

    Matches the prior-art worker so tile geometry is unchanged; only the frame
    loop is bounded.
    """
    out = tile - overlap
    starts = list(range(0, max(size - out, 0) + 1, out))
    if not starts or starts[-1] + out < size:
        starts.append(max(size - out, 0))
    return starts


def reflect_idx(i, n):
    """numpy-style 'reflect' (no edge repeat): ...f2 f1 | f0..f(n-1) | f(n-2)...

    Used for source indices outside the clip, so a window at either end of the
    clip sees exactly what the unwindowed path's mirror padding produced.
    """
    if n == 1:
        return 0
    period = 2 * n - 2
    i %= period
    return i if i < n else period - i


def plan_window(a, num_frames, window, margin):
    """Return (b, feed_start, feed_end) for the window beginning at `a`.

    feed_* are source indices, and may fall outside [0, num_frames): the caller
    reflects them. feed_end is exclusive and always reaches b + margin, which
    covers the streamer's shift_num pipeline delay because margin >= shift_num.
    """
    b = min(a + window, num_frames)
    return b, a - margin, b + margin


def window_read_plan(feed_start, feed_end, num_frames):
    """Turn the window's reflected frame list into one contiguous read.

    Returns (real_start, real_end, slots). The reader decodes real frames
    [real_start, real_end) once, in order; slots[j] lists the window positions
    that copy of real frame real_start + j fills. A frame near either end of
    the clip is reflected, so it lands in two positions; every other frame
    lands in one. Reading forward once is what lets the source come from a
    pipe instead of random access.
    """
    wanted = [reflect_idx(i, num_frames) for i in range(feed_start, feed_end)]
    real_start, real_end = min(wanted), max(wanted) + 1
    slots = [[] for _ in range(real_end - real_start)]
    for k, idx in enumerate(wanted):
        slots[idx - real_start].append(k)
    return real_start, real_end, slots


def check_tile_fits(H, W, tile, overlap):
    """Raise if `tile` cannot be cut from the reflect-padded frame.

    A tile is sliced out after the frame is padded by `overlap`, so it can be
    at most H + overlap by W + overlap. A larger tile makes the slice come
    back short, and the model then rejects it inside a skip connection --
    "Expected size 1096 but got size 1216" -- which names neither the tile nor
    the setting that caused it. For 1080p the ceiling is 1096 x 1936, and 1096
    is also the tile that yields exactly one row of output.
    """
    if tile % 4:
        raise ValueError(
            f"tile must be a multiple of 4, got {tile}: the model holds state "
            f"at half and quarter resolution, so H//2 and H//4 must be exact")
    if tile > H + overlap or tile > W + overlap:
        raise ValueError(
            f"tile {tile} exceeds the padded frame {H + overlap}x{W + overlap}: "
            f"a tile can be at most the frame plus one overlap in each axis")


def _read_exact(stream, view):
    """Fill `view` from `stream`, or raise if the stream ends first."""
    got = 0
    while got < len(view):
        n = stream.readinto(view[got:])
        if not n:
            raise EOFError(f"source pipe ended after {got} of {len(view)} bytes")
        got += n


class VspipeWindowSource:
    """Decode a window's source frames with a vspipe subprocess.

    `script` must be a VapourSynth script whose output is the RGBS clip this
    filter denoises -- the same clip, built the same way, or the windows are
    denoised from something other than what the graph claims. The builder
    checks info() against the clip before it builds the engine.
    """

    def __init__(self, script, width, height, num_frames, vspipe=None):
        self.script = os.path.abspath(script)
        self.width, self.height, self.num_frames = width, height, num_frames
        # Same order svtav1-dispatch uses (:956): the child must be the build
        # the parent chose, or it decodes with a different VapourSynth.
        self.vspipe = vspipe or os.environ.get("VSPIPE") or shutil.which("vspipe")
        if not self.vspipe:
            raise RuntimeError(
                "vspipe not found (PATH or $VSPIPE): the window source cannot be read")
        if not os.path.exists(self.script):
            raise RuntimeError(f"window source script not found: {self.script}")

    def info(self):
        """Width, height and frame count that the script actually produces."""
        out = subprocess.run([self.vspipe, "-i", self.script, "-"],
                             capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(
                f"vspipe --info failed on {self.script}:\n{out.stderr.strip()[-2000:]}")
        fields = {}
        for line in out.stdout.splitlines():
            key, _, val = line.partition(":")
            fields[key.strip()] = val.strip()
        try:
            return (int(fields["Width"]), int(fields["Height"]), int(fields["Frames"]))
        except (KeyError, ValueError):
            raise RuntimeError(f"could not read geometry from vspipe --info:\n{out.stdout}")

    def read_into(self, dest, feed_start, feed_end):
        """Fill dest[k] with source frame reflect_idx(feed_start + k)."""
        import numpy as np

        real_start, real_end, slots = window_read_plan(
            feed_start, feed_end, self.num_frames)
        cmd = [self.vspipe, "-s", str(real_start), "-e", str(real_end - 1),
               self.script, "-"]
        raw = np.empty((3, self.height, self.width), dtype=np.float32)
        view = memoryview(raw.reshape(-1).view(np.uint8))
        # stderr goes to a file, not a pipe: nothing drains a pipe while this
        # loop is reading stdout, so a chatty child would block on a full
        # stderr buffer and hang the window instead of failing.
        with tempfile.TemporaryFile() as errf:
            def tail():
                errf.seek(0)
                return errf.read().decode(errors="replace").strip()[-2000:]

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=errf, bufsize=0)
            try:
                for targets in slots:
                    _read_exact(proc.stdout, view)
                    for p in range(3):
                        plane = raw[p]
                        for k in targets:
                            dest[k, RAW_PLANE_ORDER[p]] = plane
            except EOFError as e:
                proc.kill()
                proc.wait()
                raise RuntimeError(
                    f"window source {real_start}..{real_end - 1} of "
                    f"{self.script} stopped early ({e}):\n{tail()}") from e
            finally:
                proc.stdout.close()
            rc = proc.wait()
            if rc:
                raise RuntimeError(
                    f"vspipe exited {rc} reading frames {real_start}.."
                    f"{real_end - 1} of {self.script}:\n{tail()}")


def build_bsvd_windowed_tiled(source_clip, *,
                              onnx_path: str, sigma: float, ep: str = 'TRT',
                              device_id: int = 0, fp16: bool = True,
                              variant: str = 'bsvd-64',
                              shift_num: int = 16,
                              tile: int = 576, overlap: int = 16,
                              window: int = 750, margin: int = 32,
                              source_script: str = "", vspipe: str = ""):
    """Wrap tile-sequential windowed BSVD as a VS clip via ModifyFrame.

    `source_clip` must be RGBS. Output has the same format, size and length.
    `source_script` is a VapourSynth script whose output is that same clip; it
    is what the reader subprocess runs, and its geometry is checked against
    `source_clip` before the engine is built. `source_clip` itself is used for
    frame properties and length, never for pixels.

    Two buffers are resident: the window's decoded source, (window + 2*margin)
    frames, and its output, `window` frames. Both are held in the engine's
    dtype, lossless against what the model produced. At 1080p fp16 that is
    12.4 MB a frame, so the default window of 750 costs about 19 GB and is
    constant in clip length. Lower `window` to trade throughput for memory; the
    fixed cost of the 2 * margin discarded frames grows as the window shrinks.

    Frames are served in any order within the resident window. Asking for a
    frame outside it builds that window, which is the full tile sweep, so
    non-sequential access across window boundaries is very slow. The encoder
    reads sequentially, which is the intended path.
    """
    # Argument checks first, before the heavy imports and before touching the
    # clip: these decide whether the output can be exact at all, and a bad
    # margin must fail immediately rather than after an engine build.
    if margin < shift_num:
        raise ValueError(
            f"margin {margin} is below shift_num {shift_num}: the model reads that "
            f"many future frames, so a smaller margin cannot be exact")
    if overlap % 2:
        raise ValueError(f"overlap must be even, got {overlap}")
    if window < 1:
        raise ValueError(f"window must be at least 1 frame, got {window}")
    if not source_script:
        raise ValueError(
            "source_script is required: the window's frames must be decoded out "
            "of process, because reading them from the selector deadlocks the "
            "VapourSynth thread pool")

    import threading

    import numpy as np
    import torch
    import vapoursynth as vs

    from bsvd_vs_filter import BSVDOrtStreamingV2

    if source_clip.format.id != int(vs.RGBS):
        raise ValueError(f"source_clip must be RGBS, got {source_clip.format.name}")

    H, W = source_clip.height, source_clip.width
    num_frames = source_clip.num_frames

    check_tile_fits(H, W, tile, overlap)

    # The subprocess must produce the same clip this graph claims to denoise.
    # Checked once, before the engine build, because a mismatch would show up
    # as quietly wrong pixels rather than as an error.
    reader = VspipeWindowSource(source_script, W, H, num_frames,
                                vspipe=vspipe or None)
    got = reader.info()
    if got != (W, H, num_frames):
        raise ValueError(
            f"source script {source_script} produces {got[0]}x{got[1]} "
            f"x{got[2]} frames, but the clip is {W}x{H} x{num_frames}")

    streamer = BSVDOrtStreamingV2(
        onnx_path=onnx_path, H=tile, W=tile, B=1, variant=variant,
        shift_num=shift_num, fp16=fp16, device_id=device_id, ep=ep)
    torch_dtype = streamer.torch_dtype
    device = streamer.device
    buf_dtype = np.float16 if torch_dtype == torch.float16 else np.float32

    half = overlap // 2
    out_h = out_w = tile - overlap
    tiles = [(y, x) for y in tile_origins(H, tile, overlap)
             for x in tile_origins(W, tile, overlap)]

    state = {'start': None, 'buf': None}
    # VapourSynth prefetches, so several selector calls run at once. The
    # streamer is one ONNX session with one set of state tensors: a second
    # thread entering _build_window calls reset() underneath the first and the
    # run deadlocks. bsvd_vs_filter locks its selector for the same reason.
    # Observed as GPU 0%, CPU 0.9%, 138 parked threads, exactly at the first
    # window rebuild -- a whole-clip run never rebuilds, so it never showed.
    lock = threading.Lock()

    def _build_window(a):
        b, feed_start, feed_end = plan_window(a, num_frames, window, margin)
        n_feed = feed_end - feed_start
        # Decoded once for the whole window, not once per tile: fetching inside
        # the tile loop cost twelve full ffms2 decodes plus bicubic-to-RGBS per
        # window and left the GPU at 0% while one core did colour conversion.
        src = np.empty((n_feed, 3, H, W), dtype=buf_dtype)
        reader.read_into(src, feed_start, feed_end)
        buf = np.zeros((b - a, 3, H, W), dtype=buf_dtype)
        pad = (half, half, half, half)
        for (y, x) in tiles:
            streamer.reset()
            ey, ex = min(y + out_h, H), min(x + out_w, W)
            ah, aw = ey - y, ex - x
            # Feed shift_num past the end to flush the pipeline: the output for
            # a fed frame emerges shift_num steps later.
            for k in range(n_feed + shift_num):
                if k < n_feed:
                    t = torch.from_numpy(src[k]).to(
                        device=device, dtype=torch_dtype).unsqueeze(0)
                    padded = torch.nn.functional.pad(t, pad, mode='reflect')
                    tin = padded[:, :, y:y + tile, x:x + tile].contiguous()
                else:
                    tin = torch.zeros(1, 3, tile, tile, device=device, dtype=torch_dtype)
                out = streamer.step(tin, sigma)
                oi = feed_start + k - shift_num
                if a <= oi < b:
                    crop = out[0, :, half:half + ah, half:half + aw].clamp(0, 1)
                    buf[oi - a, :, y:ey, x:ex] = crop.cpu().numpy().astype(
                        buf_dtype, copy=False)
        del src
        state['start'], state['buf'] = a, buf

    def selector(n, f):
        with lock:
            start = state['start']
            if start is None or not (start <= n < start + state['buf'].shape[0]):
                _build_window((n // window) * window)
            arr = state['buf'][n - state['start']].copy()
        fout = f.copy()
        for c in range(3):
            np.asarray(fout[c])[:] = arr[c]
        return fout

    return vs.core.std.ModifyFrame(source_clip, source_clip, selector)
