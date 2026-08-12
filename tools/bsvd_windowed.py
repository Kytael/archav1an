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


def parse_tile_arg(text):
    """Read a tile setting: "auto", "HxW", or a bare square size.

    "auto" sizes each axis to the frame, which wastes far less than a square
    tile on a 16:9 source. Inverse of tile_arg(); the split-host path sends
    this value over the wire and parses it back, so the two must agree.
    """
    text = text or "0"
    if text == "auto":
        return "auto"
    if "x" in text:
        th, tw = text.split("x", 1)
        return (int(th), int(tw))
    return int(text)


def tile_arg(tile):
    """Serialise a tile setting back to its command-line form."""
    if isinstance(tile, (tuple, list)):
        return f"{tile[0]}x{tile[1]}"
    return str(tile)


def check_tile_fits(H, W, tile_h, tile_w, overlap):
    """Raise if a tile cannot be cut from the reflect-padded frame.

    A tile is sliced out after the frame is padded by `overlap`, so it can be
    at most H + overlap by W + overlap. A larger tile makes the slice come
    back short, and the model then rejects it inside a skip connection --
    "Expected size 1096 but got size 1216" -- which names neither the tile nor
    the setting that caused it. For 1080p the ceiling is 1096 x 1936, and 1096
    is also the tile that yields exactly one row of output.
    """
    for axis, t, limit in (("height", tile_h, H + overlap),
                           ("width", tile_w, W + overlap)):
        if t % 4:
            raise ValueError(
                f"tile {axis} must be a multiple of 4, got {t}: the model holds "
                f"state at half and quarter resolution, so //2 and //4 must be exact")
        if t > limit:
            raise ValueError(
                f"tile {axis} {t} exceeds the padded frame's {limit}: a tile can "
                f"be at most the frame plus one overlap in each axis")


# 1096x1096 = 1.20 Mpx was measured working on the 2070S's 8 GB, at about
# 5.3 GB resident. That is the largest tile with a measurement behind it, so
# it is the budget rather than a computed VRAM estimate.
TILE_BUDGET_MPX = 1.2


def plan_tiling(H, W, overlap=16, budget_mpx=TILE_BUDGET_MPX, max_grid=8):
    """Pick (tile_h, tile_w) covering HxW with the least redundant area.

    A tile contributes `tile - overlap`, so covering H in `rows` passes needs
    a tile of ceil(H / rows) + overlap, rounded up to a multiple of 4. Square
    tiles cannot cover a 16:9 frame evenly, so the last one lands almost on
    top of its neighbour: at 1080p a 576 tile does 1.28x the frame's pixels
    and a 960 tile does 2.67x. Sizing each axis to the frame instead gets that
    to 1.03x, which is the single largest win available on the 2070S lane.

    Grids whose tile exceeds `budget_mpx` are skipped, so this never returns a
    tile the card cannot hold.
    """
    def up4(v):
        return -(-v // 4) * 4

    best = None
    for rows in range(1, max_grid + 1):
        for cols in range(1, max_grid + 1):
            th = up4(-(-H // rows) + overlap)
            tw = up4(-(-W // cols) + overlap)
            if th > H + overlap or tw > W + overlap:
                continue
            area = th * tw / 1e6
            if area > budget_mpx:
                continue
            # Total model pixels per frame is what costs time; fewer tiles
            # breaks a tie because each one is a separate state reset.
            key = (rows * cols * area, rows * cols)
            if best is None or key < best[0]:
                best = (key, th, tw)
    if best is None:
        raise ValueError(
            f"no tiling of {W}x{H} fits {budget_mpx} Mpx with overlap {overlap}")
    return best[1], best[2]


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
        # Set by info(); RGBS until proven otherwise, and info() is called
        # before the first read.
        self.raw_dtype = None
        self.vspipe = vspipe or os.environ.get("VSPIPE") or shutil.which("vspipe")
        if not self.vspipe:
            raise RuntimeError(
                "vspipe not found (PATH or $VSPIPE): the window source cannot be read")
        if not os.path.exists(self.script):
            raise RuntimeError(f"window source script not found: {self.script}")

    def info(self):
        """Width, height and frame count that the script actually produces.

        Also records the pipe's sample format, so the reader never assumes it.
        Guessing wrong does not raise -- it reinterprets the bytes and hands
        the model a garbled frame -- so it is read from the script itself.
        """
        import numpy as np

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
            geom = (int(fields["Width"]), int(fields["Height"]),
                    int(fields["Frames"]))
            sample, bits = fields["Sample Type"], int(fields["Bits"])
        except (KeyError, ValueError):
            raise RuntimeError(f"could not read geometry from vspipe --info:\n{out.stdout}")
        if sample != "Float" or bits not in (16, 32):
            raise RuntimeError(
                f"window source must be float RGB, got {sample} {bits}-bit "
                f"({fields.get('Format Name')}) from {self.script}")
        self.raw_dtype = np.float16 if bits == 16 else np.float32
        return geom

    def read_into(self, dest, feed_start, feed_end):
        """Fill dest[k] with source frame reflect_idx(feed_start + k)."""
        import numpy as np

        if self.raw_dtype is None:
            self.info()
        real_start, real_end, slots = window_read_plan(
            feed_start, feed_end, self.num_frames)
        cmd = [self.vspipe, "-s", str(real_start), "-e", str(real_end - 1),
               self.script, "-"]
        raw = np.empty((3, self.height, self.width), dtype=self.raw_dtype)
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
                              tile=576, overlap: int = 16,
                              window: int = 750, margin: int = 32,
                              source_script: str = "", vspipe: str = ""):
    """Wrap tile-sequential windowed BSVD as a VS clip via ModifyFrame.

    `source_clip` must be RGBS. Output has the same format, size and length.
    `source_script` is a VapourSynth script whose output is that same clip; it
    is what the reader subprocess runs, and its geometry is checked against
    `source_clip` before the engine is built. `source_clip` itself is used for
    frame properties and length, never for pixels.

    Four buffers are resident, because both the source read and the sweep run
    a window ahead: the source being swept and the next one being read, each
    (window + 2*margin) frames, plus the output being served and the next one
    being built, each `window` frames. All are held in the engine's dtype,
    lossless against what the model produced. At 1080p fp16 that is 12.4 MB a
    frame, so the default window of 750 costs about 39 GB and is constant in
    clip length. It scales linearly with `window`, so raising it is what makes
    this path expensive, not long clips. Lower `window` to trade throughput
    for memory; the fixed cost of the 2 * margin discarded frames grows as the
    window shrinks.

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

    # "auto" sizes each axis to the frame, which is worth about 1.24x on the
    # pixel work at 1080p against the square 576 this lane used to run.
    if tile == "auto":
        tile_h, tile_w = plan_tiling(H, W, overlap)
    elif isinstance(tile, (tuple, list)):
        tile_h, tile_w = tile
    else:
        tile_h = tile_w = tile
    check_tile_fits(H, W, tile_h, tile_w, overlap)

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
        onnx_path=onnx_path, H=tile_h, W=tile_w, B=1, variant=variant,
        shift_num=shift_num, fp16=fp16, device_id=device_id, ep=ep)
    torch_dtype = streamer.torch_dtype
    device = streamer.device
    buf_dtype = np.float16 if torch_dtype == torch.float16 else np.float32
    # The source script sends fp16 because the engine is normally fp16 and the
    # window is held in that dtype anyway. An fp32 engine fed from an fp16
    # pipe would be quietly denoising rounded input, so refuse rather than
    # lose precision silently.
    if np.dtype(buf_dtype).itemsize > np.dtype(reader.raw_dtype).itemsize:
        raise ValueError(
            f"engine works in {np.dtype(buf_dtype).name} but {source_script} "
            f"sends {np.dtype(reader.raw_dtype).name}: the source would be "
            f"rounded below the engine's precision")

    half = overlap // 2
    out_h, out_w = tile_h - overlap, tile_w - overlap
    tiles = [(y, x) for y in tile_origins(H, tile_h, overlap)
             for x in tile_origins(W, tile_w, overlap)]

    state = {'start': None, 'buf': None}
    # VapourSynth prefetches, so several selector calls run at once. The
    # streamer is one ONNX session with one set of state tensors: a second
    # thread entering _build_window calls reset() underneath the first and the
    # run deadlocks. bsvd_vs_filter locks its selector for the same reason.
    # Observed as GPU 0%, CPU 0.9%, 138 parked threads, exactly at the first
    # window rebuild -- a whole-clip run never rebuilds, so it never showed.
    lock = threading.Lock()

    # Reading a window's source leaves the GPU idle: measured on the 2070S at
    # about 43 s of every 231 s window, 18% of the lane. It needs no GPU and
    # no VapourSynth worker, only a pipe and numpy copies, both of which drop
    # the GIL -- so the next window's source is read on its own thread while
    # this window's tiles sweep. Costs one extra source buffer.
    ahead = {'a': None, 'src': None, 'thread': None, 'error': None}
    # The next window's finished output, swept while the encoder drains this
    # one. Only one sweep runs at a time: the streamer is a single session
    # with one set of state tensors, and a second reset() underneath it would
    # corrupt the run.
    built = {'a': None, 'buf': None, 'thread': None, 'error': None}

    def _read_source(a):
        # Decoded once for the whole window, not once per tile: fetching inside
        # the tile loop cost a full ffms2 decode plus bicubic-to-RGBS per tile
        # and left the GPU at 0% while one core did colour conversion.
        _, feed_start, feed_end = plan_window(a, num_frames, window, margin)
        src = np.empty((feed_end - feed_start, 3, H, W), dtype=buf_dtype)
        reader.read_into(src, feed_start, feed_end)
        return src

    def _drop_prefetch():
        if ahead['thread'] is not None:
            ahead['thread'].join()
        ahead.update(a=None, src=None, thread=None, error=None)

    def _start_prefetch(a):
        if a is None or a >= num_frames:
            return

        def work():
            try:
                ahead['src'] = _read_source(a)
            except Exception as exc:
                # Surfaced when this window is claimed, not here, so a read
                # failure still reaches the caller rather than vanishing.
                ahead['error'] = exc

        ahead.update(a=a, src=None, error=None)
        ahead['thread'] = threading.Thread(target=work, daemon=True)
        ahead['thread'].start()

    def _take_source(a):
        """The window's source, from the read-ahead when it is the right one."""
        if ahead['thread'] is not None and ahead['a'] == a:
            ahead['thread'].join()
            src, err = ahead['src'], ahead['error']
            ahead.update(a=None, src=None, thread=None, error=None)
            if err is not None:
                raise err
            return src
        # A seek landed outside the window that was being read ahead, so that
        # work is wasted; read what was actually asked for.
        _drop_prefetch()
        return _read_source(a)

    def _sweep(a):
        b, feed_start, feed_end = plan_window(a, num_frames, window, margin)
        n_feed = feed_end - feed_start
        src = _take_source(a)
        # The next window's source needs neither the GPU nor this thread, so
        # start it before the sweep rather than after it.
        _start_prefetch(b)
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
                    tin = padded[:, :, y:y + tile_h, x:x + tile_w].contiguous()
                else:
                    tin = torch.zeros(1, 3, tile_h, tile_w,
                                      device=device, dtype=torch_dtype)
                out = streamer.step(tin, sigma)
                oi = feed_start + k - shift_num
                if a <= oi < b:
                    crop = out[0, :, half:half + ah, half:half + aw].clamp(0, 1)
                    buf[oi - a, :, y:ey, x:ex] = crop.cpu().numpy().astype(
                        buf_dtype, copy=False)
        del src
        return buf

    def _start_build(a):
        """Sweep window `a` in the background, so the GPU works while the
        encoder drains the window already finished.

        Without this the next sweep cannot start until the selector has handed
        the encoder the last frame of the current window, and the encoder
        applies back pressure long before that. Measured on the 2070S as 23 s
        of GPU idle per 165 s window, 14% of the lane, with the encoder's CPU
        climbing through every gap.
        """
        if a is None or a >= num_frames:
            return

        def work():
            try:
                built['buf'] = _sweep(a)
            except Exception as exc:
                built['error'] = exc

        built.update(a=a, buf=None, error=None)
        built['thread'] = threading.Thread(target=work, daemon=True)
        built['thread'].start()

    def _make_current(a):
        """Publish window `a`, then queue the one after it."""
        if built['thread'] is not None and built['a'] == a:
            built['thread'].join()
            buf, err = built['buf'], built['error']
            built.update(a=None, buf=None, thread=None, error=None)
            if err is not None:
                raise err
        else:
            # A seek landed outside the window being built, so that sweep is
            # wasted. Wait for it anyway: it owns the streamer, and a second
            # sweep would call reset() underneath it.
            if built['thread'] is not None:
                built['thread'].join()
                built.update(a=None, buf=None, thread=None, error=None)
            buf = _sweep(a)
        state['start'], state['buf'] = a, buf
        _start_build(a + window)

    def selector(n, f):
        with lock:
            start = state['start']
            if start is None or not (start <= n < start + state['buf'].shape[0]):
                _make_current((n // window) * window)
            arr = state['buf'][n - state['start']].copy()
        fout = f.copy()
        for c in range(3):
            np.asarray(fout[c])[:] = arr[c]
        return fout

    return vs.core.std.ModifyFrame(source_clip, source_clip, selector)
