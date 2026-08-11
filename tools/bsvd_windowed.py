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
"""
from collections import OrderedDict

# torch, numpy, vapoursynth and the streamer are imported inside the builder,
# not here. The geometry below is plain arithmetic and decides whether output is
# exact, so it must stay unit-testable on a host with no GPU stack.


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


def build_bsvd_windowed_tiled(source_clip, *,
                              onnx_path: str, sigma: float, ep: str = 'TRT',
                              device_id: int = 0, fp16: bool = True,
                              variant: str = 'bsvd-64',
                              shift_num: int = 16,
                              tile: int = 576, overlap: int = 16,
                              window: int = 1500, margin: int = 32):
    """Wrap tile-sequential windowed BSVD as a VS clip via ModifyFrame.

    `source_clip` must be RGBS. Output has the same format, size and length.

    One window is resident at a time, held in the engine's dtype so it is
    lossless against what the model produced. At 1080p fp16 that is 12.4 MB a
    frame: about 18.6 GB at the default window of 1500, and constant in clip
    length. Lower `window` to trade throughput for memory; the per-window cost
    of the 2 * margin discarded frames grows as the window shrinks.

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

    import numpy as np
    import torch
    import vapoursynth as vs

    from bsvd_vs_filter import BSVDOrtStreamingV2, _vsframe_rgbs_to_torch

    if source_clip.format.id != int(vs.RGBS):
        raise ValueError(f"source_clip must be RGBS, got {source_clip.format.name}")

    H, W = source_clip.height, source_clip.width
    num_frames = source_clip.num_frames

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

    # Source frames are fetched once per tile pass. A small LRU keeps the
    # padded full frame across tiles at the same feed position from being
    # decoded len(tiles) times over.
    src_cache: 'OrderedDict[int, torch.Tensor]' = OrderedDict()
    SRC_CACHE = 8

    state = {'start': None, 'buf': None}

    def _padded_source(src_idx):
        hit = src_cache.get(src_idx)
        if hit is not None:
            src_cache.move_to_end(src_idx)
            return hit
        frame = source_clip.get_frame(src_idx)
        t = _vsframe_rgbs_to_torch(frame, device, torch_dtype)   # (1, 3, H, W)
        padded = torch.nn.functional.pad(t, (half, half, half, half), mode='reflect')
        src_cache[src_idx] = padded
        src_cache.move_to_end(src_idx)
        while len(src_cache) > SRC_CACHE:
            src_cache.popitem(last=False)
        return padded

    def _build_window(a):
        b, feed_start, feed_end = plan_window(a, num_frames, window, margin)
        buf = np.zeros((b - a, 3, H, W), dtype=buf_dtype)
        for (y, x) in tiles:
            streamer.reset()
            src_cache.clear()
            ey, ex = min(y + out_h, H), min(x + out_w, W)
            ah, aw = ey - y, ex - x
            # Feed shift_num past feed_end to flush the pipeline: the output for
            # a fed frame emerges shift_num steps later.
            for j in range(feed_start, feed_end + shift_num):
                if j < feed_end:
                    padded = _padded_source(reflect_idx(j, num_frames))
                    tin = padded[:, :, y:y + tile, x:x + tile].contiguous()
                else:
                    tin = torch.zeros(1, 3, tile, tile, device=device, dtype=torch_dtype)
                out = streamer.step(tin, sigma)
                oi = j - shift_num
                if a <= oi < b:
                    crop = out[0, :, half:half + ah, half:half + aw].clamp(0, 1)
                    buf[oi - a, :, y:ey, x:ex] = crop.cpu().numpy().astype(
                        buf_dtype, copy=False)
        state['start'], state['buf'] = a, buf

    def selector(n, f):
        start = state['start']
        if start is None or not (start <= n < start + state['buf'].shape[0]):
            _build_window((n // window) * window)
        arr = state['buf'][n - state['start']]
        fout = f.copy()
        for c in range(3):
            np.asarray(fout[c])[:] = arr[c]
        return fout

    return vs.core.std.ModifyFrame(source_clip, source_clip, selector)
