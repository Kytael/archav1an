# "Core freed but N bytes still allocated in framebuffers"

Status: diagnosed on encoder-host 2026-08-13. No change made. The conclusion is to leave it
alone; the reasons matter because the obvious fix breaks the pipeline.

## It is not GPU memory

The word "framebuffer" here is VapourSynth's, not the driver's. `VSCore::freeCore()` in
`src/core/vscore.cpp` checks its own host-side frame pool as the core shuts down:

```cpp
    if (memory->allocated_bytes())
        logMessage(mtWarning, "Core freed but " +
            safe_to_string(memory->allocated_bytes()) +
            " bytes still allocated in framebuffers");
```

So the message means "frames were still held when the core went away". It says nothing about
CUDA, TensorRT or MIGraphX, and it appears after every output frame has already been
delivered.

## It happens on encoder-host

Untiled BSVD through the 8060S MIGraphX lane, 300 frames of `MVI_1463`:

```
Output 300 frames in 72.81 seconds (4.12 fps)
Warning: Core freed but 522551232 bytes still allocated in framebuffers
```

522,551,232 bytes is exactly 21 frames of 1080p RGBS. The same message on gpu4 was
771,385,152 bytes, 31 frames of the same format.

The plain encode path does **not** warn. A 1200-frame `vspipe | SvtAv1EncApp` run with no
denoiser reports `Output 1200 frames in 40.09 seconds (29.94 fps)` and nothing else.

## Cause

`tools/bsvd_vs_filter.py:338` is the only `get_frame_async` call in the tree. The untiled
BSVD filter is a `std.ModifyFrame` selector that pulls its own source frames:

```python
    def _request(j):
        if 0 <= j < feed_end and j not in pending:
            pending[j] = source_clip.get_frame_async(_reflect_idx(j - shift_num, num_frames))
```

That gives the source node two consumers — the selector's own requests, and `ModifyFrame`'s
`f` argument — offset by `2 * shift_num` frames. VapourSynth keeps a frame cache on the node
so the second consumer hits instead of re-decoding. Whatever that cache still holds when the
core is freed is what the message counts.

Reproduced with no GPU, no BSVD and no TensorRT by a VPY that only mimics the shape — an
`ffms2` source, a `ModifyFrame` selector, and one frame of `get_frame_async` lookahead:

| variant | residual |
|---|---|
| plain source, 300 frames | none |
| `ModifyFrame` with `f.copy()`, no async | none |
| **async lookahead, 300 frames** | **49,766,784 B (2 frames)** |
| async, no lookahead, 300 frames | 24,883,392 B (1 frame) |
| synchronous `get_frame` instead | none |
| **async lookahead, 1200 frames** | **1,443,236,736 B (58 frames)** |

## It is bounded, and it is not an orphan

The two runs above suggested the residual grows with run length. It does not. Re-measured
2026-08-13 with the same graph shape driven by a synthetic `BlankClip` source, so run length
is the only variable:

| frames | residual | RGBS frames | wall |
|---|---|---|---|
| 300 | 99,533,568 B | 4 | 0.9 s |
| 1200 | 2,687,406,336 B | 108 | 3.6 s |
| 3000 | 2,886,473,472 B | 116 | 8.1 s |
| 10000 | 1,020,219,072 B | 41 | 26.6 s |

The residual rises, flattens well below the 4096 MB `core.max_cache_size` the generated VPY
sets, and at 10000 frames is lower than at 1200. That is a cache whose occupancy at teardown
depends on where eviction happened to be, not a leak: a leak would be monotonic.

Nothing is orphaned either. The memory is ordinary host heap in the vspipe process, and the
kernel reclaims it when vspipe exits, which is the next thing that happens. The warning is
printed on the way out.

## Do not shrink the cache

The obvious fix is to lower `core.max_cache_size` so less is held at exit. It **deadlocks**.
Dropping it from 4096 MB to 1024 MB hung a 300-frame mimic that otherwise finishes in one
second; it never produced a frame. The selector waits on `get_frame_async` from inside a
filter callback running on the VS thread pool, and the cache headroom is what keeps that from
starving. `tools/bsvd_windowed.py:44` records the synchronous version of the same hazard:
"Reading them in process deadlocks: get_frame() called from the graph".

## You do not normally see it anyway

`svtav1-dispatch.py` sends vspipe's stderr to `Temp/<tag>/<stem>/<stem>_vspipe.log`, prints
the tail only when a process exits non-zero (`run_piped`, :449-458), and deletes the log on
success (:1236). So on a run that works the message is written and thrown away. It surfaces
only in failure diagnostics — which is where it was seen, next to a real
`Failed to read packet: Input/output error`.

## Decision: ignore it

Closed 2026-08-13. The three things that would make it worth acting on are all false: it is
not GPU memory, it is not unbounded, and it is not held after the process exits. It is an
accounting note at teardown over a cache that is load-bearing — shrinking that cache
deadlocks the graph.

No code change. If the line is ever distracting in a failure log, drop it in
`_print_log_tail` rather than touching the graph or the cache size.
