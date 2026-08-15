# Window size and denoise throughput

Status: partly measured 2026-08-15. **Open question — see the TODO at the end.**

A card that cannot hold the full-frame BSVD state runs tile-sequential and
windowed. `window` is how many output frames one sweep produces. It is not a
free parameter: it sets throughput, the wait for the first frame, and host
memory, and the three do not move together.

## What a window costs

`plan_window()` in `tools/bsvd_windowed.py:93` feeds `[a - margin, b + margin)`
to produce `window` output frames, and `_sweep()` then runs `n_feed +
shift_num` steps to flush the pipeline. So each sweep pushes

    window + 2*margin + shift_num  =  window + 80   (at margin 32)

frames of GPU work through **every tile** for `window` frames of output. That
predicts throughput rising with window size, towards a ceiling.

The window also allocates `np.zeros((b - a, 3, H, W))` — **host** RAM, not
VRAM. At 1080p fp16 that is 12.4 MB per frame: 3.1 GB at window 250, 6.2 GB at
500, 9.3 GB at 750. VRAM is what forces tiling; host RAM is what a window
costs.

And the first frame cannot emerge until the first sweep completes, so
time-to-first-frame is `window / rate` plus startup. Startup itself is small:
7 to 24 s across the fleet, measured.

## Measured: gpu3, RTX 3070 Laptop, tile auto, margin 32

Full 3357-frame 1080p clip, denoise only, sink on the denoise host
(`tools/denoise-rate.py`).

| window | overhead | sustained fps | end-to-end fps | total wall |
|---:|---:|---:|---:|---:|
| 250 | 74.5 s | 4.417 | 4.215 | 796.5 s |
| **500** | 113.7 s | **5.546** | 4.800 | **699.4 s** |
| 750 | 164.1 s | 5.160 | 4.801 | 699.2 s |

**Throughput is not monotonic in window size.** It peaks at 500 and falls at
750. The `window + 80` model predicted 5.81 fps at 750 and it delivered 5.16,
missing by 11%. The model explains why 250 is poor; it does not explain 750.
The untested hypothesis is the host working set — 9.3 GB of window buffer
against 6.2 GB — but nothing here measures that.

For total time, 500 and 750 tie exactly (699.4 against 699.2 s): the better
sustained rate at 500 is cancelled by 50 s more of first-window wait at 750.
250 loses 14%.

## Measured: gpu2, RTX 5070 Laptop

| window | result |
|---|---|
| 300 | overhead 76.6 s, sustained 4.78, end-to-end 4.34, wall 773.5 s |
| 750 | **0 of 8 attempts completed** |

At 750 seven attempts faulted inside the first window and one produced exactly
one window (749 frames) before faulting at the boundary. That partial run
reached 750 frames in 163.3 s, about 5.02 fps for the first sweep against 4.78
at window 300 — the same small gain gpu3 shows between 250 and 500, and far
below what the `window + 80` model predicts.

This card faults on its own (see `docs/encode-capacity.md`); a longer sweep
appears to make it more likely, which is consistent with a fault of roughly
constant probability per unit of GPU work, but that is inference.

## For contrast: full-frame lanes pay none of this

| lane | overhead | sustained | end-to-end |
|---|---:|---:|---:|
| gpu1 4090 | 7.0 s | 18.20 | 17.37 |
| gpu4 GB10 | 12.1 s | 4.37 | 4.31 |

No window, no first-window wait, no redundant context: gpu4 loses 1.4%
end-to-end where gpu3 loses 13.5%.

## Where the roster's values came from

Not from measurement, in two of three cases.

- **gpu2 300** was adopted because the lane "dies at the first window
  boundary" at 500. That was the driver fault, not the window: it faulted at
  300 too. The rationale is void.
- **gpu3 500** carries the comment "the same windowed settings as gpu2",
  which is not true — gpu2 is 300. It happens to be the best of the three
  values tested, but not because anyone measured it.
- **2070s 750** has no recorded rationale at all, and gpu3's curve says 750
  is on the wrong side of the peak.

## TODO

1. **Sweep window size per card and record fps.** Only gpu3 has been sampled,
   at three points. Sample finely enough to locate the peak rather than
   bracket it: 300/400/500/600 on gpu3 would say whether 500 is the top or
   just the best of a coarse grid.
2. **Run every sweep on a long clip as well as this one.** Everything above is
   one 3357-frame clip, and clip length is not a neutral choice here: it is
   the denominator the fixed overhead is divided by. At 3357 frames gpu3's
   window 750 pays 164 s of first-window wait, 23% of its total; on a clip
   three times longer that falls to 8%, and the ranking of 500 against 750
   could invert -- they already tie to within 0.2 s at this length. The
   archive's clips run much longer than the bench clip, so the value the
   roster should carry is the one measured at the length that will actually
   run. Pick a real long clip from the archive rather than looping this one:
   a repeated clip would reuse the same source frames and may not stress the
   reader or the cache the same way.
3. **Test the 2070S at 500 against its configured 750.** Highest-value single
   test: it is the fleet's best remaining windowed lane and its value was
   never justified. Needs encoder-host free.
4. **Find out why 750 regresses.** Measure RSS and page-fault behaviour across
   the sweep. If it is the host working set, the peak will move with host RAM
   and with resolution, and the fleet cannot use one number everywhere.
5. **Check the interaction with `margin` and tile size.** margin 32 is used
   everywhere and the redundancy term is `2*margin` per sweep; `MIN_MARGIN` is
   16. Tile is `auto`, which chose 1112x992 here.
6. **Re-measure once any card's driver changes.** gpu2's throughput moved 30%
   on a driver update alone.

Measure with `tools/denoise-rate.py`, which separates the fixed overhead from
the sustained rate. A single fps number blends the two and depends on clip
length, which is how the same lane came to be recorded at 4.40, 5.24 and 6.72.
