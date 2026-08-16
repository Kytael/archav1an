# Window size and denoise throughput

Status: measured 2026-08-16 on gpu3, 8 runs. **The window curve this document
used to report does not exist. Window 750 does not regress. See "What was wrong
before".**

A card that cannot hold the full-frame BSVD state runs tile-sequential and
windowed. `window` is how many output frames one sweep produces. It sets
throughput, the wait for the first frame, and host memory, and the three do not
move together.

## What a window costs

`plan_window()` in `tools/bsvd_windowed.py:93` feeds `[a - margin, b + margin)`
to produce `window` output frames, and `_sweep()` then runs `n_feed +
shift_num` steps to flush the pipeline. So each sweep pushes

    window + 2*margin + shift_num  =  window + 80   (at margin 32)

frames of GPU work through **every tile** for `window` frames of output. That
predicts throughput rising with window size, towards a ceiling. It does.

The window also costs **host** RAM, not VRAM. VRAM is what forces tiling; host
RAM is what a window costs, and it costs four buffers, not one. In steady state
these are all live at the same time:

| buffer | `bsvd_windowed.py` | frames |
|---|---|---:|
| `state['buf']`, the published window the consumer is draining | 576 | `window` |
| `built['buf']`, the next window being swept in a background thread | 511 | `window` |
| `src`, the source for the sweep in flight | 465 | `window + 2*margin` |
| `ahead['src']`, the prefetch for the window after that | 480 | `window + 2*margin` |

The prefetch starts on line 510, one line before `buf` is allocated, so that
overlap is deliberate: the reader is meant to run while the GPU does. At 1080p
fp16 and 12.4 MB per frame, with margin 32, the four together predict **14.0 GB
at window 250, 26.4 GB at 500 and 38.8 GB at 750**.

Measured peak RSS on gpu3 agrees: **27.8 GB at 500 and 39.3 GB at 750**, which
is 1.3% above the prediction at 750. The four-buffer model is correct, and host
RAM is the real constraint on window size.

An earlier version of this document counted only `buf` and put 750 at 9.3 GB.
The roster comment in `tools/archive_batch/denoisers.example.toml` counted two
and put it at 19 GB. Both were wrong; use 12.4 MB per frame times four buffers.

The first frame cannot emerge until the first sweep completes, so
time-to-first-frame is `window / rate` plus startup. That is the one real cost
of a larger window, and it is reproducible to under a second: 110 s at window
500 and 161 s at 750, across four runs each.

## Measured: gpu3, RTX 3070 Laptop, tile auto, margin 32

`MVI_1463.MOV`, 6726 frames, 1080p, denoise only, sink on the denoise host
(`tools/denoise-rate.py`). Eight runs. Both window sizes were run in both
positions of a pair, and with and without a 420 s cooldown before the run,
because run order and machine state were confounded in every earlier
measurement.

| window | cooldown | position | overhead | wall | end-to-end | sustained | GPU busy |
|---:|---|---|---:|---:|---:|---:|---:|
| 500 | yes | first | 118.30 s | 1377.53 s | 4.883 | 5.097 | — |
| 500 | yes | second | 109.91 s | 1373.06 s | 4.899 | 5.079 | 1290 s |
| 500 | yes | first | 110.01 s | 1378.34 s | 4.880 | 5.057 | 1293 s |
| 500 | no | first | 108.26 s | 1707.38 s | 3.939 | 4.977 | 1597 s |
| 750 | no | second | 161.79 s | 1793.68 s | 3.750 | 3.649 | — |
| 750 | yes | first | **161.10 s** | **1326.90 s** | **5.069** | **5.317** | 1226 s |
| 750 | yes | second | 159.46 s | 1337.05 s | 5.030 | 5.262 | 1236 s |
| 750 | no | second | 162.04 s | 1359.81 s | 4.946 | 5.189 | 1256 s |

**Window 750 beats window 500 in every clean run**, on sustained rate and on
end-to-end rate both, despite paying 50 s more first-window wait. Six of the
eight runs finish in 1327-1378 s. The two that do not are discussed below, and
there is one at each window size, so dropping them does not favour either:

| window | clean runs | sustained | end-to-end |
|---:|---:|---|---|
| 500 | 3 | 5.057 - 5.097 | 4.880 - 4.899 |
| 750 | 3 | 5.189 - 5.317 | 4.946 - 5.069 |

That is 750 ahead by about 3% end-to-end and 4% sustained, with no overlap
between the two groups. The `window + 80` model predicted a gain in this
direction; the size of it is smaller than the model suggests, which is expected
once the redundant margin work is only a seventh of the sweep.

The cost of 750 is host RAM: 39.3 GB against 27.8 GB, which on gpu3's 47 GB
leaves 7.1 GB of headroom. That is what should decide the value on a given
host, not throughput. Nothing in the traces shows the headroom hurting: swap
stayed at zero, and `pgscan_direct`, `pgsteal_direct` and `allocstall_*` stayed
at **zero for every run**, sampled at 1 Hz. There is no reclaim pressure at
7 GB of headroom on this host.

## The sporadic slow run

Two of the eight runs took about 25% longer end-to-end: one at window 500
(1707 s) and one at 750 (1794 s). They correlate with nothing that was
controlled — not window size, not cooldown, not position in the pair.

The traces say the GPU was working the whole time and simply took longer to do
identical work. In the slow 500 run the card was busy for 1597 s against 1290 s
in the clean ones, at 94.6 W against 96.6 W and 1388 MHz against 1416 MHz. A 2%
clock difference cannot produce a 24% longer run. Everything else is flat:

- **Not memory.** Peak RSS, MemAvailable and major faults all match the clean
  runs of the same window.
- **Not swap or reclaim.** Zero throughout, on every run.
- **Not thermal.** 86 °C in the slow runs and 85 °C in the clean ones, with the
  same SM clock band of 1324-1546 MHz.
- **Not tile selection.** `plan_tiling()` sizes from the fixed
  `TILE_BUDGET_MPX = 1.2`, not from free VRAM, so the tile is deterministic.

What remains is a second consumer on the same card. `nvidia-smi` reports
whole-device utilisation and power, so work from anything else on the GPU —
gpu3 is a Windows laptop and the WSL2 distro does not own the card — appears
as our job being slow while the device looks busy. That is a hypothesis; the
probe records no per-process GPU accounting and cannot confirm it.

The practical consequence is about method, not about windows: **one run per
configuration cannot measure this lane.** A 1-in-4 chance of a 25% loss will
invert any ranking built from single runs, which is exactly what happened here.

## What was wrong before

Three defects. The first two are fixed; the third is a method rule.

**The sustained-rate metric rewarded a bigger window for bursting harder.** A
windowed lane publishes a whole window at once when its sweep completes, so
every frame in that window shares one timestamp. `rates()` in
`tools/denoise-rate.py` began measuring at a fixed 20% of the clip. When that
point fell inside a burst, the rest of the burst entered the average at no time
cost, and the figure climbed with window size instead of with the lane's rate.
It now skips a full window and then anchors on the *end* of the burst that
straddles the skip point, so every frame counted arrived in a later sweep.
`tests/test_denoise_rate.py` asserts that two lanes at an identical real rate
report an identical sustained rate at window 500 and at 750; the old code
failed that by 30%. Note that this defect flattered 750, so it was concealing
the result rather than producing it.

**Every figure came from a single run.** With a 1-in-4 chance of a sporadic 25%
loss, one run per cell decides the ranking by chance. The table above runs each
window four times for this reason.

**Run order and machine state were never controlled.** They turned out not to
matter — 500 in second position is within 0.4% of 500 in first position, and
the cooldown made no difference to either window — but that was not known
before it was measured, and it cost nothing to control.

The previous table read 4.417 / 5.546 / 5.160 fps at windows 250 / 500 / 750 on
a 3357-frame clip and concluded that throughput peaks at 500 and falls at 750.
**That conclusion is withdrawn.** Those numbers came from the broken metric, at
n=1, on a clip that is no longer on the fleet, so they cannot be corrected —
only discarded. Window 250 has no current figure.

## Measured: gpu2, RTX 5070 Laptop

| window | result |
|---|---|
| 300 | overhead 76.6 s, sustained 4.78, end-to-end 4.34, wall 773.5 s |
| 750 | **0 of 8 attempts completed** |

At 750 seven attempts faulted inside the first window and one produced exactly
one window (749 frames) before faulting at the boundary.

These numbers predate all three fixes above, so the sustained figure is
inflated by an unknown amount and there was one run per cell. The one fact that
survives is the fault count, which is not a rate: this card faults on its own
(see `docs/encode-capacity.md`), and a longer sweep appears to make it more
likely, which is consistent with a fault of roughly constant probability per
unit of GPU work. That is inference, not measurement.

## For contrast: full-frame lanes pay none of this

| lane | overhead | sustained | end-to-end |
|---|---:|---:|---:|
| gpu1 4090 | 7.0 s | 18.20 | 17.37 |
| gpu4 GB10 | 12.1 s | 4.37 | 4.31 |

No window, no first-window wait, no redundant context. These lanes are
unwindowed, so the burst defect never applied to them and their figures stand.

## Where the roster's values came from

- **gpu2 300** was adopted because the lane "dies at the first window
  boundary" at 500. That was the driver fault, not the window: it faulted at
  300 too. The rationale is void.
- **gpu3 500** carries the comment "the same windowed settings as gpu2",
  which is not true — gpu2 is 300. On the measurements above it is also the
  slower of the two values tested, by about 3% end-to-end. It is defensible
  only as the low-memory choice: 27.8 GB against 39.3 GB.
- **2070s 750** has no recorded rationale, but gpu3's data no longer argues
  against it. encoder-host has 124 GB against gpu3's 47, so the memory cost that
  is the real argument for 500 does not apply there. The value has still never
  been measured on that card.

## TODO

1. **Test the 2070S at 500 against its configured 750.** Highest-value single
   test: the fleet's best remaining windowed lane, on a host with enough RAM
   that 750 costs it nothing. Run each window at least three times. Needs
   encoder-host free.
2. **Find the sporadic slow run.** Sample per-process GPU use during a sweep
   (`nvidia-smi pmon`) and log what else holds the device. If it is contention
   from the Windows side, the archive batch inherits it on every windowed lane
   and the roster should account for it, not the window size.
3. **Sample window 250, 300 and 1000 on gpu3.** Both low figures were
   discarded with the old table, and the measured curve is now monotonic across
   the only two points that exist. The ceiling has not been located, and 1000
   would cost 51.2 GB, which gpu3 cannot hold — so the ceiling may be
   unreachable on this host regardless.
4. **Never record a windowed lane from one run.** Three runs minimum, and
   report the spread, not a single number. This is the rule that would have
   prevented the withdrawn table.
5. **Check the interaction with `margin` and tile size.** margin 32 is used
   everywhere and the redundancy term is `2*margin` per sweep; `MIN_MARGIN` is
   16. Tile is `auto`, which chose 1112x992 here.
6. **Re-measure once any card's driver changes.** gpu2 **lost** about 30% of
   its throughput on a driver update alone — 5.78 fps on 610.74, 4.07 on
   610.88 — and bought fault-free runs with it. See `encode-capacity.md`. The
   5.78 was measured on a driver that faulted in 6 of 12 runs, so treat it as a
   number from an unreliable lane, not as a target to get back to. Both figures
   also predate the metric fix and are single runs.

Measure with `tools/denoise-rate.py`, which separates the fixed overhead from
the sustained rate. A single fps number blends the two and depends on clip
length, which is how the same lane came to be recorded at 4.40, 5.24 and 6.72.
