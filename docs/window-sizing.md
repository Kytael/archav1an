# Window size and denoise throughput

Status: measured 2026-08-16 — 8 runs on gpu3, 4 on encoder-host; extended
2026-09-03 with gpu2's own two-point fit and 28 clips at window 750 across
gpu2 and gpu3. **The window curve this document used to report does not
exist. Window 750 does not regress; it is the faster value on every card
tested, and it is the largest that fits in 47 GB. Above the RAM ceiling the
curve bends — see "The 47 GB ceiling, and the cliff at it".**

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
| 500 | yes | first | 118.30 s | 1377.53 s | 4.883 | 4.951 | — |
| 500 | yes | second | 109.91 s | 1373.06 s | 4.899 | 4.934 | 1290 s |
| 500 | yes | first | 110.01 s | 1378.34 s | 4.880 | 4.912 | 1293 s |
| 500 | no | first | 108.26 s | 1707.38 s | 3.939 | 4.835 | 1597 s |
| 750 | no | second | 161.79 s | 1793.68 s | 3.750 | 3.545 | — |
| 750 | yes | first | **161.10 s** | **1326.90 s** | **5.069** | **5.165** | 1226 s |
| 750 | yes | second | 159.46 s | 1337.05 s | 5.030 | 5.111 | 1236 s |
| 750 | no | second | 162.04 s | 1359.81 s | 4.946 | 5.041 | 1256 s |

Overhead, wall and end-to-end are as measured. The sustained column is rescaled
by 5226/5380: these runs were taken with an interim version of the anchor fix
that started counting at frame 1346 instead of at the sweep boundary at 1500.
Both windows anchor on the same sweep at this clip length, so the correction is
one constant factor applied to every row, and it is exact to under 0.1% — the
two candidate anchors sit inside one socket flush, under a second apart in a
span above a thousand. It changes no ranking and no gap.

**Window 750 beats window 500 in every clean run**, on sustained rate and on
end-to-end rate both, despite paying 50 s more first-window wait. Six of the
eight runs finish in 1327-1378 s. The two that do not are discussed below, and
there is one at each window size, so dropping them does not favour either:

| window | clean runs | sustained | end-to-end |
|---:|---:|---|---|
| 500 | 3 | 4.912 - 4.951 | 4.880 - 4.899 |
| 750 | 3 | 5.041 - 5.165 | 4.946 - 5.069 |

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

## Measured: gpu3 on Thunderbolt 4 power — 54 W ceiling, 641-667 MHz mean

Same clip, same instrument, same tile and margin as the section above, measured
2026-08-17. The one difference is how the laptop was powered: a Thunderbolt 4
connection rather than the factory 230 W adapter. That is not a setting anyone
chose in software, and it is invisible to every query that usually answers this
question — the machine reports AC power, a 100% battery and the Balanced Windows
plan, exactly as it does on the adapter.

**What it actually does is cap the GPU at 54 W.** `nvidia-smi --query-gpu=power.limit`
returns `[N/A]` on this card, which is easy to read as "not knowable"; it is not.
`nvidia-smi -q -d POWER` reports it:

```
Current Power Limit : 54.00 W      <- Thunderbolt 4
Default Power Limit : 80.00 W      <- this card's base TGP
Max Power Limit     : 105.00 W     <- base + Dynamic Boost
```

`SW Power Cap` sits Active for the whole run. One run at each window:

| window | overhead | wall | end-to-end | sustained | mean clock | mean draw | temp | util |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 750 | 275.38 s | 3250.90 s | 2.069 | 2.743 | 667 MHz | 53.0 W | 60-81 °C | 94% |
| 500 | 192.67 s | 2614.75 s | **2.572** | **2.823** | 641 MHz | 53.5 W | 63-83 °C | 96% |

Against the full-power rows above, at the same window, cooldown and position:

| window | end-to-end | of baseline | sustained | of baseline | mean clock | of baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 750 | 2.069 vs 5.069 | 40.8% | 2.743 vs 5.165 | 53.1% | 667 vs 1292 MHz | 51.6% |
| 500 | 2.572 vs 4.899 | 52.5% | 2.823 vs 4.934 | 57.2% | 641 vs 1292 MHz | 49.6% |

**The sustained rate tracks the clock, and the clock is what 54 W buys.** 53.1%
and 57.2% of the baseline rate at 51.6% and 49.6% of the baseline clock. This is
a power ceiling and nothing else: utilisation held at 94-96% throughout, and the
card ran *cooler* than it does on the adapter — 81-83 °C against the 85-87 °C
that `docs/encode-capacity.md` records for a full-power denoise — because at 54 W
it cannot generate the heat. `HW Thermal Slowdown` never activated in either run.

Neither run has a slow patch. Sampled at 0.5 Hz and split into deciles, the clock
falls gently from about 720 MHz to about 640 MHz as the card warms, with
utilisation flat at 94-97% and no stall anywhere. The sporadic slow run this host
is known for did not appear in either.

**The window ordering reverses here, and one run each is not enough to claim it.**
At full power 750 beat 500 on both rates. On Thunderbolt power 500 is ahead:
clearly on end-to-end, by 24%, and slightly on sustained, by 2.9%. The end-to-end
half of that is not in doubt and needs no repeat — a larger window costs more
first-window wait, and at half the clock that fixed cost is paid at half the rate,
so 750 takes 275 s against 193 s to reach its first frame. The sustained half is
2.9% on single runs of a host whose clean runs spread wider than that, so treat it
as unresolved rather than as a reversal. What is safe to say is that **750's
advantage does not survive the loss of power, and its overhead penalty grows.**

For a roster entry this argues for window 500 on a Thunderbolt-powered gpu3, on
the end-to-end figure alone, which is the one that decides how long a clip takes.

These runs need no rescaling. The `sustained` column of the full-power section is
rescaled by 5226/5380 because those runs predate the final anchor fix; these were
taken on `de48247`, which carries `bfbe330`. Do not apply that factor here.

## Measured: encoder-host, RTX 2070 SUPER, tile auto, margin 32

Same clip, same instrument, 2026-08-16. Windows alternated so neither took a
favoured position, each run gated to start at 40 °C. **Two runs per window, not
the three this document asks for below** — the sweep was stopped early to give
the machine back.

| window | run | overhead | wall | end-to-end | sustained | peak RSS |
|---:|---|---:|---:|---:|---:|---:|
| 500 | 1 | 99.74 s | 1289.33 s | 5.217 | 5.769 | 29.1 GB |
| 500 | 2 | 99.79 s | 1282.41 s | 5.245 | 5.803 | 29.1 GB |
| 750 | 1 | 141.79 s | **1227.24 s** | **5.481** | **6.408** | 40.6 GB |
| 750 | 2 | 142.11 s | 1227.62 s | 5.479 | 6.410 | 40.5 GB |

**750 wins here too, by more than on gpu3**: 4.7% end-to-end and 10.6%
sustained. The card's configured value is 750, and this is the first
measurement that supports it.

The reproducibility is the other result. The two 750 runs differ by 0.03% in
wall time and the two 500 runs by 0.5%. **gpu3's sporadic slow run did not
appear in four runs here.** That is not proof it cannot, but it separates the
two hosts: the 2070S is a desktop card with its own cooling, in a machine that
owns its GPU, where gpu3 is a laptop whose card is shared with a Windows
desktop the WSL2 distro does not control. It is consistent with the contention
hypothesis in the TODO below.

Window 750 costs 40.6 GB against 29.1 GB here. On 124 GB neither matters, which
is why this host should take the faster value and gpu3 — where 750 leaves
7.1 GB spare — is the one with a real decision to make.

One anomaly, recorded because it is unexplained rather than because it mattered:
the first 500 run took 1.77 M pages of direct reclaim between minutes 5 and 11
while `MemAvailable` sat at 88 GB. Reclaiming with 88 GB free points at a
zone-constrained allocation, not at memory pressure — plausibly DMA32 or the
pinned memory the iGPU shares with the system. `allocstall_normal` stayed at 0,
so the stall counter for whichever zone it was went unsampled. The run was not
slow and the other three showed nothing.

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
windowed lane publishes a whole window when its sweep completes, and that
window then flushes down the socket far faster than it was computed. `rates()`
in `tools/denoise-rate.py` began measuring at a fixed 20% of the clip. When
that point fell inside a flush, the rest of the window entered the average at
flush speed rather than at sweep speed — up to 749 frames arriving in about a
second instead of the two and a half minutes they took to produce. The figure
therefore climbed with window size instead of with the lane's rate.

It now rounds the skip point **up to a multiple of the window**, which is where
the sweep boundaries are, so the measurement always spans whole sweeps. The
obvious alternative — find the boundary by looking for frames that share a
timestamp — does not work: the sink stamps each frame as its bytes land, so a
window's frames are milliseconds apart, and a first attempt at this fix that
compared timestamps moved the anchor by exactly one frame and did nothing.

`tests/test_denoise_rate.py` asserts that two lanes at an identical real rate
report an identical sustained rate at window 500 and at 750; the old code read
them as 1.20 and 1.27. Note that this defect flattered 750, so it was
concealing the result rather than producing it.

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

| window | result | when |
|---|---|---|
| 300 | overhead 76.6 s, sustained 4.78, end-to-end 4.34, wall 773.5 s | 2026-08-16, **sustained superseded** |
| 750 | **0 of 8 attempts completed** | 2026-08-16 |
| 300 | overhead 76.67 / 76.09 s, sustained 5.826 / 5.809, end-to-end 4.246 / 4.238 | 2026-09-03, n=2 |
| 1000 | overhead 214.62 s, sustained ~4.37, peak RSS 46.0 GB, **8-10 GB swapped** | 2026-09-03, live |
| 750 | overhead 163.1 s (n=12, 161.1-164.1), peak RSS 39.2 GB, 0 swap | 2026-09-03, live |

At 750 on 2026-08-16, seven attempts faulted inside the first window and one
produced exactly one window (749 frames) before faulting at the boundary. That
was the driver fault, not the window: the same setting ran 14 clips and about
140,000 frames clean on 2026-09-03 with no CUDA fault, Xid or segfault.

**The 4.78 sustained is withdrawn; 5.82 replaces it.** The 2026-08-16 row
predates all three fixes above, and it is the SUSTAINED column the broken
metric moved. Overhead came through untouched, which is why 76.6 s then and
76.67 s now agree to under a tenth of a second on a different clip three weeks
apart: overhead is time-to-first-frame and never went through the ramp-skipping
logic that was wrong.

### gpu2's own intercept

Two window sizes measured on this card give it the decomposition gpu3 already
had. vspipe prints the figure itself — `Script evaluation done in N seconds` is
the same quantity `denoise-rate.py` reports as `overhead_s`, and the two agree
to 0.4 s — so no benchmark is needed to collect it, only a running lane.

| window | overhead |
|---:|---:|
| 300 | 75.68 s |
| 1000 | 214.62 s |

That is **0.1985 s per window frame (5.04 fps) and a 16.1 s intercept.** The
intercept is VapourSynth startup plus the TensorRT engine load; ffms2 indexing
is not a meaningful part of it, measured at under a second. So the first window
is 79% of startup at window 300 and 92% at window 1000, and gpu2 sits about
8 s above gpu3's 8 s intercept.

The fit predicted 165 s at window 750. Twelve clips measured **163.1 s**, range
161.1-164.1. gpu3 runs consistently ~6 s higher than gpu2 at the same window,
158-175 s over eleven clips, which is a stable per-host difference rather than
noise.

### The 47 GB ceiling, and the cliff at it

Both gpu2 and gpu3 have 47 GB. The four-buffer model puts window 1000 at
51.2 GB, which does not fit, and both lanes ran it anyway on swap: 46.0 and
45.8 GB resident with 8-10 GB paged out. It cost about a quarter of the lane.

| | window 1000 | window 750 |
|---|---:|---:|
| gpu2, settled vspipe fps | 4.39 | 5.10 |
| gpu3, settled vspipe fps | 4.22 | ~5.50 |
| whole-clip fps | 4.15-4.27 | 4.49-4.89 |
| peak RSS | 45.8-46.0 GB | 39.1-39.3 GB |
| swap | 8-10 GB | 0 |

This is backwards from the model, which predicts window 1000 should be ~17%
FASTER — redundant context falls from 380/300 to 1080/1000. It is 22% slower
instead, because the swapping costs more than the larger window gains.

gpu3's overhead curve shows the same cliff independently: **0.204 s** per
window frame on the 500→750 leg, **0.230 s** on 750→1000. Linear up to the
ceiling, bending above it.

So 750 is the largest window that is still on the linear part of the curve for
a 47 GB box: `4*window + 128` frames at 12.4 MB is 38.8 GB, leaving ~8 GB. Sizing
a window is therefore a memory question first and a throughput question second,
and the memory answer is the four-buffer model, not a guess.

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
  300 too. The rationale is void. Superseded 2026-09-03: gpu2 and gpu3 both
  run **750**, chosen from the four-buffer model rather than from a fault —
  38.8 GB of 47, the largest window still below the ceiling. Measured 163 s
  overhead and ~5.1-5.5 fps settled, against 4.4 at window 1000, which does not
  fit and swaps.
- **gpu3 500** carries the comment "the same windowed settings as gpu2",
  which is not true — gpu2 was 300. Both are 750 since 2026-09-03, so the
  comment is true now for a reason it was not then. On the measurements above it is also the
  slower of the two values tested, by about 3% end-to-end. It is defensible
  only as the low-memory choice: 27.8 GB against 39.3 GB.
- **2070s 750** had no recorded rationale, and now has one: measured on the
  card, it is 4.7% faster end-to-end and 10.6% faster sustained than 500, and
  its 40.6 GB is free on a 124 GB host. Keep it. The value was right by
  accident for two years, which is not the same as being justified.

## TODO

1. **Add the third run on the 2070S.** Two per window is what the sweep got
   before it was stopped, and two cannot show a 1-in-4 event any better than
   one can. The ranking is not in doubt — the gap is 4.7% and the spread is
   0.03% — but the "no slow run on this host" claim rests on four runs.
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
   also predate the metric fix and are single runs. Re-run on 616.56 on
   2026-09-01: 0 clean runs of 12 that morning, the same Xid 13 on the same SM,
   but 22 clean runs of 22 that evening on the same driver after a reinstall.
   A clean run there is 5.9 fps sustained, so the rate is not what makes the
   lane unusable. Do not read a driver bump as a fix, and do not read one clean
   streak as one either; measure it. Details in `docs/encode-capacity.md`.
7. **Convert late, as an investigation option only.** The windowed feed sends
   RGBH over the pipe, so a 1000-frame window costs 12.4 MB a frame and the
   transport is a real share of the head: 12.31 s on gpu2, and on gpu3 a flat
   0.55 GB/s where doubling the bytes doubled the time. Moving the YUV to RGB
   conversion out of the vspipe subprocess and onto the receiving side would cut
   those bytes and, holding the source buffers as YUV, take a window-1000 lane
   from 38.9 GB to 19.0 GB -- halved, not quartered, because only the SOURCE
   buffers can be YUV; the output buffer is RGB fp16 by construction.

   **Measured 2026-09-02, and as designed it is a net loss. Do not build it
   without a new plan.** zimg on arrival is exact -- 64/64 and 32/32
   bit-identical -- but only if seven frame props travel with the frame;
   dropping them silently costs max abs 0.0977. A torch fallback is not exact
   at all: max 0.539, mean 0.0304. The blocker is the injection, not zimg. A
   `ModifyFrame` selector is a Python callback copying 3.11 MB a frame under the
   GIL, and it never approaches vspipe's 0.5 ms/frame:

   | path | ms/frame | per window-1000 sweep | share of a 181 s sweep |
   |---|---:|---:|---:|
   | sequential `get_frame` | 6.68 | 14.4 s | 8.0% |
   | `frames(prefetch=4)` | 9.28 | 20.0 s | 11.1% |
   | `frames(prefetch=8)` | 11.39 | 24.6 s | 13.6% |
   | `frames(prefetch=16)` | 5.86 | 12.7 s | 7.0% |

   Tiles are the outer loop because the sweep resets streamer state per tile, so
   a window-1000 sweep needs 2160 conversions, not 1064. Best case therefore
   costs 7-11% of the sweep to save the 2-2.8% that the memory pressure is
   actually worth (measured: 2.8% on gpu2, 1.0% on gpu3, at 39.5 GB against
   12.1 GB). Converting only the pipe, leaving buffers RGB, is likewise marginal:
   about 6.4 s of conversion against roughly 8 s of transport saved on gpu2.

   The current design is the right one and now has a reason: converting inside
   the subprocess keeps zimg fed natively by ffms2 in C, and the wider pipe is
   the price of never touching a frame from Python. The only route left is to
   call zimg's C API directly and bypass the GIL-bound frame copy. That is a
   project, not an afternoon, and it should be costed before it is started.

   **Still not worth it — but know which side of the ceiling that was measured
   on.** The 2-2.8% saving above compares 39.5 GB against 12.1 GB, and both are
   under the 47 GB the two windowed hosts have, so neither run was swapping. In
   that regime memory pressure is nearly free and halving it buys nothing. It is
   not linear: at window 1000 the same boxes sit at 46 GB with 8-10 GB paged out
   and lose about 24% of sustained throughput, which is more than the 7-11% this
   would cost. See "The 47 GB ceiling, and the cliff at it" above.

   That does not revive the idea, because there is a free alternative on this
   fleet: window 750 fits in 39 GB and stays below the cliff, so nothing needs
   buying. It would only pay on a host whose RAM ceiling arrives at a window
   size somebody actually wants — less memory than these two, or a model with a
   bigger per-frame footprint — where no free fix exists. Anyone reaching for
   this should check that condition holds first, because on the fleet as it
   stands it does not.

   Two side results worth keeping. NVDEC is a dead end here: 0.90x on gpu2, and
   on gpu3 the apparent 1.06x is BestSource against ffms2, not hardware --
   `bs_sw` 26.00 s against `bs_cuda` 26.07 s, with every decoder byte-identical
   at `4e367cdc27ba3d04`. And a per-run ONNX session buys nothing: session
   creation is 1.39 s to build plus 0.19 s to load the engine, not the ~213 s
   once assumed.

The dashboard's live fps had the same burst artefact until 2026-09-02, in a
second form. Its slope was taken between the raw ends of the sample series, and
a 2.5-sweep smoothing window holds two OR three whole bursts depending only on
where the series happened to start. At window 1000 and a true 4.20 fps that is
3.33 fps or 5.00 fps -- a 50% spread with nothing about the lane changing, and
the reason a windowed lane's live figure could sit far above the end-to-end
column beside it. The slope is now snapped to burst edges, which holds a whole
number of sweeps by construction, so the phase cancels rather than being
averaged down: the same simulation reads 4.17 fps at every phase. Smoothing
harder would only have shrunk the swing as 1/N and cost lag.

Note the lane already knows its own rate and the dashboard cannot see it:
`vspipe -p` writes `Frame: 2999/12937 (5.65 fps)` into `<stem>_vspipe.log` on
the DENOISE host, while the only log a remote lane leaves on encoder-host is
netstream's, which carries frame numbers and no rate at all.

Measure with `tools/denoise-rate.py`, which separates the fixed overhead from
the sustained rate. A single fps number blends the two and depends on clip
length, which is how the same lane came to be recorded at 4.40, 5.24 and 6.72.
