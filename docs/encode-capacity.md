# Encode capacity and the cost of denoising

Measured 2026-08-15 with `tools/encode-capacity.py`. Every figure here is from
the same byte-identical clip (`Temp/_bench/bench.MOV`, 3357 frames, 1080p) and
the same encoder build, `5fish/SVT-AV1-PSY [main] v2.3.0-C`, resolved from
`/opt/archav1an/bin` on every host.

The question is not "how fast can this CPU encode". It is "how much encoding
can this host do without spending its GPU", because every machine here except
the desktop shares a thermal and power budget between the two.

## Hosts

| host | topology | CPU |
|---|---|---|
| encoder-host | 16c/32t | AMD RYZEN AI MAX+ 395 (Radeon 8060S iGPU + RTX 2070 SUPER) |
| gpu4 | 20c/20t | GB10: 10x Cortex-X925 + 10x Cortex-A725 |
| gpu1 | 8c/16t | AMD Ryzen 7 9800X3D (RTX 4090) |
| gpu2 | 16c/16t | Intel Core Ultra 7 265H (RTX 5070 Laptop) |
| gpu3 | 8c/16t | Intel i7-11800H (RTX 3070 Laptop) |

gpu4 reports as two 10-core sockets because the two clusters have distinct
part IDs (`0xd85`, `0xd87`). It is one 20-core chip.

## Encode scaling with concurrent streams

`--preset 4 --lp 6`, encode only, idle machine, aggregate fps over the window
the streams overlapped in.

| host | 1 | 2 | 3 | 5 | peak gain |
|---|---:|---:|---:|---:|---:|
| encoder-host 16c/32t | 22.86 | 32.79 | 34.04 | 34.72 | +51.9% |
| gpu4 20c/20t | 20.57 | 25.63 | 26.65 | 26.93 | +30.9% |
| gpu1 8c/16t | 17.87 | **20.18** | 20.02 | 20.09 | +12.9% |
| gpu2 16c/16t | 15.84 | 16.23 | 15.76 | 16.45 | +3.9% |
| gpu3 8c/16t | **8.94** | 8.59 | 7.86 | 8.18 | +0.0% |

The gain tracks the gap between logical threads and physical cores:

- encoder-host has 16 idle SMT siblings for a second process to fill, and gains most.
- gpu1 has the same SMT structure with half the cores, and gains half as much.
- gpu2 has no SMT. One encoder already owns every execution context, so a
  second finds nothing free. Sixteen physical cores, almost no gain.
- gpu3 has gpu1's topology but a thermal cap; contention beats headroom and
  every added stream costs it.
- gpu4 has no SMT either, so its +31% is not an SMT effect. The likely cause
  is heterogeneity: one process schedules poorly across ten fast and ten slow
  cores, and a second picks up what the first left idle.

Three streams is the practical ceiling. Going from three to five moves encoder-host
+0.68, gpu4 +0.28, gpu1 -0.07, gpu3 +0.32, and gpu2 +0.69 inside its own
noise band.

Cost on the shared-budget boxes: gpu4's package goes 79.2C at one stream to
91.0C at three and 95.2C at five, heat that comes out of its denoise lane.

### What the batch script defaults to

Since 2026-09-19 `run_linux_dance_HQ_crf27.sh` reads the peak column above as a
per-host table rather than one number, keyed on `uname -n`:

| host | lanes |
|---|---:|
| encoder-host | 3 |
| gpu4, gpu5 | 3 |
| gpu1 | 2 |
| gpu2 | 2 |
| gpu3 | 1 |
| anything else | 1 |

An unmeasured host gets one lane. That is slower than the old blanket three,
and it is the only default that cannot thrash a machine nobody has profiled.

`--lp` is not in the table, because it follows the lane count rather than the
host: more than one lane gives `--lp 4`, one lane gives `--lp 0` and the
encoder picks its own level. The derivation runs after the override is
applied, so `JOBS=1` restores `--lp 0` as well.

What that trades is speed for memory, and it is a deliberate choice rather
than a measured optimum. `docs/lp-and-encoder-parallelism.md` measured
26.27 fps at level 6 against 23.05 at level 4 on a saturated two-slot
encoder, and the encoders here are saturated: they read a local file, not a
denoise lane. Against that, level 5 measured 2672 MB an encoder on encoder-host
where level 6 took 4795 MB, and level 4 is narrower again. Nothing here
measures level 4 at three lanes.

`JOBS` and `LP` in the environment override both, so
`JOBS=1 LP=6 ./run_linux_dance_HQ_crf27.sh` still does what it always did.

## The three-phase measurement

`--preset 4 --lp 0`, one stream, with a cooldown between phases and the encoder
looping for the whole denoise window.

| host | enc solo | enc under load | dn solo | dn under load |
|---|---:|---:|---:|---:|
| encoder-host (8060S iGPU) | 24.71 / 23.88 | 19.24 / 19.26 | 4.32 / 4.08 | 2.69 / 2.64 |
| gpu4 | 19.11 | 13.76 | 4.51 | 4.33 |
| gpu1 | 18.36 | 13.75 | 17.74 | 14.24 |
| gpu3 | 6.80 | 6.04 | 4.52 | 3.87 |

encoder-host shows two independent runs. The encode side reproduced to within 0.1%
(19.24 against 19.26), which is what gives confidence in the rest.

## encoder-host in the production configuration

`slots = 5` at `lp_level = 6`, preset 4, with the iGPU lane loaded.

| phase | rate |
|---|---:|
| encode alone, 5 streams | 35.44 fps |
| denoise alone | 3.98 fps |
| **encode while denoising** | **28.06 fps** |
| **denoise while encoding** | **2.37 fps** |

The roster reasons from "about the 26 fps the machine can do". Measured
concurrently it is 28.06, so that figure was right to within 8%. The idle
34.72 is not the comparable number.

Five slots are worth having *under load* even though the idle curve saturates
at three: one stream gives 19.26 fps concurrent and five give 28.06, +46%. The
denoiser takes CPU from the encoders, so a loaded machine has more slack for
extra slots, not less. The loaded curve was measured at its endpoints only; the
2- and 3-slot loaded points are not known.

**The roster now asks for 6, one past the measured endpoint.** A slot is held
for a whole clip -- staging and publishing included, not just the encode -- so a
slot count below the number of enabled denoisers leaves a lane blocked with
nothing to do. Six is one per lane the fleet can field, which makes the ceiling
stop being a thing that starves a card. The denoise lanes never supply frames
fast enough to saturate even five encoders, so the sixth costs nothing until a
clip occupies it. Memory is the constraint to watch rather than threads: 4.8 GB
a slot at level 6 is about 29 GB for a full six, and the 2070S lane at window
750 measured 40.6 GB peak RSS on its own, out of the same 124 GB.

## The encode pool is spread, and gpu1 has a wide slot

Changed 2026-09-02. The pool used to be six slots stacked on encoder-host at
`lp_level = 4`. It is now six spread as 2 on encoder-host, 2 on gpu4 and 2 on
gpu5, plus a seventh entry `encoder-host-lp6` at `lp_level = 6`, one slot,
reserved for gpu1 by an allowlist.

The reason for the wide slot is that gpu1 is the one lane the "a wider encoder
cannot help a starved stream" argument does not cover. Measured over its last
40 clips it does 12.67 fps of work against the 14.24 fps the card produces
under load -- about 11% short -- and the scheduler's own notes record it held
to 10.0 fps by a consumer sharing cores with a second encode. Every other lane
runs between 2.3 and 5.1 fps and cannot saturate an lp-4 slot, so they keep it.

Three mechanics are worth knowing before touching this again.

**Placement is idle-encoders-first, then first free slot; inside each pass the
lane's own host wins, and file order breaks what that leaves.** The pool is
ordered gpu4, gpu5, encoder-host so the sparks take their second slot before
the batch host takes its second -- encoder-host also stages, publishes and runs the
2070S lane, so it is the worst host to double up on. The unavoidable cost is
that encoder-host gets its *first* slot third rather than first: no single order can
make a host both earliest in round one and latest in round two.

The own-host tiebreak was added 2026-09-02. A lane that encodes where it
denoised hands its y4m to a listener on its own address, which the kernel keeps
on loopback, so none of the denoised stream crosses the LAN -- and a denoised
1080p y4m is far larger than the source, at roughly 100 MB/s per lane measured
on the netstream logs. It applies to each spark's own lane and to the two
encoder-host lanes. gpu1 and gpu3 run no encoder, so nothing changes for them.

It is a tiebreak *inside* each pass and never above it. Promoted above the idle
check it would put gpu4's lane on gpu4's own second slot while gpu5 sat
empty, which is exactly the contention `3334a11` removed.

**The idle-first check is per encoder NAME, not per host.** Two entries may
share a host, so `encoder-host` can hold its two lp-4 slots while `encoder-host-lp6`
runs gpu1: three concurrent encodes on that host, not two. Budget it as
2 x 2.1 + 4.8 = 9.0 GB.

**A new encoder entry needs a batch restart, and a new port block needs a
firewall rule.** `enabled` and the allowlists are re-read before every clip,
but the semaphore pool and the slot/port table are built once at startup, and
`_try_take` silently skips an encoder with no semaphore -- so a mid-run edit
looks applied and does nothing. Separately, blocks 5300-5339 were already open
in ufw and 5340 was not: gpu1 failed a clip with `could not connect to
10.0.0.10:5340 after 11 attempts` until
`ufw allow from 10.0.0.0/24 to any port 5340:5349 proto tcp`. The validator
checks port-block *collisions*, never *reachability*.

**End to end costs 0.5-2.7%.** Over 2077 completed clips, staging is 9-14 s and
publish ~1.1 s per clip, near-constant, so the share scales inversely with clip
length: gpu1 pays 2.7% on 547 s of work, the iGPU lane 0.5% on 2419 s. The
dashboard's live fps tracks the work figure; the number recorded per clip in
`state.jsonl` is end to end.

## Denoise lanes, end to end

Full 3357-frame clip, real encoder attached, frame count verified.

| lane | measured | recorded | note |
|---|---:|---:|---|
| 2070s (encoder-host) | **6.60** | 4.40 | 508.5 s, engine build included |
| gpu2_5070 | **4.07** | 5.78 | 824.9 s, GPU 92.8% mean, idle 1.6% |
| igpu (encoder-host) | **4.08** | 5.19 | 823.4 s; 4.32/4.08/3.98 over three runs |

Every lane re-measured on the full clip came in **below** its short-run figure
except the 2070S, which came in 50% above. The 2070S is the one card here with
its own cooling: it is the exception that shows the rule, because the others
lose to heat soak on a clip long enough to reach it.

Treat the two single-run rows with caution. gpu3 was later measured eight
times on a longer clip and about one run in four came in 25% slow for reasons
that are not heat: a 420 s cooldown before the run changed nothing, and the
clean and slow runs shared a temperature and an SM clock. See
`docs/window-sizing.md`. One run cannot distinguish a lane's rate from that
event, and only the igpu row here has more than one.

gpu2 is a separate case, and the lane stays disabled. It was recorded at
5.78 fps on driver 610.74, where it also faulted in 6 of 12 runs. On 610.88 it
completed eight consecutive runs with no fault -- six denoise-only, one full
lane-bench crossing 11 window boundaries, and one nominally under
compute-sanitizer -- at 4.07 fps, with the card GPU-bound at 92.8% either way.
Read that sanitizer run as an ordinary run: the tool instrumented nothing then,
because the WDDM debugger interface was off (see below).

**616.56 did not fix it, measured on the morning of 2026-09-01 -- but see
the non-reproduction below before trusting this paragraph.** Measured on
R615 616.56, released 2026-08-26, the lane settings the roster carried then
(tile auto, window 300, margin 32, sigma 0.05, 1700 frames; it carries window
750 since 2026-09-03): **0 clean runs of
12** with an engine that 616.56 built itself, and 1 of 12 with the engine cache
left over from 610.88. The stale cache was a real confound -- TensorRT warns
`Using an engine plan file across different models of devices ... is likely to
cause errors or deadlock` -- so it was set aside and the twelve repeated; the
result got worse, not better. The Windows log carried 38 Xid 13 and 13 Xid 153
events across those twelve runs.

The fault is the same one, not a new one. `Graphics SM Warp Exception on
(GPC 1, TPC 4, SM 0): MMU NACK Errors`, with `Graphics Exception: ESR
0x519730=0xb230020 0x519734=0x4 0x519728=0x1c81fb60 0x51972c=0x1174` -- the
same SM and the same faulting address recorded in August. The application sees
`CUDA failure 700: an illegal memory access was encountered` inside the
TensorRT kernel node, then a Myelin destructor cascade.

The shape is worth knowing before diagnosing a future run. A run started from a
healthy card faults partway, and every run after it then dies at once with zero
frames and zero bytes. That second state is the same bug, not a different one:
its first error is the same `CUDA failure 700`, raised on the first inference
instead of partway in. **`wsl --shutdown` clears it.** A restarted VM goes back
to faulting partway rather than instantly, which makes the restart the reset
procedure for a wedged card. A clean run on 616.56 is not slower than 610.88
was: 1700 frames in 395.6 s, 5.9 fps sustained. The rate is not the problem;
the lane is simply unusable. (On 616.56. It ran 28 clips clean on 610.88 on
2026-09-03 — see below.)

**Do not read the frame count at which a run dies as a window boundary.** The
lane emits frames in bursts of one window, so a fault anywhere inside a sweep
reports as `k * window` frames delivered. Every fault count being a multiple of
the window is an artifact of that granularity. Window 1700 -- one window for
the whole clip, no flush before the end -- still faulted, one minute in, with
zero frames reported and 43 GB of host RAM free.

Neither tiling nor geometry is the trigger, and both were tested on 2026-09-01
rather than argued. At 720p, where both paths fit, untiled ran 0 of 3 (faults
at 402, 169, 1348 frames) while tiled at 1312x752 ran **3 of 3 clean** at
12.8 fps, alternating, with the Windows log carrying Xid clusters at exactly
the three untiled end times and none at the tiled ones. At 1080p a sweep of
square tiles 640, 768, 896 and 1024 at margin 32 produced one clean run in
five valid attempts, and the one clean geometry faulted on its next run.
Untiled at 1080p does not run at all: the engine build dies in
`virtualMemoryBuffer.cpp::resizePhysical` with `OutOfMemory`, because
TensorRT's `cuMemCreate` allocations do not oversubscribe to host RAM the way
ordinary WDDM allocations do. Across 22 valid 1080p runs on this driver, in
every geometry and window tried, three came back clean.

One finding here applies to the healthy lanes too: **a larger window is
faster.** Window 600 measured 9.77 fps against window 300's 5.9 on the same
clip and tile, because a sweep costs `window + 80` frames of GPU work and
halving the sweep count removes that overhead. That is worth testing on the
2070S and gpu3 lanes, which do not have this fault.

**Tested on gpu2 and gpu3, 2026-09-03, and it holds only up to the host RAM
ceiling.** Both boxes have 47 GB and the four buffers need `4*window + 128`
frames at 12.4 MB, so window 1000 wants 51.2 GB. Both ran it on swap — 46 GB
resident, 8-10 GB paged out — and lost about a quarter of the lane: gpu2 4.39
settled fps against 5.10 at window 750, gpu3 4.22 against ~5.50. Larger is
faster until the buffers stop fitting, and then it is sharply slower. Size the
window from the four-buffer model first. Full numbers in
`docs/window-sizing.md`.

### 610.88, 2026-09-03: 28 clips clean

After the rollback to 610.88 and a reboot, gpu2 and gpu3 ran **28 clips and
about 285,000 frames with no CUDA fault, no Xid, no segfault and no stall**,
across window 1000 and window 750. gpu2 alone did 14 clips and ~140,000
frames, including one 39,052-frame clip in a single run.

That is by far the longest clean stretch recorded for this card, and it is the
first evidence that window 750 is survivable on it — the 0-of-8 above was on
the older driver. **It is not proof the fault is gone.** This document already
records 0 of 12 one morning and 22 of 22 that evening, and two earlier
hypotheses died on exactly this kind of streak: 616.56 was blamed and then
reproduced on 610.88, and an fp32 engine ran clean while two fp16 controls in
the same session ran clean too. A rare fault and a fixed fault look identical
over one day. What changed materially is the sample size, not the certainty.

`compute-sanitizer --tool memcheck` finds nothing. 400 frames at window 300
completed clean, `ERROR SUMMARY: 0 errors`, in 3081 s -- a 38x slowdown, which
is what proves it really instrumented this time. That clears our kernels of a
visible out-of-bounds access. It does not clear the driver: memcheck serialises
the work, and a 400-frame clip crosses one window boundary where the full clip
crosses five.

**It is NOT NVIDIA's documented TMA descriptor bug.** That was the leading
hypothesis for three weeks and it is now ruled out on 610.88 by two direct
measurements. A `gdb` run with breakpoints on `cuTensorMapEncodeTiled`,
`cuTensorMapEncodeIm2col`, `cuTensorMapEncodeIm2colWide` and
`cuTensorMapReplaceAddress` -- all four resolved -- took **zero hits across 400
frames**. TensorRT's own tactic tables say why: every kernel it evaluates for
this graph is Ampere-generation, 780 `sm80_xmma_fprop_implicit_gemm_*` names
against zero sm90, sm100 or sm120 kernels and zero mentions of TMA. The
selected tactics are the likes of
`sm80_xmma_fprop_implicit_gemm_indexed_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x128x32_stage3_warpsize4x2x1_g1_tensor16x8x16`.
TMA is Hopper and later; sm80 CASK kernels cannot use it. Reproduce with
`~/drivertest/rung1_tma.sh` and `~/drivertest/rung2b_probe.py` on gpu2; note
the probe patches `SessionOptions`, because `bsvd_vs_filter.py:101` pins
`log_severity_level = 3` and drops TensorRT's verbose build log.

What is left is an illegal memory access inside Ampere-class convolution
kernels on sm_120, at a constant SM and address. Do not cite the TMA known
issue in a bug report.

**The fault did not reproduce at all on the evening of 2026-09-01, on the same
616.56.** After the user uninstalled 616.56, ran 610.88, then reinstalled
616.56 with a reboot, the lane ran **22 consecutive clean runs, 37,400 frames,
zero faults and zero Xid events** -- against 0 clean of 12 and 1 of 12 that
same morning, on the same driver, clip, geometry and lane settings. Five arms,
each designed to isolate one variable, and every one came back clean:

| arm | what it changed | result |
|---|---|---|
| full 1-6 | 610.88-built engine plan on 616.56 | 6/6 clean |
| swap 1-4 | the exact 05:37 engine plan that was live during the cascade | 4/4 clean |
| noinstr 1-3 | coredump env vars removed | 3/3 clean |
| regoff 1-4 | debugger registry values removed -- the morning's exact config | 4/4 clean |
| fresh 1-5 | cache deleted, engine rebuilt in-process under 616.56 | 5/5 clean |

So it is not the driver version alone, not the engine plan, not the tiling, and
not instrumentation masking it. The only uncontrolled difference left is the
driver *installation*: the morning's 616.56 was installed over 610.74, the
evening's was a fresh install over 610.88 with a reboot between. That is a
hypothesis, not a finding -- it was not tested, because testing it means
reinstalling over the top again.

Two secondary results from that campaign. Three builds of the same ONNX at the
same geometry produced **three different engines** (md5 `a620d52e`, `fefbd1bb`,
`1d59ebf9`; 20,276,068 / 20,131,756 / 20,219,364 bytes), because TensorRT
tactic selection is timing-based -- so which kernels land in a plan is partly a
lottery per build, and "the same engine" is never the same file. And a run
truncated with `vspipe -e N` aborts at teardown with `rc=134` and a torch
`SetDevice` error reading `CUDA error: no error`; full runs exit 0. That abort
is a pipe-truncation artifact and not a fault.

**Read the JSON `rc` and frame count, never the summary line's fps.**
`denoise-rate.py`'s wrapper exit code is 0 even on failure, and a faulted run
still reports a `sustained_fps` for the frames it managed. Run 1 of the 05:36
campaign shows `"sustained_fps": 2.973` next to `"rc": 1` and `"frames": 1197`
of 1700. It looks clean in the summary and is not.

Rung 3 -- naming the faulting kernel under `cuda-gdb` -- could not run, because
nothing faulted. Two mechanics are worth keeping for whoever retries it.
`cuda-gdb` stops on a benign `SIGSEGV` raised inside `dlopen()` from `cuInit()`
during ORT's `cudaSetDevice`, before any kernel exists; pass it with
`handle SIGSEGV nostop noprint pass` (`~/drivertest/rung3_cudagdb2.sh`).
Better, skip the debugger: `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` plus
`CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1` catches the exception with no debugger
attached, for about 9% throughput cost (5.05 fps against 5.53), and the dump is
read offline with `cuda-gdb -c`. Scripts for all five arms are in
`~/drivertest/rung3_*.sh` on gpu2.

Given 0 of 12 in the morning and 22 of 22 at night, a rare bug and a fixed bug
look identical here. Keep gpu2 on 610.88 rather than call 616.56 fixed.

compute-sanitizer needed no Nsight install and no WSL restart -- only two
registry values on the Windows side, which enable the WDDM debugger interface:
`HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm /v EnableDebugInterface` and
`HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger /v EnableInterface`, both
REG_DWORD 1. Without them the tool prints `Failed to initialize WDDM debugger
interface`, instruments nothing, and runs at full speed -- which reads as a
clean sanitizer pass and is not one.

## Thermal behaviour

- **gpu1** needs no cooldown at any phase and peaks at 70C. It is the only
  host whose CPU and GPU do not compete.
- **gpu3** holds 85-87C for an entire denoise at a mean 1292 MHz against a
  1935 MHz peak -- throttling about a third below its own boost, sustained.
  That is where its missing denoise throughput goes, not to the encoder.
  **This row assumes the factory 230 W adapter.** On Thunderbolt 4 power the
  same host is capped at 54 W, runs at 641-667 MHz, and sits *cooler* at
  81-83C because it cannot generate the heat -- for roughly half the denoise
  rate. Check the power ceiling before reading a slow gpu3 as a thermal
  problem: `nvidia-smi -q -d POWER`, not `--query-gpu=power.limit`, which
  returns `[N/A]` on this card. See `docs/window-sizing.md`.
- **gpu4** takes the heat on the CPU side: 78.3C denoising alone, 92.3C mean
  and 96.9C peak once an encoder joins, while its GPU clock barely moves.
- **encoder-host** reaches its ceiling inside the run itself. The package sits at
  84-86C throughout a denoise regardless of the starting temperature, and the
  iGPU clocks 1943-2151 MHz against a 2609-2640 MHz peak. Pre-cooling cannot
  change a sustained rate on this host; it only ever mattered for a phase short
  enough to finish before the machine heats up.

## Corrections to earlier figures

- **Preset.** The pipeline runs `--speed 4`. Anything measured at preset 8 is
  void: it made gpu1 read 51.7 fps against encoder-host's documented 26, a machine
  with half the cores appearing twice as fast.
- **encoder-host's iGPU lane is not 5.19 fps.** Three full-clip runs measured 4.32,
  4.08 and 3.98. 5.19 is a short-run figure taken before the die heat-soaks.
  For a 2.5 TB archive the sustained rate is what counts, so the roster
  overstates this lane by about 20%, and that feeds the pool total.
- **`--lp` is not neutral.** At one stream gpu3 goes 6.80 fps at `--lp 0` to
  8.94 at `--lp 6`, +31%; gpu1 goes the other way, 18.36 to 17.87. The auto
  level leaves a third of gpu3's encoder unused. Production hardcodes 6.
- **Concurrent denoise figures measured before 2026-08-15 are void.** A single
  600-frame encode against a 3357-frame denoise left the denoiser unloaded for
  88% of the window on gpu3 and 94% on gpu4, so the "under load" number
  described a denoiser that was mostly alone.

## Measuring this again

Two traps are built into `tools/encode-capacity.py` because both produced wrong
numbers first:

- The cooldown gates on the hottest sensor available, not the GPU alone. On
  encoder-host the iGPU edge read 44C directly after an encode that left the
  package at 83C, so a GPU-only gate passed instantly.
- On encoder-host `nvidia-smi` describes the 2070S, which is idle while the 8060S
  works, and there are no `/sys/class/thermal` zones at all. `amd_sample()`
  reads the amdgpu hwmon and `gpu_busy_percent`; `cpu_temp()` falls back to
  `k10temp`.

Write results somewhere durable. The first set of raw JSON for this document
was lost with a temporary directory.
