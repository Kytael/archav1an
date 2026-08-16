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

`slots = 5` at `lp_level = 6`, preset 4, with the iGPU lane loaded -- what the
roster actually asks for.

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

gpu2 is a separate case. It was recorded at 5.78 fps on driver 610.74, where
it also faulted in 6 of 12 runs. On 610.88 it has completed eight consecutive
runs with no fault -- six denoise-only, one full lane-bench crossing 11 window
boundaries, and one more under compute-sanitizer -- at 4.07 fps, with the card
GPU-bound at 92.8% either way. The lane traded about 30% of its throughput for
stability. Whether that trade is necessary is not known; R580 is the newest
driver branch whose notes do not carry NVIDIA's open TMA descriptor bug, and
testing it would show whether stability and speed can be had together.

## Thermal behaviour

- **gpu1** needs no cooldown at any phase and peaks at 70C. It is the only
  host whose CPU and GPU do not compete.
- **gpu3** holds 85-87C for an entire denoise at a mean 1292 MHz against a
  1935 MHz peak -- throttling about a third below its own boost, sustained.
  That is where its missing denoise throughput goes, not to the encoder.
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
