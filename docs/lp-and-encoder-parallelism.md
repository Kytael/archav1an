# `--lp` is a level, not a thread count

Status: measured on encoder-host 2026-08-13. No code changed yet; the recommendations at the
end are open.

## What the encoder actually accepts

```
--lp   Amount of parallelism to use. 0 means choose the level based on machine core count.
       Refer to Appendix A.1 of the user guide, default is 0 [0, 6]
--pin  Pin the execution to the first N cores. default is 0 [0, core count of the machine]
```

`--lp` takes a **level 0-6**. `--pin` is the parameter that takes a core count. Passing a
core count to `--lp` does not error. It clamps, with two warnings that scroll past in a
normal run:

```
Svt[warn]: Level of parallelism supports levels [0-6]. Setting maximum parallelism level.
Svt[warn]: Level of parallelism does not correspond to a target number of processors to use.
Svt[info]: Level of Parallelism: 6
```

`Docs/Parameters.md` describes what the levels do: they raise both the thread count and the
number of pictures in the pipeline, and "in CRF mode, levels 4 and higher will process extra
mini-gops in parallel as well, leading to higher speed, but much higher memory."

## Where the repo gets it wrong

The av1an paths are correct: every `run_linux_*.sh` and `av1an-batch-*.sh` passes `--lp 3`,
a real level, because av1an already runs many chunk workers.

Two places on the single-pass path pass a core count instead:

- `run_linux_dance_HQ_crf27.sh:13` — `LP=$(nproc)`, capped at 64, passed as `--lp "$LP"`.
  On a 32-thread box that is `--lp 32`, clamped to level 6.
- `tools/archive_batch/dispatch_cmd.py:42` — `"--lp", str(encode.threads_per_slot)`. With
  the default `threads_per_slot = 16` every slot sends `--lp 16`, clamped to level 6.

The consequence is that `roster.py`'s thread budget (`slots * threads_per_slot <= cores`,
and the `oversubscribe` ceiling above it) constrains nothing. Each slot runs at level 6
whatever the number says.

## Measurements

One encoder, 1200 frames of `MVI_1463`, dance-HQ parameters, `/opt/archav1an/bin/SvtAv1EncApp`
(5fish/SVT-AV1-PSY v2.3.0-C), 32-thread Ryzen AI MAX+ 395. Source decode ran at 29.9 fps, so
the encoder is the bottleneck at every level.

| `--lp` | fps | encoder RSS | threads | PPCS | max latency |
|---|---|---|---|---|---|
| 1 | 2.38 | 1045 MB | 17 | 74 | 42.3 s |
| 2 | 5.48 | 1316 MB | 63 | 74 | 16.2 s |
| 3 | 9.77 | 1621 MB | 77 | 74 | 9.5 s |
| 4 | 15.01 | 2069 MB | 87 | 107 | 8.6 s |
| 5 | 17.31 | 2672 MB | 91 | 140 | 10.0 s |
| 6 | 22.90 | 4795 MB | 95 | 305 | 21.0 s |
| 0 (auto) | 22.11 | 4827 MB | 95 | 305 | 21.8 s |
| 32 (clamps to 6) | 20.97 | 4850 MB | 95 | 305 | 23.1 s |

**Every level produced a bit-identical bitstream**: 59,145,867 bytes, md5
`29982f1799b73265bb60e21f656a3e4f`. `--lp` costs nothing in quality or size. It buys speed
with memory.

Auto picks level 6 on this box, so `--lp 0`, `--lp 16` and `--lp 32` are the same encoder
configuration. The PPCS column is where the memory goes: 74 buffers up to level 3, then 107,
140 and 305 as the extra mini-GOPs arrive.

### Concurrent slots

Same clip, N encoders at once, aggregate fps and summed encoder RSS.

| slots | `--lp` | aggregate fps | summed RSS |
|---|---|---|---|
| 2 | 6 | 26.27 | 9558 MB |
| 2 | 16 (clamps to 6) | 25.66 | 9617 MB |
| 2 | 5 | 24.19 | 5396 MB |
| 2 | 4 | 23.05 | 4154 MB |
| 2 | 3 | 16.10 | 3184 MB |
| 3 | 6 | 26.07 | 14372 MB |
| 3 | 5 | 25.54 | 7996 MB |
| 3 | 4 | **25.88** | **6176 MB** |
| 4 | 6 | 25.31 | 19077 MB |
| 4 | 4 | 25.91 | 8231 MB |
| 4 | 3 | 24.21 | 6367 MB |

The machine ceiling is about 26 fps and several configurations reach it. `--lp 16` and
`--lp 6` at 2 slots agree within run-to-run noise, which is the direct confirmation that 16
clamps to 6.

The memory-efficient point on the plateau is **3 slots at level 4**: 25.88 fps for 6.2 GB,
against 26.27 fps for 9.6 GB at 2 slots and level 6. Level 3 is below the floor — aggregate
throughput falls 38% because the encoders stop filling the machine.

Caveat: these runs had the box to themselves. In the archive batch the encoders share the
CPU with the denoise lanes, and a lower level has less slack to absorb that.

## Correction to the design document

the design notes, which are not part of this tree says the earlier sweep
compared thread counts:

| encoder-host, 1 encoder, `--lp 32` | 15.26 fps |
| encoder-host, 2 encoders, `--lp 16` | 18.12 fps |
| encoder-host, 3 encoders, `--lp 10` | 18.21 fps |

All three of 32, 16 and 10 clamp to level 6, so that was a slot-count sweep with the encoder
pinned at maximum parallelism throughout. The slot conclusion still stands. The
`threads_per_slot` framing does not.

The same section blames `pardenoise.py` for setting `lp = min(os.cpu_count(), 64)` and calls
it "2x oversubscription" that "explains most of the 37% contention loss". That mechanism is
wrong: 32 clamps to level 6, which is exactly what `--lp 16` and the encoder's own default
also give. Whatever caused the contention loss, it was not a thread request that the
recommended configuration did not also make.

## Recommendations

1. **Stop passing a core count.** In `run_linux_dance_HQ_crf27.sh`, delete the `LP=$(nproc)`
   block and pass `--lp 0`, or drop the flag and let the encoder default. Measured cost:
   22.11 fps against 22.90, inside noise.
2. **Pass a level in `dispatch_cmd.py`.** `--lp 4` with `slots = 3` matches the 3-denoiser
   roster, so no denoiser waits for an encoder slot, and costs 1.5% of the ceiling for 35%
   less memory. If you prefer the fewest moving parts, keep `slots = 2` and pass `--lp 6`.
3. **Fix or drop the thread budget in `roster.py`.** `slots * threads_per_slot <= cores` is
   arithmetic over a value the encoder ignores. Either rename the field to `lp_level`,
   validate `0 <= lp_level <= 6`, and drop the ceiling, or keep a core budget and spend it on
   `--pin`, which is the parameter that accepts one.
4. **Re-benchmark before trusting the archive-run estimate.** Any change here moves the
   encode rate the schedule is built on.
