# `--lp` is a level, not a thread count

Status: measured on encoder-host 2026-08-13, auto-level behaviour confirmed on gpu1, gpu2 and
gpu3 the same day. Applied — see "What changed" at the end.

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

Two places on the single-pass path passed a core count instead. Both are fixed; they are
recorded here because the mistake is easy to repeat:

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

## What `--lp 0` picks, and why it is not a synonym for 6

`load_default_buffer_configuration_settings` in `Source/Lib/Globals/enc_handle.c:306-331`
maps the logical processor count straight onto a level:

| logical cores | auto level |
|---|---|
| 1 | 1 |
| 2 | 2 |
| 3 - 5 | 3 |
| 6 - 11 | 4 |
| 12 - 23 | 5 |
| 24 or more | 6 |

Two details matter. The count is `get_num_processors()`, so it is logical threads, not
physical cores. And `--pin N` is applied first: `pin_threads` replaces the core count when it
is smaller, so `--pin` steers the auto level as well as the affinity.

Measured across the fleet 2026-08-13 by reading the encoder's own banner:

| host | logical cores | encoder | `--lp 0` gives |
|---|---|---|---|
| encoder-host | 32 | PSY v2.3.0-C | **6** |
| gpu1 | 16 | mainline v4.1.0 | **5** |
| gpu2 | 16 | PSY v2.3.0-C | **5** |
| gpu3 | 16 | mainline v4.1.0 | **5** |

Both encoder builds agree, so the mapping is not PSY-specific. The practical consequence:
every 16-thread host in the fleet drops from level 6 to level 5 when a hardcoded core count
is replaced by `--lp 0`. That is the encoder's own judgement and it costs memory rather than
quality — on encoder-host level 5 measured 2672 MB against 4795 MB — but it is a real change in
encoder configuration, not a no-op.

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

## What changed

Applied 2026-08-13.

1. **`run_linux_dance_HQ_crf27.sh` passes `--lp 0`.** The `LP=$(nproc)` block is gone. On
   encoder-host this is the same level 6 as before, at 22.11 fps against 22.90, inside noise. On
   the 16-thread hosts it is level 5 rather than the clamped 6 — see the auto table above.
2. **`roster.py` field `threads_per_slot` is now `lp_level`, default 4.** `dispatch_cmd.py`
   passes it straight to `--lp`, so a roster now names a level instead of a thread count.
   The default was 6 while the pool ran 2 slots, on the reading that at 2 slots level 6
   measured 26.27 fps against 23.05 at level 4, and only 3 or more slots closed that gap.

   **That reading was retired on 2026-08-16, and the table above is not what decides it.**
   Every row here is a saturated encoder, fed as fast as it will take frames. In the archive
   batch it never is: an encoder consumes frames at the rate of the denoiser feeding it, and
   the fastest lane in the fleet is gpu1's 4090 at 14.24 fps under load, with every other
   lane between 2.4 and 6.6 (`docs/encode-capacity.md:62-82`). A wider encoder cannot speed up
   a starved stream. The width still costs memory — 4.8 GB a slot against 2.1 GB — which
   across the 6-slot pool is 29 GB against 13 GB.

   One gap in the evidence, left open deliberately: there is no single-slot level-4 row above,
   so whether one level-4 stream keeps up with 14.24 fps while other slots are busy is
   untested. If the 4090 lane looks throttled on the first real run, raise the level. It is
   changeable per clip now, so that costs nothing but the decision.
3. **The thread budget is gone, and `oversubscribe` with it.** `slots * threads_per_slot <=
   cores * oversubscribe` was arithmetic over a value the encoder ignores, and no thread
   budget replaces it: a single level-4 encoder already asks for 87 threads on 32 cores.
   `_validate` now rejects an `lp_level` outside `[0, 6]`, because the encoder would only
   clamp it to 6 and warn. `load_roster` no longer takes `core_count`.

This is a config-key rename. A roster that still says `threads_per_slot` silently falls back
to the default `lp_level = 6`; `.archive-run/denoisers.toml` was migrated in place. The
`.archive-bench*` rosters are historical run artifacts and were left alone.

**Re-benchmark before trusting the archive-run estimate.** The 2-slot encode rate the
schedule is built on came from clamped runs that were already at level 6, so it should hold,
but no combined denoise-plus-encode run has been measured since.

Not done: nothing spends a core budget on `--pin`. `--pin` is the parameter that accepts a
core count, and it also steers `--lp 0`, so it is the right lever if the encoders ever need
to be confined. No measurement exists for it.
