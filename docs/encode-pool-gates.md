# Remote encode pool: gate results

Run 2026-08-26 on the full fleet, against the design notes, which are not part of this tree
section 9. Code at `8271cba` when the gates started.

The gates were run in an isolated `ARCHIVE_RUN_DIR=.gate-run`, never against the live
`.archive-run`, for the reason `archive-batch.py:39` gives. 69 real archive clips were
staged from gpu1, encoded and published to `/mnt/media/dance/encoded/`.

## Verdicts

| Gate | Verdict | Evidence |
|---|---|---|
| 1 Parity | **PASS**, bit-identical | Same clip on encoder-host (AMD) and gpu2 (Intel): md5 `45c6c6d8…` on both extracted bitstreams, 900/900 frames each |
| 2 Local path unchanged | **PASS** | gpu1 split lane to the local encoder, `netstream -> 10.0.0.10:5301 \| SvtAv1EncApp (local)`, 900/900 frames, 59.5 s |
| 3 Same-host pairing | **PASS** | gpu4 denoise + gpu4 encode: only non-ssh socket in 53 samples was `10.0.0.14:47696 ↔ 10.0.0.14:5320`, one staged source, 900/900 frames. Repeated 6× through the scheduler |
| 4 Cross pairing | **PASS** | gpu1 denoise, gpu4 encode, 900/900 frames, 66.3 s. Repeated 11× through the scheduler |
| 5 Allowlist | **PASS** | Over 60 published clips: gpu1_4090 took 11, **0 on gpu2**; the allowlisted gpu4 lane took 9, **0 on gpu2**. gpu2 served 18 clips in the same run, every one from an unrestricted lane |
| 6 Off-LAN refusal | **PASS**, both halves | 6a: gpu2 refused `10.44.0.7:5310` with `Errno 99`, bound nothing else, wrote no output. 6b: the scheduler quarantined gpu2 for 300 s, charged no clip for it, and finished 8 clips on the remaining encoders with 0 failures |
| 7 Address healing | **PASS** | gpu2 resolved to `10.0.0.17`, never its tailnet address; a CIDR it does not hold raised `NoTrustedAddress` |
| 8 Slot accounting | **PASS** | ~1000 lane-state samples across three runs, **zero** in `waiting_for_slot`. Concurrency is capped by lane count, not slots — see below |

Gate 8's wording ("concurrently working lanes equals the enabled slot total") cannot hold
as written: 6 lanes against 13 slots ceilings at 6. The testable half — no lane waiting
while a slot is free in its allowlist — held throughout.

## Measured throughput

Final full-fleet run, 6 lanes, `lp_level = 4`:

| lane | fps (work) |
|---|---|
| gpu1_4090 | 10.67 |
| 2070s | 4.63 |
| gpu3_3070 | 4.38 |
| gpu5 | 4.35 |
| gpu4 | 4.16 |
| igpu | 2.82 |
| **pool** | **33.74** |

Without gpu1 the same pool measured 22.95 fps, so gpu1 contributes almost exactly its own
rate and the pool composes additively at this size.

**A remote encode can be faster than a local one.** Gate 1's clip took 388 s encoding on
encoder-host and 239 s encoding on gpu2, same denoiser, same source. Offloading stopped the
encoder competing with the iGPU denoiser for encoder-host's cores.

## Bandwidth

y4m at 1080p YUV420P10 is **6.22 MB/frame**, so 1 fps of denoise throughput costs ~50 Mbps.
Measured directly: `[netstream] sent 20460 MB in 317.3s (64 MB/s)` for one 3289-frame clip
gpu1→gpu4, i.e. **512 Mbps sustained for one lane**.

gpu4, gpu5, gpu3 and gpu2 share a **2.5 Gbps uplink**; encoder-host and gpu1 have 10 Gbps.
gpu3 and gpu2 are both on the same WiFi, so a gpu3→gpu2 pairing crosses the air twice and
is the worst placement available, not a cheap one. The picker is topology-blind; spec section 10
deferred bandwidth-aware admission and that is still the gap.

## Unplanned: the hard-kill test

The scheduler was SIGKILLed mid-run deliberately, to see what an abrupt death leaves behind.

- **It leaks every in-flight child**, local and remote, and those children keep their ports.
  Four orphans survived on gpu4 including a `netstream recv` whose command line says neither
  `encode-serve` nor `denoise-serve`, so an obvious `pgrep` misses it.
- **Recovery is safe but not automatic.** The restarted run assigned the same slot, and the
  encode half refused with `cannot bind 10.0.0.14:5320 ([Errno 98] Address already in use)`
  rather than mis-routing. No cross-clip corruption. But the denoise half still connected to the
  stale listener and pushed frames into it — nothing reclaims a port or signals the denoise side.
- **No clip was lost.** All three in-flight clips requeued and published on other lanes.

Clearing the orphans by PID was enough for the pool to heal itself completely.

## Defects found and fixed

All four were found by running the gates, not by reading the code.

| commit | defect |
|---|---|
| `18af8d2` | Live fps was blind to every remote-encode lane: `_netstream.log` is written on the encode host, and the counter that does reach this host is in `_encode.log` in the encoder's own format |
| `3334a11` | The picker took the first free slot in roster order, piling a second encode onto a busy host while another sat idle. gpu4 carried two encodes at load 18/20 feeding gpu1 at 10.0 fps against 14.2, while gpu5 sat at load 1.7 with no encode at all |
| `d163918` | Every remote-encode failure was blamed on the denoise host, because `log_tail` never read `_encode.log` and the dispatch's own `Error:` marker was not a recognised root cause |
| `7653db2` | A windowed lane displayed 23.6 fps against a true 4.6, measuring the encoder draining a delivered window rather than the lane filling one |

## Still open

- **The picker is topology-blind.** It has no notion of the shared 2.5 Gbps uplink or of the
  WiFi hosts behind it, so it can route a lane's y4m across the worst available path.
- **gpu1 has no `[[encoder]]` entry**, so the fleet's largest flow always crosses the wire.
- **gpu1 shows TensorRT drift**: `vstrt: TensorRT version mismatch, built with 110100 but
  loaded with 110201`. A warning today, and the same 11.2.1 upgrade recorded in the encoder-host
  fallout note.
- **gpu2's GPU cannot denoise.** It failed a clip with `CUDA failure 700: an illegal memory
  access`. It encodes flawlessly and bit-identically, so it belongs in the pool as an encoder
  and not as a lane.
- **A plain no-denoise encode is broken on VS R77.** The source VPY at
  `tools/svtav1-dispatch.py:1718` omits the `LoadPlugin` preamble every other generated VPY
  carries, so `core.ffms2` is missing. Production never hits it because every rostered lane
  denoises. Pre-existing, unrelated to the pool.
