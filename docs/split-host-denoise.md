# Split-host denoise: why it looks like this

Status: applied. `--remote-denoise` / `--denoise-serve` in `tools/svtav1-dispatch.py`, transport in `tools/netstream.py`.

## Problem

BSVD denoise and SVT-AV1 encode want different hardware, and on this fleet they live on different machines. Measured at 1080p (2026-08-03):

| stage | host | rate |
|---|---|---|
| BSVD, ORT-MIGraphX, full-frame | encoder-host Radeon 8060S | 5.19 fps |
| BSVD, ORT-TRT, full-frame | gpu1 RTX 4090 | 21.6 fps (pure inference), 16.2 fps through `vspipe` |
| SVT-AV1, dance-HQ preset | encoder-host (16c Zen5) | 15.9 fps |

Denoising on the 4090 and encoding on the 16-core box puts the two stages within 2% of each other, so a pipe between them runs both flat out — roughly 3x the all-local rate. The dance-HQ preset is single-pass `vspipe | SvtAv1EncApp` (no av1an chunking), so it can be streamed; the av1an paths cannot, because per-scene CRF needs a seekable denoised file.

## Transport: TCP, not ssh

Throughput encoder-host↔gpu1, same 10G LAN:

| path | throughput |
|---|---|
| ssh over tailscale | 168 MB/s |
| ssh over the LAN IP | 155 MB/s — ssh is the cap, not the wire or the cipher |
| tailscale raw TCP | 185 MB/s (WireGuard cap) |
| LAN raw TCP | 669-935 MB/s |

The stream is yuv420p10 at 6.22 MB/frame, so 16 fps needs ~99 MB/s: ssh would run at 60% utilization with no headroom for 4K, while a plain socket has 7x. So ssh carries only the control channel — launch, stderr, exit status — and the frames get their own connection.

## Which end listens

The **encoder** listens; the denoise host connects out. Three reasons:

1. The listener's lifetime is owned by the process that also owns the encoder. When the remote listens, it has to be spawned over ssh and detached, and a local crash strands a process holding a port on the remote box.
2. The ssh remote command is then the actual work (`vspipe`), so its exit status describes the denoise. If the remote were a server, its exit status would describe the server.
3. The firewall rule lands on a Linux box we control, scoped to one source IP, instead of on a Windows host where rules are per network profile and silently stop applying when the profile flips.

`send` retries only the *connect*, so the two ends may start in either order. It never retries mid-stream: BSVD carries state across frames and the encoder sees one continuous y4m, so a broken connection must fail loudly rather than resume with a hole.

## Truncation safety

A remote that dies mid-stream closes the socket cleanly. The local encoder therefore finalizes a short IVF and exits 0 — `run_piped`'s source check cannot see it. Two guards catch this instead: the ssh exit-status check in `run_remote_denoise`, and definitively the frame-count verification before the mux. Do not remove either.

## Remote invocation

The remote half is this same script re-invoked over ssh with `--denoise-serve`, so it builds its own VPY with its own paths and no translation is needed. It is launched as `ssh host bash -s` with the command on stdin: the remote login shell is fish, and `ssh host bash -c '...'` mangles quoting silently.

## Verification

```bash
# transport only, no firewall change needed (reverse-tunnel the callback)
ssh -f -N -R 5300:127.0.0.1:5300 <remote>
python3 tools/svtav1-dispatch.py -i Input/clip.MOV -o Output/clip-av1.mkv \
  --quality 40 --speed 8 --denoise-bsvd --bsvd-sigma 0.05 \
  --remote-denoise <remote> --remote-callback 127.0.0.1
```

Expect `Frame count verified: N frames.` and a muxed output. The remote's log is `Temp/<stem>/<stem>_remote.log`.

## Known rough edges

- A remote that fails to start is only reported after the receiver's 300 s accept timeout. The remote log tail is printed then, so the cause is visible, but the wait is dead time.
- `ssh host bash -s` is non-interactive and non-login, so it reads no shell rc: `$VSPIPE` and friends are unset on the remote. A host that needs a pinned vspipe (encoder-host's py3.12 MIGraphX shim) cannot currently be the denoise end.
- The staged source under `<remote-root>/Temp/_remote/` is not cleaned up between runs.
- IPv4 only.
