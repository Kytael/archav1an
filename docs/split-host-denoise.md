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

`--remote-encode` moves the encoder to a third host and changes none of that. The encode
host listens (`--encode-serve IP:PORT`) and the source connects out, whether that source is
a local `vspipe` or a second host running `--denoise-serve`. Only two things differ: which
machine holds the listener, and that the finished IVF comes back by rsync instead of being
written where it was muxed.

The source may also be a `vspipe` with **no denoise pass at all** — an encode job. Until
2026-09-04 dispatch refused that combination outright, on the grounds that only the denoise
branches routed to the remote and the plain source branch would encode locally while
reporting success. That was true of the code, and the fix was to route the branch rather
than keep the refusal: `--remote-encode` is what archive-batch passes for every encoder
whose roster `host` is not `local`, so the refusal made every encode job a local-encoder
job and answered the rest of the pool with one error line. Verified on gpu5 the same day:
a 120-frame clip decoded on encoder-host, streamed 83 MB, encoded there, and came back with the
submitted preset's settings intact.

`send` retries only the *connect*, so the two ends may start in either order. It never retries mid-stream: BSVD carries state across frames and the encoder sees one continuous y4m, so a broken connection must fail loudly rather than resume with a hole.

Both ends set `SO_KEEPALIVE` (60 s idle, 6 probes 10 s apart) so that "fail loudly" also covers a peer that vanishes **without** closing. A receiver has nothing to send, so otherwise it never learns the peer is gone: the socket stays `established` and `recv()` blocks until the lane burns its whole dispatch timeout, which is `3600 + frames/1.0` seconds — 4.5 h on a 12427-frame clip. Measured on gpu3 2026-08-31, whose ethernet renegotiates at a random speed on every link flap; the socket sat established with `lastrcv` at 5.8 minutes. It is keepalive and not an application stall timeout because a tile-sequential lane legitimately emits nothing until the last tile pass of a window, so a timeout would fail those lanes for working correctly.

netstream is spawned from **each side's own checkout** — locally via `os.path.dirname(__file__)`, remotely under `remote_root` — so a transport change is only in effect on a host once that host has pulled it.

## Truncation safety

A remote that dies mid-stream closes the socket cleanly. The local encoder therefore finalizes a short IVF and exits 0 — `run_piped`'s source check cannot see it. Two guards catch this instead: the ssh exit-status check in `run_remote_denoise`, and definitively the frame-count verification before the mux. Do not remove either.

A remote encoder keeps both guards, one per ssh session, but weights them differently.
`run_remote_encode` and `run_remote_encode_split` treat the ENCODE host's nonzero exit as
fatal: that host tears down no VS core and no TRT session after the last frame, so it has
no teardown race to forgive. The denoise host's exit stays advisory for the reason above.
The frame count is still verified on the host that holds the source, against that source,
before the mux -- the IVF is fetched first for exactly that reason.

## Remote invocation

Every remote half is this same script re-invoked over ssh, so it builds its own VPY and its own encoder command with its own paths and no translation is needed. The denoise half runs with `--denoise-serve IP:PORT`, the encode half with `--encode-serve IP:PORT`; case 4 runs both, one per host, out of `run_remote_encode_split`.

Each is launched as `ssh host bash -s` with the command line written to stdin. Never `ssh host bash -c '...'`: a remote login shell on this fleet may be fish, which mangles the quoting of that form silently -- a wrong result, not an error. `build_remote_shell_cmd` writes the line for both halves (`build_encode_serve_cmd` calls it), so the `cd` into the checkout, the tilde handling in `quote_remote_path` and `PYTHONUNBUFFERED=1` cannot drift apart between them.

That line deliberately sets no PATH. `main()` calls `prefer_prefix_bin()` as its first statement, and that prepends `$VS_PREFIX/bin` and appends `$VS_PREFIX/lib` together, which they must be. Setting `PATH=/opt/archav1an/bin` here as well would hardcode one prefix while `prefer_prefix_bin` honours another, and prefix binaries against system libraries share a soname and differ in ABI -- the mismatch that made `ffmpeg` write a complete output file and then exit 127 on 2026-08-12, and read as a bad-clip problem for two runs. Measured on gpu4, whose non-interactive ssh PATH omits the prefix: `shutil.which` answers `None` for `vspipe` and `SvtAv1EncApp` before that call and the real prefix paths after it, so no lane ever needed the PATH line.

The checkout differs per host, so `--remote-root` and `--remote-encode-root` carry it from the roster's `Denoiser.root` and `Encoder.root`. The interpreter does not: `--remote-python` is one value for both halves and defaults to `/opt/archav1an/venv/bin/python`, so every pool host must have its interpreter at that exact path.

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
- `ssh host bash -s` is non-interactive and non-login, so it reads no shell rc: `$VSPIPE` and friends are unset on the remote. A host that needs a pinned vspipe (encoder-host's py3.12 MIGraphX shim) cannot currently be the denoise end. `prefer_prefix_bin()` does not lift this either: it decides which `vspipe` is found by name, not which one a host pins through `$VSPIPE`, which the remote command still does not set.
- The staged source under `<remote-root>/Temp/_remote/` is not cleaned up between runs.
- IPv4 only.
