# archav1an

Neural video denoising and AV1 encoding, on one machine or spread across several.

This started as a Linux port of Auto-Boost-Av1an, and its per-scene CRF boosting is still the core of the two-pass scripts below. Three things have grown around it since:

- **Neural denoising** with BSVD or STA-SUNet, running inside VapourSynth through ONNX Runtime — TensorRT on NVIDIA, MIGraphX on AMD. Tiled and windowed modes let cards without enough memory for a full 1080p frame still run the model.
- **Split-host denoising**, where the denoiser runs on a remote GPU and streams frames back over the network to whichever machine encodes. The two do not have to be the same machine, or the same vendor.
- **Batch processing** of a whole archive across several GPUs at once, with a per-clip scheduler, resume after interruption, and per-lane throughput accounting.

This guide explains how to set it up and run it on Linux (Arch-based distros like CachyOS, and Ubuntu/Debian).

Both x86_64 and arm64 are supported. On arm64 the setup skips the x86-only assemblers, builds the VapourSynth plugins from source where no distro package exists, and takes TensorRT from apt — no index publishes an arm64 TensorRT wheel.

---

## Which Script Should I Choose?

Pick based on your content type and desired quality:

### 🎌 ANIME
| Script | Quality | Description |
|--------|---------|-------------|
| `run_linux_anime_crf32.sh` | **Standard** | ✅ Recommended starting point |
| `run_linux_anime_crf25.sh` | High | Higher quality, larger files |
| `run_linux_anime_crf18.sh` | Archival | Maximum quality, largest files |
| `run_linux_anime_crf15.sh` | Archival+ | Aggressive settings, largest files |

### 🎬 LIVE ACTION / MOVIES / TV SHOWS
| Script | Quality | Description |
|--------|---------|-------------|
| `run_linux_live_crf32.sh` | **Standard** | ✅ Recommended starting point |
| `run_linux_live_crf25.sh` | High | Higher quality, larger files |
| `run_linux_live_crf18.sh` | Archival | Maximum quality, largest files |
| `run_linux_live_crf15.sh` | Archival+ | Maximum fidelity, largest files |

### ⚽ SPORTS / FAST MOTION
| Script | Quality | Description |
|--------|---------|-------------|
| `run_linux_sports_crf27.sh` | Optimized | ✅ Best for high-motion content |

### 💃 DANCE / PERFORMANCE
| Script | Quality | Description |
|--------|---------|-------------|
| `run_linux_dance_crf27.sh` | Standard | Dance/performance footage |
| `run_linux_dance_HQ_crf27.sh` | High | Single-pass whole-clip encode via `svtav1-dispatch.py`; forwards extra args (e.g. `--denoise-bsvd`) |

### 🎞️ DIRECT ENCODE (SINGLE PASS, NO BOOST)
| Script | Quality | Description |
|--------|---------|-------------|
| `av1an-batch-anime-crf32.sh` | Standard | Single-pass anime encode, no Auto-Boost |
| `av1an-batch-liveaction-crf32.sh` | Standard | Single-pass live action encode, no Auto-Boost |
| `av1an-batch-dance-crf27.sh` | Standard | Single-pass dance encode, no Auto-Boost |

### 🧹 DENOISERS (single-pass path)

`tools/svtav1-dispatch.py` — used by `run_linux_dance_HQ_crf27.sh` and reachable from the single-pass wrappers — can GPU-denoise ahead of the encoder:

| Flag | Backend |
|------|---------|
| `--denoise-bsvd` | BSVD V2 stateful streaming via ONNX Runtime (TensorRT/CUDA on NVIDIA, MIGraphX on AMD). Default `--bsvd-sigma auto` picks σ per clip |
| `--denoise-bsvd-smdegrain` | BSVD as SMDegrain prefilter (hybrid; helps dark clips) |
| `--denoise-scunet` | SCUNet via vs-mlrt (TRT/MIGX backend) |
| `--denoise-stasunet` | STA-SUNet TensorRT engine (fixed 512/768 tiles; see `--denoise-stasunet-engine`) |
| `--denoise-smdegrain` | Classical mvtools SMDegrain |

Model assets and Python deps are staged by `./setup.sh --install denoiser`.

### 🌐 SPLIT-HOST DENOISE (REMOTE GPU)

The BSVD denoise stage can run on another machine's GPU while SVT-AV1 encodes locally — useful when the fast GPU and the fast CPU are different boxes. ssh carries only control; the denoised y4m comes back over a plain TCP socket, because a single ssh stream caps around 1.3 Gbps while a socket on the same 10G LAN sustains 5+ Gbps.

| Flag | Meaning |
|------|---------|
| `--remote-denoise SSH_TARGET` | Denoise on this ssh host (alias or `user@host`), encode here. BSVD paths only |
| `--remote-callback IP` | Address the remote streams back to. Defaults to this host's IP toward the remote — pass the LAN IP explicitly if ssh reaches the remote over a VPN |
| `--remote-port N` | Listening port on this host (default 5300) |
| `--remote-root PATH` | Repo checkout on the remote (default `~/archav1an`) |
| `--remote-python PATH` | Interpreter on the remote (default `/opt/archav1an/venv/bin/python`) |

```bash
# open the port to the denoise host once (example: ufw)
sudo ufw allow from <remote-lan-ip> to any port 5300 proto tcp
# then any single-pass wrapper takes the flag
./run_linux_dance_HQ_crf27.sh --denoise-bsvd --remote-denoise gpu1 --remote-callback <this-host-lan-ip>
```

Requirements: key-based ssh plus `rsync` on both hosts, and the same repo checkout at `--remote-root` on the remote with `./setup.sh --install denoiser` already run there. The source file is staged to `<remote-root>/Temp/_remote/` per run. `--denoise-serve HOST:PORT` is the remote half of the protocol — the dispatcher invokes it over ssh; you never pass it by hand.

### 🚀 PROGRESSION BOOST (PER-SCENE OPTIMIZATION)
| Script | Quality | Description |
|--------|---------|-------------|
| `Progression-Boost-SSIMU2-anime.sh` | **Auto** | Analyzes each scene and optimizes settings for Anime |
| `Progression-Boost-SSIMU2-liveaction.sh` | **Auto** | Analyzes each scene and optimizes settings for Live Action |
> **Note:** Progression Boost uses SSIMULACRA2 metrics to target a visual quality score (Default: 82), adjusting bitrate (crf) on a per scene basis to target that quality score. It benchmarks your CPU/RAM on first run to set optimal workers.

> **TIP:** Start with CRF 30. If quality isn't sufficient, try CRF 25. For archival purposes, use CRF 18.

### What is CRF?
CRF stands for "Constant Rate Factor." It determines the balance between Video Quality and File Size:
- **Lower CRF** (e.g., 18) = Higher Quality, Larger File Size
- **Higher CRF** (e.g., 30) = Lower Quality, Smaller File Size

---

## Prerequisites

### Automatic Installation (recommended)

Everything builds into a single isolated prefix at `/opt/archav1an/` so nothing collides with pacman-owned paths. For the full inventory (versions, plugins, system packages, Python deps), see [DEPENDENCIES.md](DEPENDENCIES.md).

**Install everything:**
```bash
chmod +x setup.sh
./setup.sh --install A
```

`setup.sh` prompts for sudo **once** at the start to create `/opt/archav1an/` and chown it to you; everything after that runs as your user. If `uv` (the Python venv/pip replacement) isn't installed, the script fetches Astral's official installer and drops it in `~/.local/bin`. The only other step that needs root is installing distro packages, and `system_deps` escalates for the package manager itself — so a fresh host may prompt a second time, and an already-provisioned one will not prompt at all. Running the whole script under `sudo` is supported too, and is the better choice for an unattended `-y` run; see below.

Or selectively:
```bash
./setup.sh --install system_deps   # distro packages via pacman/apt (escalates itself)
./setup.sh --install python_libs   # uv venv at /opt/archav1an/venv
./setup.sh --install ffmpeg        # source-built ffmpeg w/ NVENC into prefix
./setup.sh --install vapoursynth   # VS R79 + FFMS2 + BestSource
./setup.sh --install denoiser      # BSVD/SCUNet/SMDegrain/RVRT/STA-SUNet plugins + models
./setup.sh --install wwxd vszip subtext  # core VS plugins
```

The full target list is `system_deps python_libs svt_av1 ffmpeg vapoursynth av1an ffvship oxipng fssimu2 wwxd vszip subtext denoiser`. Dependencies resolve automatically, so naming a late component pulls in what it needs.

`system_deps` is the one target that installs distro packages, and the only one that needs root. You do not have to hand it root up front — it escalates for the package manager by itself and stays as your user for everything else.

Running the whole script under `sudo` is equally supported, and it is the right call for an unattended `-y` run on a host that still needs packages: sudo's credential cache expires after 15 minutes by default, and the denoiser wants root for `/etc/ld.so.conf.d` an hour into a full build. A root run hands `/opt/archav1an/` and `build_tmp/` back to `$SUDO_USER` when it exits, so the venv and the next unprivileged run stay writable, and the AUR steps drop back to `$SUDO_USER` because `makepkg`/`paru` refuse to run as root at all.

It checks first and escalates only if something is missing — `pacman -Qi` on Arch, `dpkg -s` on Debian/Ubuntu — so on an already-provisioned host the whole install runs with no package-manager prompt. AUR packages remain manual: `paru` or `yay` is installed for you where one is a repo package (CachyOS has both; plain Arch has neither), and named in a warning where it is not.

### cuDNN and TensorRT on Debian/Ubuntu

Ubuntu ships **no cuDNN and no TensorRT at any version**, and its `nvidia-cuda-toolkit` is CUDA 12.0. The BSVD denoise path wants cuDNN 9 and TensorRT 10 for the onnxruntime TensorRT EP, so on Debian/Ubuntu those come from NVIDIA's own CUDA repository or not at all. Without it `--denoise-bsvd` still works, on the slower CUDA EP, and setup says so.

Adding that repo is a separate, opt-in target. It is **not** part of `--install A`, because it installs a third-party repository and signing key:

```bash
./setup.sh --install nvidia_repo
```

It installs NVIDIA's `cuda-keyring`, refreshes apt, installs `cuda-toolkit`, then re-runs `system_deps` to pick up cuDNN and TensorRT. It deliberately installs `cuda-toolkit` rather than the `cuda` metapackage, which would pull `cuda-drivers` and can replace a working display driver. On WSL2 it selects NVIDIA's `wsl-ubuntu` repo, which carries no driver packages — the GPU driver there belongs to Windows, and installing a Linux one breaks CUDA.

Arch needs none of this: cuDNN comes from pacman and TensorRT from the AUR, both already handled.

**Activate the env to use vspipe / Python tools from your shell:**
```bash
source activate-venv.sh
```

`activate-venv.sh` sources the venv, prepends `/opt/archav1an/bin` to PATH, sets `LD_LIBRARY_PATH=/opt/archav1an/lib` (the source-built R79 VapourSynth shares pacman's SONAME `libvapoursynth.so.4`, so LD_LIBRARY_PATH precedence makes R79 win only inside this activated env — pacman's copy remains the global default outside it), and sets `VAPOURSYNTH_EXTRA_PLUGIN_PATH=/opt/archav1an/lib/vapoursynth`.

**Choosing the Python version:**
```bash
PYTHON_VERSION=3.13 ./setup.sh --install python_libs   # pin to 3.13 (uv downloads if needed)
./setup.sh --install python_libs                       # default: whatever `python3` resolves to
```
If a Python bump (typically pacman to a new minor) breaks a binary dep, pin to a known-good version via `PYTHON_VERSION`. The installer warns and rebuilds the venv when the requested version differs from what's already there; rerun `--install vapoursynth` after a Python change because the VS module is binary-linked to the venv's interpreter.

For manual / step-by-step installation see [DEPENDENCIES.md](DEPENDENCIES.md) for the full component list and source URLs.

## Verification

After `source activate-venv.sh`:

```bash
# Core tools (should resolve to /opt/archav1an/bin)
which vspipe ffmpeg av1an SvtAv1EncApp
vspipe --version    # should report "Core R79"
ffmpeg -version | head -n 1
SvtAv1EncApp --help | grep -i "SVT-AV1"

# VapourSynth Python (should resolve to /opt/archav1an/lib/python3.X/site-packages)
python -c "import vapoursynth as v; print(v.__file__); print(v.core.version().split(chr(10))[0])"

# Plugins
python -c "
from vapoursynth import core
checks = ['wwxd','vszip','sub','ffms2','bs','knlm','trt','mv']
for c in checks: print(f'{c:8s}: {hasattr(core, c)}')
"

# Isolation check — pacman owns nothing inside our prefix
pacman -Qkk vapoursynth     # must report "0 altered files"
pacman -Qo /opt/archav1an/bin/vspipe   # must say "No package owns ..."
```
## Usage

1.  Place your source files (e.g., `.mkv` or `.mp4`) into the `Input/` folder.
    *   *Note: The script will create this folder automatically if it doesn't exist.*
    *   *Note: Files do NOT need to be renamed to `*-source.mkv` anymore.*
2.  Make the scripts executable (if not already):
    ```bash
    chmod +x run_linux_anime_*.sh
    chmod +x run_linux_live_*.sh
    ```
3.  Run the script variant of your choice.
    *   Your final encoded files will appear in the `Output/` folder.
We provide variants based on content type (Anime vs Live Action) and quality. All scripts support **Auto-BT.709 Detection**.

**Anime Variants:**
*   **Standard (CRF 32)**: `./run_linux_anime_crf32.sh` - Balanced speed/quality.
*   **High (CRF 25)**: `./run_linux_anime_crf25.sh` - Slower, Tune 0.
*   **Highest (CRF 18)**: `./run_linux_anime_crf18.sh` - Aggressive boosting.
*   **Archival (CRF 15)**: `./run_linux_anime_crf15.sh` - Maximum quality.

**Live Action Variants (Auto-Crop Enabled):**
*   **Standard (CRF 32)**: `./run_linux_live_crf32.sh` - Auto-crop, Tune 3.
*   **High (CRF 25)**: `./run_linux_live_crf25.sh` - Tune 3, Variance Boost 2.
*   **Highest (CRF 18)**: `./run_linux_live_crf18.sh` - Maximum fidelity.
*   **Archival (CRF 15)**: `./run_linux_live_crf15.sh` - Maximum fidelity.

**Sports / High-Motion Content:**
*   **Standard (CRF 27)**: `./run_linux_sports_crf27.sh` - Optimized for high-motion content with extra temporal filtering.

The script will:
1.  Detect Scene Changes.
2.  Start Av1an with the optimized parameters.
3.  Automatically calculate worker count based on your hardware (on first run).
4.  Run Auto-Boost-Av1an (Fast Pass -> Metrics -> Zones -> Final Encode).
5.  Mux audio/subtitles back.
6.  Tag the output file.
7.  Cleanup temporary files.
8.  Final outputs are in the `Output/` folder.

## Audio Encoding (Standalone)

We include an `audio-encoding/` folder for batch audio conversion workflows:

| Script | Description |
|--------|-------------|
| `encode-ac3-audio.sh` | Converts audio tracks to **AC3** (Dolby Digital) - for legacy devices |
| `encode-eac3-audio.sh` | Converts audio tracks to **EAC3** (Dolby Digital Plus) - recommended |
| `encode-opus-audio.sh` | Converts audio tracks to **Opus** - best quality/size ratio |

> **2.1 Channel Support:** `encode-opus-audio.sh` (192k) and `encode-ac3-audio.sh` (320k) now support 2.1 channel detection and optimization.


*Usage:*
```bash
cd audio-encoding
# Place your .mkv files in this folder
./encode-eac3-audio.sh
```

Settings files (`settings-encode-*.txt`) control bitrates per channel configuration.

## Extras (Linux)

We include an `extras/` folder with helper scripts for advanced workflows:

| Script | Description |
|--------|-------------|
| `lossless-intermediary.sh` | Converts video to lossless 10-bit x265 intermediates |
| `compare.sh` | Generates comparison screenshots via `comp.py` |
| `light-denoise.sh` | Applies DFTTest denoise + x265 lossless encoding |
| `light-denoise-nvidia.sh` | GPU-accelerated NVEncC denoise (NVIDIA required) |
| `forced-aspect-remux.sh` | Copies aspect ratio from source to encoded output |
| `disk-usage.sh` | Reports disk usage (Linux replacement for NTFS compress) |

*Usage:*
```bash
cd extras
./light-denoise.sh
```

## Prefilter (Deband Scripts)

The `prefilter/` folder contains scripts for applying deband filters before encoding:

| Script | Description |
|--------|-------------|
| `nvidia-deband.sh` | NVIDIA GPU deband using NVEncC + libplacebo |
| `x265-lossless-deband.sh` | CPU deband using VapourSynth + x265 lossless |

Edit `prefilter/settings.txt` to customize filter settings.

*Requirements:*
- For NVIDIA scripts: NVEncC installed and in PATH
- For x265 scripts: VapourSynth with placebo plugin, x265

## Encode dashboard

`tools/encode-dash.py` is a status page for a batch run. Start it from the repo
root:

```bash
/opt/archav1an/venv/bin/python tools/encode-dash.py
```

It binds this host's Tailscale address on port 9328 and serves the page at `/`,
the whole snapshot as JSON at `/api/status`, and Prometheus text at `/metrics`.
`--host` and `--port` override the defaults. `--grafana-url` points the page's
"history" button at wherever you keep the long-term charts; leave it out and the
button is hidden, because where the history lives is a property of your
deployment rather than of this program.

### Starting and stopping a run

The page's "Start the run" button posts to `POST /api/run/start`; the daemon
spawns `tools/archive-batch.py` with its own session (`start_new_session=True`),
so the daemon's Ctrl-C does not reach the batch.
A second start while a run is live answers `409` — one run at a time. The
"Stop the run" button posts to `POST /api/run/stop`; stopping is graceful,
through the batch's control directory, and the batch clears its liveness file
as it exits. A daemon crash mid-run is a non-event: on restart the supervisor
adopts the live batch from `batch.json` and the page keeps reporting the run it
already knows about. A stale liveness file left by a SIGKILLed batch is
cleared, so a fresh start is not refused forever. Liveness is decided with the
process's `/proc` starttime (`pid_start`, compared by
`tools/archive_batch/pidfile.py`), so a recycled pid reads as stale instead of
blocking starts or being adopted by mistake. A record written before that
field existed — a batch that was already running when you upgraded the
checkout — carries no starttime, so it is checked against the process's
command line instead: a live pre-upgrade run keeps reading as live, and the
page does not offer Start over it. If the run directory is not writable by the
daemon, the page still comes up read-only and Start answers `503` naming the
reason.

To keep it across reboots, install it as a systemd user service:

```bash
./tools/install-encode-dash-service.sh
```

That writes `~/.config/systemd/user/encode-dash.service`, enables it and starts
it. Re-run it after moving the checkout or the venv; it rewrites the unit and
restarts the daemon. `VENV_PY` overrides the interpreter, and `GRAFANA_URL`
puts `--grafana-url` on the generated `ExecStart` — passed at install time
rather than defaulted in the script, so no site's hostname is committed here.
Afterwards use
`systemctl --user restart encode-dash` to pick up a code change, and
`journalctl --user -u encode-dash` to read the log.

`setup.sh` does not install this, deliberately. `setup.sh` builds encode
dependencies on every host that denoises or encodes; the dashboard runs on one
host and serves the whole fleet, so installing it everywhere would leave four
idle daemons fighting for one port. The unit is generated rather than committed
because `ExecStart` needs this checkout's absolute path, and the hosts do not
agree on it.

It runs beside the batch, not inside it: it reads `.archive-run/`, spawns or
adopts the batch, and then talks to it only through files — the control
directory, and `batch.json`, which the supervisor writes to claim a live run.
Stopping the unit kills only the daemon (`KillMode=process`), so the batch
survives and is adopted by the next daemon.

Two batch-side behaviours exist to feed it, and they are on by default:

- `vspipe -p` and `netstream recv --progress` leave a frame counter in each
  clip's log, which is where the **live** rate comes from. A rate computed from
  finished clips would be up to three hours stale on the longest clip in the
  archive, because a record only lands when a clip completes.
- Each worker writes `.archive-run/lanes/<name>.json` naming the clip it holds
  and whether it is denoising or waiting for an encode slot.

Reading it: a blank rate is not zero. An absent measurement and a measured zero
are different facts everywhere in this page, so a lane that has not started
never looks like a lane that has stalled.

### The Failed panel

Its heading reads `N exhausted · M listed`, because the two numbers are
different questions and it used to answer only the first. The list is the last
50 failed *attempts*, most of which get retried and succeed; `exhausted` counts
the clips the run has actually given up on. A healthy run therefore shows
"0 exhausted" over a list of fifty rows, which read as a broken page until the
heading said both. Tick **out of attempts only** to see just the clips that
need a decision — those are also the only ones with a `retry` button, since a
clip with an attempt left is already coming back on its own.

The list is capped, but never at the cost of the rows that matter: it carries
the most recent failures *plus* every exhausted one. Recency alone was the
wrong cap, because a clip stops failing at the moment it runs out of attempts
and then drifts out of a newest-50 window while clips that keep retrying stay
in it. The filter emptied the panel instead of narrowing it — at exactly the
point the panel had something to say.

**Retry all** puts back every clip that is out of attempts, in one press. It
reads `state.jsonl`, not the rows on screen — the list above is a capped
preview, so a button driven from the rendered rows would quietly do less than
its label says while still reporting success. The number it acts on is the
`N exhausted` in the heading beside it. Like Stop, it arms on the first click
and acts on the second, because it re-queues work measured in days.

**Retry works while the run is stopped**, and that is the case it exists for. A
clip out of attempts is dropped from the queue, so a run with nothing else left
prints `nothing to do` and exits; the batch then applies any retry queued
against it on the next start, before it decides whether it has work. So the
order is: press `retry` on the rows you want back, then **Start**. Retries and
submissions are the two requests that survive a stopped run — a queued `yield`
or `stop` is still discarded at startup, because each names a live lane or a
live run and a stale one would fire into a run it was never meant for.

### Encode jobs: a folder with no denoise pass

The **Encode a folder** form queues work that skips the GPU entirely: every
`.MOV` and `.MP4` directly in one folder, decoded on the batch host and encoded
with SVT-AV1 on an encode host's CPU. One level only — a folder inside the one
you name is not included, because its files would be published to a destination
chosen for their parent.

The form takes three fields: `host` is the ssh alias that holds the folder (or
`local`), `path` is an **absolute POSIX path on that host**, and `dest` is a
subpath under `encoded/`. A Windows drive letter is refused, by the daemon and
again by the probe: the path is handed to a shell on the holding host, and the
probe pipes a bash script to it, so the WSL side of a box is the reachable
target even when the files live on an NTFS drive. `M:\Media\Dance\SetA\2026`
on gpu1 is `/mnt/media/dance/SetA/2026`, with `host = gpu1`.

The archive run is **not** one of the choices here. It is driven by
`manifest-raw.tsv`, it denoises, and it is started by **Start the run**. It was
briefly listed as a folder, which offered a choice and then refused half of it:
selecting it blanked the fields and disabled the button, which reads worse than
not listing it at all.

Submitting writes a `submit` control request. The batch probes the folder with
one `ffprobe` per file — over ssh when the folder is on another host — and that
walk is slow, so it acks twice: "probing" at once, then the count when the walk
finishes. A control handler that blocked would stop the run answering a stop.

**Queueing a folder works while the run is stopped**, and on a drained archive
it is the only way that works at all: the control poller that answers a
submission starts only *after* the run has decided it has work, so a folder
queued against a finished archive used to be swept away at the next **Start**
with `nothing to do` as the only sign anything had happened. The batch now
lifts queued submissions out before it clears the stale requests, probes each
one during startup, and counts the jobs it found as work to do. So the order is
the same as a retry's: queue the folder, then **Start**. The probe is serial
there rather than threaded — nothing is polling yet, and a run that started
before its own queue was known would report `nothing to do` and exit.

Jobs are appended to `manifest-encode.tsv`, which is read at startup like
`manifest-raw.tsv`, so a submitted folder survives a restart. It is a separate
file with three extra columns — the host holding the source, the destination
under `encoded/`, and the preset. An archive clip's destination *is* its own
parent directory; a submitted job's is not, and nothing may derive one from the
other. The preset sits third, among the columns the batch writes: the `ffprobe`
columns trail off, because a source with an embedded thumbnail emits a second
rate column, so nothing read from the end has a fixed meaning.

An encode job runs on any host in the pool, local or remote. On a remote encoder the
batch host decodes and streams y4m over the same `--remote-encode` path a denoise lane
uses; only the denoise pass is absent. This was refused until 2026-09-04, which made every
encode job a job for whichever encoder has `host = "local"` — see `docs/split-host-denoise.md`.

**A preset picks the encoder settings for that folder.** The dropdown lists
every `run_linux_*.sh` in the repo, and the scripts *are* the catalogue — there
is no separate list to keep in step, which is how `ENCODER_PARAMS` and
`run_linux_dance_HQ_crf27.sh` drifted apart before. A preset supplies four
values: `--quality`, `--photon-noise`, `--speed` and `--encoder-params`. Its
own `--lp` is dropped, because that is a memory-for-parallelism choice
belonging to the encode host, and the roster's `lp_level` already passes one.
Leave the preset empty for the fleet-fixed settings, which is what every encode
job got before this existed. Presets reach encode jobs only: an archive clip's
output must not depend on which device took it, and a per-job setting there
would make it depend on when it was queued instead.

**Denoise jobs win the contest for slots.** A slot worker leaves the last free
slot on an encoder alone while archive work is still queued or in flight, since
a lane can only reach that host through a slot. An encoder with `slots = 1`
therefore runs encode jobs only once the denoise work is done. A running encode
job is never preempted; a lane can wait one out, bounded by that job's length.

That reserve is held only for a lane that could actually claim it: some enabled
denoiser whose allowlist names that encoder. Without the check, every encoder
kept a slot open for lanes that are switched off, or that route elsewhere — and
in an encode-only run, where the archive backlog never drains because no lane
can take it, that slot stayed reserved for the whole run.

Each slot row carries its own live rate, progress bar and ETA, read the same
way a lane's is: a slot worker writes the same heartbeat, and dispatch writes
the same `<stem>_vspipe.log`. The **Encode hosts** table carries the figure per
host — live fps summed across that host's slots, and the completed-job average,
which stays per clip so a two-slot host does not read as twice as fast as it is.
A single slot's rate cannot be compared with another host running two, and that
comparison is what the table is for.

Submitting into a drained run queues nothing useful. Slot workers end when the
queue empties, exactly as lane workers do, so a submission has a worker waiting
for it only while the run still has work.

### Stopping and starting an encode host

The **Encode hosts** table lists every `[[encoder]]` in the roster, enabled or
not, each with the same switch a lane has. Turning a host off kills nothing in
flight: it finishes what it is encoding, lanes stop picking it at their next
clip, and it goes quiet. Turning it back on needs the row to exist while the
host is off, which is why this table lists disabled hosts and the slot count in
the status line does not — that count answers how many clips can encode now.

It is a roster write, `POST /api/encoder/<name>/enabled`, and a separate route
from the lane switch on purpose. The two tables are separate namespaces: this
fleet's roster carries a denoiser `gpu4` *and* an encoder `gpu4`, the same
machine doing two different jobs. One route taking a bare name would have to
guess which was meant, and it would guess wrong on exactly the hosts that do
both — turning off a CPU encoder would stop a GPU lane mid-clip.

### Routing a lane to encode hosts

Each lane's edit form carries an **encoders** box per host in the pool. Ticking
none means any enabled host, which is the default and what an absent `encoders`
key says in the roster. Ticking some pins the lane to those, and the scheduler
will wait for one of them rather than spill elsewhere.

Checkboxes rather than a text field because the roster validates these names
against the `[[encoder]]` table: a typo in a typed list is refused only after
the whole list has been typed. A lane that names a host since removed from the
pool still shows it, ticked and marked — that roster is already refused by the
validator, and hiding the name would make the refusal unexplainable from the
page.

Routing and the switch are one contract. A lane pinned to a single host, with
that host then switched off, would wait for ever while the rest of the fleet
drains the queue, so the roster refuses it and the page repeats the message —
which names all three ways out: enable one of its encoders, widen its
allowlist, or disable the lane. Switching off the **last** enabled encoder is
allowed, because that is an operator halting the pool rather than one lane
starving beside working ones.

### Adding and editing an encode host

Each host row carries **edit** and **remove**, and the section has **Add an
encode host**. The fields are the `[[encoder]]` ones — `host`, `root`,
`stream_ip` or `stream_net`, `port_base`, `slots`, `lp_level` — and a new host
arrives switched off, for the same reason a new lane does: adding it does not
test it.

Give a new host a port block no other host uses; slot N listens on
`port_base + N`. A remote host needs `stream_ip` or `stream_net`, and the
validator says so rather than letting the lane fail at its first clip. Prefer
`stream_net` for a WSL2 box under mirrored networking, whose address is not
reserved and goes stale on a lease change.

The name is read-only on an edit, and refused by the daemon. It is stricter
than the lane rule: every lane's `encoders` allowlist names encoders by name,
so a rename would leave each of those pointing at a host that no longer
exists. **remove** is refused for the same reason while any lane still routes
there — clear that routing first. Removing the last encoder is refused too,
since a roster with no pool cannot run.

The table is hidden on a legacy `[encode]` roster. That format has no
`[[encoder]]` blocks — the loader synthesizes one named `local` from the table —
and there is no block to write, so a switch there would answer 404.

### Adding and editing a lane

"Add a lane" opens a form that writes one `[[denoiser]]` block into the roster.
The block always arrives with `enabled = false`, because adding a lane does not
test it.

Each row's **edit** button opens the same form filled from that lane, and saving
posts `POST /api/lane/<name>`. Only the values change: the block's spacing, its
aligned `=` column and any comment inside it survive, and no other lane is
touched. A change takes effect at the next clip, exactly like the enable switch
— nothing in flight is killed. Clearing a field removes the key, which is how a
lane goes back to a default; the roster validator still has the last word, so an
edit that would not load is refused with its message and nothing is written.

The name is read-only. It keys the worker's heartbeat at
`.archive-run/lanes/<name>.json`, dispatch's `--temp-tag` and the scheduler's
in-flight bookkeeping, so a rename mid-run would orphan all three while the lane
kept working. To rename, remove the lane and add it again.

When adding, the form starts with a **known lane** picker. Choosing a host fills every field
from `tools/encode_dash/static/lane-presets.json`, a catalogue of the lanes this
fleet has actually benchmarked — backend, device, tiling, window, margin,
checkout path — and shows the measurement the settings come from. The fields
stay editable, so a preset is a starting point rather than a decision. The
picker is hidden if the catalogue cannot be read, and the form still takes a
hand-typed lane.

Benchmarked a new host? Add it to that file. Each entry needs `id`, `label`,
`note`, `source` and a `fields` object whose keys are the `Denoiser` fields.
`tests/test_lane_presets.py` puts every entry through the real roster writer and
the real validator, singly and all together, so a preset that names an unknown
key or omits a window on a tiled card fails the suite rather than the operator.
A preset cannot carry `port`: the y4m listener lives on the encode host now, so
a port belongs to an `[[encoder]]` entry as `port_base`, and the loader refuses
any `[[denoiser]]` that sets one.

History and alerting live in the Prometheus and Grafana on the Pi rather than in
this page, which is why it carries no charts. Design and the part 2 plan:
the design notes, which are not part of this tree.

## Reference docs

Longer write-ups live in `docs/`:

| Doc | Covers |
|-----|--------|
| [lp-and-encoder-parallelism.md](docs/lp-and-encoder-parallelism.md) | `--lp` is a level in [0, 6], not a thread count. Measured fps/memory per level, what `--lp 0` picks from the core count, and how many encoder slots to run |
| [split-host-denoise.md](docs/split-host-denoise.md) | Rationale and measurements behind `--remote-denoise` |
| [encode-pool-gates.md](docs/encode-pool-gates.md) | What the remote encode pool was verified to do on real hardware: the eight gate results, measured pool throughput and y4m bandwidth, what a hard kill leaves behind, and what is still open |
| [vapoursynth-isolation.md](docs/vapoursynth-isolation.md) | How the `/opt/archav1an` prefix keeps its VapourSynth from colliding with the distro's |
| [framebuffer-warning.md](docs/framebuffer-warning.md) | The VapourSynth "framebuffer" message at the end of a run, and why it is not a leak |

## Troubleshooting

-   **Missing Tools**: Ensure `av1an`, `SvtAv1EncApp`, `ffmpeg`, `mkvmerge`, `mkvpropedit` are in your PATH.
-   **VapourSynth Errors**: Ensure you have the required plugins (`ffms2`) installed and accessible to VapourSynth.
-   **Permissions**: Ensure you have write permissions in the folder.

## Attribution

This work derives from **Auto-Boost-Av1an-Linux** by
[abdalrahmanx9](https://github.com/abdalrahmanx9/Auto-Boost-Av1an-Linux), and
carries contributions from that project's history by LastBreeze and Line. The
scene-detection and progression-boost scripts under `tools/` come from that
lineage.

The upstream project declares no licence, so no licence is claimed or granted
here either. Treat this as published for reference and discussion. If you want
to reuse any of it, ask the upstream author first.

## Notes

**Host names in this repository are examples.** `encoder-host`, `gpu1`, `gpu2`
and so on are role names standing in for whatever machines you run on. Put
your own SSH aliases in the roster and the example configuration.

**Model weights are distributed separately** and are not in the tree. The
pipeline expects them under `models/`; see the setup scripts for the paths it
looks in.

**The batch tooling's example corpus is fictional.** `SetA` and `SetB`, the
folder names under them, the years and the file counts are placeholders. The
batch reads whatever tree you point `ARCHIVE_ROOT` at; only the two-level
`<set>/<year>/<folder>` shape matters, and only to the ordering rule in
`tools/archive_batch/manifest.py`.

**Design notes are not published.** Several comments refer to them; they
discuss the private fleet the code was written for.
