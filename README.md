# Auto-Boost-Av1an for Linux

This guide explains how to set up and run Auto-Boost-Av1an on Linux (Arch-based distros like CachyOS, and Ubuntu/Debian).

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

`setup.sh` prompts for sudo **once** at the start to create `/opt/archav1an/` and chown it to you; everything after that runs as your user. If `uv` (the Python venv/pip replacement) isn't installed, the script fetches Astral's official installer and drops it in `~/.local/bin`. No further sudo unless a system package is genuinely missing — in that case the script prints the exact `sudo pacman -S …` command and exits.

Or selectively:
```bash
sudo ./setup.sh --install system_deps    # distro packages (the only step needing root)
./setup.sh --install python_libs   # uv venv at /opt/archav1an/venv
./setup.sh --install ffmpeg        # source-built ffmpeg w/ NVENC into prefix
./setup.sh --install vapoursynth   # VS R76 + FFMS2 + BestSource
./setup.sh --install denoiser      # BSVD/SCUNet/SMDegrain/RVRT/STA-SUNet plugins + models
./setup.sh --install wwxd vszip subtext  # core VS plugins
```

The full target list is `system_deps python_libs svt_av1 ffmpeg vapoursynth av1an ffvship oxipng fssimu2 wwxd vszip subtext denoiser`. Dependencies resolve automatically, so naming a late component pulls in what it needs.

`system_deps` is the one target that installs distro packages, and it is what the other components tell you to run when something is missing. On Arch the setup runs `pacman -Q <pkg>` before any `pacman -S`, so already-installed packages skip cleanly; anything it genuinely needs (most often `cudnn`, `tensorrt`, AUR `vapoursynth-plugin-ctmf-git`) fails with the exact `sudo pacman -S …` command to run. On Debian/Ubuntu it checks with `dpkg -s` and installs the missing set through apt, printing `sudo apt install -y …` if it was not started as root.

**Activate the env to use vspipe / Python tools from your shell:**
```bash
source activate-venv.sh
```

`activate-venv.sh` sources the venv, prepends `/opt/archav1an/bin` to PATH, sets `LD_LIBRARY_PATH=/opt/archav1an/lib` (the source-built R76 VapourSynth shares pacman v75's SONAME `libvapoursynth.so.4`, so LD_LIBRARY_PATH precedence makes R76 win only inside this activated env — pacman's v75 remains the global default outside it), and sets `VAPOURSYNTH_EXTRA_PLUGIN_PATH=/opt/archav1an/lib/vapoursynth`.

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
vspipe --version    # should report "Core R76"
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

## Reference docs

Longer write-ups live in `docs/`:

| Doc | Covers |
|-----|--------|
| [lp-and-encoder-parallelism.md](docs/lp-and-encoder-parallelism.md) | `--lp` is a level in [0, 6], not a thread count. Measured fps/memory per level, what `--lp 0` picks from the core count, and how many encoder slots to run |
| [split-host-denoise.md](docs/split-host-denoise.md) | Rationale and measurements behind `--remote-denoise` |
| [vapoursynth-isolation.md](docs/vapoursynth-isolation.md) | How the `/opt/archav1an` prefix keeps its VapourSynth from colliding with the distro's |
| [framebuffer-warning.md](docs/framebuffer-warning.md) | The VapourSynth "framebuffer" message at the end of a run, and why it is not a leak |

## Troubleshooting

-   **Missing Tools**: Ensure `av1an`, `SvtAv1EncApp`, `ffmpeg`, `mkvmerge`, `mkvpropedit` are in your PATH.
-   **VapourSynth Errors**: Ensure you have the required plugins (`ffms2`) installed and accessible to VapourSynth.
-   **Permissions**: Ensure you have write permissions in the folder.