# Auto-Boost-Av1an for Linux

This guide explains how to set up and run Auto-Boost-Av1an on Linux (Arch-based distros like CachyOS, and Ubuntu/Debian).

---

## Which Script Should I Choose?

Pick based on your content type and desired quality:

### 🎌 ANIME
| Script | Quality | Description |
|--------|---------|-------------|
| `run_linux_anime_crf30.sh` | **Standard** | ✅ Recommended starting point |
| `run_linux_anime_crf25.sh` | High | Higher quality, larger files |
| `run_linux_anime_crf18.sh` | Archival | Maximum quality, largest files |

### 🎬 LIVE ACTION / MOVIES / TV SHOWS
| Script | Quality | Description |
|--------|---------|-------------|
| `run_linux_live_crf30.sh` | **Standard** | ✅ Recommended starting point |
| `run_linux_live_crf25.sh` | High | Higher quality, larger files |
| `run_linux_live_crf18.sh` | Archival | Maximum quality, largest files |

### ⚽ SPORTS / FAST MOTION
| Script | Quality | Description |
|--------|---------|-------------|
| `run_linux_sports_crf33.sh` | Optimized | ✅ Best for high-motion content |

### 🎞️ DIRECT ENCODE (SINGLE PASS, NO BOOST)
| Script | Quality | Description |
|--------|---------|-------------|
| `av1an-batch-anime-crf30.sh` | Standard | Single-pass anime encode, no Auto-Boost |
| `av1an-batch-liveaction-crf30.sh` | Standard | Single-pass live action encode, no Auto-Boost |

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

**One-time bootstrap (the only step requiring sudo):**
```bash
sudo install -d -o "$USER" -g "$USER" /opt/archav1an
```

This creates the prefix and chowns it to your user. Every subsequent install step runs as your user — no `sudo` required.

**Prerequisite: `uv`** (Python venv + pip replacement). Install once:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install everything:**
```bash
chmod +x setup.sh
./setup.sh --install A
```

Or selectively:
```bash
./setup.sh --install python_libs   # uv venv at /opt/archav1an/venv
./setup.sh --install ffmpeg        # source-built ffmpeg w/ NVENC into prefix
./setup.sh --install vapoursynth   # VS R73 + FFMS2 + BestSource
./setup.sh --install denoiser      # SCUNet/SMDegrain/RVRT/STA-SUNet plugins
./setup.sh --install wwxd vszip subtext  # core VS plugins
```

The setup runs `pacman -Q <pkg>` before any `pacman -S` so already-installed system packages skip cleanly. Packages it genuinely needs to install (most often `cudnn`, `tensorrt`, AUR `vapoursynth-plugin-ctmf-git`) will fail with a clear message telling you the exact `sudo pacman -S ...` command to run — install those manually and re-run.

**Activate the env to use vspipe / Python tools from your shell:**
```bash
source activate-venv.sh
```

`activate-venv.sh` sources the venv, prepends `/opt/archav1an/bin` to PATH, sets `LD_LIBRARY_PATH=/opt/archav1an/lib` (the source-built R73 VapourSynth library deliberately has no SONAME so it's only visible inside this activated env — pacman's system v75 remains the global default outside it), and sets `VAPOURSYNTH_PLUGIN_PATH=/opt/archav1an/lib/vapoursynth`.

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
vspipe --version    # should report "Core R73"
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

### 3. Usage

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
*   **Standard (CRF 30)**: `./run_linux_anime_crf30.sh` - Balanced speed/quality (Tune 3).
*   **High (CRF 25)**: `./run_linux_anime_crf25.sh` - Slower, Tune 0.
*   **Highest (CRF 18)**: `./run_linux_anime_crf18.sh` - Aggressive boosting.

**Live Action Variants (Auto-Crop Enabled):**
*   **Standard (CRF 30)**: `./run_linux_live_crf30.sh` - Auto-crop, Tune 3.
*   **High (CRF 25)**: `./run_linux_live_crf25.sh` - Tune 3, Variance Boost 2.
*   **Highest (CRF 18)**: `./run_linux_live_crf18.sh` - Maximum fidelity.

**Sports / High-Motion Content:**
*   **Low Quality (CRF 33)**: `./run_linux_sports_crf33.sh` - Optimized for high-motion content with extra temporal filtering.

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
| `encode-opus-audio.sh` | Legacy audio encoding (use `audio-encoding/` instead) |
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

## Troubleshooting

-   **Missing Tools**: Ensure `av1an`, `SvtAv1EncApp`, `ffmpeg`, `mkvmerge`, `mkvpropedit` are in your PATH.
-   **VapourSynth Errors**: Ensure you have the required plugins (`ffms2`) installed and accessible to VapourSynth.
-   **Permissions**: Ensure you have write permissions in the folder.