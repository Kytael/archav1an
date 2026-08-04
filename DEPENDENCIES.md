# Auto-Boost-Av1an Dependencies (Linux)

The `setup.sh` script builds and installs everything into a single isolated prefix at `/opt/archav1an/` (overridable via `VS_PREFIX`). pacman/AUR system packages stay where they are; nothing archav1an writes lands in `/usr/local` or pacman-owned paths.

## Layout under `$VS_PREFIX` (default `/opt/archav1an`)

| Path | Contents |
| :--- | :--- |
| `bin/` | `vspipe`, `ffmpeg`, `ffprobe`, `ffplay`, `SvtAv1EncApp`, `av1an`, `oxipng`, `FFVship`, `dav1d`, `ffmsindex` |
| `lib/libvapoursynth.so.4`, `libvapoursynth-script.so` | R76 core (SONAME `libvapoursynth.so.4`; wins over pacman's v75 via `LD_LIBRARY_PATH` set by `activate-venv.sh`) |
| `lib/vapoursynth/*.so` | VS plugins (see table below) |
| `lib/python3.X/site-packages/vapoursynth/` | R76 Python module (versioned per interpreter, currently 3.14), wired into the venv via `_vapoursynth_native.pth` |
| `lib/python3/site-packages/` (deprecated) | (Empty placeholder; the actual site-packages is versioned per-Python.) |
| `include/{vapoursynth,ffms2,libav*,libsw*,dav1d,svt-av1}/` | Headers for downstream plugin builds |
| `lib/pkgconfig/*.pc` | pkg-config metadata for the prefix |
| `venv/` | uv-managed Python venv (default: latest `python3` on PATH; override with `PYTHON_VERSION=<x.y>`) |

## Core Tools (source-built into `$VS_PREFIX`)

| Software | Version | Source |
| :--- | :--- | :--- |
| **VapourSynth** | **R76 (pinned)** | [vapoursynth/vapoursynth](https://github.com/vapoursynth/vapoursynth) — meson build, installed under `$VS_PREFIX/lib/python<X.Y>/site-packages/vapoursynth/` (R74+ ships everything as a Python package) and bridged into the traditional `bin/lib/include` layout via symlinks |
| **FFMS2** | tag `5.0` | [FFMS/ffms2](https://github.com/FFMS/ffms2) |
| **BestSource** | latest git (master) | [vapoursynth/bestsource](https://github.com/vapoursynth/bestsource) (meson+ninja, native opts) |
| **FFmpeg** | latest git (master) | [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) — Clang + PGO + LTO + NVENC/NVDEC/CUDA (auto-detected) |
| **nv-codec-headers** | latest git (master) | [FFmpeg/nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers) — required for FFmpeg NVENC/NVDEC |
| **SVT-AV1-PSY** | tag `v2.3.0-C` | [5fish/svt-av1-psy](https://github.com/5fish/svt-av1-psy) — Clang, PGO, LTO, AVX512, NATIVE |
| **dav1d** | tag `1.5.3` | [VideoLAN/dav1d](https://code.videolan.org/videolan/dav1d) — required for FFmpeg `--enable-libdav1d` |
| **Av1an** | latest git (master) | [rust-av/Av1an](https://github.com/rust-av/Av1an) |
| **oxipng** | latest crates.io | `cargo install oxipng` |
| **fssimu2** | tag `0.1.3` | [gianni-rosato/fssimu2](https://github.com/gianni-rosato/fssimu2) — Zig 0.15.2 build |
| **vship / FFVship** | tag `v5.0.1` | [Line-fr/Vship](https://codeberg.org/Line-fr/Vship) — GPU SSIMU2 (CUDA/Vulkan; HIP discouraged by upstream) |

Plugin tags (built into `$VS_PREFIX/lib/vapoursynth/`):

| Plugin | Version |
| :--- | :--- |
| WWXD | tag `v1.0` (only tag) |
| VSZIP | tag `R13` |
| SubText | tag `R6` |

**Pin policy:** dependencies that ship release tags are pinned to a known-good tag; tag-less projects (FFmpeg, BestSource, nv-codec-headers, Av1an) track master. To bump a pin, edit the `--branch <tag>` arg in the relevant `setup/*.sh` file and re-run `./setup.sh --install <component>`.

## System packages (pacman / paru)

These are installed via the host package manager. `setup.sh` runs `pacman -Q <pkg>` first and only invokes `pacman -S` (which requires sudo) when something's missing.

**Required on every install:**
- `mkvtoolnix-cli` / `mkvtoolnix-gui` (mkvmerge/mkvpropedit) — Arch `pacman -S mkvtoolnix-cli mkvtoolnix-gui` / Debian `apt install mkvtoolnix`
- `x264` (scene-detection pass) — `pacman -S x264` / `apt install x264`
- `opus-tools` — `pacman -S opus-tools` / `apt install opus-tools`
- `xclip` (clipboard) — `pacman -S xclip` / `apt install xclip`
- `mediainfo` (BT.709 auto-detection)
- `python` 3.13+ on the system (uv will download Python 3.13 anyway if pacman is on 3.14, see [Python](#python))

**NVIDIA path (any TensorRT/CUDA denoising):**
- `cudnn`, `tensorrt` (AUR), `opencl-icd-loader` (or `ocl-icd`)
- CUDA toolkit at `/opt/cuda` (pacman's `cuda` package) or `/usr/local/cuda` (NVIDIA installer)

**SMDegrain / ContraSharpening (`--denoise-smdegrain` flag):**
- `vapoursynth-plugin-mvtools`
- `vapoursynth-plugin-removegrain-git` (AUR)
- `vapoursynth-plugin-ctmf-git` (AUR) — for ContraSharpening; without it, SMDegrain runs but without sharpening
- `boost` (only when building bm3d from source)

**AMD ROCm path (alternative to TensorRT):**
- `rocm-migraphx` or `migraphx`

**Split-host denoise (`--remote-denoise` flag):**
- `openssh` — client on the encoder host, server on the denoise host; key-based auth (the dispatcher must not hit a password prompt)
- `rsync` — both hosts; stages the source to `<remote-root>/Temp/_remote/`
- The denoise host needs this repo checked out at `--remote-root` with `./setup.sh --install denoiser` run there, and `models/*.onnx` present (git-tracked, so a clone is enough)
- One inbound TCP port on the encoder host (default 5300). Setup does not touch firewalls; open it yourself, scoped to the denoise host
- `tools/netstream.py` carries the y4m stream and is stdlib-only — no extra Python packages

## VapourSynth plugins (built into `$VS_PREFIX/lib/vapoursynth/`)

| Plugin | Source | Notes |
| :--- | :--- | :--- |
| **FFMS2** | (above) | Symlinked from `$VS_PREFIX/lib/libffms2.so` |
| **BestSource** | (above) | Symlinked from the meson install location |
| **WWXD** | [dubhater/vapoursynth-wwxd](https://github.com/dubhater/vapoursynth-wwxd) | Scene detection; linked with `-lm` |
| **VSZIP** | [dnjulek/vapoursynth-zip](https://github.com/dnjulek/vapoursynth-zip) | SSIMULACRA2/XPSNR metrics; Zig 0.15.2 build |
| **SubText** | [vapoursynth/subtext](https://github.com/vapoursynth/subtext) | Subtitles |
| **vs-mlrt (vstrt)** | [AmusementClub/vs-mlrt](https://github.com/AmusementClub/vs-mlrt) | TensorRT backend for SCUNet/STA-SUNet (NVIDIA) |
| **vs-mlrt (vsmigx)** | (same repo) | MIGraphX backend (AMD ROCm) |
| **KNLMeansCL** | [Khanattila/KNLMeansCL](https://github.com/Khanattila/KNLMeansCL) | OpenCL spatial+temporal denoise |
| **mvtools** (symlink) | pacman | Symlinked from `/usr/lib/vapoursynth/libmvtools.so` |
| **removegrain** (symlink) | pacman/AUR | Symlinked from `/usr/lib/vapoursynth/libremovegrain.so` |
| **ctmf** (symlink) | AUR | Optional, for SMDegrain ContraSharpening |

The pacman v75 vapoursynth library coexists with our R76 build by both having SONAME `libvapoursynth.so.4` (the v74+ ABI). `activate-venv.sh` sets `LD_LIBRARY_PATH=$VS_PREFIX/lib`, which the dynamic linker searches *before* the ldconfig cache — so inside the activated env you get R76, outside the env you get pacman's v75. The bridge deliberately skips creating `$VS_PREFIX/lib/libvsscript.so`: `libvsscript` uses `dladdr()` to find itself and looks the result up in `~/.config/vapoursynth/vapoursynth.toml`; a symlink would be loaded via `LD_LIBRARY_PATH` first, `dladdr` would return the symlink path, and the toml lookup (keyed by the real path) would miss.

## Python <a id="python"></a>

The venv at `$VS_PREFIX/venv` is created and managed by **uv** (not pip + `python -m venv`).

- Default Python: whatever `python3` resolves to on PATH (typically the latest pacman version).
- Override: `PYTHON_VERSION=3.13 ./setup.sh --install python_libs` — uv downloads the specified interpreter if missing. Useful when a newer Python breaks a binary dep (e.g. PyO3-based packages typically lag a release behind).
- Upgrade detection: if you set `PYTHON_VERSION` to something different from what's already in the venv, `install_python_libs` warns and rebuilds the venv; you must rerun `--install vapoursynth` after because the source-built VS module is binary-linked to the venv's interpreter.
- The pip-published `vapoursynth` stub package is removed from the venv on every install, otherwise it would shadow the source-built R76 module.

### Python packages

Installed via `VIRTUAL_ENV=$VENV_DIR uv pip install ...`:

- `vsjetpack` (bundle: `vstools`, `vsdenoise`, etc.)
- `vstools`
- `numpy`, `Cython`, `psutil`
- `rich`, `colorama`, `natsort`, `anitopy`, `pyperclip`
- `requests`, `requests_toolbelt`
- `vsscunet`, `onnx`, `onnxscript`, `adjust` (SCUNet denoiser stack)
- `vsrvrt` (RVRT denoiser; installed with `--no-deps` to avoid pulling the PyPI vapoursynth stub)
- `mvsfunc_pkg` (havsfunc dependency; installed manually from GitHub)
- `havsfunc_legacy` (r33 tag, patched for VS R73+ kwarg quirks)
- `onnxruntime-gpu` — ONNX Runtime with CUDA + TensorRT EPs, used by the BSVD V2 stateful streaming path (`--denoise-bsvd` / `--denoise-bsvd-smdegrain` in `tools/svtav1-dispatch.py`). On AMD/encoder-host, replace with the encoder-host ORT-ROCm from-source build (memory `encoder-host_ort_rocm_build.md`).
- `av` (PyAV) — used by `tools/bsvd_optsig.py` to decode the warmup window for the V3 optimal-σ predictor.

## Build tools

- **uv** — required (Python venv manager + pip replacement). Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Rust** (stable, via rustup or pacman) — for Av1an and FFVship.
- **Zig** (latest stable, fetched automatically by VSZIP's `build.sh`).
- **CMake, Ninja, Meson** (auto-installed by setup if missing on Arch).
- **clang/clang++**, **NASM/YASM**, **autotools** — standard build essentials.
- **Git** — obvious.

## First-time bootstrap

```bash
# One-time: create the prefix and chown to your user
sudo install -d -o "$USER" -g "$USER" /opt/archav1an

# Install components (no sudo needed after the prefix exists)
./setup.sh --install A          # everything
./setup.sh --install vapoursynth  # just VS + FFMS2 + BestSource
./setup.sh --install denoiser     # plugins + denoise stack

# Activate the env when you want to run vspipe / vsmlrt / etc. from your shell
source activate-venv.sh
```

`setup.sh` allows non-root invocation when the user has write access to `$VS_PREFIX`. Individual install scripts that genuinely need root (e.g. `pacman -S` for system packages) will fail with a clear error suggesting the manual sudo command.
