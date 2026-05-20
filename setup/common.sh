#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Ensure user-installed tools (uv, cargo, etc.) are visible even when the
# script was invoked via sudo (which strips $PATH) or from a fresh
# non-login shell that didn't source the user's fish/bash rc. Probe
# common install locations under the invoking user's home and prepend
# whichever exist.
_invoking_user_home="${SUDO_USER:+/home/$SUDO_USER}"
_invoking_user_home="${_invoking_user_home:-$HOME}"
for _extra_bin in "$_invoking_user_home/.local/bin" "$_invoking_user_home/.cargo/bin"; do
    if [ -d "$_extra_bin" ] && [[ ":$PATH:" != *":$_extra_bin:"* ]]; then
        export PATH="$_extra_bin:$PATH"
    fi
done
unset _invoking_user_home _extra_bin

# Install uv (Python venv + pip replacement) into ~/.local/bin if missing.
# Uses Astral's official installer; no sudo required.
ensure_uv() {
    if command -v uv &>/dev/null; then
        return 0
    fi
    log_info "uv not found; installing via https://astral.sh/uv/install.sh..."
    if ! command -v curl &>/dev/null; then
        log_error "curl is required to bootstrap uv. Install curl (e.g. 'sudo pacman -S curl' or 'sudo apt install curl') and re-run."
        return 1
    fi
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        log_error "uv installer failed. Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi
    # The installer writes to $HOME/.local/bin (or $XDG_BIN_HOME). Our PATH
    # block above already includes ~/.local/bin, so a fresh probe should work.
    if ! command -v uv &>/dev/null; then
        # Last-ditch: source the installer's env file if it created one.
        [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if command -v uv &>/dev/null; then
        log_success "uv installed: $(uv --version)"
        return 0
    fi
    log_error "uv installed but still not on PATH. Open a new shell or check ~/.local/bin."
    return 1
}

check_root() {
    # If the user owns $VS_PREFIX (set in this file above), they don't need
    # sudo for builds that land inside the prefix. Individual install tasks
    # that genuinely need root (e.g. install_system_deps invoking pacman)
    # will fail with a clear permission error if invoked without sudo.
    if [ "$EUID" -eq 0 ]; then
        return 0
    fi
    if [ -n "$VS_PREFIX" ] && [ -w "$VS_PREFIX" ]; then
        log_info "Running as $USER with write access to $VS_PREFIX (sudo not required for user-prefix builds)."
        return 0
    fi
    # Prefix doesn't exist (or isn't writable by us). The only step that
    # needs root is creating it with the right owner; do that here so
    # plain `./setup.sh --install A` works as the first command.
    log_info "Bootstrapping $VS_PREFIX (one-time sudo prompt; the rest of setup runs as $USER)..."
    if sudo install -d -o "$USER" -g "$USER" "$VS_PREFIX"; then
        log_success "Created $VS_PREFIX owned by $USER."
        return 0
    fi
    log_error "Failed to create $VS_PREFIX. Create it manually: sudo install -d -o \$USER -g \$USER $VS_PREFIX"
    exit 1
}

check_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_LIKE=$ID_LIKE
    else
        log_error "Cannot detect Linux distribution."
        exit 1
    fi

    log_info "Detected Distribution: $DISTRO"

    if command -v pacman &> /dev/null; then
        DISTRO_FAMILY="arch"
        log_success "Detected Arch-based system ($DISTRO)."
    elif [ -f /etc/debian_version ]; then
        DISTRO_FAMILY="debian"
        log_success "Detected Debian/Ubuntu-based system."
    else
        log_error "Error: This script supports Arch-based (CachyOS, Manjaro, etc.) and Debian/Ubuntu systems."
        log_info "Detected: $DISTRO"
        exit 1
    fi

    export DISTRO_FAMILY
}

# Helper: get the Python site-packages directory dynamically
get_python_site_packages() {
    python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || {
        if [ "$DISTRO_FAMILY" = "arch" ]; then
            python3 -c "import sys; print(f'/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')" 2>/dev/null || echo "/usr/lib/python3/site-packages"
        else
            echo "/usr/lib/python3/dist-packages"
        fi
    }
}

# Install prefix for the source-built native stack (VapourSynth + plugins +
# ffmpeg + SVT-AV1 + venv). Chosen under /opt so pacman never owns anything
# inside it; deliberately *not* /usr/local. See plans/2026-05-20-vapoursynth-isolation.md.
VS_PREFIX="${VS_PREFIX:-/opt/archav1an}"
export VS_PREFIX

# Helper: get the VapourSynth plugin path. Single source of truth: $VS_PREFIX.
get_vs_plugin_path() {
    echo "$VS_PREFIX/lib/vapoursynth"
}

# Virtual environment path for Python dependencies
VENV_DIR="${VENV_DIR:-$VS_PREFIX/venv}"
export VENV_DIR

# Set native build optimization flags for all source builds
set_native_build_flags() {
    export CC="clang"
    export CXX="clang++"
    export CFLAGS="-march=native -O3 -flto"
    export CXXFLAGS="-march=native -O3 -flto"
    export LDFLAGS="-flto -fuse-ld=lld"
    export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
    export PKG_CONFIG_PATH="$VS_PREFIX/lib/pkgconfig:/usr/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export LD_LIBRARY_PATH="$VS_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$VS_PREFIX/lib:${LIBRARY_PATH:-}"
}

# Set up build_tmp on tmpfs (RAM) for faster compilation
# Falls back to disk if tmpfs mount fails (e.g., not enough RAM or no permissions)
setup_build_tmpfs() {
    local build_dir="${1:-build_tmp}"
    mkdir -p "$build_dir"
    if mountpoint -q "$build_dir" 2>/dev/null; then
        log_info "build_tmp is already a tmpfs mount."
        return 0
    fi
    local ram_gb=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
    if [ "$ram_gb" -ge 16 ]; then
        local tmpfs_size="8G"
        if [ "$ram_gb" -ge 64 ]; then
            tmpfs_size="16G"
        fi
        if mount -t tmpfs -o size="$tmpfs_size" tmpfs "$build_dir" 2>/dev/null; then
            log_info "build_tmp mounted as tmpfs (${tmpfs_size} RAM disk)."
            return 0
        fi
    fi
    log_info "Using disk-backed build_tmp (tmpfs unavailable or not enough RAM)."
    return 0
}

# Helper: detect GPU vendor and set up HIP environment for AMD GPUs
# Sets GPU_VENDOR to "amd", "nvidia", or "unknown"
# For AMD GPUs, detects the gfx target and sets HSA_OVERRIDE_GFX_VERSION
detect_gpu() {
    GPU_VENDOR="unknown"
    GPU_GFX_TARGET=""

    # Check for AMD GPU via lspci or /sys
    if lspci 2>/dev/null | grep -qi "VGA.*AMD\|Display.*AMD\|3D.*AMD"; then
        GPU_VENDOR="amd"
    elif [ -d "/sys/class/drm" ]; then
        for card in /sys/class/drm/card*/device/vendor; do
            if [ -f "$card" ] && [ "$(cat "$card")" = "0x1002" ]; then
                GPU_VENDOR="amd"
                break
            fi
        done
    fi

    # Check for NVIDIA GPU (lspci, nvidia-smi, or CUDA libs for WSL2)
    if lspci 2>/dev/null | grep -qi "VGA.*NVIDIA\|Display.*NVIDIA\|3D.*NVIDIA" || \
       nvidia-smi &>/dev/null || \
       /usr/lib/wsl/lib/nvidia-smi &>/dev/null || \
       [ -f /usr/lib/wsl/lib/libcuda.so ] || \
       [ -f /opt/cuda/bin/nvcc ]; then
        if [ "$GPU_VENDOR" = "amd" ]; then
            GPU_VENDOR="both"
        else
            GPU_VENDOR="nvidia"
        fi
    fi

    # For AMD GPUs, detect gfx target and set HSA_OVERRIDE_GFX_VERSION
    if [ "$GPU_VENDOR" = "amd" ] || [ "$GPU_VENDOR" = "both" ]; then
        # Try to get gfx target from ROCm agent info
        if command -v rocminfo &> /dev/null; then
            GPU_GFX_TARGET=$(rocminfo 2>/dev/null | grep -oP 'gfx[0-9a-f]+' | head -1)
        fi

        # Fallback: check amdgpu kernel driver ip_discovery
        if [ -z "$GPU_GFX_TARGET" ]; then
            local ip_major ip_minor ip_rev
            ip_major=$(cat /sys/class/drm/card*/device/ip_discovery/die/*/GC/*/major 2>/dev/null | head -1)
            ip_minor=$(cat /sys/class/drm/card*/device/ip_discovery/die/*/GC/*/minor 2>/dev/null | head -1)
            ip_rev=$(cat /sys/class/drm/card*/device/ip_discovery/die/*/GC/*/revision 2>/dev/null | head -1)
            if [ -n "$ip_major" ] && [ -n "$ip_minor" ] && [ -n "$ip_rev" ]; then
                GPU_GFX_TARGET="gfx${ip_major}${ip_minor}${ip_rev}"
            fi
        fi

        # Fallback: parse dmesg for gfx target
        if [ -z "$GPU_GFX_TARGET" ]; then
            GPU_GFX_TARGET=$(dmesg 2>/dev/null | grep -oP 'gfx[0-9a-f]+' | tail -1)
        fi

        # Set HSA_OVERRIDE_GFX_VERSION based on detected target
        # gfx format: gfxMAJORMINORREV (e.g., gfx900=9.0.0, gfx1030=10.3.0, gfx1100=11.0.0, gfx1151=11.5.1)
        if [ -n "$GPU_GFX_TARGET" ]; then
            local gfx_num="${GPU_GFX_TARGET#gfx}"
            local num_len=${#gfx_num}
            if [ "$num_len" -ge 3 ]; then
                local rev="${gfx_num: -1}"
                local minor="${gfx_num: -2:1}"
                local major="${gfx_num:0:$((num_len-2))}"
                export HSA_OVERRIDE_GFX_VERSION="${major}.${minor}.${rev}"
                log_info "AMD GPU detected: $GPU_GFX_TARGET (HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION})"
            else
                log_warn "AMD GPU detected ($GPU_GFX_TARGET) but could not parse version."
            fi
        else
            log_warn "AMD GPU detected but could not determine gfx target."
            log_warn "You may need to set HSA_OVERRIDE_GFX_VERSION manually."
        fi
    fi

    if [ "$GPU_VENDOR" = "nvidia" ] || [ "$GPU_VENDOR" = "both" ]; then
        if [ -d "/opt/cuda/bin" ]; then
            export PATH="/opt/cuda/bin:$PATH"
            if [ ! -f /etc/profile.d/cuda.sh ]; then
                echo 'export PATH="/opt/cuda/bin:$PATH"' > /etc/profile.d/cuda.sh
                log_info "Added /opt/cuda/bin to system PATH via /etc/profile.d/cuda.sh"
            fi
        fi
    fi

    if [ "$GPU_VENDOR" = "nvidia" ]; then
        log_info "NVIDIA GPU detected."
    elif [ "$GPU_VENDOR" = "amd" ]; then
        log_info "AMD GPU detected."
    elif [ "$GPU_VENDOR" = "both" ]; then
        log_info "Both AMD and NVIDIA GPUs detected."
    else
        log_warn "No supported GPU detected."
    fi

    export GPU_VENDOR GPU_GFX_TARGET
}

# Helper: detect if running under WSL2
is_wsl2() {
    uname -r | grep -qi microsoft
}

# WSL2 CUDA fix: the NVIDIA stubs in /usr/lib/wsl/lib (libcuda.so.1) have malformed
# ELF hash tables that crash glibc's ld.so, breaking nvcc and CUDA runtime init.
# Disables WSL's auto-ldconfig and creates clean symlinks at /usr/local/lib/wsl-cuda.
setup_wsl2_cuda() {
    if ! is_wsl2; then
        return 0
    fi
    if [ -f /etc/ld.so.conf.d/ld.wsl.conf ]; then
        log_info "Disabling WSL2 auto-ldconfig (broken NVIDIA stubs crash build tools)..."
        if ! grep -q "ldconfig = false" /etc/wsl.conf 2>/dev/null; then
            if grep -q "\[automount\]" /etc/wsl.conf 2>/dev/null; then
                sed -i "/\[automount\]/a ldconfig = false" /etc/wsl.conf
            else
                printf '\n[automount]\nldconfig = false\n' >> /etc/wsl.conf
            fi
        fi
        rm -f /etc/ld.so.conf.d/ld.wsl.conf
        ldconfig
        log_success "WSL2 ldconfig fixed."
    fi

    # Ensure WSL2 libcuda takes priority over nvidia-utils stub in /usr/lib.
    # nvidia-utils installs libcuda.so.595.x (native Linux stub) which doesn't work in WSL2.
    # Putting /usr/lib/wsl/lib first in ldconfig makes the real WSL2 driver win.
    if [ ! -f /etc/ld.so.conf.d/00-wsl2-cuda.conf ]; then
        echo "/usr/lib/wsl/lib" > /etc/ld.so.conf.d/00-wsl2-cuda.conf
        ldconfig
        log_info "WSL2 CUDA ldconfig priority set (/usr/lib/wsl/lib before /usr/lib)."
    fi

    # Clean CUDA symlinks: .so.1 → .so bypasses the malformed hash table in .so.1
    local wsl_cuda_dir="/usr/local/lib/wsl-cuda"
    if [ -f /usr/lib/wsl/lib/libcuda.so ] && [ ! -d "$wsl_cuda_dir" ]; then
        log_info "Creating clean CUDA symlinks for WSL2..."
        mkdir -p "$wsl_cuda_dir"
        ln -sf /usr/lib/wsl/lib/libcuda.so "$wsl_cuda_dir/libcuda.so"
        ln -sf /usr/lib/wsl/lib/libcuda.so "$wsl_cuda_dir/libcuda.so.1"
        ln -sf /usr/lib/wsl/lib/libcuda.so "$wsl_cuda_dir/libcuda.so.1.1"
        for f in /usr/lib/wsl/lib/libnvcuvid* /usr/lib/wsl/lib/libnvidia-encode* \
                 /usr/lib/wsl/lib/libnvidia-gpucomp* /usr/lib/wsl/lib/libdxcore* \
                 /usr/lib/wsl/lib/libd3d12*; do
            [ -f "$f" ] && ln -sf "$f" "$wsl_cuda_dir/$(basename "$f")"
        done
        log_success "Clean CUDA symlinks created at $wsl_cuda_dir"
    fi
}

AUTO_YES=false

ask_yes_no() {
    local prompt="$1"
    local default="$2" # Y or N
    stty sane 2>/dev/null

    if [ "$AUTO_YES" = true ]; then
        echo "$prompt [auto-yes]"
        return 0
    fi

    local yn_prompt="[y/n]"
    if [ "$default" == "Y" ]; then yn_prompt="[Y/n]"; fi
    if [ "$default" == "N" ]; then yn_prompt="[y/N]"; fi

    read -p "$prompt $yn_prompt " -n 1 -r
    echo ""

    if [ -z "$REPLY" ]; then
        if [ "$default" == "Y" ]; then return 0; fi
        if [ "$default" == "N" ]; then return 1; fi
    fi

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}
