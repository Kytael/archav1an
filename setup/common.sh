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

check_root() {
    if [ "$EUID" -ne 0 ]; then 
        log_error "Please run as root (sudo)."
        exit 1
    fi
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
    python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/usr/lib/python3/dist-packages"
}

# Helper: get the VapourSynth plugin path
get_vs_plugin_path() {
    if command -v pkg-config &> /dev/null && pkg-config --exists vapoursynth 2>/dev/null; then
        echo "$(pkg-config --variable=libdir vapoursynth)/vapoursynth"
    elif [ "$DISTRO_FAMILY" = "arch" ]; then
        echo "/usr/lib/vapoursynth"
    else
        echo "/usr/lib/x86_64-linux-gnu/vapoursynth"
    fi
}

# Virtual environment path for Python dependencies
VENV_DIR="/opt/auto-boost-av1an/venv"
export VENV_DIR

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

    # Check for NVIDIA GPU
    if lspci 2>/dev/null | grep -qi "VGA.*NVIDIA\|Display.*NVIDIA\|3D.*NVIDIA"; then
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

AUTO_YES=false

ask_yes_no() {
    local prompt="$1"
    local default="$2" # Y or N

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
