#!/bin/bash
# Activate the Auto-Boost-Av1an Python virtual environment
# Source this from any script that needs to call python3

VENV_DIR="${VENV_DIR:-/opt/auto-boost-av1an/venv}"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "[WARN] Python venv not found at $VENV_DIR. Run setup.sh first."
    echo "       Falling back to system python3."
fi

# Ensure source-built libraries and VapourSynth plugins are found at runtime
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
export VAPOURSYNTH_PLUGIN_PATH="/usr/local/lib/vapoursynth"
export PATH="/usr/local/bin:$PATH"

# WSL2: use clean CUDA symlinks (originals in /usr/lib/wsl/lib crash glibc's ld.so)
if uname -r | grep -qi microsoft; then
    if [ -d /usr/local/lib/wsl-cuda ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *:/usr/local/lib/wsl-cuda:*) ;;
            *) export LD_LIBRARY_PATH="/usr/local/lib/wsl-cuda:${LD_LIBRARY_PATH:-}" ;;
        esac
    fi
fi

# AMD ROCm: set HSA_OVERRIDE_GFX_VERSION if not already set and an AMD GPU is present
if [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ] && [ -d /opt/rocm ]; then
    _gfx=""
    if command -v rocminfo &>/dev/null; then
        _gfx=$(rocminfo 2>/dev/null | grep -oP 'gfx[0-9a-f]+' | head -1)
    fi
    if [ -z "$_gfx" ]; then
        _gfx=$(cat /sys/class/drm/card*/device/ip_discovery/die/*/GC/*/major 2>/dev/null | head -1 | xargs -I{} bash -c 'printf "gfx%s%s%s" {} $(cat /sys/class/drm/card*/device/ip_discovery/die/*/GC/*/minor 2>/dev/null | head -1) $(cat /sys/class/drm/card*/device/ip_discovery/die/*/GC/*/revision 2>/dev/null | head -1)' 2>/dev/null)
    fi
    if [ -n "$_gfx" ]; then
        _num="${_gfx#gfx}"
        _len="${#_num}"
        if [ "$_len" -ge 3 ]; then
            _rev="${_num: -1}"; _minor="${_num: -2:1}"; _major="${_num:0:$((_len-2))}"
            export HSA_OVERRIDE_GFX_VERSION="${_major}.${_minor}.${_rev}"
        fi
    fi
    unset _gfx _num _len _rev _minor _major
fi

# Use mimalloc for faster multi-threaded memory allocation (SVT-AV1, av1an, etc.)
MIMALLOC_PATH=""
if [ -f /usr/lib/libmimalloc.so ]; then
    MIMALLOC_PATH="/usr/lib/libmimalloc.so"
elif [ -f /usr/lib/x86_64-linux-gnu/libmimalloc.so ]; then
    MIMALLOC_PATH="/usr/lib/x86_64-linux-gnu/libmimalloc.so"
fi
if [ -n "$MIMALLOC_PATH" ]; then
    export LD_PRELOAD="${MIMALLOC_PATH}${LD_PRELOAD:+:$LD_PRELOAD}"
fi
