#!/bin/bash
# Activate the Auto-Boost-Av1an Python virtual environment
# Source this from any script that needs to call python3

VENV_DIR="/opt/auto-boost-av1an/venv"

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

# Use mimalloc for faster multi-threaded memory allocation (SVT-AV1, av1an, etc.)
if [ -f /usr/lib/libmimalloc.so ]; then
    export LD_PRELOAD="/usr/lib/libmimalloc.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
