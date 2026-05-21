#!/bin/bash
# Activate the Auto-Boost-Av1an Python virtual environment
# Source this from any script that needs to call python3

# Install prefix for the isolated archav1an native stack. Sourced standalone
# (this file does NOT source setup/common.sh), so we define VS_PREFIX inline.
VS_PREFIX="${VS_PREFIX:-/opt/archav1an}"
VENV_DIR="${VENV_DIR:-$VS_PREFIX/venv}"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "[WARN] Python venv not found at $VENV_DIR. Run setup.sh first."
    echo "       Falling back to system python3."
fi

# libvapoursynth.so.4 is symlinked into $VS_PREFIX/lib from the package dir.
# LD_LIBRARY_PATH=$VS_PREFIX/lib puts our R76 build ahead of pacman's system
# v75 (which is cached by ldconfig under the same SONAME) only inside this
# activated env; outside the env, the system v75 stays the default.
export LD_LIBRARY_PATH="$VS_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# R76 renamed VAPOURSYNTH_PLUGIN_PATH -> VAPOURSYNTH_EXTRA_PLUGIN_PATH and
# also auto-loads plugins from <libvapoursynth-dir>/plugins/. Set both so
# scripts written against the R73 env var keep working.
export VAPOURSYNTH_PLUGIN_PATH="$VS_PREFIX/lib/vapoursynth"
export VAPOURSYNTH_EXTRA_PLUGIN_PATH="$VS_PREFIX/lib/vapoursynth"
export PATH="$VS_PREFIX/bin:$PATH"

# Cover Python embedders (vspipe and friends) that need libpython3.X.so.1.0
# when the venv interpreter lives outside $VS_PREFIX (typical for uv-managed
# Pythons under ~/.local/share/uv/python/...). The build-time rpath should
# handle this for our own binaries, but other tools that load libvsscript
# don't always get the rpath treatment.
#
# Skip when the venv's libpython lives in a global libdir like /usr/lib —
# prepending those would put pacman's libvapoursynth.so.4 (v75) ahead of
# our R76 build (which is symlinked in via $VS_PREFIX/lib) and break the
# isolation. The system loader will find /usr/lib via its default search
# regardless, so adding it to LD_LIBRARY_PATH only loses us, never helps.
if [ -x "$VENV_DIR/bin/python" ]; then
    _uv_py_libdir="$("$VENV_DIR/bin/python" -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))' 2>/dev/null)"
    case "$_uv_py_libdir" in
        /usr/lib|/usr/lib64|/lib|/lib64|/usr/local/lib|/usr/local/lib64)
            # System libdir; loader already searches it. Do nothing.
            ;;
        *)
            if [ -n "$_uv_py_libdir" ] && [ -d "$_uv_py_libdir" ]; then
                export LD_LIBRARY_PATH="$_uv_py_libdir:$LD_LIBRARY_PATH"
            fi
            ;;
    esac
    unset _uv_py_libdir
fi

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
