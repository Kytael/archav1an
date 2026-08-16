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

# Make cargo usable, not merely present, for the components built from crates.
#
# `command -v cargo` is not the test it looks like. rustup installs a proxy at
# ~/.cargo/bin/cargo (and /usr/bin/cargo when it comes from a distro package)
# that carries no toolchain of its own, so cargo answers on PATH from the
# moment rustup exists and every invocation then fails with "rustup could not
# choose a version of cargo to run, because one wasn't specified explicitly,
# and no default is configured". A fresh host hit that at the first cargo
# build, having skipped the install branch because cargo was "already there".
ensure_rust() {
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v cargo &> /dev/null; then
        if command -v pacman &> /dev/null; then
            local _sudo
            _sudo=$(pkg_manager_sudo) || { log_error "Rust is missing and neither root nor sudo is available to install it."; return 1; }
            $_sudo pacman -S --needed --noconfirm rust || { log_error "Failed to install Rust"; return 1; }
        else
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y || { log_error "Failed to install Rust via rustup"; return 1; }
        fi
        [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    # Cheapest command that a toolchain-less proxy fails.
    if ! cargo --version &> /dev/null; then
        if ! command -v rustup &> /dev/null; then
            log_error "cargo is on PATH but does not run, and rustup is not there to repair it. Reinstall Rust."
            return 1
        fi
        log_info "rustup has no default toolchain; installing stable."
        rustup default stable || { log_error "'rustup default stable' failed; set a toolchain by hand and re-run."; return 1; }
        cargo --version &> /dev/null || { log_error "cargo still does not run after 'rustup default stable'."; return 1; }
    fi
    return 0
}

# Pick a host C++ compiler that the installed nvcc accepts.
#
# nvcc trails gcc by roughly 6-12 months on major releases. Distros ship
# the latest gcc immediately (Arch had gcc 16 within weeks of release),
# so the default g++ is often ahead of nvcc's supported range. Symptom is
# usually a libstdc++ header pulling in a compiler builtin that nvcc
# doesn't recognize (e.g. __builtin_is_virtual_base_of in libstdc++ 16).
#
# Strategy: probe the default g++ with a real nvcc compile of a header
# that exercises libstdc++. If it works, use it (i.e. once nvcc catches
# up to gcc 16, this function becomes a no-op). Otherwise enumerate all
# side-installed g++-N on PATH, try them in descending order of N, and
# pick the newest one that nvcc accepts.
nvcc_pick_ccbin() {
    if ! command -v nvcc &> /dev/null; then
        return 0
    fi

    # The probe must include something that pulls in libstdc++. A bare
    # `int main(){return 0;}` compiles even when nvcc + gcc 16 are
    # mutually unhappy because nothing in it touches the offending
    # libstdc++ headers. <memory> is small but exercises the new C++26
    # builtins that gcc 16 uses internally.
    local _probe='#include <memory>\nint main(){return 0;}'
    local _probe_obj
    _probe_obj="$(mktemp --suffix=.o)"

    _nvcc_test() {
        local _ccbin="$1"
        local _flag=""
        [ -n "$_ccbin" ] && _flag="-ccbin $_ccbin"
        # shellcheck disable=SC2086
        printf '%b' "$_probe" | nvcc $_flag -x cu -c -o "$_probe_obj" - &>/dev/null
    }

    if _nvcc_test ""; then
        rm -f "$_probe_obj"
        unset -f _nvcc_test
        return 0
    fi

    # Enumerate g++-N on PATH and sort by N descending.
    local _candidates _candidate
    _candidates="$(compgen -c g++- 2>/dev/null \
        | awk -F- '/^g\+\+-[0-9]+$/ {print $NF}' \
        | sort -u -nr)"

    for _candidate in $_candidates; do
        if _nvcc_test "g++-$_candidate"; then
            export NVCC_PREPEND_FLAGS="-ccbin g++-$_candidate ${NVCC_PREPEND_FLAGS:-}"
            log_info "nvcc: using host compiler g++-$_candidate (default g++ is too new for this CUDA toolkit; will switch back to default once nvcc gains support)."
            rm -f "$_probe_obj"
            unset -f _nvcc_test
            return 0
        fi
    done

    rm -f "$_probe_obj"
    unset -f _nvcc_test
    log_warn "nvcc: default g++ rejected by the installed CUDA toolkit and no compatible g++-N found on PATH. Install a supported version (e.g. 'sudo pacman -S gcc15' on Arch when CUDA 13.x is installed) if nvcc builds fail."
    return 0
}

# Locate a pacman-installed VapourSynth plugin .so. The layout moved
# between v74 and v75+/R76: v74 shipped /usr/lib/vapoursynth/lib*.so,
# v75+ ships /usr/lib/python3.X/site-packages/vapoursynth/plugins/*.so
# (no `lib` prefix). Some AUR packages also install to /usr/vapoursynth
# (e.g. vapoursynth-plugin-ctmf-git, a PKGBUILD quirk).
#
# Usage: find_pacman_vs_plugin mvtools  ->  prints the path on stdout
find_pacman_vs_plugin() {
    local name="$1"
    local _candidate
    # Use compgen to dodge zsh's "no matches found" failure on unmatched
    # globs and bash's literal-string behavior on the same.
    for _candidate in \
        $(compgen -G "/usr/lib/python3.*/site-packages/vapoursynth/plugins/${name}.so" 2>/dev/null) \
        "/usr/lib/vapoursynth/lib${name}.so" \
        "/usr/lib/vapoursynth/${name}.so" \
        "/usr/vapoursynth/lib${name}.so" \
        "/usr/vapoursynth/${name}.so"; do
        if [ -f "$_candidate" ]; then
            echo "$_candidate"
            return 0
        fi
    done
    return 1
}

# Authenticate once, up front, for every privileged step this run will take.
#
# Exactly two things need root: creating $VS_PREFIX owned by you, and letting
# the package manager install missing distro packages. Both are decidable
# before any work starts, so decide them here and take a single `sudo -v`.
# Every later escalation then hits sudo's credential cache and stays silent.
#
# The point is the failure mode as much as the prompt count. Authentication
# happens at second zero, so a wrong password or a user not in sudoers aborts
# immediately rather than after a long build. On a provisioned host nothing is
# missing, so this returns without prompting at all.
#
# Caveat: a sudoers policy with timestamp_timeout=0 disables the cache, and
# then each escalation prompts again. Nothing here can prevent that.
preflight_sudo() {
    [ "$EUID" -eq 0 ] && return 0

    local _reasons=()
    if [ -n "$VS_PREFIX" ] && [ ! -w "$VS_PREFIX" ]; then
        _reasons+=("create $VS_PREFIX owned by $USER")
    fi
    # system_deps_missing needs DISTRO_FAMILY, and on Arch a GPU_VENDOR too, so
    # the caller runs check_distro and detect_gpu before this.
    if declare -F system_deps_missing >/dev/null && [ -n "$(system_deps_missing)" ]; then
        _reasons+=("install missing distro packages")
    fi
    [ "${#_reasons[@]}" -eq 0 ] && return 0

    local _why="${_reasons[0]}${_reasons[1]:+ and ${_reasons[1]}}"
    if ! command -v sudo &>/dev/null; then
        log_error "This run needs root to $_why, but sudo is not installed. Re-run as root."
        return 1
    fi

    log_info "This run needs root once, to $_why."
    log_info "Authenticating now so nothing prompts part-way through the build."
    if ! sudo -v; then
        log_error "sudo authentication failed — aborting before any work is done."
        return 1
    fi
}

# Prints the AUR helper to drive, or nothing when the host has none. paru and
# yay take the same -S --needed --noconfirm arguments the denoiser uses, so
# either works and whichever is installed wins.
aur_helper() {
    local _h
    for _h in paru yay; do
        command -v "$_h" &>/dev/null && { echo "$_h"; return 0; }
    done
    return 1
}

# Hand $VS_PREFIX back to the user who ran sudo.
#
# check_root returns early on EUID 0, so under `sudo ./setup.sh` the prefix is
# created by whichever component reaches it first -- as root. Everything after
# that assumes it is yours: the venv is pip-installed into on later runs and
# --update rewrites the manifest, both unprivileged. spark2 ended a successful
# `sudo ./setup.sh --install A -y` with 21k root-owned files under the prefix.
#
# Refusing to run as root would be the wrong fix. sudo's credential cache
# expires (15 minutes by default) and the denoiser wants root for
# /etc/ld.so.conf.d an hour into a full build, so a root run is the honest way
# to start something and walk away. Fix the result instead of the invocation.
#
# The prefix is the obvious tree. build_tmp is the one that gets missed: every
# source build clones and compiles there, so a root run leaves it root-owned
# and the next unprivileged run dies at the clone rather than at the install,
# which reads like a git problem. encoder-host, spark2 and gpu3 all ended up that
# way. It lives beside setup.sh, and the installers put it under the directory
# the script was called from, so check both.
restore_ownership() {
    [ "$EUID" -eq 0 ] || return 0
    [ -n "${SUDO_USER:-}" ] || return 0

    local _grp _d _seen=""
    _grp="$(id -gn "$SUDO_USER" 2>/dev/null || printf '%s' "$SUDO_USER")"

    local _targets=()
    [ -n "${VS_PREFIX:-}" ] && _targets+=("$VS_PREFIX")
    [ -n "${BASE_DIR:-}" ] && _targets+=("$BASE_DIR/build_tmp")
    _targets+=("$PWD/build_tmp")

    for _d in "${_targets[@]}"; do
        [ -d "$_d" ] || continue
        case " $_seen " in *" $_d "*) continue ;; esac
        _seen="$_seen $_d"
        # -quit on the first hit: this runs on every exit, and the common case
        # is a tree that is already entirely the user's.
        [ -n "$(find "$_d" ! -user "$SUDO_USER" -print -quit 2>/dev/null)" ] || continue
        log_info "Returning $_d to $SUDO_USER:$_grp (this run was root)."
        chown -R "$SUDO_USER:$_grp" "$_d" \
            || log_warn "Could not chown $_d to $SUDO_USER; later unprivileged runs will fail to write it."
    done
}

check_root() {
    # If the user owns $VS_PREFIX (set in this file above), they don't need
    # sudo for builds that land inside the prefix. Individual install tasks
    # that genuinely need root (e.g. install_system_deps invoking pacman)
    # escalate for that one command; preflight_sudo has already authenticated.
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
    log_info "Bootstrapping $VS_PREFIX (the rest of setup runs as $USER)..."
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
# inside it; deliberately *not* /usr/local.
VS_PREFIX="${VS_PREFIX:-/opt/archav1an}"
export VS_PREFIX

# Helper: get the VapourSynth plugin path. Single source of truth: $VS_PREFIX.
get_vs_plugin_path() {
    echo "$VS_PREFIX/lib/vapoursynth"
}

# The pip-published vapoursynth wheel bundles its own core and sits in the venv
# ahead of the source-built module that vapoursynth.sh wires in through .pth.
# When it wins, "import vapoursynth" warns "the VapourSynth Python module
# version is R79 but the VapourSynth core library is R76" and Python-side code
# talks to a different core than vspipe does.
#
# python_libs removes it, but any later pip install that depends on vapoursynth
# -- vsscunet and friends, from the denoiser -- pulls it back, so the removal
# has to run again after those.
drop_pip_vapoursynth_stub() {
    [ -x "$VENV_DIR/bin/pip" ] || return 0
    "$VENV_DIR/bin/pip" show vapoursynth &>/dev/null || return 0
    log_info "Removing pip-installed vapoursynth stub (it shadows the source-built module)..."
    if command -v uv &>/dev/null; then
        VIRTUAL_ENV="$VENV_DIR" uv pip uninstall vapoursynth >/dev/null 2>&1 && return 0
    fi
    "$VENV_DIR/bin/pip" uninstall -y vapoursynth >/dev/null 2>&1 \
        || { log_warn "Could not remove the pip vapoursynth stub; importing vapoursynth may report a core/module version mismatch."; return 0; }

    # Re-register the built module. Installing the wheel rewrites
    # ~/.config/vapoursynth/vapoursynth.toml to name its own libvsscript.so, so
    # the entry install_vapoursynth wrote is gone by now -- and once the wheel
    # is uninstalled that path does not exist either, leaving vspipe dead with
    # "Python executable and library path couldn't be determined despite
    # automatic configuration".
    "$VENV_DIR/bin/python" -m vapoursynth config &>/dev/null \
        && log_info "Re-wrote vapoursynth.toml for the source-built libvsscript.so." \
        || log_warn "Removed the pip stub but could not re-run 'python -m vapoursynth config'; vspipe may fail to initialise VSScript."
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
    # Prefix lib dir goes in LDFLAGS as an explicit -L, NOT in LIBRARY_PATH:
    # pkgconf treats LIBRARY_PATH entries as system dirs and elides their -L
    # from --libs output, while clang searches LIBRARY_PATH only after
    # /usr/lib — the combination silently links system libs (e.g. ffms2
    # picking system libavcodec over the prefix build).
    export LDFLAGS="-L$VS_PREFIX/lib -flto -fuse-ld=lld"
    # rustc links through `cc`, which is gcc. The cc crate builds a package's C
    # dependencies with the CFLAGS above, so CC=clang plus -flto makes those
    # objects LLVM bitcode, and GNU ld then refuses the resulting rlib:
    # "liblibgit2_sys-*.rlib: error adding symbols: file format not recognized"
    # while linking av1an's build script. Arch escapes this only by accident --
    # its llvm package drops LLVMgold.so into /usr/lib/bfd-plugins, which GNU
    # ld auto-loads. Ubuntu ships the same plugin as LLVMgold-18.so and ld does
    # not pick it up. Link with the compiler that produced the bitcode instead
    # of depending on a plugin filename.
    export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C linker=clang -C link-arg=-fuse-ld=lld"
    # Debian/Ubuntu keep their .pc files under /usr/lib/<triplet>/pkgconfig, so
    # naming only /usr/lib/pkgconfig would drop them. pkg-config searches its
    # built-in path as well, so this stays additive on both distros.
    export PKG_CONFIG_PATH="$VS_PREFIX/lib/pkgconfig:/usr/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export LD_LIBRARY_PATH="$VS_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    unset LIBRARY_PATH
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
        # /opt/cuda is where Arch's cuda package lands. Every other packaging
        # (NVIDIA's own .deb repos, DGX OS) uses /usr/local/cuda. Only checking
        # the Arch path left nvcc off PATH on Ubuntu, so ffvship logged
        # "Neither nvcc nor hipcc found" and fell back to a Vulkan build on a
        # host with a working CUDA 13 toolkit sitting in /usr/local/cuda/bin.
        local _cuda_bin
        for _cuda_bin in /opt/cuda/bin /usr/local/cuda/bin; do
            [ -x "$_cuda_bin/nvcc" ] || continue
            case ":$PATH:" in *":$_cuda_bin:"*) ;; *) export PATH="$_cuda_bin:$PATH" ;; esac
            if [ ! -f /etc/profile.d/cuda.sh ]; then
                if [ "$EUID" -eq 0 ]; then
                    echo "export PATH=\"$_cuda_bin:\$PATH\"" > /etc/profile.d/cuda.sh
                    log_info "Added $_cuda_bin to system PATH via /etc/profile.d/cuda.sh"
                else
                    log_warn "Not root — skipped writing /etc/profile.d/cuda.sh (PATH exported for this session only)."
                fi
            fi
            break
        done
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
    # Everything below writes /etc or /usr/local — as non-root the redirects
    # fail but the function still logged success. Skip loudly instead.
    if [ "$EUID" -ne 0 ]; then
        log_warn "setup_wsl2_cuda: not root — skipping /etc ldconfig + CUDA symlink fixes (re-run under sudo if CUDA init fails)."
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

# ---------------------------------------------------------------------------
# Update / health machinery
#
# SOURCES["component:name"]="url|ref" — declared by each setup/ module at file
# top-level (single source of truth for pins; clone_src and the update checker
# both read it). ref is a tag/branch, or the literal "pinned" for immutable
# non-git artifacts (release downloads, local staging).
# ARTIFACTS["component"]="rel/path ..." — $VS_PREFIX-relative artifacts to
# health-check. Entries under lib/vapoursynth/ additionally get a VS load probe.
# Manifest: $MANIFEST_DIR/<component>.src, lines "name|url|ref|commit".
# ---------------------------------------------------------------------------
MANIFEST_DIR="${MANIFEST_DIR:-$VS_PREFIX/share/archav1an/manifest}"
declare -gA SOURCES
declare -gA ARTIFACTS

# record_src <component> <name> <url> <ref> <commit>
# Replaces any existing manifest line for <name>, keeps the rest.
record_src() {
    local component=$1 name=$2 url=$3 ref=$4 commit=$5
    local mf="$MANIFEST_DIR/$component.src"
    mkdir -p "$MANIFEST_DIR"
    if [ -f "$mf" ]; then
        grep -v "^${name}|" "$mf" > "$mf.tmp" || true
        mv "$mf.tmp" "$mf"
    fi
    echo "${name}|${url}|${ref}|${commit}" >> "$mf"
}

# record_system_src <component> <name> <pkg> <version>
# For a source the distro supplies rather than one we clone. On Arch the three
# SMDegrain plugins come from pacman and the AUR, so nothing was ever cloned
# and nothing was ever recorded -- and src_update_status then reported
# "no install record" for all three forever, which made the whole denoiser
# component read UNKNOWN on a machine where it was correctly installed.
# The entry is marked with the ref "system" so the status check knows there is
# no upstream commit to compare against.
record_system_src() {
    record_src "$1" "$2" "system:$3" system "$4"
}

# clone_src <component> <name> <destdir> [extra git-clone args...]
# Fresh shallow clone of SOURCES[component:name] into destdir + manifest record.
clone_src() {
    local component=$1 name=$2 dest=$3
    shift 3
    local spec="${SOURCES[$component:$name]:-}"
    if [ -z "$spec" ]; then
        log_error "clone_src: no SOURCES entry for $component:$name"
        return 1
    fi
    local url="${spec%%|*}" ref="${spec##*|}"
    rm -rf "$dest"
    if ! git clone --branch "$ref" --depth 1 "$@" "$url" "$dest"; then
        log_error "Failed to clone $name ($url @ $ref)"
        return 1
    fi
    local commit
    commit=$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo unknown)
    # Bookkeeping must not decide whether the clone succeeded. record_src ends
    # in a write to $MANIFEST_DIR, so an unwritable prefix used to make this
    # function return non-zero after a perfectly good clone -- and callers
    # report that as "Failed to clone", which sends you hunting the network.
    record_src "$component" "$name" "$url" "$ref" "$commit" \
        || log_warn "clone_src: cloned $name but could not record it in $MANIFEST_DIR"
    return 0
}

# src_update_status <component>
# Prints one line per SOURCES entry: "name|OK/UPDATE/UNKNOWN/WARN|detail".
# Returns 0 when everything is OK/WARN, 1 when any source needs a rebuild
# (UPDATE or UNKNOWN). Tags are treated as immutable; branches are compared
# against the remote head (strict pacman-style).
src_update_status() {
    local component=$1
    local mf="$MANIFEST_DIR/$component.src"
    local rc=0 key name spec url ref line rec_url rec_ref rec_commit
    local ls_out remote_sha
    for key in "${!SOURCES[@]}"; do
        [[ "$key" == "$component:"* ]] || continue
        name="${key#*:}"
        spec="${SOURCES[$key]}"
        url="${spec%%|*}"; ref="${spec##*|}"
        if [ "$ref" = "pinned" ]; then
            echo "$name|OK|pinned artifact"
            continue
        fi
        line=$(grep "^${name}|" "$mf" 2>/dev/null | head -n 1)
        if [ -z "$line" ]; then
            echo "$name|UNKNOWN|no install record"
            rc=1
            continue
        fi
        IFS='|' read -r _ rec_url rec_ref rec_commit <<< "$line"
        # A distro-supplied source has no upstream commit to compare, so the
        # url/ref check below would call it UPDATE on every run. The package
        # manager owns its currency; report what is installed and move on.
        if [ "$rec_ref" = "system" ]; then
            echo "$name|OK|${rec_url#system:} $rec_commit"
            continue
        fi
        if [ "$url" != "$rec_url" ] || [ "$ref" != "$rec_ref" ]; then
            echo "$name|UPDATE|pin changed: $rec_ref -> $ref"
            rc=1
            continue
        fi
        case "$url" in
            crates.io:*)
                local crate="${url#crates.io:}" latest
                latest=$(curl -sf -A archav1an-setup "https://crates.io/api/v1/crates/$crate" \
                    | sed -n 's/.*"max_stable_version":"\([^"]*\)".*/\1/p')
                if [ -z "$latest" ]; then
                    echo "$name|WARN|crates.io unreachable"
                elif [ "$latest" != "$rec_commit" ]; then
                    echo "$name|UPDATE|$rec_commit -> $latest"
                    rc=1
                else
                    echo "$name|OK|$rec_commit"
                fi
                continue
                ;;
        esac
        if ls_out=$(git ls-remote "$url" "refs/heads/$ref" 2>/dev/null); then
            remote_sha=$(echo "$ls_out" | cut -f1)
            if [ -z "$remote_sha" ]; then
                # No such branch: ref is a tag — immutable, matches the record.
                echo "$name|OK|pinned $ref"
            elif [ "$remote_sha" != "$rec_commit" ]; then
                echo "$name|UPDATE|$ref moved: ${rec_commit:0:8} -> ${remote_sha:0:8}"
                rc=1
            else
                echo "$name|OK|$ref @ ${rec_commit:0:8}"
            fi
        else
            echo "$name|WARN|remote unreachable: $url"
        fi
    done
    return $rc
}

# check_linkage <component>
# ldd over the component's present artifacts; prints "artifact|BROKEN|missing: ..."
# per broken artifact. Returns 0 healthy, 1 broken.
check_linkage() {
    local component=$1 rc=0 a path missing
    local ldpath="$VS_PREFIX/lib"
    [ -d /usr/lib/wsl/lib ] && ldpath="$ldpath:/usr/lib/wsl/lib"
    # libvsscript.so lives only inside the VapourSynth package dir, and
    # vapoursynth.sh must not symlink it out (dladdr + vapoursynth.toml; see
    # the comment there). vspipe finds it at runtime through its own $ORIGIN
    # rpath, but $VS_PREFIX/bin/vspipe is a symlink, so ldd expands $ORIGIN to
    # bin/ and reports the library missing -- a healthy vapoursynth came back
    # BROKEN and every sweep offered to rebuild it. Arch hid this too: pacman
    # leaves /usr/lib/libvsscript.so behind for ldd to find.
    local _vs_pkg
    _vs_pkg=$(vs_pkg_dir) && ldpath="$ldpath:$_vs_pkg"
    for a in ${ARTIFACTS[$component]:-}; do
        path="$VS_PREFIX/$a"
        [ -e "$path" ] || continue
        missing=$(env LD_LIBRARY_PATH="$ldpath" ldd "$path" 2>/dev/null \
            | awk '/not found/{print $1}' | sort -u | tr '\n' ' ')
        if [ -n "$missing" ]; then
            echo "$a|BROKEN|missing: ${missing% }"
            rc=1
        fi
    done
    return $rc
}

# vs_pkg_dir — where VapourSynth R76 installs itself, or nothing if absent.
#
# Anchored on the venv's Python version rather than a glob, for the reason
# setup/vapoursynth.sh:100 gives: a venv rebuilt at a new Python leaves the
# prior install_dir behind, and the stale one would name the wrong build.
# activate-venv.sh repeats this rather than calling it, because that file is
# sourced standalone and does not read this one.
vs_pkg_dir() {
    [ -x "$VENV_DIR/bin/python" ] || return 1
    local d
    d="$VS_PREFIX/lib/$("$VENV_DIR/bin/python" -c 'import sys;print("python%d.%d" % sys.version_info[:2])' 2>/dev/null)/site-packages/vapoursynth"
    [ -d "$d" ] || return 1
    printf '%s\n' "$d"
}

# vs_core_ok — can the prefix Python import the VapourSynth core at all?
# Cached; load probes are meaningless (and all fail) when the core is broken.
_VS_CORE_OK=""
vs_core_ok() {
    if [ -z "$_VS_CORE_OK" ]; then
        if LD_LIBRARY_PATH="$VS_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
           "$VENV_DIR/bin/python" -c "import vapoursynth; vapoursynth.core.std" >/dev/null 2>&1; then
            _VS_CORE_OK=yes
        else
            _VS_CORE_OK=no
        fi
    fi
    [ "$_VS_CORE_OK" = "yes" ]
}

# probe_plugin_load <plugin.so> — try LoadPlugin in the prefix VS core.
# "already loaded" (autoloaded healthy plugin) counts as success.
probe_plugin_load() {
    local so=$1
    LD_LIBRARY_PATH="$VS_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    "$VENV_DIR/bin/python" - "$so" <<'PYEOF'
import sys
import vapoursynth as vs
try:
    vs.core.std.LoadPlugin(sys.argv[1])
except vs.Error as e:
    if "already loaded" in str(e).lower():
        sys.exit(0)
    print(e, file=sys.stderr)
    sys.exit(1)
PYEOF
}

# component_health <component>
# ldd + (for lib/vapoursynth/*.so) VS load probe. Prints "artifact|BROKEN|reason"
# lines; returns 0 healthy, 1 broken. Skips load probes (with a note) when the
# VS core itself cannot import.
component_health() {
    local component=$1 rc=0 a so ldd_broken="" probe_err probe_rc mismatch
    local out
    out=$(check_linkage "$component") || rc=1
    [ -n "$out" ] && echo "$out"
    ldd_broken=$(echo "$out" | cut -d'|' -f1)
    for a in ${ARTIFACTS[$component]:-}; do
        case "$a" in
            lib/vapoursynth/*.so) ;;
            *) continue ;;
        esac
        so="$VS_PREFIX/$a"
        [ -e "$so" ] || continue
        # Already flagged by ldd — don't double-report.
        echo "$ldd_broken" | grep -qx "$a" && continue
        if ! vs_core_ok; then
            echo "$a|SKIPPED|VS core not importable, load probe skipped"
            continue
        fi
        probe_err=$(probe_plugin_load "$so" 2>&1 >/dev/null); probe_rc=$?
        if [ "$probe_rc" -ne 0 ]; then
            echo "$a|BROKEN|fails to load in VapourSynth"
            rc=1
            continue
        fi
        # A plugin that loads can still be built against a different TensorRT
        # than it loads, and vstrt says so itself on stderr: "built with 110100
        # but loaded with 110201; continue but fingers crossed". Take it at its
        # word. The same drift on gpu4 -- built 10.14.01, loaded 10.16.01 --
        # did not warn, it segfaulted the process, and nothing in this sweep saw
        # it coming because the plugin loaded fine everywhere else.
        mismatch=$(printf '%s' "$probe_err" \
            | sed -n 's/.*built with \([0-9]*\) but loaded with \([0-9]*\).*/built against TensorRT \1, loading \2/p' | head -1)
        if [ -n "$mismatch" ]; then
            echo "$a|BROKEN|$mismatch — rebuild against the installed TensorRT"
            rc=1
        fi
    done
    # A VapourSynth linked against a static libstdc++ exports the whole C++
    # runtime, locale facets included, and the Python module loads it
    # RTLD_GLOBAL. Every C++ library dlopened after it binds to those copies,
    # and libnvinfer segfaults in its own static constructor when it does --
    # which is how both Sparks lost libvstrt.so while every other plugin loaded
    # fine. Every host built before 2026-08-15 has this link; the x86 ones
    # survive it by luck, so flag them too and let the relink fix it. An
    # exported ios_base::Init is the tell -- it comes only from libstdc++.a,
    # and --exclude-libs makes it local. See install_vapoursynth for the cause.
    if [ "$component" = "vapoursynth" ]; then
        local _vs_so="$VS_PREFIX/lib/libvapoursynth.so.4"
        if [ -e "$_vs_so" ] && command -v nm &>/dev/null \
           && nm -D --defined-only "$_vs_so" 2>/dev/null | grep -q '_ZNSt8ios_base4InitC1Ev'; then
            echo "lib/libvapoursynth.so.4|BROKEN|static libstdc++ exported into the global scope — breaks TensorRT; relink"
            rc=1
        fi
    fi
    return $rc
}

AUTO_YES=false

ask_yes_no() {
    local prompt="$1"
    local default="$2" # Y or N
    stty sane 2>/dev/null

    if [ "$AUTO_YES" = true ]; then
        # Default-N prompts are safety gates (full rebuilds, destructive
        # removals) — -y takes the safe default there instead of forcing yes.
        if [ "$default" == "N" ]; then
            echo "$prompt [auto: taking default N]"
            return 1
        fi
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
