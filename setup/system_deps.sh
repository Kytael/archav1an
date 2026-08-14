#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

# Prints the prefix needed to run the package manager: empty when already root,
# "sudo" otherwise. Fails only when neither applies.
#
# setup.sh runs as your user by design -- check_root() takes one sudo to create
# $VS_PREFIX owned by you, and everything after that builds unprivileged. The
# package installs below are the only other steps that need root, so they
# escalate themselves rather than making the caller do it. Before this, a
# non-root `--install A` on a fresh host aborted at the first component with a
# copy-paste command, which made the documented one-command install work only
# on a host that was already provisioned.
#
# Escalating the whole script instead would be worse, and used to be the rule
# here: makepkg and paru refuse to run as root outright, uv and cargo live in
# your home and vanish under sudo's $HOME, and a root-owned prefix leaves the
# venv unwritable for every later run.
pkg_manager_sudo() {
    [ "$EUID" -eq 0 ] && return 0
    if command -v sudo &>/dev/null; then
        echo sudo
        return 0
    fi
    return 1
}

# The pacman package list, one per line. Extracted from install_system_deps_arch
# so the preflight can ask what is missing without installing anything, the way
# debian_system_deps() already allows on the apt side. Quiet on purpose: it runs
# during the preflight, where progress logging would be noise.
arch_system_deps() {
    local DEPS=(
        # Build tools
        base-devel cmake pkgconf autoconf automake libtool
        yasm nasm clang llvm lld openmp rust meson ninja git curl wget
        # Runtime libraries (kept as pacman deps, source-built FFmpeg at /usr/local shadows these)
        ffmpeg x264 x265 libass freetype2 fribidi fontconfig opus
        zimg libjpeg-turbo libwebp libavif xxhash dav1d
        # FFmpeg link dependencies
        libvpx lame libvorbis libsoxr gnutls srt
        vid.stab libbluray svt-av1
        # Python and utilities
        python python-pip python-numpy python-psutil python-rich cython
        jq mediainfo mkvtoolnix-cli mkvtoolnix-gui xclip opus-tools
        # Vulkan and VA-API (for FFVship and hardware decode)
        vulkan-headers vulkan-icd-loader libva
        # Performance
        mimalloc
    )

    # This function's stdout IS the package list, so detect_gpu's logging must
    # not land in it. Skip the probe entirely when the caller already ran it,
    # and send it to stderr otherwise.
    [ -n "$GPU_VENDOR" ] || detect_gpu >&2

    if [ "$GPU_VENDOR" = "nvidia" ] || [ "$GPU_VENDOR" = "both" ]; then
        DEPS+=(cuda)
    fi

    if [ "$GPU_VENDOR" = "amd" ] || [ "$GPU_VENDOR" = "both" ]; then
        # Check if hipcc is already available (e.g., from opencl-amd or opencl-amd-dev)
        if ! command -v hipcc &> /dev/null && [ ! -f "/opt/rocm/bin/hipcc" ]; then
            DEPS+=(hip-runtime-amd)
        fi
    fi

    printf '%s\n' "${DEPS[@]}"
}

arch_system_deps_missing() {
    local _dep
    while read -r _dep; do
        [ -n "$_dep" ] || continue
        pacman -Qi "$_dep" &>/dev/null || printf '%s\n' "$_dep"
    done < <(arch_system_deps)
}

# Which distro packages are missing, whichever family this is. Prints nothing
# when the host is already provisioned, which is what lets the preflight stay
# silent and prompt-free there.
system_deps_missing() {
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        arch_system_deps_missing
    else
        debian_system_deps_missing
    fi
}

install_system_deps_arch() {
    log_info "Installing build tools and libraries (pacman)..."

    # Precheck which DEPS are missing so non-root invocations don't
    # uselessly hit pacman -S (which requires sudo) when everything is
    # already installed — the common case under FORCE_REINSTALL=1.
    local _missing=()
    mapfile -t _missing < <(arch_system_deps_missing)
    local DEPS=()
    mapfile -t DEPS < <(arch_system_deps)
    if [ "${#_missing[@]}" -eq 0 ]; then
        log_info "All ${#DEPS[@]} system packages already installed; skipping pacman -S."
    else
        log_info "Installing ${#_missing[@]} missing system packages via pacman -S: ${_missing[*]}"
        local _sudo
        if ! _sudo="$(pkg_manager_sudo)"; then
            log_error "Need root for pacman -S, and sudo is not installed. Run as root, or: pacman -S --needed --noconfirm ${_missing[*]}"
            return 1
        fi
        [ -n "$_sudo" ] && log_info "Escalating for pacman only (sudo may prompt); the rest of setup stays as $USER."
        # Sync + full upgrade only when we actually need to install something:
        # installing against a stale db (or after a bare -Sy) is the classic
        # Arch partial-upgrade hazard.
        log_info "Updating system packages..."
        $_sudo pacman -Syu --noconfirm || { log_error "pacman -Syu failed"; return 1; }
        $_sudo pacman -S --needed --noconfirm "${_missing[@]}" || { log_error "Failed to install system dependencies via pacman"; return 1; }
    fi

    # AUR helper. denoiser.sh installs three AUR packages (tensorrt,
    # vapoursynth-plugin-removegrain-git, vapoursynth-plugin-ctmf-git) with
    # `paru` and nothing used to check that it exists, so a host without it
    # failed inside that component with a misleading error. pacman cannot
    # install these: they are AUR-only, which is the whole reason a helper is
    # needed. paru itself IS a repo package on CachyOS; on plain Arch it is not,
    # and bootstrapping it means a makepkg build we do not do unattended.
    if ! command -v paru &>/dev/null; then
        local _sudo_paru
        _sudo_paru="$(pkg_manager_sudo)" || _sudo_paru=""
        if pacman -Si paru &>/dev/null && { [ "$EUID" -eq 0 ] || [ -n "$_sudo_paru" ]; }; then
            log_info "Installing AUR helper paru..."
            $_sudo_paru pacman -S --needed --noconfirm paru \
                || log_warn "Failed to install paru — AUR packages in the denoiser component will be skipped"
        else
            log_warn "AUR helper 'paru' not found and not in the repos. Install it before setup.sh --install denoiser, or that component skips its AUR packages: git clone https://aur.archlinux.org/paru.git && cd paru && makepkg -si"
        fi
    fi

    log_success "Build tools and system libraries installed."
}

# The onnxruntime TRT EP hard-links libnvinfer.so.10, but the apt candidate for
# libnvinfer10 is an 11.x build whose soname is .so.11 — installing the
# candidate leaves the EP unloadable. Pick the newest 10.x build for the CUDA
# major we actually have. Prints nothing when no such version exists.
pick_tensorrt10_version() {
    local cuda_ver="$1"                  # full release, e.g. 13.0
    local cuda_major="${cuda_ver%%.*}"
    local madison v
    madison="$(apt-cache madison libnvinfer10 2>/dev/null)"
    # Match the exact CUDA release first. A loose major match takes the newest
    # 10.x for any 13.x, and on a CUDA 13.0 host that is 10.16.1.11+cuda13.2,
    # whose libnvinfer-dev depends on libnvinfer-safe-headers-dev. That package
    # is absent from the cuda13.0 line, so apt resolves it to the 11.x
    # candidate and the whole transaction fails on a broken-packages error.
    v="$(printf '%s\n' "$madison" \
        | awk -v c="+cuda${cuda_ver}" '$3 ~ /^10\./ && index($3, c) {print $3; exit}')"
    [ -n "$v" ] || v="$(printf '%s\n' "$madison" \
        | awk -v c="+cuda${cuda_major}." '$3 ~ /^10\./ && index($3, c) {print $3; exit}')"
    printf '%s' "$v"
}

# Single source of truth for the Debian/Ubuntu package set. Prints one spec per
# line; a spec is either a bare name or name=version for the pinned entries.
#
# is_installed and the installer both read this, so adding a dependency here is
# enough for a host that already ran setup to pick it up. The previous design
# hand-picked five packages for is_installed to test, which meant a host that
# had those five reported the component as done and silently skipped everything
# added later -- exactly how libssl-dev went uninstalled after being added.
debian_system_deps() {
    local DEPS=(
        # Build tools
        build-essential cmake pkg-config autoconf automake libtool
        clang lld llvm libomp-dev meson ninja-build git curl wget
        # Runtime libraries (the source-built FFmpeg at $VS_PREFIX shadows these)
        ffmpeg libx264-dev libx265-dev libass-dev libfreetype-dev
        libfribidi-dev libfontconfig-dev libopus-dev
        libzimg-dev libjpeg-turbo8-dev libwebp-dev libavif-dev libxxhash-dev
        libdav1d-dev
        # FFmpeg link dependencies. libssl-dev is not optional even though
        # nothing links OpenSSL directly: srt.pc carries
        # "Requires.private: openssl libcrypto", so without openssl.pc and
        # libcrypto.pc every pkg-config query for srt fails and FFmpeg's
        # configure reports "srt >= 1.3.0 not found" with no hint of the real
        # cause. Arch never hits this because its openssl package ships the
        # .pc files and is always present.
        libvpx-dev libmp3lame-dev libvorbis-dev libsoxr-dev libgnutls28-dev
        libsrt-openssl-dev libssl-dev libvidstab-dev libbluray-dev
        # Python and utilities
        python3 python3-pip python3-venv python3-dev python3-numpy
        python3-psutil python3-rich cython3
        jq mediainfo mkvtoolnix mkvtoolnix-gui xclip opus-tools
        # Vulkan and VA-API (for FFVship and hardware decode)
        libvulkan-dev vulkan-tools libva-dev
        # Denoiser component: OpenCL for KNLMeansCL, boost for vs-mlrt. The
        # arch branch installs these from inside the denoiser, which needs root
        # halfway through a component the user is running unprivileged. Asking
        # for them here keeps every apt call in one transaction.
        ocl-icd-opencl-dev opencl-headers
        libboost-filesystem-dev libboost-system-dev
        # fftw3f, which MVTools' meson.build requires outright. On Arch it
        # arrives as a dependency of the AUR mvtools package.
        libfftw3-dev
        # Performance
        libmimalloc-dev
    )

    # yasm and nasm assemble x86 only. Every arm64 build that would use them
    # falls back to its C path, so requesting them off x86 installs two
    # packages that nothing can call.
    case "$(uname -m)" in
        x86_64|i?86) DEPS+=(yasm nasm) ;;
    esac

    # Every emitter in here goes to stderr: this function's stdout IS the
    # package list, and log_info writes to stdout, so an unredirected line
    # would be parsed as a package name.
    detect_gpu >&2

    if [ "$GPU_VENDOR" = "nvidia" ] || [ "$GPU_VENDOR" = "both" ]; then
        # Ubuntu's nvidia-cuda-toolkit is CUDA 12 and would fight a toolkit
        # already installed from NVIDIA's own repo (DGX OS ships CUDA 13).
        # Only ask for it when the host has no nvcc at all.
        local nvcc_bin=""
        if command -v nvcc &>/dev/null; then
            nvcc_bin="nvcc"
        elif [ -x /usr/local/cuda/bin/nvcc ]; then
            nvcc_bin="/usr/local/cuda/bin/nvcc"
        else
            log_info "No nvcc found; adding Ubuntu's nvidia-cuda-toolkit." >&2
            DEPS+=(nvidia-cuda-toolkit)
        fi

        if [ -n "$nvcc_bin" ]; then
            local cuda_ver cuda_major
            cuda_ver="$("$nvcc_bin" --version 2>/dev/null \
                | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
            cuda_major="${cuda_ver%%.*}"
            if [ -n "$cuda_major" ]; then
                # cuDNN and TensorRT are what the BSVD onnxruntime path needs.
                # aarch64 has no wheel for either, so apt is the only source;
                # on x86 these are also the cheapest way to get them.
                DEPS+=("libcudnn9-cuda-${cuda_major}")
                local trt_ver
                trt_ver="$(pick_tensorrt10_version "$cuda_ver")"
                if [ -n "$trt_ver" ]; then
                    # Two rules, both learned by breaking them:
                    #
                    # 1. Every TensorRT package goes in ONE apt transaction.
                    #    Installing the runtime here and the headers later, from
                    #    the denoiser that needs them to build vstrt, lets the
                    #    second apt resolve to the 11.x candidate and drag the
                    #    runtime up with it, undoing the pin.
                    # 2. The pin must cover the whole dependency closure, not
                    #    just what we name. apt does not back off the newest
                    #    candidate for an unpinned dependency -- it reports a
                    #    conflict and fails the transaction. Pinning
                    #    libnvinfer-bin alone produced exactly that against
                    #    libnvinfer-lean10, -vc-plugin10, -dispatch10 and
                    #    libnvonnxparsers10.
                    #
                    # So this is libnvinfer-bin's full Depends closure plus the
                    # dev packages vstrt compiles against. Verify any change
                    # with: apt-get install -s -y <the whole list>
                    #
                    # libnvinfer-bin carries trtexec, which vsmlrt uses to cache
                    # TRT engines; without it the denoiser warns and rebuilds
                    # every engine on every run.
                    log_info "Pinning TensorRT to $trt_ver for the onnxruntime TRT EP." >&2
                    local _trt_pkg
                    for _trt_pkg in libnvinfer10 libnvinfer-plugin10 \
                                    libnvinfer-lean10 libnvinfer-vc-plugin10 \
                                    libnvinfer-dispatch10 libnvonnxparsers10 \
                                    libnvinfer-dev libnvinfer-headers-dev \
                                    libnvinfer-plugin-dev libnvinfer-headers-plugin-dev \
                                    libnvinfer-bin; do
                        DEPS+=("$_trt_pkg=$trt_ver")
                    done
                else
                    log_warn "No TensorRT 10.x build for CUDA $cuda_major in apt — --denoise-bsvd will fall back to the slower CUDA EP." >&2
                fi
            fi
        fi
    fi

    printf '%s\n' "${DEPS[@]}"
}

# Prints the specs from debian_system_deps that dpkg does not have. A pinned
# entry (name=version) is tested by name alone; apt re-pins it when the
# installed version differs.
debian_system_deps_missing() {
    local spec name
    while IFS= read -r spec; do
        [ -n "$spec" ] || continue
        name="${spec%%=*}"
        dpkg -s "$name" &>/dev/null || printf '%s\n' "$spec"
    done < <(debian_system_deps)
}

install_system_deps_debian() {
    log_info "Installing build tools and libraries (apt)..."
    local _missing=()
    mapfile -t _missing < <(debian_system_deps_missing)

    if [ "${#_missing[@]}" -eq 0 ]; then
        log_info "All system packages already installed; skipping apt install."
    else
        local _sudo
        if ! _sudo="$(pkg_manager_sudo)"; then
            log_error "Need root for apt install, and sudo is not installed. Run as root, or: apt install -y ${_missing[*]}"
            return 1
        fi
        [ -n "$_sudo" ] && log_info "Escalating for apt only (sudo may prompt); the rest of setup stays as $USER."
        log_info "Updating apt..."
        $_sudo apt update || { log_error "apt update failed"; return 1; }
        log_info "Installing ${#_missing[@]} missing system packages: ${_missing[*]}"
        $_sudo apt install -y "${_missing[@]}" || { log_error "Failed to install system dependencies via apt"; return 1; }
        ldconfig
    fi

    log_success "Build tools and system libraries installed."
}

install_system_deps() {
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        install_system_deps_arch
    else
        install_system_deps_debian
    fi
}

uninstall_system_deps() {
    log_warn "Uninstalling system dependencies can break your system!"
    log_warn "This will remove packages like ffmpeg, python3, git, gcc, etc."
    if ask_yes_no "Are you ABSOLUTELY SURE you want to continue?" "N"; then
        if [ "$DISTRO_FAMILY" = "arch" ]; then
            local DEPS=(
                ffmpeg x264 mkvtoolnix-cli mkvtoolnix-gui
                python python-pip git curl wget cmake pkgconf
                autoconf automake libtool yasm nasm clang
                zimg python-numpy python-psutil python-rich jq mediainfo
                opus-tools x265 xclip meson ninja libass cuda
                libjpeg-turbo libwebp libavif xxhash
            )
            pacman -Rns --noconfirm "${DEPS[@]}"
        else
            local DEPS=(
                software-properties-common ffmpeg x264 mkvtoolnix mkvtoolnix-gui
                python3 python3-pip git curl wget build-essential cmake pkg-config
                autoconf automake libtool yasm nasm clang libavcodec-dev libavformat-dev
                libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev
                libzimg-dev python3-numpy python3-psutil python3-rich jq mediainfo
                opus-tools x265 xclip meson ninja-build libass-dev nvidia-cuda-toolkit
                libjpeg-turbo8-dev libwebp-dev libavif-dev
            )
            apt remove -y "${DEPS[@]}"
        fi
        log_success "System packages removed (hopefully you knew what you were doing)."
    else
        log_info "Uninstall aborted."
    fi
}
