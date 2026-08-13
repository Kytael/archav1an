#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_system_deps_arch() {
    log_info "Installing build tools and libraries (pacman)..."
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

    # Detect GPU and add appropriate packages
    detect_gpu

    if [ "$GPU_VENDOR" = "nvidia" ] || [ "$GPU_VENDOR" = "both" ]; then
        log_info "Adding NVIDIA CUDA packages..."
        DEPS+=(cuda)
    fi

    if [ "$GPU_VENDOR" = "amd" ] || [ "$GPU_VENDOR" = "both" ]; then
        # Check if hipcc is already available (e.g., from opencl-amd or opencl-amd-dev)
        if command -v hipcc &> /dev/null || [ -f "/opt/rocm/bin/hipcc" ]; then
            log_info "AMD HIP compiler (hipcc) already available, skipping ROCm HIP install."
        else
            log_info "Adding AMD ROCm/HIP packages..."
            DEPS+=(hip-runtime-amd)
        fi
    fi

    # Precheck which DEPS are missing so non-root invocations don't
    # uselessly hit pacman -S (which requires sudo) when everything is
    # already installed — the common case under FORCE_REINSTALL=1.
    local _missing=()
    for _dep in "${DEPS[@]}"; do
        pacman -Qi "$_dep" &>/dev/null || _missing+=("$_dep")
    done
    if [ "${#_missing[@]}" -eq 0 ]; then
        log_info "All ${#DEPS[@]} system packages already installed; skipping pacman -S."
    else
        log_info "Installing ${#_missing[@]} missing system packages via pacman -S: ${_missing[*]}"
        if [ "$EUID" -ne 0 ]; then
            log_error "Need root for pacman -S. Run: sudo pacman -S --needed --noconfirm ${_missing[*]}"
            return 1
        fi
        # Sync + full upgrade only when we actually need to install something:
        # installing against a stale db (or after a bare -Sy) is the classic
        # Arch partial-upgrade hazard.
        log_info "Updating system packages..."
        pacman -Syu --noconfirm || { log_error "pacman -Syu failed"; return 1; }
        pacman -S --needed --noconfirm "${_missing[@]}" || { log_error "Failed to install system dependencies via pacman"; return 1; }
    fi

    # AUR helper. denoiser.sh installs three AUR packages (tensorrt,
    # vapoursynth-plugin-removegrain-git, vapoursynth-plugin-ctmf-git) with
    # `paru` and nothing used to check that it exists, so a host without it
    # failed inside that component with a misleading error. pacman cannot
    # install these: they are AUR-only, which is the whole reason a helper is
    # needed. paru itself IS a repo package on CachyOS; on plain Arch it is not,
    # and bootstrapping it means a makepkg build we do not do unattended.
    if ! command -v paru &>/dev/null; then
        if pacman -Si paru &>/dev/null && [ "$EUID" -eq 0 ]; then
            log_info "Installing AUR helper paru..."
            pacman -S --needed --noconfirm paru \
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
    local cuda_major="$1"
    apt-cache madison libnvinfer10 2>/dev/null \
        | awk -v c="+cuda${cuda_major}" '$3 ~ /^10\./ && index($3, c) {print $3; exit}'
}

install_system_deps_debian() {
    log_info "Installing build tools and libraries (apt)..."
    local DEPS=(
        # Build tools
        build-essential cmake pkg-config autoconf automake libtool
        clang lld llvm libomp-dev meson ninja-build git curl wget
        # Runtime libraries (the source-built FFmpeg at $VS_PREFIX shadows these)
        ffmpeg libx264-dev libx265-dev libass-dev libfreetype-dev
        libfribidi-dev libfontconfig-dev libopus-dev
        libzimg-dev libjpeg-turbo8-dev libwebp-dev libavif-dev libxxhash-dev
        libdav1d-dev
        # FFmpeg link dependencies
        libvpx-dev libmp3lame-dev libvorbis-dev libsoxr-dev libgnutls28-dev
        libsrt-openssl-dev libvidstab-dev libbluray-dev
        # Python and utilities
        python3 python3-pip python3-venv python3-dev python3-numpy
        python3-psutil python3-rich cython3
        jq mediainfo mkvtoolnix mkvtoolnix-gui xclip opus-tools
        # Vulkan and VA-API (for FFVship and hardware decode)
        libvulkan-dev vulkan-tools libva-dev
        # Performance
        libmimalloc-dev
    )

    # yasm and nasm assemble x86 only. Every arm64 build that would use them
    # falls back to its C path, so requesting them off x86 installs two
    # packages that nothing can call.
    case "$(uname -m)" in
        x86_64|i?86) DEPS+=(yasm nasm) ;;
    esac

    detect_gpu

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
            log_info "No nvcc found; adding Ubuntu's nvidia-cuda-toolkit."
            DEPS+=(nvidia-cuda-toolkit)
        fi

        if [ -n "$nvcc_bin" ]; then
            local cuda_major
            cuda_major="$("$nvcc_bin" --version 2>/dev/null \
                | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)"
            if [ -n "$cuda_major" ]; then
                # cuDNN and TensorRT are what the BSVD onnxruntime path needs.
                # aarch64 has no wheel for either, so apt is the only source;
                # on x86 these are also the cheapest way to get them.
                DEPS+=("libcudnn9-cuda-${cuda_major}")
                local trt_ver
                trt_ver="$(pick_tensorrt10_version "$cuda_major")"
                if [ -n "$trt_ver" ]; then
                    log_info "Pinning TensorRT to $trt_ver for the onnxruntime TRT EP."
                    DEPS+=("libnvinfer10=$trt_ver" "libnvinfer-plugin10=$trt_ver")
                else
                    log_warn "No TensorRT 10.x build for CUDA $cuda_major in apt — --denoise-bsvd will fall back to the slower CUDA EP."
                fi
            fi
        fi
    fi

    # Precheck which DEPS are missing so a non-root invocation under
    # FORCE_REINSTALL=1 does not uselessly hit apt install when everything is
    # already there. A pinned entry (name=version) is checked by name alone;
    # apt re-pins it when the installed version differs.
    local _missing=()
    local _dep _name
    for _dep in "${DEPS[@]}"; do
        _name="${_dep%%=*}"
        dpkg -s "$_name" &>/dev/null || _missing+=("$_dep")
    done
    if [ "${#_missing[@]}" -eq 0 ]; then
        log_info "All ${#DEPS[@]} system packages already installed; skipping apt install."
    else
        if [ "$EUID" -ne 0 ]; then
            log_error "Need root for apt install. Run: sudo apt install -y ${_missing[*]}"
            return 1
        fi
        log_info "Updating apt..."
        apt update || { log_error "apt update failed"; return 1; }
        log_info "Installing ${#_missing[@]} missing system packages: ${_missing[*]}"
        apt install -y "${_missing[@]}" || { log_error "Failed to install system dependencies via apt"; return 1; }
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
