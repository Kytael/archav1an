#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_system_deps_arch() {
    log_info "Updating system packages..."
    pacman -Syu --noconfirm

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
        # OpenCL runtime (for KNLMeansCL denoiser)
        opencl-icd-loader
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

    pacman -S --needed --noconfirm "${DEPS[@]}" || { log_error "Failed to install system dependencies via pacman"; return 1; }

    log_success "Build tools and system libraries installed."
}

install_system_deps_debian() {
    log_info "Updating apt..."
    apt update

    log_info "Installing System Packages (apt)..."
    local DEPS=(
        software-properties-common ffmpeg x264 mkvtoolnix mkvtoolnix-gui
        python3 python3-pip git curl wget build-essential cmake pkg-config
        autoconf automake libtool yasm nasm clang libavcodec-dev libavformat-dev
        libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev
        libzimg-dev python3-numpy python3-psutil python3-rich jq mediainfo
        opus-tools x265 xclip meson ninja-build libass-dev nvidia-cuda-toolkit cython3
        libjpeg-turbo8-dev libwebp-dev libavif-dev libxxhash-dev libdav1d-dev
        libvpx-dev libmp3lame-dev libvorbis-dev libsoxr-dev libgnutls28-dev
        libsrt-openssl-dev libvidstab-dev libbluray-dev libva-dev
        libfribidi-dev libfontconfig-dev
        # OpenCL runtime (for KNLMeansCL denoiser)
        ocl-icd-opencl-dev
    )

    apt install -y "${DEPS[@]}" || { log_error "Failed to install system dependencies via apt"; return 1; }

    log_info "Compiling FFmpeg master from source to satisfy BestSource (libavcodec >= 61.19.0)..."
    local CDIR="/tmp/ffmpeg_master_build"
    local ORIG_DIR="$(pwd)"
    mkdir -p "$CDIR"
    cd "$CDIR"
    if [ -d "ffmpeg" ]; then rm -rf ffmpeg; fi
    git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git ffmpeg || { cd "$ORIG_DIR"; log_error "Failed to clone ffmpeg repo"; return 1; }
    cd ffmpeg
    ./configure \
      --prefix="/usr/local" \
      --enable-shared \
      --enable-gpl \
      --enable-libx264 \
      --enable-libx265 \
      --enable-libass \
      --enable-libfreetype \
      --disable-doc \
      --disable-programs || { cd "$ORIG_DIR"; log_error "FFmpeg configure failed"; return 1; }
    make -j"$(nproc)" || { cd "$ORIG_DIR"; log_error "FFmpeg make failed"; return 1; }
    make install || { cd "$ORIG_DIR"; log_error "FFmpeg make install failed"; return 1; }
    ldconfig
    cd "$ORIG_DIR"

    log_success "System packages and FFmpeg libraries installed."
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
