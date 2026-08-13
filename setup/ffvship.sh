#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

SOURCES["ffvship:vship"]="https://codeberg.org/Line-fr/Vship.git|v5.0.1"
ARTIFACTS["ffvship"]="bin/FFVship lib/vapoursynth/libvship.so"

install_ffvship() {
    if command -v FFVship &> /dev/null && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "FFVship is already installed."
        return 0
    fi

    log_info "Compiling FFVship..."

    # Ensure pkg-config can find locally-built libraries (ffms2, ffmpeg, etc.)
    export PKG_CONFIG_PATH="$VS_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

    # pkg-config supplies -L$VS_PREFIX/lib -lffms2, which is enough to find
    # libffms2 itself and not enough to link. libffms2's DT_NEEDED names
    # libavcodec.so.63, libavutil.so.61 and friends from the same prefix, and ld
    # resolves a shared library's own dependencies through -rpath-link or
    # LD_LIBRARY_PATH -- never through -L. This function never called
    # set_native_build_flags, so neither was set and the link died on
    # "undefined reference to av_malloc@LIBAVUTIL_61".
    #
    # Arch hid this: pacman's ffmpeg also ships libavcodec.so.63, so ld found
    # those instead and linked FFVship against the SYSTEM libraries, same
    # soname and older ABI. That is the wrong-library trap from 5fbe07c, and it
    # only failed loudly here because Ubuntu ships .so.60 and has nothing to
    # accidentally match.
    #
    # LD_RUN_PATH records the prefix in the built binary, so FFVship keeps
    # resolving to our libraries at runtime without an env var.
    export LD_LIBRARY_PATH="$VS_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export LD_RUN_PATH="$VS_PREFIX/lib:${LD_RUN_PATH:-}"

    # Ensure ROCm/HIP tools and environment are set up
    if [ -d "/opt/rocm" ]; then
        export PATH="/opt/rocm/bin:$PATH"
        export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
        export HIP_PATH="${HIP_PATH:-/opt/rocm}"
    fi

    # Detect GPU if not already done
    if [ -z "$GPU_VENDOR" ]; then
        detect_gpu
    fi

    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    clone_src ffvship vship Vship || { cd "$ORIG_DIR"; return 1; }
    cd Vship || { cd "$ORIG_DIR"; log_error "Failed to cd into Vship"; return 1; }

    # WSL2: nvcc -arch=native can't query the GPU, so detect via Windows nvidia-smi
    if is_wsl2 && [ -f /mnt/c/Windows/system32/nvidia-smi.exe ]; then
        local cc
        cc=$(/mnt/c/Windows/system32/nvidia-smi.exe --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
        if [ -n "$cc" ]; then
            local sm="sm_${cc//./}"
            log_info "WSL2: detected GPU compute capability $cc ($sm)"
            sed -i "s/-arch=native/-arch=$sm/" Makefile
        fi
    fi

    # GPU_BACKEND override: set to "cuda", "hip", or "vulkan" to force a specific build
    # e.g.: GPU_BACKEND=vulkan FORCE_REINSTALL=1 sudo ./setup.sh --install ffvship
    if [ "${GPU_BACKEND,,}" = "cuda" ]; then
        log_info "Building FFVship with CUDA (forced via GPU_BACKEND)..."
        nvcc_pick_ccbin
        make buildcuda || { cd "$ORIG_DIR"; log_error "FFVship buildcuda failed"; return 1; }
    elif [ "${GPU_BACKEND,,}" = "hip" ]; then
        log_info "Building FFVship with HIP (forced via GPU_BACKEND)..."
        make build || { cd "$ORIG_DIR"; log_error "FFVship HIP build failed"; return 1; }
    elif [ "${GPU_BACKEND,,}" = "vulkan" ]; then
        log_info "Building FFVship with Vulkan (forced via GPU_BACKEND)..."
        make buildVulkan || { cd "$ORIG_DIR"; log_error "FFVship Vulkan build failed"; return 1; }
    elif command -v nvcc &> /dev/null; then
        log_info "Building FFVship with CUDA (NVIDIA)..."
        nvcc_pick_ccbin
        make buildcuda || { cd "$ORIG_DIR"; log_error "FFVship buildcuda failed"; return 1; }
    elif command -v hipcc &> /dev/null; then
        log_info "Building FFVship with HIP (AMD)..."
        make build || { cd "$ORIG_DIR"; log_error "FFVship HIP build failed"; return 1; }
    else
        log_warn "Neither nvcc nor hipcc found. Attempting Vulkan build."
        make buildVulkan || { cd "$ORIG_DIR"; log_error "FFVship Vulkan build failed"; return 1; }
    fi

    make buildFFVSHIP || { cd "$ORIG_DIR"; log_error "FFVship make buildFFVSHIP failed"; return 1; }
    make install PREFIX="$VS_PREFIX" || { cd "$ORIG_DIR"; log_error "FFVship make install failed"; return 1; }

    # Ensure libvship.so is in the VapourSynth plugin path
    local VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"
    if [ -f "$VS_PREFIX/lib/vapoursynth/libvship.so" ] && [ "$VS_PLUGIN_PATH" != "$VS_PREFIX/lib/vapoursynth" ]; then
        log_info "Linking libvship.so to VapourSynth plugin path ($VS_PLUGIN_PATH)..."
        ln -sf "$VS_PREFIX/lib/vapoursynth/libvship.so" "$VS_PLUGIN_PATH/libvship.so"
    elif [ -f "$VS_PREFIX/lib/libvship.so" ]; then
        ln -sf "$VS_PREFIX/lib/libvship.so" "$VS_PLUGIN_PATH/libvship.so"
    fi

    cd "$ORIG_DIR"

    log_success "FFVship installed."
}

uninstall_ffvship() {
    log_info "Uninstalling FFVship..."
    rm -vf "$VS_PREFIX/bin/FFVship"
    rm -f "$MANIFEST_DIR/ffvship.src"
    log_success "FFVship uninstalled."
}
