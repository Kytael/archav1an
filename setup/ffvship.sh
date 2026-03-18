#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_ffvship() {
    if command -v FFVship &> /dev/null; then
        log_info "FFVship is already installed."
        return 0
    fi

    log_info "Compiling FFVship..."

    # Ensure pkg-config can find locally-built libraries (ffms2, ffmpeg, etc.)
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

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

    if [ -d "Vship" ]; then rm -rf Vship; fi
    git clone --branch v5.0.0 --depth 1 https://codeberg.org/Line-fr/Vship.git || { cd "$ORIG_DIR"; log_error "Failed to clone Vship"; return 1; }
    cd Vship || { cd "$ORIG_DIR"; log_error "Failed to cd into Vship"; return 1; }

    if command -v nvcc &> /dev/null; then
        log_info "Building FFVship with CUDA (NVIDIA)..."
        make buildcuda || { cd "$ORIG_DIR"; log_error "FFVship buildcuda failed"; return 1; }
    elif command -v hipcc &> /dev/null; then
        log_info "Building FFVship with HIP (AMD)..."
        make build || { cd "$ORIG_DIR"; log_error "FFVship HIP build failed"; return 1; }
    else
        log_warn "Neither nvcc nor hipcc found. Attempting Vulkan build."
        make buildVulkan || { cd "$ORIG_DIR"; log_error "FFVship Vulkan build failed"; return 1; }
    fi

    make buildFFVSHIP || { cd "$ORIG_DIR"; log_error "FFVship make buildFFVSHIP failed"; return 1; }
    make install PREFIX=/usr/local || { cd "$ORIG_DIR"; log_error "FFVship make install failed"; return 1; }

    # Ensure libvship.so is in the VapourSynth plugin path
    local VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"
    if [ -f /usr/local/lib/vapoursynth/libvship.so ] && [ "$VS_PLUGIN_PATH" != "/usr/local/lib/vapoursynth" ]; then
        log_info "Linking libvship.so to VapourSynth plugin path ($VS_PLUGIN_PATH)..."
        ln -sf /usr/local/lib/vapoursynth/libvship.so "$VS_PLUGIN_PATH/libvship.so"
    elif [ -f /usr/local/lib/libvship.so ]; then
        ln -sf /usr/local/lib/libvship.so "$VS_PLUGIN_PATH/libvship.so"
    fi

    cd "$ORIG_DIR"

    log_success "FFVship installed."
}

uninstall_ffvship() {
    log_info "Uninstalling FFVship..."
    rm -vf /usr/local/bin/FFVship
    log_success "FFVship uninstalled."
}
