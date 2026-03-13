#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_ffvship() {
    if ! command -v FFVship &> /dev/null; then
        log_info "Compiling FFVship..."

        # Ensure pkg-config can find locally-built libraries (ffms2, ffmpeg, etc.)
        export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"

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

        mkdir -p build_tmp
        cd build_tmp || exit 1

        if [ -d "Vship" ]; then rm -rf Vship; fi
        git clone --branch v5.0.0 --depth 1 https://codeberg.org/Line-fr/Vship.git || { log_error "Failed to clone Vship"; cd ..; return 1; }
        cd Vship || { log_error "Failed to cd into Vship"; cd ..; cd ..; return 1; }

        if command -v nvcc &> /dev/null; then
            log_info "Building FFVship with CUDA (NVIDIA)..."
            make buildcuda || { log_error "FFVship buildcuda failed"; cd ..; cd ..; return 1; }
        elif command -v hipcc &> /dev/null; then
            log_info "Building FFVship with HIP (AMD)..."
            make build || { log_error "FFVship HIP build failed"; cd ..; cd ..; return 1; }
        else
            log_warn "Neither nvcc nor hipcc found. Attempting Vulkan build (slower)."
            log_warn "For AMD GPUs, install hip-runtime-amd for better performance."
            make buildVulkan || { log_error "FFVship buildVulkan failed (no Vulkan SDK?)"; cd ..; cd ..; return 1; }
        fi

        make buildFFVSHIP || { log_error "FFVship make buildFFVSHIP failed"; cd ..; cd ..; return 1; }
        make install PREFIX=/usr/local || { log_error "FFVship make install failed"; cd ..; cd ..; return 1; }
        cd ..
        cd .. # Exit build_tmp

        log_success "FFVship installed."
    else
        log_info "FFVship is already installed."
    fi
}

uninstall_ffvship() {
    log_info "Uninstalling FFVship..."
    rm -vf /usr/local/bin/FFVship
    log_success "FFVship uninstalled."
}
