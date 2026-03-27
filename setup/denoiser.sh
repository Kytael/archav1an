#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_denoiser() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    # 1. OpenCL runtime (required by KNLMeansCL)
    log_info "Installing OpenCL runtime..."
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        pacman -S --needed --noconfirm opencl-icd-loader || { log_error "Failed to install opencl-icd-loader"; return 1; }
    else
        apt install -y ocl-icd-opencl-dev || { log_error "Failed to install ocl-icd-opencl-dev"; return 1; }
    fi

    # 2. Python packages: vsmlrt and havsfunc
    log_info "Installing vsmlrt and havsfunc into venv..."
    "$VENV_DIR/bin/pip" install vsmlrt havsfunc || { log_error "Failed to install vsmlrt/havsfunc"; return 1; }

    # 3. KNLMeansCL VapourSynth plugin
    log_info "Compiling KNLMeansCL..."
    mkdir -p build_tmp
    cd build_tmp || return 1

    if [ -d "KNLMeansCL" ]; then rm -rf KNLMeansCL; fi
    git clone --depth 1 https://github.com/Khanattila/KNLMeansCL.git || { log_error "Failed to clone KNLMeansCL"; cd ..; return 1; }
    cd KNLMeansCL
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release || { log_error "KNLMeansCL cmake failed"; cd ../../..; return 1; }
    make -j"$(nproc)" || { log_error "KNLMeansCL make failed"; cd ../../..; return 1; }

    if [ -f "libknlmeanscl.so" ]; then
        cp "libknlmeanscl.so" "$VS_PLUGIN_PATH/"
        log_success "KNLMeansCL installed to $VS_PLUGIN_PATH/"
    else
        log_error "KNLMeansCL compilation failed — libknlmeanscl.so not found"
        cd ../../..
        return 1
    fi

    cd ../../..
    ldconfig

    log_success "Denoiser dependencies installed (vsmlrt, havsfunc, KNLMeansCL)."
}

uninstall_denoiser() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"

    log_info "Removing vsmlrt and havsfunc from venv..."
    "$VENV_DIR/bin/pip" uninstall -y vsmlrt havsfunc || true

    log_info "Removing KNLMeansCL plugin..."
    rm -f "$VS_PLUGIN_PATH/libknlmeanscl.so" || true

    log_success "Denoiser dependencies removed."
}
