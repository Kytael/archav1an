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
    # Accept either opencl-icd-loader or ocl-icd (they conflict, both provide OpenCL)
    log_info "Checking OpenCL runtime..."
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        if pacman -Qi opencl-icd-loader &>/dev/null || pacman -Qi ocl-icd &>/dev/null; then
            log_info "OpenCL ICD loader already installed, skipping."
        else
            pacman -S --needed --noconfirm opencl-icd-loader || { log_error "Failed to install opencl-icd-loader"; return 1; }
        fi
    else
        apt install -y ocl-icd-opencl-dev || { log_error "Failed to install ocl-icd-opencl-dev"; return 1; }
    fi

    # 2. Python packages: onnxruntime + havsfunc
    log_info "Installing onnxruntime and havsfunc into venv..."
    "$VENV_DIR/bin/pip" install onnxruntime havsfunc || { log_error "Failed to install onnxruntime/havsfunc"; return 1; }

    # Optional: ROCm execution provider for AMD GPUs (skip if ROCm not installed)
    if command -v rocm-smi &>/dev/null || [ -d /opt/rocm ]; then
        log_info "ROCm detected — installing onnxruntime-rocm..."
        "$VENV_DIR/bin/pip" install onnxruntime-rocm || log_warn "onnxruntime-rocm install failed, falling back to CPU"
    fi

    # 3. SCUNet ONNX models (from vs-mlrt release)
    local MODELS_DIR="$BASE_DIR/models/scunet"
    mkdir -p "$MODELS_DIR"

    log_info "Downloading SCUNet ONNX models..."
    local RELEASE_URL
    RELEASE_URL=$(curl -sf "https://api.github.com/repos/AmusementClub/vs-mlrt/releases/latest" \
        | python3 -c "import sys,json; r=json.load(sys.stdin); print(next(a['browser_download_url'] for a in r['assets'] if 'models' in a['name'] and 'contrib' not in a['name']))" 2>/dev/null)

    if [ -z "$RELEASE_URL" ]; then
        log_error "Could not find models release URL"
        return 1
    fi

    local MODELS_TMP="/tmp/vsmlrt-models.7z"
    curl -L "$RELEASE_URL" -o "$MODELS_TMP" || { log_error "Failed to download models archive"; return 1; }

    # Extract only SCUNet models
    7z e "$MODELS_TMP" "models/scunet/*.onnx" -o"$MODELS_DIR" -y || { log_error "Failed to extract SCUNet models"; return 1; }
    rm -f "$MODELS_TMP"

    if ls "$MODELS_DIR"/scunet_color_*.onnx &>/dev/null; then
        log_success "SCUNet models installed to $MODELS_DIR"
    else
        log_error "SCUNet model files not found after extraction"
        return 1
    fi

    # 4. KNLMeansCL VapourSynth plugin
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

    log_success "Denoiser installed (onnxruntime, SCUNet models, KNLMeansCL)."
}

uninstall_denoiser() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"

    log_info "Removing onnxruntime and havsfunc from venv..."
    "$VENV_DIR/bin/pip" uninstall -y onnxruntime onnxruntime-rocm havsfunc || true

    log_info "Removing KNLMeansCL plugin..."
    rm -f "$VS_PLUGIN_PATH/libknlmeanscl.so" || true

    log_info "Removing SCUNet models..."
    rm -rf "$BASE_DIR/models/scunet" || true

    log_success "Denoiser dependencies removed."
}
