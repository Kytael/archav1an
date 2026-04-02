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

    # 2. PyTorch (ROCm for AMD, CPU fallback) + vsscunet + havsfunc
    log_info "Installing PyTorch + vsscunet + havsfunc into venv..."
    if command -v rocm-smi &>/dev/null || [ -d /opt/rocm ]; then
        local rocm_ver
        rocm_ver=$(cat /opt/rocm/.info/version 2>/dev/null | grep -oP '^\d+\.\d+' || echo "")
        if [ -z "$rocm_ver" ]; then
            log_warn "Could not detect ROCm version, defaulting to rocm6.2 index"
            rocm_ver="6.2"
        fi
        log_info "ROCm $rocm_ver detected — installing PyTorch ROCm build..."
        "$VENV_DIR/bin/pip" install torch torchvision --index-url "https://download.pytorch.org/whl/rocm${rocm_ver}" || { log_error "Failed to install PyTorch ROCm ${rocm_ver}"; return 1; }
    else
        log_info "No ROCm/CUDA detected — installing PyTorch CPU build (needed for ONNX export)..."
        "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu || { log_error "Failed to install PyTorch CPU"; return 1; }
    fi
    "$VENV_DIR/bin/pip" install vsscunet havsfunc || { log_error "Failed to install vsscunet/havsfunc"; return 1; }

    # Pre-download all SCUNet .pth model weights
    log_info "Pre-downloading SCUNet model weights..."
    "$VENV_DIR/bin/python" -m vsscunet || { log_error "Failed to download SCUNet models"; return 1; }

    # 3. Export SCUNet models to ONNX for MIGraphX
    log_info "Exporting SCUNet models to ONNX..."
    local ONNX_DIR="$VS_PLUGIN_PATH/models/scunet"
    mkdir -p "$ONNX_DIR"
    "$VENV_DIR/bin/python3" -c "
import torch, sys
from pathlib import Path
import vsscunet
model_dir = Path(vsscunet.__file__).parent / 'models'
out_dir = Path(sys.argv[1])
from vsscunet.network_scunet import SCUNet

for strength in [15, 25, 50]:
    name = f'scunet_color_{strength}'
    pth = model_dir / f'{name}.pth'
    if not pth.exists():
        print(f'  skip {name}.pth (not found)', flush=True)
        continue
    out = out_dir / f'{name}.onnx'
    if out.exists() and out.stat().st_size > 1024:
        print(f'  {name}.onnx already exists, skipping', flush=True)
        continue
    print(f'  exporting {name}...', flush=True)
    m = SCUNet(config=[4,4,4,4,4,4,4])
    m.load_state_dict(torch.load(str(pth), map_location='cpu', mmap=True))
    m = m.eval()
    dummy = torch.zeros(1, 3, 256, 256)
    ep = torch.onnx.export(m, (dummy,), dynamo=True)
    ep.save(str(out))
    print(f'  {name}.onnx done', flush=True)
" "$ONNX_DIR" || { log_error "ONNX export failed"; return 1; }
    # Make model dir writable so vsmlrt.py can cache .mxr files next to the .onnx files
    chmod -R o+w "$ONNX_DIR"
    log_success "SCUNet ONNX models exported to $ONNX_DIR"

    # 4. MIGraphX package (AMD graph compiler)
    log_info "Installing MIGraphX..."
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        pacman -S --needed --noconfirm rocm-migraphx || { log_error "Failed to install rocm-migraphx"; return 1; }
    else
        log_warn "Debian/Ubuntu: install rocm-migraphx manually from your ROCm repo"
    fi

    # 5. Build libvsmigx.so from vs-mlrt source
    log_info "Building libvsmigx.so from vs-mlrt source..."
    local VS_INCLUDE_DIR
    VS_INCLUDE_DIR="$(pkg-config --variable=includedir vapoursynth 2>/dev/null || echo /usr/local/include)"
    mkdir -p build_tmp && cd build_tmp || return 1

    if [ -d "vs-mlrt" ]; then rm -rf vs-mlrt; fi
    git clone --depth 1 https://github.com/AmusementClub/vs-mlrt.git || { log_error "Failed to clone vs-mlrt"; cd ..; return 1; }
    cd vs-mlrt/vsmigx
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja -DVAPOURSYNTH_INCLUDE_DIRS="$VS_INCLUDE_DIR" || { log_error "vsmigx cmake failed"; cd ../../..; return 1; }
    ninja || { log_error "vsmigx build failed"; cd ../../..; return 1; }
    cp libvsmigx.so "$VS_PLUGIN_PATH/"
    cd ../../..
    cd ..
    log_success "libvsmigx.so installed to $VS_PLUGIN_PATH/"

    # 6. Install vsmlrt.py and patch the alter_mxr_path cache bug
    log_info "Installing vsmlrt.py..."
    local VSMLRT_PY
    VSMLRT_PY="$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo /usr/lib/python3/site-packages)/vsmlrt.py"
    curl -fsSL "https://raw.githubusercontent.com/AmusementClub/vs-mlrt/master/scripts/vsmlrt.py" -o "$VSMLRT_PY" || { log_error "Failed to download vsmlrt.py"; return 1; }
    # Patch bug: alter_mxr_path cache check used wrong variable name (mxr_path instead of alter_mxr_path)
    sed -i 's/os.access(alter_mxr_path, mode=os.R_OK) and os.path.getsize(mxr_path)/os.access(alter_mxr_path, mode=os.R_OK) and os.path.getsize(alter_mxr_path)/' "$VSMLRT_PY"
    log_success "vsmlrt.py installed to $VSMLRT_PY"

    # Symlink migraphx-driver where vsmlrt.py expects it
    mkdir -p "$VS_PLUGIN_PATH/vsmlrt-hip"
    ln -sf "$(command -v migraphx-driver)" "$VS_PLUGIN_PATH/vsmlrt-hip/migraphx-driver"

    # 7. Boost (required by KNLMeansCL)
    log_info "Checking Boost..."
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        pacman -S --needed --noconfirm boost || { log_error "Failed to install boost"; return 1; }
    else
        apt install -y libboost-filesystem-dev libboost-system-dev || { log_error "Failed to install boost"; return 1; }
    fi

    # 8. KNLMeansCL VapourSynth plugin (Meson build, OpenCL spatial+temporal denoiser)
    log_info "Compiling KNLMeansCL..."
    mkdir -p build_tmp && cd build_tmp || return 1

    if [ -d "KNLMeansCL" ]; then rm -rf KNLMeansCL; fi
    git clone --depth 1 https://github.com/Khanattila/KNLMeansCL.git || { log_error "Failed to clone KNLMeansCL"; cd ..; return 1; }
    cd KNLMeansCL
    meson setup build --buildtype=release || { log_error "KNLMeansCL meson setup failed"; cd ../..; return 1; }
    ninja -C build || { log_error "KNLMeansCL build failed"; cd ../..; return 1; }

    if [ -f "build/libknlmeanscl.so" ]; then
        cp "build/libknlmeanscl.so" "$VS_PLUGIN_PATH/"
        log_success "KNLMeansCL installed to $VS_PLUGIN_PATH/"
    else
        log_error "KNLMeansCL compilation failed — build/libknlmeanscl.so not found"
        cd ../..; return 1
    fi

    cd ../..
    ldconfig

    log_success "Denoiser installed (PyTorch, vsscunet, MIGraphX, libvsmigx.so, vsmlrt.py, KNLMeansCL)."
}

uninstall_denoiser() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"

    log_info "Removing torch, vsscunet, havsfunc from venv..."
    "$VENV_DIR/bin/pip" uninstall -y torch vsscunet havsfunc || true

    log_info "Removing vs-mlrt files..."
    rm -f "$VS_PLUGIN_PATH/libvsmigx.so" || true
    rm -f "$VS_PLUGIN_PATH/vsmlrt-hip/migraphx-driver" || true
    local VSMLRT_PY
    VSMLRT_PY="$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo /usr/lib/python3/site-packages)/vsmlrt.py"
    rm -f "$VSMLRT_PY" || true

    log_info "Removing SCUNet ONNX models..."
    rm -rf "$VS_PLUGIN_PATH/models/scunet" || true

    log_info "Removing KNLMeansCL plugin..."
    rm -f "$VS_PLUGIN_PATH/libknlmeanscl.so" || true

    log_success "Denoiser dependencies removed."
}
