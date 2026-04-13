#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_denoiser() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    # Detect GPU if not already done
    if [ -z "$GPU_VENDOR" ]; then
        detect_gpu
    fi

    local VS_INCLUDE_DIR
    VS_INCLUDE_DIR="$(pkg-config --variable=includedir vapoursynth 2>/dev/null || echo /usr/local/include)"
    local _aur_user="${SUDO_USER:-}"

    # =========================================================================
    # 1. OpenCL runtime (required by KNLMeansCL)
    # =========================================================================
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

    # =========================================================================
    # 2. GPU backend: NVIDIA (TensorRT) or AMD (MIGraphX) or CPU
    # =========================================================================
    if [ "$GPU_VENDOR" = "nvidia" ] || [ "$GPU_VENDOR" = "both" ]; then
        log_info "NVIDIA GPU detected — installing TensorRT backend..."

        # 2a. cuDNN (official repo)
        if [ "$DISTRO_FAMILY" = "arch" ]; then
            pacman -S --needed --noconfirm cudnn || { log_error "Failed to install cudnn"; return 1; }
        else
            log_warn "Debian/Ubuntu: install libcudnn9-cuda-12 manually from NVIDIA repos"
        fi

        # 2b. TensorRT (AUR — cannot build as root, use SUDO_USER)
        if ! pacman -Qi tensorrt &>/dev/null; then
            if [ -z "$_aur_user" ] || [ "$_aur_user" = "root" ]; then
                log_error "Cannot install AUR package 'tensorrt' as root. Set SUDO_USER or run: sudo -u <user> paru -S tensorrt"
                return 1
            fi
            log_info "Installing tensorrt from AUR as $_aur_user (this may take a while)..."
            # Pass CUDA toolkit path so cmake can find nvcc (WSL2: nvcc lives at /opt/cuda/bin)
            local _cuda_bin=""
            for _d in /opt/cuda/bin /usr/local/cuda/bin; do
                [ -x "$_d/nvcc" ] && { _cuda_bin="$_d"; break; }
            done
            sudo -u "$_aur_user" env \
                PATH="${_cuda_bin:+$_cuda_bin:}${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}" \
                CUDAToolkit_ROOT="${_cuda_bin%/bin}" \
                paru -S --needed --noconfirm tensorrt || { log_error "Failed to install tensorrt from AUR"; return 1; }
        else
            log_info "tensorrt already installed."
        fi

        # 2c. PyTorch CUDA
        log_info "Installing PyTorch CUDA (cu128) + vsscunet + havsfunc into venv..."
        "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || { log_error "Failed to install PyTorch CUDA"; return 1; }

        # 2d. Build libvstrt.so from vs-mlrt source
        log_info "Building libvstrt.so from vs-mlrt source..."
        local ORIG_DIR="$(pwd)"
        mkdir -p build_tmp && cd build_tmp || return 1

        if [ -d "vs-mlrt" ]; then rm -rf vs-mlrt; fi
        git clone --depth 1 https://github.com/AmusementClub/vs-mlrt.git || { log_error "Failed to clone vs-mlrt"; cd "$ORIG_DIR"; return 1; }

        cd vs-mlrt/vstrt
        mkdir -p build && cd build
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -G Ninja \
            -DVAPOURSYNTH_INCLUDE_DIRECTORY="$VS_INCLUDE_DIR" \
            -DCMAKE_CXX_FLAGS="-ffast-math" \
            || { log_error "vstrt cmake failed"; cd "$ORIG_DIR"; return 1; }
        ninja || { log_error "vstrt build failed"; cd "$ORIG_DIR"; return 1; }

        # Output is libvstrt.so (standard TRT) or libvstrt_rtx.so (TRT RTX)
        local _vstrt_lib=""
        if [ -f "libvstrt_rtx.so" ]; then
            _vstrt_lib="libvstrt_rtx.so"
        elif [ -f "libvstrt.so" ]; then
            _vstrt_lib="libvstrt.so"
        else
            log_error "vstrt build succeeded but no libvstrt*.so found"
            cd "$ORIG_DIR"; return 1
        fi
        cp "$_vstrt_lib" "$VS_PLUGIN_PATH/libvstrt.so"
        log_success "libvstrt.so installed to $VS_PLUGIN_PATH/"
        cd "$ORIG_DIR"

        # 2e. Symlink trtexec for vsmlrt.py
        mkdir -p "$VS_PLUGIN_PATH/vsmlrt-cuda"
        if command -v trtexec &>/dev/null; then
            ln -sf "$(command -v trtexec)" "$VS_PLUGIN_PATH/vsmlrt-cuda/trtexec"
            log_info "Symlinked trtexec to $VS_PLUGIN_PATH/vsmlrt-cuda/"
        else
            log_warn "trtexec not found in PATH — vsmlrt TRT engine caching may not work"
        fi

    elif [ "$GPU_VENDOR" = "amd" ]; then
        log_info "AMD GPU detected — installing MIGraphX backend..."

        # 2a. PyTorch ROCm
        local rocm_ver
        rocm_ver=$(cat /opt/rocm/.info/version 2>/dev/null | grep -oP '^\d+\.\d+' || echo "")
        if [ -z "$rocm_ver" ]; then
            log_warn "Could not detect ROCm version, defaulting to rocm6.2 index"
            rocm_ver="6.2"
        fi
        log_info "ROCm $rocm_ver detected — installing PyTorch ROCm build..."
        "$VENV_DIR/bin/pip" install torch torchvision --index-url "https://download.pytorch.org/whl/rocm${rocm_ver}" || { log_error "Failed to install PyTorch ROCm ${rocm_ver}"; return 1; }

        # 2b. MIGraphX package
        log_info "Installing MIGraphX..."
        if [ "$DISTRO_FAMILY" = "arch" ]; then
            pacman -S --needed --noconfirm rocm-migraphx 2>/dev/null \
                || pacman -S --needed --noconfirm migraphx \
                || { log_error "Failed to install migraphx (tried rocm-migraphx and migraphx)"; return 1; }
        else
            log_warn "Debian/Ubuntu: install rocm-migraphx manually from your ROCm repo"
        fi

        # 2c. Build libvsmigx.so from vs-mlrt source, or symlink existing install
        if [ -f "$VS_PLUGIN_PATH/libvsmigx.so" ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
            log_info "libvsmigx.so already in $VS_PLUGIN_PATH, skipping. (FORCE_REINSTALL=1 to rebuild)"
        elif [ -f "/usr/lib/vapoursynth/libvsmigx.so" ] && [ "$VS_PLUGIN_PATH" != "/usr/lib/vapoursynth" ]; then
            ln -sf /usr/lib/vapoursynth/libvsmigx.so "$VS_PLUGIN_PATH/libvsmigx.so"
            log_success "Symlinked existing libvsmigx.so to $VS_PLUGIN_PATH/"
        else
            log_info "Building libvsmigx.so from vs-mlrt source..."
            local ORIG_DIR="$(pwd)"
            mkdir -p build_tmp && cd build_tmp || return 1

            if [ -d "vs-mlrt" ]; then rm -rf vs-mlrt; fi
            git clone --depth 1 https://github.com/AmusementClub/vs-mlrt.git || { log_error "Failed to clone vs-mlrt"; cd "$ORIG_DIR"; return 1; }

            cd vs-mlrt/vsmigx
            mkdir -p build && cd build
            cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja -DVAPOURSYNTH_INCLUDE_DIRS="$VS_INCLUDE_DIR" \
                || { log_error "vsmigx cmake failed"; cd "$ORIG_DIR"; return 1; }
            ninja || { log_error "vsmigx build failed"; cd "$ORIG_DIR"; return 1; }
            cp libvsmigx.so "$VS_PLUGIN_PATH/"
            log_success "libvsmigx.so installed to $VS_PLUGIN_PATH/"
            cd "$ORIG_DIR"
        fi

        # 2d. Symlink migraphx-driver for vsmlrt.py
        mkdir -p "$VS_PLUGIN_PATH/vsmlrt-hip"
        ln -sf "$(command -v migraphx-driver)" "$VS_PLUGIN_PATH/vsmlrt-hip/migraphx-driver"

    else
        log_info "No AMD/NVIDIA GPU detected — installing PyTorch CPU build (needed for ONNX export)..."
        "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu || { log_error "Failed to install PyTorch CPU"; return 1; }
    fi

    # =========================================================================
    # 3. vsscunet + onnx + havsfunc_legacy + mvsfunc_pkg (shared)
    # =========================================================================
    # NOTE: do NOT install 'havsfunc' from PyPI — v34 is a different package that lacks SMDegrain.
    # We install v33 (the original Holy's AviSynth port) manually as havsfunc_legacy below.
    "$VENV_DIR/bin/pip" install vsscunet onnx onnxscript adjust || { log_error "Failed to install vsscunet/onnx/adjust"; return 1; }

    local _site
    _site="$("$VENV_DIR/bin/python3" -c "import sysconfig; print(sysconfig.get_path('purelib'))")"

    # mvsfunc_pkg — havsfunc dependency (PyPI mvsfunc is unmaintained/broken; install from GitHub as a package)
    if [ ! -d "$_site/mvsfunc_pkg" ] || [ "${FORCE_REINSTALL:-0}" = "1" ]; then
        log_info "Installing mvsfunc_pkg from GitHub..."
        mkdir -p "$_site/mvsfunc_pkg"
        curl -fsSL "https://raw.githubusercontent.com/HomeOfVapourSynthEvolution/mvsfunc/master/mvsfunc/mvsfunc.py" \
            -o "$_site/mvsfunc_pkg/mvsfunc.py" || { log_error "Failed to download mvsfunc"; return 1; }
        # Remove relative _metadata import that breaks standalone use
        sed -i '/^from \._metadata import/d' "$_site/mvsfunc_pkg/mvsfunc.py"
        printf 'from .mvsfunc import *\n' > "$_site/mvsfunc_pkg/__init__.py"
        log_success "mvsfunc_pkg installed to $_site/mvsfunc_pkg/"
    else
        log_info "mvsfunc_pkg already installed."
    fi

    # havsfunc_legacy — v33 from original GitHub repo, patched to use mvsfunc_pkg
    # NOTE: master is now v34 (restructured, no SMDegrain). Pin to r33 tag.
    if [ ! -f "$_site/havsfunc_legacy.py" ] || [ "${FORCE_REINSTALL:-0}" = "1" ]; then
        log_info "Installing havsfunc_legacy (r33) from GitHub..."
        curl -fsSL "https://raw.githubusercontent.com/HomeOfVapourSynthEvolution/havsfunc/refs/tags/r33/havsfunc.py" \
            -o "$_site/havsfunc_legacy.py" || { log_error "Failed to download havsfunc r33"; return 1; }
        sed -i 's/^import mvsfunc as mvf$/import mvsfunc_pkg as mvf/' "$_site/havsfunc_legacy.py"
        sed -i 's/from mvsfunc import /from mvsfunc_pkg import /g' "$_site/havsfunc_legacy.py"
        # VS R73+ no longer strips leading underscores from reserved-word kwargs;
        # havsfunc r33 uses _global= but MVTools expects global_ (trailing underscore).
        sed -i 's/_global=/global_=/g' "$_site/havsfunc_legacy.py"
        log_success "havsfunc_legacy installed to $_site/"
    else
        log_info "havsfunc_legacy already installed."
    fi

    # =========================================================================
    # 3.5  SMDegrain plugins: MVTools + RemoveGrain (needed for --denoise-smdegrain)
    # =========================================================================
    log_info "Installing MVTools and RemoveGrain VapourSynth plugins..."
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        pacman -S --needed --noconfirm vapoursynth-plugin-mvtools || { log_error "Failed to install vapoursynth-plugin-mvtools"; return 1; }
        # Pacman installs to /usr/lib/vapoursynth; symlink if VS_PLUGIN_PATH differs
        if [ -f "/usr/lib/vapoursynth/libmvtools.so" ] && [ "$VS_PLUGIN_PATH" != "/usr/lib/vapoursynth" ]; then
            ln -sf /usr/lib/vapoursynth/libmvtools.so "$VS_PLUGIN_PATH/libmvtools.so"
            log_info "Symlinked libmvtools.so to $VS_PLUGIN_PATH/"
        fi

        if ! pacman -Qi vapoursynth-plugin-removegrain &>/dev/null && ! pacman -Qi vapoursynth-plugin-removegrain-git &>/dev/null; then
            if [ -z "$_aur_user" ] || [ "$_aur_user" = "root" ]; then
                log_warn "Cannot install vapoursynth-plugin-removegrain-git as root. Run manually: sudo -u <user> paru -S vapoursynth-plugin-removegrain-git"
            else
                log_info "Installing vapoursynth-plugin-removegrain-git from AUR as $_aur_user..."
                sudo -u "$_aur_user" paru -S --needed --noconfirm vapoursynth-plugin-removegrain-git || \
                    log_warn "Failed to install vapoursynth-plugin-removegrain-git (SMDegrain chroma may not work)"
            fi
        else
            log_info "vapoursynth-plugin-removegrain already installed."
        fi
        if [ -f "/usr/lib/vapoursynth/libremovegrain.so" ] && [ "$VS_PLUGIN_PATH" != "/usr/lib/vapoursynth" ]; then
            ln -sf /usr/lib/vapoursynth/libremovegrain.so "$VS_PLUGIN_PATH/libremovegrain.so"
            log_info "Symlinked libremovegrain.so to $VS_PLUGIN_PATH/"
        fi

        # CTMF — median filter needed by ContraSharpening in havsfunc
        if ! pacman -Qi vapoursynth-plugin-ctmf-git &>/dev/null; then
            if [ -z "$_aur_user" ] || [ "$_aur_user" = "root" ]; then
                log_warn "Cannot install vapoursynth-plugin-ctmf-git as root. Run manually: sudo -u <user> paru -S vapoursynth-plugin-ctmf-git"
            else
                log_info "Installing vapoursynth-plugin-ctmf-git from AUR as $_aur_user..."
                sudo -u "$_aur_user" paru -S --needed --noconfirm vapoursynth-plugin-ctmf-git || \
                    log_warn "Failed to install vapoursynth-plugin-ctmf-git (SMDegrain ContraSharpening may not work)"
            fi
        else
            log_info "vapoursynth-plugin-ctmf already installed."
        fi
        if [ -f "/usr/lib/vapoursynth/libctmf.so" ] && [ "$VS_PLUGIN_PATH" != "/usr/lib/vapoursynth" ]; then
            ln -sf /usr/lib/vapoursynth/libctmf.so "$VS_PLUGIN_PATH/libctmf.so"
            log_info "Symlinked libctmf.so to $VS_PLUGIN_PATH/"
        fi
    else
        log_warn "Debian/Ubuntu: install vapoursynth-mvtools, vapoursynth-removegrain, and vapoursynth-ctmf manually for --denoise-smdegrain support"
    fi

    # Pre-download all SCUNet .pth model weights
    log_info "Pre-downloading SCUNet model weights..."
    "$VENV_DIR/bin/python3" -m vsscunet || { log_error "Failed to download SCUNet models"; return 1; }

    # =========================================================================
    # 4. Export SCUNet models to ONNX (shared — needed for all backends)
    # =========================================================================
    log_info "Exporting SCUNet color models to ONNX..."
    local ONNX_DIR="$VS_PLUGIN_PATH/models/scunet"
    # Symlink models dir from /usr/lib/vapoursynth if it exists there but not in VS_PLUGIN_PATH
    if [ -d "/usr/lib/vapoursynth/models" ] && [ "$VS_PLUGIN_PATH" != "/usr/lib/vapoursynth" ] && [ ! -e "$VS_PLUGIN_PATH/models" ]; then
        ln -sf /usr/lib/vapoursynth/models "$VS_PLUGIN_PATH/models"
        log_info "Symlinked existing models dir to $VS_PLUGIN_PATH/models"
    fi
    mkdir -p "$ONNX_DIR"
    "$VENV_DIR/bin/python3" -c "
import torch, sys
from pathlib import Path
import vsscunet
model_dir = Path(vsscunet.__file__).parent / 'models'
out_dir = Path(sys.argv[1])
from vsscunet.network_scunet import SCUNet

import torch.export as _tex
_h = _tex.Dim('height', min=64, max=2048)
_w = _tex.Dim('width',  min=64, max=2048)

for name in ['scunet_color_15', 'scunet_color_25', 'scunet_color_50',
             'scunet_color_real_psnr', 'scunet_color_real_gan']:
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
    ep = torch.onnx.export(m, (dummy,), dynamo=True, opset_version=18, dynamic_shapes=({2: _h, 3: _w},), input_names=['input'], output_names=['output'])
    ep.save(str(out))
    print(f'  {name}.onnx done', flush=True)
" "$ONNX_DIR" || { log_error "ONNX color export failed"; return 1; }

    # Download gray model .pth files
    log_info "Downloading gray SCUNet model weights (optional)..."
    local GRAY_PTH_DIR="$ONNX_DIR/gray_pth"
    mkdir -p "$GRAY_PTH_DIR"
    local _gray_base="https://github.com/cszn/SCUNet/releases/download/v1.0"
    for _sigma in 15 25 50; do
        local _fname="scunet_gray_${_sigma}.pth"
        if [ -f "$GRAY_PTH_DIR/$_fname" ] && [ -s "$GRAY_PTH_DIR/$_fname" ]; then
            log_info "  $_fname already present, skipping"
            continue
        fi
        if curl -fsSL "$_gray_base/$_fname" -o "$GRAY_PTH_DIR/$_fname" 2>/dev/null; then
            log_info "  Downloaded $_fname"
        else
            log_warn "  Could not download $_fname — gray models will be unavailable"
            rm -f "$GRAY_PTH_DIR/$_fname"
        fi
    done

    "$VENV_DIR/bin/python3" -c "
import torch, sys
from pathlib import Path
from vsscunet.network_scunet import SCUNet

out_dir = Path(sys.argv[1])
gray_dir = Path(sys.argv[2])

for sigma in [15, 25, 50]:
    name = f'scunet_gray_{sigma}'
    pth = gray_dir / f'{name}.pth'
    if not pth.exists():
        print(f'  skip {name}.pth (not found)', flush=True)
        continue
    out = out_dir / f'{name}.onnx'
    if out.exists() and out.stat().st_size > 1024:
        print(f'  {name}.onnx already exists, skipping', flush=True)
        continue
    print(f'  exporting {name}...', flush=True)
    m = SCUNet(in_nc=1, config=[4,4,4,4,4,4,4])
    m.load_state_dict(torch.load(str(pth), map_location='cpu', mmap=True))
    m = m.eval()
    dummy = torch.zeros(1, 1, 256, 256)
    import torch.export as _tex2; _gh = _tex2.Dim('height', min=64, max=2048); _gw = _tex2.Dim('width', min=64, max=2048)
    ep = torch.onnx.export(m, (dummy,), dynamo=True, opset_version=18, dynamic_shapes=({2: _gh, 3: _gw},), input_names=['input'], output_names=['output'])
    ep.save(str(out))
    print(f'  {name}.onnx done', flush=True)
" "$ONNX_DIR" "$GRAY_PTH_DIR" || log_warn "Gray ONNX export failed (non-fatal)"

    # Make model dir writable so vsmlrt.py can cache engine files next to the .onnx files
    chmod -R o+w "$ONNX_DIR"
    log_success "SCUNet ONNX models exported to $ONNX_DIR"

    # =========================================================================
    # 5. Build tools (cmake, ninja, meson — needed for KNLMeansCL)
    # =========================================================================
    log_info "Installing build tools (cmake, ninja, meson)..."
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        pacman -S --needed --noconfirm cmake ninja meson || { log_error "Failed to install build tools"; return 1; }
    else
        apt install -y cmake ninja-build meson || { log_error "Failed to install build tools"; return 1; }
    fi

    # =========================================================================
    # 6. Install vsmlrt.py (shared — supports TRT, MIGX, ORT, etc.)
    # =========================================================================
    log_info "Installing vsmlrt.py..."
    local VSMLRT_PY
    VSMLRT_PY="$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo /usr/lib/python3/site-packages)/vsmlrt.py"
    curl -fsSL "https://raw.githubusercontent.com/AmusementClub/vs-mlrt/master/scripts/vsmlrt.py" -o "$VSMLRT_PY" || { log_error "Failed to download vsmlrt.py"; return 1; }
    # Patch bug: alter_mxr_path cache check used wrong variable name
    sed -i 's/os.access(alter_mxr_path, mode=os.R_OK) and os.path.getsize(mxr_path)/os.access(alter_mxr_path, mode=os.R_OK) and os.path.getsize(alter_mxr_path)/' "$VSMLRT_PY"
    log_success "vsmlrt.py installed to $VSMLRT_PY"

    # =========================================================================
    # 7. Boost (required by KNLMeansCL)
    # =========================================================================
    log_info "Checking Boost..."
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        pacman -S --needed --noconfirm boost || { log_error "Failed to install boost"; return 1; }
    else
        apt install -y libboost-filesystem-dev libboost-system-dev || { log_error "Failed to install boost"; return 1; }
    fi

    # =========================================================================
    # 8. KNLMeansCL VapourSynth plugin (OpenCL spatial+temporal denoiser)
    # =========================================================================
    log_info "Compiling KNLMeansCL..."
    local ORIG_DIR2="$(pwd)"
    mkdir -p build_tmp && cd build_tmp || return 1

    if [ -d "KNLMeansCL" ]; then rm -rf KNLMeansCL; fi
    git clone --depth 1 https://github.com/Khanattila/KNLMeansCL.git || { log_error "Failed to clone KNLMeansCL"; cd "$ORIG_DIR2"; return 1; }
    cd KNLMeansCL
    meson setup build --buildtype=release || { log_error "KNLMeansCL meson setup failed"; cd "$ORIG_DIR2"; return 1; }
    ninja -C build || { log_error "KNLMeansCL build failed"; cd "$ORIG_DIR2"; return 1; }

    if [ -f "build/libknlmeanscl.so" ]; then
        cp "build/libknlmeanscl.so" "$VS_PLUGIN_PATH/"
        log_success "KNLMeansCL installed to $VS_PLUGIN_PATH/"
    else
        log_error "KNLMeansCL compilation failed — build/libknlmeanscl.so not found"
        cd "$ORIG_DIR2"; return 1
    fi

    cd "$ORIG_DIR2"
    ldconfig

    if [ "$GPU_VENDOR" = "nvidia" ] || [ "$GPU_VENDOR" = "both" ]; then
        log_success "Denoiser installed (PyTorch CUDA, vsscunet, TensorRT, libvstrt.so, vsmlrt.py, KNLMeansCL)."
    else
        log_success "Denoiser installed (PyTorch ROCm, vsscunet, MIGraphX, libvsmigx.so, vsmlrt.py, KNLMeansCL)."
    fi
}

uninstall_denoiser() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"

    log_info "Removing torch, vsscunet, havsfunc from venv..."
    "$VENV_DIR/bin/pip" uninstall -y torch vsscunet havsfunc || true

    log_info "Removing vs-mlrt files..."
    rm -f "$VS_PLUGIN_PATH/libvstrt.so" || true
    rm -f "$VS_PLUGIN_PATH/libvsmigx.so" || true
    rm -f "$VS_PLUGIN_PATH/vsmlrt-cuda/trtexec" || true
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
