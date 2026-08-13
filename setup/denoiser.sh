#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

SOURCES["denoiser:vs-mlrt"]="https://github.com/AmusementClub/vs-mlrt.git|master"
SOURCES["denoiser:knlmeanscl"]="https://github.com/Khanattila/KNLMeansCL.git|master"
SOURCES["denoiser:vsmlrt-py"]="https://github.com/AmusementClub/vs-mlrt.git|master"
SOURCES["denoiser:mvsfunc"]="https://github.com/HomeOfVapourSynthEvolution/mvsfunc.git|master"
SOURCES["denoiser:havsfunc-legacy"]="https://github.com/HomeOfVapourSynthEvolution/havsfunc.git|r33"
SOURCES["denoiser:scunet-weights"]="https://github.com/cszn/SCUNet/releases/download/v1.0|pinned"
# The three SMDegrain plugins. Arch takes these from the AUR; Ubuntu has no
# AUR, so it builds the same upstreams the AUR packages track. mvtools is
# pinned to v29 to match Arch's vapoursynth-plugin-mvtools 29, and the other
# two follow master because their Arch counterparts are -git packages.
SOURCES["denoiser:mvtools"]="https://github.com/dubhater/vapoursynth-mvtools.git|v29"
SOURCES["denoiser:removegrain"]="https://github.com/vapoursynth/vs-removegrain.git|master"
SOURCES["denoiser:ctmf"]="https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CTMF.git|master"
ARTIFACTS["denoiser"]="lib/vapoursynth/libvstrt.so lib/vapoursynth/libvsmigx.so lib/vapoursynth/libknlmeanscl.so lib/vapoursynth/libmvtools.so lib/vapoursynth/libremovegrain.so lib/vapoursynth/libctmf.so"

# require_debian_pkgs <what> <pkg>...
# Confirms apt packages the denoiser needs. They all live in
# debian_system_deps, so the normal path is that this only verifies them. The
# previous code ran a bare `apt install` mid-component, which fails for a user
# running setup unprivileged -- and it was the very first step of the denoiser,
# so the whole component died on it. Installing is still allowed when already
# root; otherwise the error names the one command that fixes it.
require_debian_pkgs() {
    local what="$1"; shift
    local missing=() p
    for p in "$@"; do
        dpkg -s "$p" &>/dev/null || missing+=("$p")
    done
    if [ "${#missing[@]}" -eq 0 ]; then
        log_info "$what already installed."
        return 0
    fi
    if [ "$EUID" -eq 0 ]; then
        apt install -y "${missing[@]}" \
            || { log_error "Failed to install $what (${missing[*]})"; return 1; }
        return 0
    fi
    log_error "$what missing: ${missing[*]}. Run: sudo ./setup.sh --install system_deps"
    return 1
}

# build_meson_vs_plugin <sources-name> <expected .so>
# Builds one meson-based VapourSynth plugin from SOURCES into the prefix plugin
# dir. Only the Debian/Ubuntu path calls this; Arch gets the same three plugins
# from the AUR.
build_meson_vs_plugin() {
    local name="$1" lib="$2"
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    if [ -f "$VS_PLUGIN_PATH/$lib" ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "$name already installed."
        return 0
    fi

    # Prefer the venv meson, as install_vapoursynth does, so every meson build
    # in the prefix runs one version; fall back to whatever apt provided.
    local MESON="$VENV_DIR/bin/meson"
    [ -x "$MESON" ] || MESON="meson"

    local ORIG_DIR="$(pwd)"
    mkdir -p build_tmp && cd build_tmp || return 1
    log_info "Compiling $name from source..."
    clone_src denoiser "$name" "$name" || { cd "$ORIG_DIR"; return 1; }
    cd "$name" || { cd "$ORIG_DIR"; log_error "Failed to cd into $name"; return 1; }

    "$MESON" setup build --prefix="$VS_PREFIX" --libdir="lib/vapoursynth" \
        --buildtype=release \
        || { cd "$ORIG_DIR"; log_error "$name meson setup failed"; return 1; }
    ninja -C build || { cd "$ORIG_DIR"; log_error "$name build failed"; return 1; }

    # Copy the built plugin ourselves rather than running `ninja install`.
    # RemoveGrain and CTMF ignore --libdir and install into
    # $(pkg-config --variable=libdir vapoursynth)/vapoursynth, and our bridged
    # vapoursynth.pc points libdir at the Python package directory -- so they
    # landed in .../site-packages/vapoursynth/vapoursynth/ where nothing looks
    # for them. Taking the artifact straight out of the build tree is what
    # wwxd.sh already does and does not depend on each project's install logic.
    local _built
    _built="$(find build -maxdepth 2 -name '*.so' -type f 2>/dev/null | head -1)"
    if [ -z "$_built" ]; then
        cd "$ORIG_DIR"
        log_error "$name built but no .so found under build/"
        return 1
    fi
    cp "$_built" "$VS_PLUGIN_PATH/$lib" \
        || { cd "$ORIG_DIR"; log_error "Failed to copy $_built to $VS_PLUGIN_PATH/$lib"; return 1; }
    cd "$ORIG_DIR"
    log_success "$name installed to $VS_PLUGIN_PATH/$lib (from $_built)"
}

install_denoiser() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    # Detect GPU if not already done
    if [ -z "$GPU_VENDOR" ]; then
        detect_gpu
    fi

    set_native_build_flags
    # VS_INCLUDE_DIR is set after set_native_build_flags above, which adjusts
    # PKG_CONFIG_PATH to prefer $VS_PREFIX/lib/pkgconfig over pacman's R75 pc.
    # Defining it earlier made pkg-config find pacman's vapoursynth.pc first and
    # return /usr/lib/python3.14/site-packages/vapoursynth/include (no VapourSynth.h).
    local VS_INCLUDE_DIR
    VS_INCLUDE_DIR="$(pkg-config --variable=includedir vapoursynth 2>/dev/null || echo "$VS_PREFIX/include/vapoursynth")"
    log_info "VS_INCLUDE_DIR resolved to $VS_INCLUDE_DIR"
    # AUR builds (paru / makepkg) cannot run as root, so we need a non-root
    # user to drop into when invoked via sudo. When the script is run by a
    # regular user (the recommended path now), $USER is already that user
    # and `sudo -u $USER paru ...` is a no-op-equivalent — paru just runs
    # in the same uid, prompting for the targeted user's sudo password
    # only when it needs to escalate for the system-install phase.
    local _aur_user
    if [ "$EUID" -eq 0 ]; then
        _aur_user="${SUDO_USER:-}"
    else
        _aur_user="$USER"
    fi

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
        require_debian_pkgs "OpenCL ICD loader" ocl-icd-opencl-dev opencl-headers || return 1
    fi

    # =========================================================================
    # 2. GPU backend: NVIDIA (TensorRT) or AMD (MIGraphX) or CPU
    # =========================================================================
    if [ "$GPU_VENDOR" = "nvidia" ] || [ "$GPU_VENDOR" = "both" ]; then
        log_info "NVIDIA GPU detected — installing TensorRT backend..."

        # 2a. cuDNN (official repo)
        if [ "$DISTRO_FAMILY" = "arch" ]; then
            if pacman -Q cudnn &>/dev/null; then
                log_info "cudnn already installed."
            else
                pacman -S --needed --noconfirm cudnn || { log_error "Failed to install cudnn. Run: sudo pacman -S cudnn"; return 1; }
            fi
        elif dpkg -l 'libcudnn9-cuda-*' 2>/dev/null | grep -q '^ii'; then
            log_info "cudnn already installed."
        else
            log_warn "No libcudnn9-cuda-* package. Run: sudo setup.sh --install system_deps. Both the ORT TRT EP and the CUDA EP need libcudnn.so.9."
        fi

        # 2b. TensorRT (AUR — cannot build as root, use SUDO_USER)
        #
        # This is the SYSTEM TensorRT (currently 11.x), and ONLY the vstrt /
        # SCUNet path in 2d needs it, at compile time, for its headers. The BSVD
        # lane does not use it at all — section 9 installs its own TensorRT 10
        # runtime into the venv, because the ORT provider hard-links .so.10.
        # So a failure here must NOT abort the component: it used to `return 1`
        # and take section 9 down with it, which left every host that cannot
        # build the AUR package with no BSVD TRT EP for no reason. Failure now
        # skips 2d and nothing else.
        local _have_sys_trt=1
        if [ "$DISTRO_FAMILY" != "arch" ]; then
            # Ubuntu has no AUR. The headers vstrt compiles against come from
            # apt instead, pinned alongside the runtime by system_deps so one
            # transaction fixes the TensorRT version. Nothing to build here.
            if dpkg -s libnvinfer-headers-dev &>/dev/null; then
                log_info "TensorRT headers already installed (apt)."
            else
                log_warn "No libnvinfer-headers-dev — skipping the vstrt/SCUNet plugin build. Run: sudo setup.sh --install system_deps. The BSVD lane is unaffected."
                _have_sys_trt=0
            fi
        elif ! pacman -Qi tensorrt &>/dev/null; then
            local _cuda_bin=""
            for _d in /opt/cuda/bin /usr/local/cuda/bin; do
                [ -x "$_d/nvcc" ] && { _cuda_bin="$_d"; break; }
            done
            if [ -z "$_aur_user" ] || [ "$_aur_user" = "root" ]; then
                log_warn "Cannot build AUR package 'tensorrt' as root. Set SUDO_USER or run: sudo -u <user> paru -S tensorrt"
                _have_sys_trt=0
            elif ! command -v paru &>/dev/null; then
                # system_deps.sh installs paru; this only catches a host that
                # skipped that component or lost the package since.
                log_warn "AUR helper 'paru' not found — cannot install the system tensorrt. Run: setup.sh --install system_deps"
                _have_sys_trt=0
            else
                log_info "Installing tensorrt from AUR as $_aur_user (this may take a while)..."
                # Pass CUDA toolkit path so cmake can find nvcc (WSL2: nvcc lives at /opt/cuda/bin)
                sudo -u "$_aur_user" env \
                    PATH="${_cuda_bin:+$_cuda_bin:}${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}" \
                    CUDAToolkit_ROOT="${_cuda_bin%/bin}" \
                    paru -S --needed --noconfirm tensorrt \
                    || { log_warn "Failed to install tensorrt from AUR"; _have_sys_trt=0; }
            fi
        else
            log_info "tensorrt already installed."
        fi
        [ "$_have_sys_trt" -eq 1 ] \
            || log_warn "No system TensorRT — skipping the vstrt/SCUNet plugin build. The BSVD lane is unaffected and is wired below."

        # 2c. PyTorch CUDA
        log_info "Installing PyTorch CUDA (cu128) + vsscunet + havsfunc into venv..."
        "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || { log_error "Failed to install PyTorch CUDA"; return 1; }

        # 2d. Build libvstrt.so from vs-mlrt source. Needs the system TensorRT
        # headers from 2b, so it is skipped when that package is absent.
        if [ "$_have_sys_trt" -eq 1 ]; then
            log_info "Building libvstrt.so from vs-mlrt source..."
            local ORIG_DIR="$(pwd)"
            mkdir -p build_tmp && cd build_tmp || return 1

            clone_src denoiser vs-mlrt vs-mlrt || { cd "$ORIG_DIR"; return 1; }

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
            #
            # Only the AUR tensorrt puts trtexec on PATH, at /usr/sbin/trtexec.
            # Debian's libnvinfer-bin installs it to /usr/src/tensorrt/bin,
            # which is on no PATH, so a PATH-only probe reported it missing on a
            # host that had just installed it. Without trtexec, vsmlrt rebuilds
            # every TRT engine on every run instead of caching them.
            mkdir -p "$VS_PLUGIN_PATH/vsmlrt-cuda"
            local _trtexec=""
            if command -v trtexec &>/dev/null; then
                _trtexec="$(command -v trtexec)"
            else
                local _c
                for _c in /usr/src/tensorrt/bin/trtexec /usr/local/tensorrt/bin/trtexec; do
                    [ -x "$_c" ] && { _trtexec="$_c"; break; }
                done
            fi
            if [ -n "$_trtexec" ]; then
                ln -sf "$_trtexec" "$VS_PLUGIN_PATH/vsmlrt-cuda/trtexec"
                log_info "Symlinked trtexec ($_trtexec) to $VS_PLUGIN_PATH/vsmlrt-cuda/"
            else
                log_warn "trtexec not found — vsmlrt TRT engine caching may not work"
            fi
        fi

    fi

    if [ "$GPU_VENDOR" = "amd" ] || [ "$GPU_VENDOR" = "both" ]; then
        log_info "AMD GPU detected — installing MIGraphX backend (best-effort)..."

        # 2a. PyTorch ROCm — pure-AMD hosts only. On a dual-GPU "both" host the
        # CUDA torch installed by the NVIDIA block above runs the shared SCUNet
        # ONNX export; a rocm wheel would clobber it and break export on the
        # NVIDIA card, so we keep cu128 there.
        if [ "$GPU_VENDOR" = "amd" ]; then
            local rocm_ver
            rocm_ver=$(cat /opt/rocm/.info/version 2>/dev/null | grep -oP '^\d+\.\d+' || echo "")
            if [ -z "$rocm_ver" ]; then
                log_warn "Could not detect ROCm version, defaulting to rocm6.2 index"
                rocm_ver="6.2"
            fi
            log_info "ROCm $rocm_ver detected — installing PyTorch ROCm build..."
            "$VENV_DIR/bin/pip" install torch torchvision --index-url "https://download.pytorch.org/whl/rocm${rocm_ver}" || { log_error "Failed to install PyTorch ROCm ${rocm_ver}"; return 1; }
        fi

        # 2b. MIGraphX package
        log_info "Installing MIGraphX..."
        if [ "$DISTRO_FAMILY" = "arch" ]; then
            if pacman -Q rocm-migraphx &>/dev/null || pacman -Q migraphx &>/dev/null; then
                log_info "migraphx already installed."
            else
                pacman -S --needed --noconfirm rocm-migraphx 2>/dev/null \
                    || pacman -S --needed --noconfirm migraphx \
                    || log_warn "Failed to install migraphx (tried rocm-migraphx and migraphx). libvsmigx.so may fail to build/load. Run: sudo pacman -S migraphx"
            fi
        else
            log_warn "Debian/Ubuntu: install rocm-migraphx manually from your ROCm repo"
        fi

        # 2c. Build libvsmigx.so from vs-mlrt source, or symlink existing install
        if [ -f "$VS_PLUGIN_PATH/libvsmigx.so" ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
            log_info "libvsmigx.so already in $VS_PLUGIN_PATH, skipping. (FORCE_REINSTALL=1 to rebuild)"
        elif [ -f "/usr/lib/vapoursynth/libvsmigx.so" ] && [ "$VS_PLUGIN_PATH" != "/usr/lib/vapoursynth" ] \
             && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
            ln -sf /usr/lib/vapoursynth/libvsmigx.so "$VS_PLUGIN_PATH/libvsmigx.so"
            log_success "Symlinked existing libvsmigx.so to $VS_PLUGIN_PATH/"
        else
            log_info "Building libvsmigx.so from vs-mlrt source..."
            # Best-effort: a failure warns and continues so the NVIDIA/TRT side
            # and the shared components (KNLMeansCL, SMDegrain plugins) still
            # install on a dual-GPU host with a half-broken system migraphx.
            local ORIG_DIR_MIGX="$(pwd)"
            if mkdir -p build_tmp && cd build_tmp \
               && clone_src denoiser vs-mlrt vs-mlrt \
               && cd vs-mlrt/vsmigx \
               && mkdir -p build && cd build \
               && cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja -DVAPOURSYNTH_INCLUDE_DIRECTORY="$VS_INCLUDE_DIR" \
               && ninja \
               && [ -f libvsmigx.so ]; then
                # --remove-destination: the existing entry may be a symlink to
                # /usr/lib/vapoursynth/, which plain cp would write through.
                cp --remove-destination libvsmigx.so "$VS_PLUGIN_PATH/"
                log_success "libvsmigx.so installed to $VS_PLUGIN_PATH/"
            else
                log_warn "libvsmigx.so build failed — MIGraphX native plugin unavailable on this host (SCUNet-via-migx won't work; BSVD uses the ORT shim path instead)."
            fi
            cd "$ORIG_DIR_MIGX"
        fi

        # 2d. Symlink migraphx-driver for vsmlrt.py (guard: absent if migraphx pkg failed)
        mkdir -p "$VS_PLUGIN_PATH/vsmlrt-hip"
        local _migx_driver
        _migx_driver="$(command -v migraphx-driver 2>/dev/null || true)"
        if [ -n "$_migx_driver" ]; then
            ln -sf "$_migx_driver" "$VS_PLUGIN_PATH/vsmlrt-hip/migraphx-driver"
        else
            log_warn "migraphx-driver not found — vsmlrt.py MIGraphX engine caching may not work."
        fi
    fi

    if [ "$GPU_VENDOR" != "nvidia" ] && [ "$GPU_VENDOR" != "both" ] && [ "$GPU_VENDOR" != "amd" ]; then
        log_info "No AMD/NVIDIA GPU detected — installing PyTorch CPU build (needed for ONNX export)..."
        "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu || { log_error "Failed to install PyTorch CPU"; return 1; }
    fi

    # =========================================================================
    # 3. vsscunet + onnx + havsfunc_legacy + mvsfunc_pkg (shared)
    # =========================================================================
    # NOTE: do NOT install 'havsfunc' from PyPI — v34 is a different package that lacks SMDegrain.
    # We install v33 (the original Holy's AviSynth port) manually as havsfunc_legacy below.
    "$VENV_DIR/bin/pip" install vsscunet onnx onnxscript adjust || { log_error "Failed to install vsscunet/onnx/adjust"; return 1; }

    # RVRT (vsrvrt): --no-deps avoids pulling PyPI vapoursynth stub, which shadows the
    # source-built VapourSynth and breaks ffms2/other plugins.
    # TODO: revisit RVRT perf before enabling.
    VIRTUAL_ENV="$VENV_DIR" uv pip install --no-deps vsrvrt || log_warn "Failed to install vsrvrt (RVRT denoising unavailable)"

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
        record_src denoiser mvsfunc "https://github.com/HomeOfVapourSynthEvolution/mvsfunc.git" master "$(git ls-remote https://github.com/HomeOfVapourSynthEvolution/mvsfunc.git refs/heads/master 2>/dev/null | cut -f1)"
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
        record_src denoiser havsfunc-legacy "https://github.com/HomeOfVapourSynthEvolution/havsfunc.git" r33 r33
        log_success "havsfunc_legacy installed to $_site/"
    else
        log_info "havsfunc_legacy already installed."
    fi

    # =========================================================================
    # 3.5  SMDegrain plugins: MVTools + RemoveGrain (needed for --denoise-smdegrain)
    # =========================================================================
    log_info "Installing MVTools and RemoveGrain VapourSynth plugins..."
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        if pacman -Q vapoursynth-plugin-mvtools &>/dev/null; then
            log_info "vapoursynth-plugin-mvtools already installed."
        else
            pacman -S --needed --noconfirm vapoursynth-plugin-mvtools || { log_error "Failed to install vapoursynth-plugin-mvtools. Run: sudo pacman -S vapoursynth-plugin-mvtools"; return 1; }
        fi
        # Pacman's plugin install location moved from /usr/lib/vapoursynth/
        # to /usr/lib/python3.X/site-packages/vapoursynth/plugins/ between
        # v74 and v75+. find_pacman_vs_plugin probes both layouts.
        _mvtools_src="$(find_pacman_vs_plugin mvtools)"
        if [ -n "$_mvtools_src" ]; then
            ln -sf "$_mvtools_src" "$VS_PLUGIN_PATH/libmvtools.so"
            log_info "Symlinked $_mvtools_src to $VS_PLUGIN_PATH/libmvtools.so"
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
        _rg_src="$(find_pacman_vs_plugin removegrain)"
        if [ -n "$_rg_src" ]; then
            ln -sf "$_rg_src" "$VS_PLUGIN_PATH/libremovegrain.so"
            log_info "Symlinked $_rg_src to $VS_PLUGIN_PATH/libremovegrain.so"
        fi

        # CTMF — median filter needed by ContraSharpening in havsfunc.
        # The AUR PKGBUILD still uses the V3 API (#include <VapourSynth.h>),
        # which neither R76 (our source build) nor pacman v75 ships in their
        # installed headers — both apply meson.build's `exclude_files` rule.
        # We backfill the V3 headers into $VS_PREFIX/include/vapoursynth/
        # during install_vapoursynth, so pass them through to paru's build
        # env via CPPFLAGS. Once the AUR package is updated for V4 API this
        # can be dropped.
        if ! pacman -Qi vapoursynth-plugin-ctmf-git &>/dev/null; then
            if [ -z "$_aur_user" ] || [ "$_aur_user" = "root" ]; then
                log_warn "Cannot install vapoursynth-plugin-ctmf-git as root. Run manually: sudo -u <user> paru -S vapoursynth-plugin-ctmf-git"
            else
                log_info "Installing vapoursynth-plugin-ctmf-git from AUR as $_aur_user (with V3-header CPPFLAGS bridge)..."
                sudo -u "$_aur_user" \
                    CPPFLAGS="-I$VS_PREFIX/include/vapoursynth ${CPPFLAGS:-}" \
                    paru -S --needed --noconfirm vapoursynth-plugin-ctmf-git || \
                        log_warn "Failed to install vapoursynth-plugin-ctmf-git (SMDegrain ContraSharpening may not work)"
            fi
        else
            log_info "vapoursynth-plugin-ctmf already installed."
        fi
        _ctmf_src="$(find_pacman_vs_plugin ctmf)"
        if [ -n "$_ctmf_src" ]; then
            ln -sf "$_ctmf_src" "$VS_PLUGIN_PATH/libctmf.so"
            log_info "Symlinked $_ctmf_src to $VS_PLUGIN_PATH/libctmf.so"
        fi
    else
        # All three build with meson, which is how their AUR packages build
        # them too (arch-meson is meson plus distro defaults). A failure here
        # costs --denoise-smdegrain and nothing else, so none of them aborts
        # the component.
        build_meson_vs_plugin mvtools libmvtools.so \
            || log_warn "mvtools build failed (--denoise-smdegrain will not work)"
        build_meson_vs_plugin removegrain libremovegrain.so \
            || log_warn "removegrain build failed (SMDegrain chroma may not work)"
        # CTMF still includes <VapourSynth.h>, the V3 API header, which neither
        # our R76 build nor the distro package installs -- both apply
        # meson.build's exclude_files rule. install_vapoursynth backfills the
        # V3 headers into $VS_PREFIX/include/vapoursynth, so point the compiler
        # at them, exactly as the arch branch does through paru's build env.
        CPPFLAGS="-I$VS_PREFIX/include/vapoursynth ${CPPFLAGS:-}" \
            build_meson_vs_plugin ctmf libctmf.so \
            || log_warn "ctmf build failed (SMDegrain ContraSharpening may not work)"
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
    record_src denoiser scunet-weights "https://github.com/cszn/SCUNet/releases/download/v1.0" pinned v1.0

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
        local _missing=()
        for _p in cmake ninja meson; do
            pacman -Q "$_p" &>/dev/null || _missing+=("$_p")
        done
        if [ ${#_missing[@]} -eq 0 ]; then
            log_info "KNLMeansCL build deps (cmake ninja meson) already installed."
        else
            pacman -S --needed --noconfirm "${_missing[@]}" \
                || { log_error "Failed to install KNLMeansCL build deps (${_missing[*]}). Run: sudo pacman -S ${_missing[*]}"; return 1; }
        fi
        unset _missing _p
    else
        require_debian_pkgs "KNLMeansCL build deps" cmake ninja-build meson || return 1
    fi

    # =========================================================================
    # 6. Install vsmlrt.py (shared — supports TRT, MIGX, ORT, etc.)
    # =========================================================================
    log_info "Installing vsmlrt.py..."
    local VSMLRT_PY
    # Land vsmlrt.py inside the venv's site-packages, not the system one
    # (system /usr/lib/python3.X/site-packages requires sudo and would
    # pollute the pacman-owned tree).
    VSMLRT_PY="$("$VENV_DIR/bin/python" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "$VENV_DIR/lib/python3/site-packages")/vsmlrt.py"
    curl -fsSL "https://raw.githubusercontent.com/AmusementClub/vs-mlrt/master/scripts/vsmlrt.py" -o "$VSMLRT_PY" || { log_error "Failed to download vsmlrt.py"; return 1; }
    # Patch bug: alter_mxr_path cache check used wrong variable name
    sed -i 's/os.access(alter_mxr_path, mode=os.R_OK) and os.path.getsize(mxr_path)/os.access(alter_mxr_path, mode=os.R_OK) and os.path.getsize(alter_mxr_path)/' "$VSMLRT_PY"
    record_src denoiser vsmlrt-py "https://github.com/AmusementClub/vs-mlrt.git" master "$(git ls-remote https://github.com/AmusementClub/vs-mlrt.git refs/heads/master 2>/dev/null | cut -f1)"
    log_success "vsmlrt.py installed to $VSMLRT_PY"

    # =========================================================================
    # 7. Boost (required by KNLMeansCL)
    # =========================================================================
    log_info "Checking Boost..."
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        if pacman -Q boost &>/dev/null; then
            log_info "boost already installed."
        else
            pacman -S --needed --noconfirm boost || { log_error "Failed to install boost. Run: sudo pacman -S boost"; return 1; }
        fi
    else
        require_debian_pkgs "boost" libboost-filesystem-dev libboost-system-dev || return 1
    fi

    # =========================================================================
    # 8. KNLMeansCL VapourSynth plugin (OpenCL spatial+temporal denoiser)
    # =========================================================================
    log_info "Compiling KNLMeansCL..."
    local ORIG_DIR2="$(pwd)"
    mkdir -p build_tmp && cd build_tmp || return 1

    clone_src denoiser knlmeanscl KNLMeansCL || { cd "$ORIG_DIR2"; return 1; }
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

    # =========================================================================
    # 9. BSVD denoiser (V2 stateful streaming via ORT-TRT/ORT-MIGraphX EP)
    #    Python-side ORT (no VS plugin) + staged model assets.
    # =========================================================================
    log_info "Installing onnxruntime for BSVD denoise path..."
    # PyAV is required by the default --bsvd-sigma=auto path
    # (tools/bsvd_optsig.py imports av to decode warmup frames).
    "$VENV_DIR/bin/pip" install -U av \
        || log_warn "Failed to install PyAV — --denoise-bsvd with sigma=auto will fail; pass --bsvd-sigma <float>"
    if [ "$GPU_VENDOR" = "nvidia" ] || [ "$GPU_VENDOR" = "both" ]; then
        # onnxruntime-gpu ships CUDA + TensorRT EPs (the latter is what we use
        # for BSVD V2 stateful streaming).
        # PyPI has no aarch64 build of onnxruntime-gpu, and never has, for any
        # release. NVIDIA's jetson-ai-lab devpi is the only index that carries
        # one, and it is keyed by CUDA version (sbsa/cu130 for CUDA 13.0).
        local _ort_index=()
        if [ "$(uname -m)" = "aarch64" ]; then
            local _ort_cu
            _ort_cu="$( { command -v nvcc >/dev/null && nvcc --version || /usr/local/cuda/bin/nvcc --version; } 2>/dev/null \
                | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1\2/p' | head -1)"
            if [ -n "$_ort_cu" ]; then
                _ort_index=(--extra-index-url "https://pypi.jetson-ai-lab.io/sbsa/cu${_ort_cu}")
                log_info "aarch64: adding the jetson-ai-lab sbsa/cu${_ort_cu} index for onnxruntime-gpu."
            else
                log_warn "aarch64 with no detectable CUDA version — onnxruntime-gpu will not resolve on PyPI."
            fi
        fi
        "$VENV_DIR/bin/pip" install -U "${_ort_index[@]}" onnxruntime-gpu \
            || log_warn "Failed to install onnxruntime-gpu — --denoise-bsvd will fail at dispatch"

        # TensorRT RUNTIME for the ORT TRT EP. onnxruntime-gpu's provider .so
        # (libonnxruntime_providers_tensorrt.so) hard-links libnvinfer.so.10 and
        # libnvonnxparser.so.10 (TensorRT *10*). The EP is LISTED by ORT regardless,
        # so dispatch's ctypes.util.find_library("nvinfer") sees the system AUR
        # `tensorrt` (now 11.x → soname .so.11), picks the TRT EP, then the
        # InferenceSession dies at create with "libnvinfer.so.10 => not found".
        # Fix: install the TRT-10 runtime libs (py-agnostic wheel — ORT needs only
        # the .so's, not the cp-specific python bindings) INTO the managed venv and
        # register them with the dynamic loader, so both ORT's dlopen and dispatch's
        # find_library("nvinfer") resolve .so.10 with no manual LD_LIBRARY_PATH.
        # Pinned <11 to match the provider soname (the system 11.x libs are left
        # untouched for the vstrt/SCUNet + standalone-engine paths that use them).
        log_info "Installing TensorRT 10 runtime libs for the onnxruntime TRT EP (BSVD)..."
        local _trt_libdir=""
        local _trt_on_loader_path=0
        if ldconfig -p 2>/dev/null | grep -q 'libnvinfer\.so\.10'; then
            # A system TensorRT 10 already satisfies the provider. This is the
            # normal Ubuntu case: system_deps pins libnvinfer10 from apt, which
            # lands in the default loader path. It is also the ONLY route on
            # aarch64, where no index publishes a TensorRT wheel at all.
            # Arch is unaffected: its AUR tensorrt is 11.x, soname .so.11, so
            # this test fails there and the pip path below still runs.
            _trt_on_loader_path=1
        elif "$VENV_DIR/bin/pip" install -U 'tensorrt-cu12-libs<11'; then
            _trt_libdir="$("$VENV_DIR/bin/python" -c "import tensorrt_libs,os;print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null)"
        fi
        # The ORT provider needs libcudnn.so.9 as well as libnvinfer.so.10, and
        # registering only the tensorrt dir leaves the session dying on cudnn.
        # Arch hosts get it from pacman `cudnn` in /usr/lib, but that package is
        # not universal, and torch already pulls a copy into the venv. Register
        # both dirs so the EP does not depend on which of the two is present.
        local _cudnn_libdir
        _cudnn_libdir="$(echo "$VENV_DIR"/lib/python3.*/site-packages/nvidia/cudnn/lib)"
        [ -f "$_cudnn_libdir/libcudnn.so.9" ] || _cudnn_libdir=""
        if [ "$_trt_on_loader_path" -eq 1 ]; then
            log_success "BSVD TRT EP wired: libnvinfer.so.10 is already on the loader path (system TensorRT)."
        elif [ -n "$_trt_libdir" ] && [ -f "$_trt_libdir/libnvinfer.so.10" ]; then
            if { [ "$EUID" -eq 0 ] || [ -w /etc/ld.so.conf.d ]; } \
               && printf '%s\n' "$_trt_libdir" ${_cudnn_libdir:+"$_cudnn_libdir"} \
                    > /etc/ld.so.conf.d/archav1an-tensorrt.conf 2>/dev/null \
               && ldconfig 2>/dev/null; then
                log_success "BSVD TRT EP wired: libnvinfer.so.10 installed + registered on the loader path ($_trt_libdir${_cudnn_libdir:+, $_cudnn_libdir})."
            else
                log_warn "BSVD TRT EP: libnvinfer.so.10 installed at $_trt_libdir but not registered globally (need root). Run --denoise-bsvd with:  LD_LIBRARY_PATH=$_trt_libdir${_cudnn_libdir:+:$_cudnn_libdir}:\$LD_LIBRARY_PATH"
            fi
        elif [ "$(uname -m)" = "aarch64" ]; then
            # Naming the wheel here would be wrong advice: no index publishes a
            # TensorRT wheel for aarch64, so apt is the only source.
            log_warn "BSVD TRT EP: no libnvinfer.so.10 on the loader path. aarch64 has no TensorRT wheel, so install it with: sudo setup.sh --install system_deps. --denoise-bsvd will fall back to the slower CUDA EP."
        else
            log_warn "BSVD TRT EP: could not install libnvinfer.so.10 (tensorrt-cu12-libs<11). --denoise-bsvd will fall back to the slower CUDA EP."
        fi
    fi

    if [ "$GPU_VENDOR" = "amd" ] || [ "$GPU_VENDOR" = "both" ]; then
        # MIGraphX ORT (BSVD V2 stateful streaming on the AMD GPU) lives in an
        # EXTERNAL py3.12 venv shim, NOT the managed py3.14 venv: onnxruntime_migraphx
        # ships cp312-only wheels, and python_libs.sh rebuilds $VS_PREFIX/venv on any
        # python-version drift (which would nuke a 3.12 venv placed there). setup.sh
        # therefore neither builds nor rebuilds the shim — it only DETECTS it and
        # prints the dispatch wiring, so normal --install/--update never touch it.
        # See memory encoder-host-migraphx-env-restored.md for how to (re)build the shim.
        local _migx_venv="" _cand
        for _cand in "${MIGX_VENV:-}" \
                     "/home/${SUDO_USER:-$USER}/reposetc/bsvd/migraphx-venv" \
                     "$VS_PREFIX/venv-migx"; do
            [ -n "$_cand" ] && [ -x "$_cand/bin/python" ] && { _migx_venv="$_cand"; break; }
        done
        if [ -n "$_migx_venv" ] \
           && "$_migx_venv/bin/python" -c "import onnxruntime as o; assert 'MIGraphXExecutionProvider' in o.get_available_providers()" 2>/dev/null; then
            log_success "MIGraphX ORT shim found: $_migx_venv (MIGraphXExecutionProvider available)"
            log_info    "  Run BSVD dispatch on the AMD GPU with:"
            log_info    "    VSPIPE=$_migx_venv/bin/vspipe $_migx_venv/bin/python tools/svtav1-dispatch.py -i IN -o OUT.mkv --denoise-bsvd ..."
        else
            log_warn "No working MIGraphX ORT venv (checked \$MIGX_VENV, ~/reposetc/bsvd/migraphx-venv, $VS_PREFIX/venv-migx)."
            log_warn "  --denoise-bsvd on the AMD GPU needs a py3.12 venv with onnxruntime_migraphx; set MIGX_VENV=<path> or rebuild per memory encoder-host-migraphx-env-restored.md."
        fi
    fi

    # Split-host denoise (--remote-denoise): report readiness only. The remote's
    # install is this same `setup.sh --install denoiser`, run on that box, and
    # opening the inbound port is the user's call — setup never touches firewalls.
    if command -v ssh >/dev/null 2>&1 && command -v rsync >/dev/null 2>&1; then
        log_success "ssh + rsync present — split-host denoise (--remote-denoise) usable"
        log_info    "  Remote host needs this repo at --remote-root (default ~/archav1an) with --install denoiser run there."
        log_info    "  Open the return port on THIS host, scoped to the denoise host, e.g.:"
        log_info    "    sudo ufw allow from <remote-lan-ip> to any port 5300 proto tcp"
    else
        log_warn "ssh and/or rsync missing — --remote-denoise (split-host denoise) will not work."
    fi

    log_info "Staging BSVD model assets (ft_ep5 ONNX + σ-estimator)..."
    local _bsvd_repo
    if [ "$EUID" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
        _bsvd_repo="/home/$SUDO_USER/gitproj/bsvd"
    else
        _bsvd_repo="$HOME/gitproj/bsvd"
    fi
    local _archav1an_models
    # BASH_SOURCE, not $0: this file is sourced by setup.sh, so $0 points at
    # setup.sh's caller and would stage models into the repo's parent dir.
    _archav1an_models="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/models"
    mkdir -p "$_archav1an_models"
    local _src _dst _pair
    for _pair in \
        "$_bsvd_repo/runs/bsvd_v4_finetune/ft_ep5_stateful_v2_dyn_fp16.onnx:$_archav1an_models/bsvd_ft_ep5_stateful_v2_dyn_fp16.onnx" \
        "$_bsvd_repo/runs/sigma_estimator_v3.pth:$_archav1an_models/bsvd_sigma_estimator_v3.pth"
    do
        _src="${_pair%:*}"
        _dst="${_pair#*:}"
        if [ -f "$_dst" ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
            log_info "  $(basename "$_dst") already present"
        elif [ -f "$_src" ]; then
            cp "$_src" "$_dst" && log_success "  staged $(basename "$_dst") from $_src"
        else
            log_warn "  source missing: $_src — copy manually if you want --denoise-bsvd on this host"
        fi
    done

    # This component's pip installs (vsscunet and friends) depend on
    # vapoursynth, so pip reinstalls the PyPI wheel that python_libs removed.
    # It bundles its own core and shadows the source-built module, which shows
    # up as "Python module version is R79 but the core library is R76".
    drop_pip_vapoursynth_stub

    case "$GPU_VENDOR" in
        both)   log_success "Denoiser installed (CUDA+TensorRT libvstrt.so AND MIGraphX libvsmigx.so [best-effort], vsscunet, vsmlrt.py, onnxruntime-gpu, KNLMeansCL, BSVD assets staged)." ;;
        nvidia) log_success "Denoiser installed (PyTorch CUDA, vsscunet, TensorRT, libvstrt.so, vsmlrt.py, onnxruntime-gpu, KNLMeansCL, BSVD assets staged)." ;;
        amd)    log_success "Denoiser installed (PyTorch ROCm, vsscunet, MIGraphX, libvsmigx.so, vsmlrt.py, KNLMeansCL, BSVD assets staged)." ;;
        *)      log_success "Denoiser installed (PyTorch CPU, vsscunet, vsmlrt.py, KNLMeansCL, BSVD assets staged)." ;;
    esac
}

uninstall_denoiser() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"

    log_info "Removing denoiser Python packages from venv..."
    # Mirror what install_denoiser pip-installs (havsfunc is never pip-installed
    # under that name — it is curled in as havsfunc_legacy.py below).
    "$VENV_DIR/bin/pip" uninstall -y torch torchvision vsscunet vsrvrt \
        onnxruntime-gpu onnx onnxscript adjust av || true

    log_info "Removing havsfunc_legacy + mvsfunc_pkg from venv site-packages..."
    local _site
    _site="$("$VENV_DIR/bin/python3" -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null)"
    if [ -n "$_site" ]; then
        rm -f "$_site/havsfunc_legacy.py" || true
        rm -rf "$_site/mvsfunc_pkg" || true
    fi

    log_info "Removing vs-mlrt files..."
    rm -f "$VS_PLUGIN_PATH/libvstrt.so" || true
    rm -f "$VS_PLUGIN_PATH/libvsmigx.so" || true
    rm -f "$VS_PLUGIN_PATH/vsmlrt-cuda/trtexec" || true
    rm -f "$VS_PLUGIN_PATH/vsmlrt-hip/migraphx-driver" || true
    local VSMLRT_PY
    # Land vsmlrt.py inside the venv's site-packages, not the system one
    # (system /usr/lib/python3.X/site-packages requires sudo and would
    # pollute the pacman-owned tree).
    VSMLRT_PY="$("$VENV_DIR/bin/python" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "$VENV_DIR/lib/python3/site-packages")/vsmlrt.py"
    rm -f "$VSMLRT_PY" || true

    log_info "Removing SCUNet ONNX models..."
    rm -rf "$VS_PLUGIN_PATH/models/scunet" || true

    log_info "Removing KNLMeansCL plugin..."
    rm -f "$VS_PLUGIN_PATH/libknlmeanscl.so" || true

    log_info "Removing SMDegrain plugin symlinks (mvtools/removegrain/ctmf)..."
    # Symlinks only — the pacman packages they point at are left installed.
    rm -f "$VS_PLUGIN_PATH/libmvtools.so" "$VS_PLUGIN_PATH/libremovegrain.so" \
        "$VS_PLUGIN_PATH/libctmf.so" || true

    # Staged BSVD model assets in models/ are git-tracked repo content — left in place.

    rm -f "$MANIFEST_DIR/denoiser.src"
    log_success "Denoiser dependencies removed."
}
