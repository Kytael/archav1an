#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

# NVIDIA's CUDA apt repository — opt-in, never part of `--install A`.
#
# Ubuntu ships no cuDNN and no TensorRT at all, and its nvidia-cuda-toolkit is
# CUDA 12.0. The BSVD denoise path needs cuDNN 9 and TensorRT 10 for the
# onnxruntime TRT EP, so on Debian/Ubuntu this repo is the only source. Without
# it BSVD still runs, on the slower CUDA EP, which the denoiser reports.
#
# This is deliberately a separate target rather than something system_deps does
# for you. It installs a third-party repository and signing key, and it is the
# one action here that can break an unrelated part of the machine.

# True when the CUDA repo is already configured. Used both as this component's
# is_installed test and by debian_system_deps to decide whether cuDNN and
# TensorRT are reachable at all.
nvidia_cuda_repo_present() {
    grep -rqsi 'developer\.download\.nvidia\.com/compute/cuda' \
        /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null
}

# Prints the repo path segment, e.g. "ubuntu2404/x86_64". Empty when this host
# has no matching repo upstream.
nvidia_cuda_repo_path() {
    local _id _ver _distro _arch
    _id="$(. /etc/os-release 2>/dev/null && echo "$ID")"
    _ver="$(. /etc/os-release 2>/dev/null && echo "$VERSION_ID")"

    case "$(uname -m)" in
        x86_64)  _arch="x86_64" ;;
        # Server-class arm64 (Grace, GB10) uses sbsa. Jetson is a different
        # repo entirely (L4T/JetPack) and is not handled here.
        aarch64) _arch="sbsa" ;;
        *) return 1 ;;
    esac

    # WSL2 has its own repo, and it matters: the GPU driver comes from Windows,
    # so the normal repo's driver packages must never be installed there. The
    # wsl-ubuntu repo omits them. x86_64 only.
    if is_wsl2; then
        [ "$_arch" = "x86_64" ] || return 1
        echo "wsl-ubuntu/x86_64"
        return 0
    fi

    case "$_id" in
        ubuntu) _distro="ubuntu${_ver//./}" ;;
        debian) _distro="debian${_ver%%.*}" ;;
        *) return 1 ;;
    esac
    echo "${_distro}/${_arch}"
}

install_nvidia_repo() {
    if [ "$DISTRO_FAMILY" = "arch" ]; then
        log_error "nvidia_repo is a Debian/Ubuntu target. On Arch, cuDNN comes from pacman and TensorRT from the AUR; setup.sh --install system_deps and --install denoiser already handle both."
        return 1
    fi

    [ -z "$GPU_VENDOR" ] && detect_gpu
    if [ "$GPU_VENDOR" != "nvidia" ] && [ "$GPU_VENDOR" != "both" ]; then
        log_warn "No NVIDIA GPU detected — skipping the CUDA repo. Nothing here would be used."
        return 0
    fi

    local _path
    if ! _path="$(nvidia_cuda_repo_path)"; then
        log_error "No NVIDIA CUDA repo for this OS/architecture ($(. /etc/os-release && echo "$ID $VERSION_ID"), $(uname -m)). Install cuDNN and TensorRT by hand, or run without --denoise-bsvd's TensorRT EP."
        return 1
    fi

    local _sudo
    if ! _sudo="$(pkg_manager_sudo)"; then
        log_error "Adding an apt repository needs root, and sudo is not installed. Re-run as root."
        return 1
    fi

    if nvidia_cuda_repo_present; then
        log_info "NVIDIA CUDA repo already configured."
    else
        # The keyring package is NVIDIA's own bootstrap: it drops both the
        # signing key and the .sources file, so nothing here hand-writes apt
        # config or pipes a key through apt-key.
        local _base="https://developer.download.nvidia.com/compute/cuda/repos/${_path}"
        local _deb="cuda-keyring_1.1-1_all.deb"
        local _tmp
        _tmp="$(mktemp -d)" || return 1
        log_info "Fetching NVIDIA's keyring from ${_base}/${_deb}..."
        if ! curl -fsSL -o "$_tmp/$_deb" "${_base}/${_deb}"; then
            rm -rf "$_tmp"
            log_error "Could not download $_deb from $_base. Check the URL in a browser: the repo path for this host is '$_path'."
            return 1
        fi
        log_info "Installing the keyring and repo definition..."
        if ! $_sudo dpkg -i "$_tmp/$_deb"; then
            rm -rf "$_tmp"
            log_error "dpkg -i $_deb failed"
            return 1
        fi
        rm -rf "$_tmp"
        $_sudo apt update || { log_error "apt update failed after adding the repo"; return 1; }
        log_success "NVIDIA CUDA repo added ($_path)."
    fi

    # cuda-toolkit, NOT the `cuda` metapackage. `cuda` pulls cuda-drivers, which
    # competes with the distro's own nvidia-driver-NNN and can swap the running
    # driver out from under a working desktop. On WSL2 installing a Linux GPU
    # driver breaks CUDA outright, because the driver belongs to Windows.
    if [ -x /usr/local/cuda/bin/nvcc ] || command -v nvcc &>/dev/null; then
        log_info "nvcc already present; not installing cuda-toolkit."
    else
        log_info "Installing cuda-toolkit (no driver packages)..."
        $_sudo apt install -y cuda-toolkit \
            || { log_error "Failed to install cuda-toolkit"; return 1; }
    fi

    # Now that cuDNN and TensorRT are reachable, the normal package list asks
    # for them. Re-running system_deps is what actually installs them.
    log_info "Re-running system_deps to pick up cuDNN and TensorRT..."
    install_system_deps_debian || return 1

    log_success "NVIDIA CUDA repo ready — --denoise-bsvd can use the TensorRT EP."
}

uninstall_nvidia_repo() {
    local _sudo
    _sudo="$(pkg_manager_sudo)" || _sudo=""
    log_warn "Removing the CUDA repo does NOT remove cuDNN or TensorRT; apt will simply stop offering updates for them."
    $_sudo rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/cuda*.sources
    $_sudo apt-get remove -y cuda-keyring 2>/dev/null
    $_sudo apt update 2>/dev/null
    log_success "CUDA repo definition removed."
}
