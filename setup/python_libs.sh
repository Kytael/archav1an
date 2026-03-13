#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_python_libs() {
    log_info "Installing Python Libraries..."

    local PIP_FLAGS=""
    if [ "$DISTRO_FAMILY" = "debian" ]; then
        # Debian/Ubuntu uses externally-managed Python; need this flag
        PIP_FLAGS="--break-system-packages --ignore-installed"
    fi

    pip3 install vsjetpack numpy rich vstools psutil anitopy pyperclip requests \
        requests_toolbelt natsort colorama wakepy Cython \
        $PIP_FLAGS || { log_error "Failed to install Python libraries"; return 1; }

    # Remove the pip-installed vapoursynth which conflicts with the source build we are about to do
    log_info "Removing pip-installed VapourSynth to avoid version mismatch..."
    pip3 uninstall -y vapoursynth $PIP_FLAGS || true

    log_success "Python libraries installed."
}

uninstall_python_libs() {
    log_info "Uninstalling Python Libraries..."

    local PIP_FLAGS=""
    if [ "$DISTRO_FAMILY" = "debian" ]; then
        PIP_FLAGS="--break-system-packages"
    fi

    pip3 uninstall -y vsjetpack numpy rich vstools psutil anitopy pyperclip \
        requests requests_toolbelt natsort colorama wakepy Cython \
        $PIP_FLAGS
    log_success "Python libraries uninstalled."
}
