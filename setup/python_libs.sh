#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_python_libs() {
    log_info "Installing Python Libraries into venv ($VENV_DIR)..."

    # Create venv if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment at $VENV_DIR..."
        mkdir -p "$(dirname "$VENV_DIR")"
        python3 -m venv "$VENV_DIR" --system-site-packages || { log_error "Failed to create venv"; return 1; }
    fi

    # Install packages into venv
    "$VENV_DIR/bin/pip" install --upgrade pip || log_warn "pip upgrade failed, continuing..."
    "$VENV_DIR/bin/pip" install vsjetpack numpy rich vstools psutil anitopy pyperclip requests \
        requests_toolbelt natsort colorama Cython vsmlrt havsfunc \
        || { log_error "Failed to install Python libraries"; return 1; }

    # Remove the pip-installed vapoursynth which conflicts with the source build we are about to do
    log_info "Removing pip-installed VapourSynth to avoid version mismatch..."
    "$VENV_DIR/bin/pip" uninstall -y vapoursynth || true

    log_success "Python libraries installed in venv."
}

uninstall_python_libs() {
    log_info "Uninstalling Python Libraries..."

    if [ -d "$VENV_DIR" ]; then
        "$VENV_DIR/bin/pip" uninstall -y vsjetpack numpy rich vstools psutil anitopy pyperclip \
            requests requests_toolbelt natsort colorama Cython
        log_success "Python libraries uninstalled from venv."
    else
        log_warn "Venv not found at $VENV_DIR, nothing to uninstall."
    fi
}
