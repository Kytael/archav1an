#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_python_libs() {
    log_info "Installing Python Libraries into venv ($VENV_DIR)..."

    ensure_uv || return 1

    # Target Python version:
    #   - default: whatever `python3` resolves to on PATH (latest pacman).
    #   - override: PYTHON_VERSION=<x.y> (e.g. 3.13) — uv downloads if missing.
    local TARGET_PY="${PYTHON_VERSION:-python3}"

    # Detect current venv's Python version (if venv exists). We rebuild only
    # when the user EXPLICITLY pinned PYTHON_VERSION to a version that
    # doesn't match what's already in the venv — otherwise leave the venv
    # alone (uv may have picked a different minor than the system python3
    # due to wheel availability, and we don't want to churn the venv on
    # every re-run).
    local CURRENT_PY=""
    if [ -x "$VENV_DIR/bin/python" ]; then
        CURRENT_PY="$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    fi

    if [ -n "$PYTHON_VERSION" ] && [ -n "$CURRENT_PY" ] && [ "$CURRENT_PY" != "$PYTHON_VERSION" ]; then
        log_warn "Python version change requested: venv has $CURRENT_PY, PYTHON_VERSION=$PYTHON_VERSION. Rebuilding venv at $VENV_DIR."
        log_warn "VapourSynth source-build is binary-linked to the venv's Python; rerun '--install vapoursynth' after this completes."
        rm -rf "$VENV_DIR"
    fi

    # Create venv if it doesn't exist (or just removed for upgrade).
    # Clean venv (no --system-site-packages): the source-built VapourSynth
    # module is exposed via a .pth written by setup/vapoursynth.sh, not by
    # inheriting system site-packages.
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating uv-managed venv (Python $TARGET_PY) at $VENV_DIR..."
        mkdir -p "$(dirname "$VENV_DIR")"
        uv venv --python "$TARGET_PY" "$VENV_DIR" \
            || { log_error "Failed to create venv with uv at Python $TARGET_PY. To pin a specific version: PYTHON_VERSION=3.13 ./setup.sh --install python_libs"; return 1; }
    fi

    VIRTUAL_ENV="$VENV_DIR" uv pip install --upgrade pip \
        || log_warn "pip upgrade failed, continuing..."

    if ! VIRTUAL_ENV="$VENV_DIR" uv pip install \
            vsjetpack numpy rich vstools psutil anitopy pyperclip requests \
            requests_toolbelt natsort colorama Cython; then
        local _failed_py
        _failed_py="$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")"
        log_error "uv pip install failed under Python $_failed_py."
        log_error "If a binary dep lags behind the latest Python release (e.g. PyO3-based packages on Python bumps),"
        log_error "pin to a known-good version with: PYTHON_VERSION=3.13 ./setup.sh --install python_libs"
        return 1
    fi

    # Remove the pip-published vapoursynth stub which would shadow the source-built
    # R73 module that vapoursynth.sh wires via .pth.
    log_info "Removing pip-installed vapoursynth stub (avoid R73 module shadow)..."
    VIRTUAL_ENV="$VENV_DIR" uv pip uninstall vapoursynth || true

    # Report the venv's actual installed Python, not the target spec.
    local INSTALLED_PY
    INSTALLED_PY="$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || echo "unknown")"
    log_success "Python libraries installed in venv (Python $INSTALLED_PY)."
}

uninstall_python_libs() {
    log_info "Uninstalling Python Libraries..."

    if [ -d "$VENV_DIR" ]; then
        VIRTUAL_ENV="$VENV_DIR" uv pip uninstall \
            vsjetpack numpy rich vstools psutil anitopy pyperclip \
            requests requests_toolbelt natsort colorama Cython \
            || log_warn "Some packages failed to uninstall (already removed?)"
        log_success "Python libraries uninstalled from venv."
    else
        log_warn "Venv not found at $VENV_DIR, nothing to uninstall."
    fi
}
