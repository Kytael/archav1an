#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_python_libs() {
    log_info "Installing Python Libraries into venv ($VENV_DIR)..."

    if ! command -v uv &> /dev/null; then
        log_error "uv not found on PATH. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi

    # Target Python version:
    #   - default: whatever `python3` resolves to on PATH (latest pacman).
    #   - override: PYTHON_VERSION=<x.y> (e.g. 3.13) — uv downloads if missing.
    local TARGET_PY="${PYTHON_VERSION:-python3}"

    # Detect current venv's Python version (if venv exists). When it differs
    # from the target, rebuild — the venv is binary-incompatible across
    # Python minors, and the source-built VS module is too.
    local CURRENT_PY=""
    local TARGET_PY_CANON=""
    if [ -x "$VENV_DIR/bin/python" ]; then
        CURRENT_PY="$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    fi
    # Canonicalize target to x.y for comparison (resolves "python3" -> actual version)
    if [ "$TARGET_PY" = "python3" ] && command -v python3 &> /dev/null; then
        TARGET_PY_CANON="$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    else
        TARGET_PY_CANON="$TARGET_PY"
    fi

    if [ -n "$CURRENT_PY" ] && [ -n "$TARGET_PY_CANON" ] && [ "$CURRENT_PY" != "$TARGET_PY_CANON" ]; then
        log_warn "Python version change detected: venv has $CURRENT_PY, target is $TARGET_PY_CANON. Rebuilding venv at $VENV_DIR."
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
        log_error "uv pip install failed under Python $TARGET_PY_CANON."
        log_error "If a binary dep lags behind the latest Python release (e.g. PyO3-based packages on Python bumps),"
        log_error "pin to a known-good version with: PYTHON_VERSION=3.13 ./setup.sh --install python_libs"
        return 1
    fi

    # Remove the pip-published vapoursynth stub which would shadow the source-built
    # R73 module that vapoursynth.sh wires via .pth.
    log_info "Removing pip-installed vapoursynth stub (avoid R73 module shadow)..."
    VIRTUAL_ENV="$VENV_DIR" uv pip uninstall vapoursynth || true

    log_success "Python libraries installed in venv (Python $TARGET_PY_CANON)."
}

uninstall_python_libs() {
    log_info "Uninstalling Python Libraries..."

    if [ -d "$VENV_DIR" ]; then
        VIRTUAL_ENV="$VENV_DIR" uv pip uninstall \
            vsjetpack numpy rich vstools psutil anitopy pyperclip \
            requests requests_toolbelt natsort colorama Cython \
            || true
        log_success "Python libraries uninstalled from venv."
    else
        log_warn "Venv not found at $VENV_DIR, nothing to uninstall."
    fi
}
