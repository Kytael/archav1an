#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_python_libs() {
    log_info "Installing Python Libraries into venv ($VENV_DIR)..."

    ensure_uv || return 1

    # Target Python interpreter:
    #   - default: the system `python3` (absolute path, not the bare name —
    #     uv interprets a bare "python3" as "any python3" and would happily
    #     reuse a previously-cached uv-managed interpreter at an older
    #     minor, which is not what the user expects when they don't pin).
    #   - override: PYTHON_VERSION=<x.y> (e.g. 3.13) — uv downloads if missing.
    local TARGET_PY="${PYTHON_VERSION:-}"
    if [ -z "$TARGET_PY" ]; then
        TARGET_PY="$(command -v python3 2>/dev/null || true)"
        if [ -z "$TARGET_PY" ]; then
            log_error "No python3 found on PATH and PYTHON_VERSION is unset. Install python3 or set PYTHON_VERSION=<x.y>."
            return 1
        fi
    fi

    # Compute the target's major.minor for upgrade detection. PYTHON_VERSION
    # is already "x.y"; for a path, ask the interpreter itself.
    local TARGET_PY_MM
    if [[ "$TARGET_PY" == /* ]]; then
        TARGET_PY_MM="$("$TARGET_PY" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    else
        TARGET_PY_MM="$TARGET_PY"
    fi

    local CURRENT_PY=""
    if [ -x "$VENV_DIR/bin/python" ]; then
        CURRENT_PY="$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    fi

    if [ -n "$CURRENT_PY" ] && [ -n "$TARGET_PY_MM" ] && [ "$CURRENT_PY" != "$TARGET_PY_MM" ]; then
        log_warn "Python version mismatch: venv has $CURRENT_PY, target is $TARGET_PY_MM. Rebuilding venv at $VENV_DIR."
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
            vsjetpack numpy scipy rich vstools psutil anitopy pyperclip requests \
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
    drop_pip_vapoursynth_stub

    # Report the venv's actual installed Python, not the target spec.
    local INSTALLED_PY
    INSTALLED_PY="$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || echo "unknown")"
    log_success "Python libraries installed in venv (Python $INSTALLED_PY)."
}

# pip is its own update resolver — "update" here is just install -U over the
# same package set (used by ./setup.sh --update; a no-op when all current).
update_python_libs() {
    [ -d "$VENV_DIR" ] || { log_warn "venv missing — run --install python_libs instead."; return 1; }
    log_info "Upgrading python_libs packages in venv..."
    VIRTUAL_ENV="$VENV_DIR" uv pip install -U \
            vsjetpack numpy scipy rich vstools psutil anitopy pyperclip requests \
            requests_toolbelt natsort colorama Cython \
        || { log_error "uv pip install -U failed."; return 1; }
    VIRTUAL_ENV="$VENV_DIR" uv pip uninstall vapoursynth || true
    log_success "python_libs packages upgraded."
}

uninstall_python_libs() {
    log_info "Uninstalling Python Libraries..."

    if [ -d "$VENV_DIR" ]; then
        VIRTUAL_ENV="$VENV_DIR" uv pip uninstall \
            vsjetpack numpy scipy rich vstools psutil anitopy pyperclip \
            requests requests_toolbelt natsort colorama Cython \
            || log_warn "Some packages failed to uninstall (already removed?)"
        log_success "Python libraries uninstalled from venv."
    else
        log_warn "Venv not found at $VENV_DIR, nothing to uninstall."
    fi
}
