#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

SOURCES["oxipng:oxipng"]="crates.io:oxipng|latest"
ARTIFACTS["oxipng"]="bin/oxipng"

install_oxipng() {
    if [ -f "$VS_PREFIX/bin/oxipng" ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "oxipng (source-built) is already installed."
        return 0
    fi

    log_info "Compiling oxipng from source with native optimizations..."
    set_native_build_flags

    ensure_rust || return 1

    cargo install --force oxipng || { log_error "Failed to install oxipng via cargo"; return 1; }  # see install_av1an

    if [ -f "$HOME/.cargo/bin/oxipng" ]; then
        cp "$HOME/.cargo/bin/oxipng" "$VS_PREFIX/bin/oxipng"
        chmod +x "$VS_PREFIX/bin/oxipng"
        rm -f "$HOME/.cargo/bin/oxipng"   # see install_av1an: one copy, in the prefix

        local _oxipng_ver
        _oxipng_ver=$("$VS_PREFIX/bin/oxipng" --version 2>/dev/null | awk 'NR==1{print $2}')
        record_src oxipng oxipng "crates.io:oxipng" latest "${_oxipng_ver:-unknown}"

        log_success "oxipng installed with LTO and -march=native."
    else
        log_warn "oxipng binary not found in cargo bin after install?"
        return 1
    fi
}

uninstall_oxipng() {
    log_info "Uninstalling oxipng..."
    rm -vf "$VS_PREFIX/bin/oxipng"
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    cargo uninstall oxipng 2>/dev/null || true
    rm -f "$MANIFEST_DIR/oxipng.src"
    log_success "oxipng uninstalled."
}
