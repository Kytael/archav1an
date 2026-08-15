#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

SOURCES["av1an:av1an"]="https://github.com/rust-av/Av1an.git|master"
ARTIFACTS["av1an"]="bin/av1an"

install_av1an() {
    if [ -f "$VS_PREFIX/bin/av1an" ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "av1an (source-built) is already installed."
        return 0
    fi

    log_info "Compiling av1an from source with native optimizations..."
    set_native_build_flags

    ensure_rust || return 1

    cargo install --git https://github.com/rust-av/Av1an.git --bin av1an || { log_error "Failed to install av1an via cargo"; return 1; }

    local _av1an_sha
    _av1an_sha=$(git ls-remote https://github.com/rust-av/Av1an.git refs/heads/master 2>/dev/null | cut -f1)
    record_src av1an av1an "https://github.com/rust-av/Av1an.git" master "${_av1an_sha:-unknown}"

    if [ -f "$HOME/.cargo/bin/av1an" ]; then
        cp "$HOME/.cargo/bin/av1an" "$VS_PREFIX/bin/av1an"
        chmod +x "$VS_PREFIX/bin/av1an"
        # One av1an, in the prefix. cargo drops its own copy in ~/.cargo/bin,
        # which sits on PATH ahead of $VS_PREFIX/bin in a plain shell, so a
        # later prefix-only reinstall would leave you running the old binary
        # with no sign of it. uninstall_av1an already treats the prefix copy
        # as the artifact and deletes this one.
        rm -f "$HOME/.cargo/bin/av1an"
        log_success "av1an installed with LTO and -march=native."
    else
        log_warn "av1an binary not found in cargo bin after install?"
        return 1
    fi
}

uninstall_av1an() {
    log_info "Uninstalling av1an..."
    rm -vf "$VS_PREFIX/bin/av1an"
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    cargo uninstall av1an 2>/dev/null || true
    rm -vf "$HOME/.cargo/bin/av1an"
    rm -f "$MANIFEST_DIR/av1an.src"
    log_success "av1an uninstalled."
}
