#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_av1an() {
    if [ -f /usr/local/bin/av1an ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "av1an (source-built) is already installed."
        return 0
    fi

    log_info "Compiling av1an from source with native optimizations..."
    set_native_build_flags

    # Ensure Rust is available
    if ! command -v cargo &> /dev/null; then
        if command -v pacman &> /dev/null; then
            pacman -S --needed --noconfirm rust || { log_error "Failed to install Rust"; return 1; }
        else
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y || { log_error "Failed to install Rust via rustup"; return 1; }
            source "$HOME/.cargo/env"
        fi
    fi
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    export PATH="$HOME/.cargo/bin:$PATH"

    cargo install --git https://github.com/rust-av/Av1an.git --bin av1an || { log_error "Failed to install av1an via cargo"; return 1; }

    if [ -f "$HOME/.cargo/bin/av1an" ]; then
        cp "$HOME/.cargo/bin/av1an" /usr/local/bin/av1an
        chmod +x /usr/local/bin/av1an
        log_success "av1an installed with LTO and -march=native."
    else
        log_warn "av1an binary not found in cargo bin after install?"
        return 1
    fi
}

uninstall_av1an() {
    log_info "Uninstalling av1an..."
    rm -vf /usr/local/bin/av1an
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    cargo uninstall av1an 2>/dev/null || true
    rm -vf "$HOME/.cargo/bin/av1an"
    log_success "av1an uninstalled."
}
