#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_oxipng() {
    if [ -f /usr/local/bin/oxipng ]; then
        log_info "oxipng (source-built) is already installed."
        return 0
    fi

    log_info "Compiling oxipng from source with native optimizations..."
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

    cargo install oxipng || { log_error "Failed to install oxipng via cargo"; return 1; }

    if [ -f "$HOME/.cargo/bin/oxipng" ]; then
        cp "$HOME/.cargo/bin/oxipng" /usr/local/bin/oxipng
        chmod +x /usr/local/bin/oxipng
        log_success "oxipng installed with LTO and -march=native."
    else
        log_warn "oxipng binary not found in cargo bin after install?"
        return 1
    fi
}

uninstall_oxipng() {
    log_info "Uninstalling oxipng..."
    rm -vf /usr/local/bin/oxipng
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    cargo uninstall oxipng 2>/dev/null || true
    log_success "oxipng uninstalled."
}
