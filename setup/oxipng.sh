#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_oxipng() {
    if command -v pacman &> /dev/null; then
        if ! pacman -Qi oxipng &> /dev/null; then
            log_info "Installing oxipng via pacman..."
            pacman -S --noconfirm oxipng || { log_error "Failed to install oxipng with pacman"; return 1; }
        else
            log_info "oxipng is already installed (pacman)."
        fi
    else
        if ! command -v oxipng &> /dev/null; then
            log_info "Installing oxipng via cargo..."
            [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
            cargo install oxipng || { log_error "Failed to install oxipng via cargo"; return 1; }

            if [ -f "$HOME/.cargo/bin/oxipng" ]; then
                cp "$HOME/.cargo/bin/oxipng" /usr/local/bin/oxipng
                chmod +x /usr/local/bin/oxipng
                log_success "oxipng installed."
            fi
        else
            log_info "oxipng is already installed."
        fi
    fi
}

uninstall_oxipng() {
    log_info "Uninstalling oxipng..."
    if command -v pacman &> /dev/null; then
        pacman -R --noconfirm oxipng 2>/dev/null || true
    fi
    rm -vf /usr/local/bin/oxipng
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    cargo uninstall oxipng 2>/dev/null || true
    log_success "oxipng uninstalled."
}
