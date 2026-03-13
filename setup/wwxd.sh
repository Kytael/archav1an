#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_wwxd() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    mkdir -p build_tmp
    cd build_tmp || exit 1

    log_info "Compiling VapourSynth-WWXD..."
    if [ -d "vapoursynth-wwxd" ]; then rm -rf vapoursynth-wwxd; fi
    git clone --branch v1.0 --depth 1 https://github.com/dubhater/vapoursynth-wwxd.git || { log_error "Failed to clone WWXD"; cd ..; return 1; }
    cd vapoursynth-wwxd || { log_error "Failed to cd into vapoursynth-wwxd"; cd ..; cd ..; return 1; }

    # Find VapourSynth headers dynamically
    local VS_INCLUDE=""
    if command -v pkg-config &> /dev/null && pkg-config --exists vapoursynth 2>/dev/null; then
        VS_INCLUDE="$(pkg-config --cflags vapoursynth)"
    elif [ -d "/usr/local/include/vapoursynth" ]; then
        VS_INCLUDE="-I/usr/local/include/vapoursynth"
    elif [ -d "/usr/include/vapoursynth" ]; then
        VS_INCLUDE="-I/usr/include/vapoursynth"
    else
        log_error "VapourSynth headers not found. Please install VapourSynth first."
        cd ..; cd ..; return 1
    fi

    gcc -o libwwxd.so -fPIC -shared -O3 -Wall -Wextra -I. $VS_INCLUDE src/*.c -lm || \
        { log_error "Compilation failed"; cd ..; cd ..; return 1; }

    cp libwwxd.so "$VS_PLUGIN_PATH/" || { log_error "Failed to copy libwwxd.so"; cd ..; cd ..; return 1; }
    cd ..
    cd .. # Exit build_tmp

    log_success "WWXD installed."
}

uninstall_wwxd() {
    log_info "Uninstalling WWXD..."
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    find "$VS_PLUGIN_PATH" /usr/local/lib/vapoursynth -name "libwwxd.so" -delete 2>/dev/null
    log_success "WWXD uninstalled."
}
