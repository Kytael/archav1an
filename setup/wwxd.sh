#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_wwxd() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    set_native_build_flags

    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    log_info "Compiling VapourSynth-WWXD..."
    if [ -d "vapoursynth-wwxd" ]; then rm -rf vapoursynth-wwxd; fi
    git clone --branch v1.0 --depth 1 https://github.com/dubhater/vapoursynth-wwxd.git || { cd "$ORIG_DIR"; log_error "Failed to clone WWXD"; return 1; }
    cd vapoursynth-wwxd || { cd "$ORIG_DIR"; log_error "Failed to cd into vapoursynth-wwxd"; return 1; }

    # Find VapourSynth headers dynamically
    local VS_INCLUDE=""
    if command -v pkg-config &> /dev/null && pkg-config --exists vapoursynth 2>/dev/null; then
        VS_INCLUDE="$(pkg-config --cflags vapoursynth)"
    elif [ -d "/usr/local/include/vapoursynth" ]; then
        VS_INCLUDE="-I/usr/local/include/vapoursynth"
    elif [ -d "/usr/include/vapoursynth" ]; then
        VS_INCLUDE="-I/usr/include/vapoursynth"
    else
        cd "$ORIG_DIR"; log_error "VapourSynth headers not found. Please install VapourSynth first."; return 1
    fi

    clang -o libwwxd.so -fPIC -shared -march=native -O3 -flto -fuse-ld=lld -Wall -Wextra -I. $VS_INCLUDE src/*.c -lm || \
        { cd "$ORIG_DIR"; log_error "Compilation failed"; return 1; }

    cp libwwxd.so "$VS_PLUGIN_PATH/" || { cd "$ORIG_DIR"; log_error "Failed to copy libwwxd.so"; return 1; }
    cd "$ORIG_DIR"

    log_success "WWXD installed."
}

uninstall_wwxd() {
    log_info "Uninstalling WWXD..."
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    find "$VS_PLUGIN_PATH" /usr/local/lib/vapoursynth -name "libwwxd.so" -delete 2>/dev/null
    log_success "WWXD uninstalled."
}
