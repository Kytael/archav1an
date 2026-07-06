#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

SOURCES["wwxd:vapoursynth-wwxd"]="https://github.com/dubhater/vapoursynth-wwxd.git|v1.0"
ARTIFACTS["wwxd"]="lib/vapoursynth/libwwxd.so"

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
    clone_src wwxd vapoursynth-wwxd vapoursynth-wwxd || { cd "$ORIG_DIR"; return 1; }
    cd vapoursynth-wwxd || { cd "$ORIG_DIR"; log_error "Failed to cd into vapoursynth-wwxd"; return 1; }

    # Find VapourSynth headers dynamically
    local VS_INCLUDE=""
    if command -v pkg-config &> /dev/null && pkg-config --exists vapoursynth 2>/dev/null; then
        VS_INCLUDE="$(pkg-config --cflags vapoursynth)"
    elif [ -d "$VS_PREFIX/include/vapoursynth" ]; then
        VS_INCLUDE="-I$VS_PREFIX/include/vapoursynth"
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
    find "$VS_PLUGIN_PATH" "$VS_PREFIX/lib/vapoursynth" -name "libwwxd.so" -delete 2>/dev/null
    rm -f "$MANIFEST_DIR/wwxd.src"
    log_success "WWXD uninstalled."
}
