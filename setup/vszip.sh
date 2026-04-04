#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_vszip() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    log_info "Compiling VSZIP..."
    if [ -d "vszip" ]; then rm -rf vszip; fi
    git clone --branch R13 --depth 1 https://github.com/dnjulek/vapoursynth-zip.git vszip || { cd "$ORIG_DIR"; log_error "Failed to clone VSZIP"; return 1; }
    cd vszip || { cd "$ORIG_DIR"; log_error "Failed to cd into vszip"; return 1; }

    cd build-help || { cd "$ORIG_DIR"; log_error "Failed to cd into build-help"; return 1; }
    chmod +x build.sh
    ./build.sh || { cd "$ORIG_DIR"; log_error "VSZIP build.sh failed"; return 1; }

    if [ -f "../zig-out/lib/libvszip.so" ]; then
        cp "../zig-out/lib/libvszip.so" "$VS_PLUGIN_PATH/libvszip.so" || { cd "$ORIG_DIR"; log_error "Failed to copy libvszip.so"; return 1; }
    else
        cd "$ORIG_DIR"; log_error "VSZIP Compilation failed!"; return 1
    fi

    ldconfig
    cd "$ORIG_DIR"

    log_success "VSZIP installed."
}

uninstall_vszip() {
    log_info "Uninstalling VSZIP..."
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    find "$VS_PLUGIN_PATH" /usr/local/lib/vapoursynth -name "libvszip.so" -delete 2>/dev/null
    log_success "VSZIP uninstalled."
}
