#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_vszip() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    mkdir -p build_tmp
    cd build_tmp || exit 1

    log_info "Compiling VSZIP..."
    if [ -d "vszip" ]; then rm -rf vszip; fi
    git clone --branch R13 --depth 1 https://github.com/dnjulek/vapoursynth-zip.git vszip || { log_error "Failed to clone VSZIP"; cd ..; return 1; }
    cd vszip || { log_error "Failed to cd into vszip"; cd ..; cd ..; return 1; }

    cd build-help
    chmod +x build.sh
    ./build.sh || { log_error "VSZIP build.sh failed"; cd ..; cd ..; cd ..; return 1; }

    if [ -f "../zig-out/lib/libvszip.so" ]; then
        cp "../zig-out/lib/libvszip.so" "$VS_PLUGIN_PATH/libvszip.so" || { log_error "Failed to copy libvszip.so"; cd ..; cd ..; cd ..; return 1; }
    else
        log_error "VSZIP Compilation failed!"
    fi
    cd ../..

    ldconfig
    cd .. # Exit build_tmp

    log_success "VSZIP installed."
}

uninstall_vszip() {
    log_info "Uninstalling VSZIP..."
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    find "$VS_PLUGIN_PATH" /usr/local/lib/vapoursynth -name "libvszip.so" -delete 2>/dev/null
    log_success "VSZIP uninstalled."
}
