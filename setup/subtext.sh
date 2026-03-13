#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_subtext() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    mkdir -p build_tmp
    cd build_tmp || exit 1

    log_info "Compiling SubText..."
    if [ -d "subtext" ]; then rm -rf subtext; fi
    git clone --branch R5 --depth 1 https://github.com/vapoursynth/subtext.git || { log_error "Failed to clone SubText"; cd ..; return 1; }
    cd subtext || { log_error "Failed to cd into subtext"; cd ..; cd ..; return 1; }

    # avcodec_close() was removed in FFmpeg 6.0; replace with avcodec_free_context()
    sed -i 's/avcodec_close(d->avctx)/avcodec_free_context(\&d->avctx)/g' src/image.cpp

    mkdir build && cd build
    meson setup .. --buildtype=release || { log_error "SubText meson setup failed"; cd ..; cd ..; cd ..; return 1; }
    ninja || { log_error "SubText ninja build failed"; cd ..; cd ..; cd ..; return 1; }

    if [ -f "libsubtext.so" ]; then
        cp "libsubtext.so" "$VS_PLUGIN_PATH/" || { log_error "Failed to copy libsubtext.so"; cd ..; cd ..; cd ..; return 1; }
    else
        log_error "SubText compilation failed!"
    fi

    cd ../../..
    cd .. # Exit build_tmp

    log_success "SubText installed."
}

uninstall_subtext() {
    log_info "Uninstalling SubText..."
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    find "$VS_PLUGIN_PATH" /usr/local/lib/vapoursynth -name "libsubtext.so" -delete 2>/dev/null
    log_success "SubText uninstalled."
}
