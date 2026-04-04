#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_subtext() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    set_native_build_flags

    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    log_info "Compiling SubText..."
    if [ -d "subtext" ]; then rm -rf subtext; fi
    git clone --branch R5 --depth 1 https://github.com/vapoursynth/subtext.git || { cd "$ORIG_DIR"; log_error "Failed to clone SubText"; return 1; }
    cd subtext || { cd "$ORIG_DIR"; log_error "Failed to cd into subtext"; return 1; }

    # avcodec_close() was removed in FFmpeg 6.0; replace with avcodec_free_context()
    sed -i 's/avcodec_close(d->avctx)/avcodec_free_context(\&d->avctx)/g' src/image.cpp

    mkdir build && cd build || { cd "$ORIG_DIR"; log_error "Failed to create/enter build dir"; return 1; }
    CC=clang CXX=clang++ meson setup .. --buildtype=release \
        -Dc_args="-march=native -O3" \
        -Dcpp_args="-march=native -O3" \
        -Db_lto=true || { cd "$ORIG_DIR"; log_error "SubText meson setup failed"; return 1; }
    ninja || { cd "$ORIG_DIR"; log_error "SubText ninja build failed"; return 1; }

    if [ -f "libsubtext.so" ]; then
        cp "libsubtext.so" "$VS_PLUGIN_PATH/" || { cd "$ORIG_DIR"; log_error "Failed to copy libsubtext.so"; return 1; }
    else
        cd "$ORIG_DIR"; log_error "SubText compilation failed!"; return 1
    fi

    cd "$ORIG_DIR"

    log_success "SubText installed."
}

uninstall_subtext() {
    log_info "Uninstalling SubText..."
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    find "$VS_PLUGIN_PATH" /usr/local/lib/vapoursynth -name "libsubtext.so" -delete 2>/dev/null
    log_success "SubText uninstalled."
}
