#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

SOURCES["subtext:subtext"]="https://github.com/vapoursynth/subtext.git|R6"
ARTIFACTS["subtext"]="lib/vapoursynth/libsubtext.so"

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
    clone_src subtext subtext subtext || { cd "$ORIG_DIR"; return 1; }
    cd subtext || { cd "$ORIG_DIR"; log_error "Failed to cd into subtext"; return 1; }

    # avcodec_close() was removed in FFmpeg 6.0; rewrite to avcodec_free_context().
    # R6's src/image.cpp uses the new API already, so this sed is a no-op there;
    # kept for older tag pins.
    sed -i 's/avcodec_close(d->avctx)/avcodec_free_context(\&d->avctx)/g' src/image.cpp

    # Use the venv's meson — system /usr/sbin/meson is shebanged to system
    # /usr/bin/python (3.14 on current Arch), and SubText's meson.build calls
    # find_installation('python', 'python3') which would resolve via that
    # meson's sys.executable rather than the venv interpreter (same trap as
    # the VS R76 build).
    CC=clang CXX=clang++ "$VENV_DIR/bin/meson" setup build --buildtype=release \
        -Dc_args="-march=native -O3" \
        -Dcpp_args="-march=native -O3" \
        -Db_lto=true || { cd "$ORIG_DIR"; log_error "SubText meson setup failed"; return 1; }
    "$VENV_DIR/bin/meson" compile -C build || { cd "$ORIG_DIR"; log_error "SubText meson compile failed"; return 1; }

    # R5 (autotools) produced libsubtext.so; R6 (meson shared_module with
    # name_prefix: '') produces subtext.so. Accept either, install as the
    # canonical libsubtext.so VS plugin name.
    local _built
    for _built in build/subtext.so build/libsubtext.so; do
        if [ -f "$_built" ]; then
            cp "$_built" "$VS_PLUGIN_PATH/libsubtext.so" \
                || { cd "$ORIG_DIR"; log_error "Failed to copy $_built"; return 1; }
            break
        fi
    done
    if [ ! -f "$VS_PLUGIN_PATH/libsubtext.so" ]; then
        cd "$ORIG_DIR"; log_error "SubText build produced no output .so under build/."; return 1
    fi

    cd "$ORIG_DIR"

    log_success "SubText installed."
}

uninstall_subtext() {
    log_info "Uninstalling SubText..."
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    find "$VS_PLUGIN_PATH" "$VS_PREFIX/lib/vapoursynth" -name "libsubtext.so" -delete 2>/dev/null
    rm -f "$MANIFEST_DIR/subtext.src"
    log_success "SubText uninstalled."
}
