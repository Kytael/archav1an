#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_vapoursynth() {
    if [ -f /usr/local/bin/vspipe ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "VapourSynth (source-built) is already installed."
        return 0
    fi

    log_info "Compiling VapourSynth from source with native optimizations..."
    set_native_build_flags

    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    # 1. VapourSynth
    if [ -d "vapoursynth" ]; then rm -rf vapoursynth; fi
    git clone --branch R73 --depth 1 https://github.com/vapoursynth/vapoursynth.git || { cd "$ORIG_DIR"; log_error "Failed to clone VapourSynth"; return 1; }
    cd vapoursynth || { cd "$ORIG_DIR"; log_error "Failed to cd into vapoursynth"; return 1; }
    ./autogen.sh || { cd "$ORIG_DIR"; log_error "VapourSynth autogen failed"; return 1; }
    ./configure || { cd "$ORIG_DIR"; log_error "VapourSynth configure failed"; return 1; }
    make -j "$(nproc)" || { cd "$ORIG_DIR"; log_error "VapourSynth make failed"; return 1; }
    make install || { cd "$ORIG_DIR"; log_error "VapourSynth make install failed"; return 1; }
    cd "$BUILD_DIR"

    # Link Python module if not found
    local SITE_PKG_DIR
    SITE_PKG_DIR="$(get_python_site_packages)"
    local VS_SO_SEARCH
    VS_SO_SEARCH="$(find /usr/local/lib -name 'vapoursynth.so' -type f 2>/dev/null | head -1)"

    if [ -n "$VS_SO_SEARCH" ] && [ ! -f "$SITE_PKG_DIR/vapoursynth.so" ]; then
        log_info "Linking VapourSynth Python module to $SITE_PKG_DIR..."
        mkdir -p "$SITE_PKG_DIR"
        ln -sf "$VS_SO_SEARCH" "$SITE_PKG_DIR/vapoursynth.so"
    fi

    # 2. FFMS2
    log_info "Compiling FFMS2 with native optimizations..."
    if [ -d "ffms2" ]; then rm -rf ffms2; fi
    git clone --branch 5.0 --depth 1 https://github.com/FFMS/ffms2.git || { cd "$ORIG_DIR"; log_error "Failed to clone FFMS2"; return 1; }
    cd ffms2 || { cd "$ORIG_DIR"; log_error "Failed to cd into ffms2"; return 1; }
    ./autogen.sh || { cd "$ORIG_DIR"; log_error "FFMS2 autogen failed"; return 1; }
    ./configure --enable-shared || { cd "$ORIG_DIR"; log_error "FFMS2 configure failed"; return 1; }
    make -j "$(nproc)" || { cd "$ORIG_DIR"; log_error "FFMS2 make failed"; return 1; }
    make install || { cd "$ORIG_DIR"; log_error "FFMS2 make install failed"; return 1; }
    cd "$BUILD_DIR"

    # Symlink FFMS2 to VapourSynth plugin path
    local VS_PLUGIN_PATH="/usr/local/lib/vapoursynth"
    mkdir -p "$VS_PLUGIN_PATH"

    if [ -f "/usr/local/lib/libffms2.so" ]; then
        log_info "Linking FFMS2 to VapourSynth plugin folder..."
        ln -sf "/usr/local/lib/libffms2.so" "$VS_PLUGIN_PATH/libffms2.so"
    fi

    # 3. BestSource
    log_info "Compiling BestSource with native optimizations..."
    if [ -d "bestsource" ]; then rm -rf bestsource; fi
    git clone --depth 1 --recurse-submodules https://github.com/vapoursynth/bestsource.git || { cd "$ORIG_DIR"; log_error "Failed to clone BestSource"; return 1; }
    cd bestsource || { cd "$ORIG_DIR"; log_error "Failed to cd into bestsource"; return 1; }

    if ! command -v meson &> /dev/null && [ -d "$VENV_DIR" ]; then
        "$VENV_DIR/bin/pip" install meson || true
    fi
    meson setup build --buildtype=release \
        -Dc_args="-march=native -O3" \
        -Dcpp_args="-march=native -O3" \
        -Db_lto=true || { cd "$ORIG_DIR"; log_error "BestSource meson setup failed"; return 1; }
    ninja -C build || { cd "$ORIG_DIR"; log_error "BestSource ninja build failed"; return 1; }
    ninja -C build install || { cd "$ORIG_DIR"; log_error "BestSource ninja install failed"; return 1; }

    local BS_SO
    BS_SO="$(find /usr/local/lib -name 'libbestsource*' -type f 2>/dev/null | head -1)"
    if [ -n "$BS_SO" ]; then
        log_info "Linking BestSource to VapourSynth plugin folder..."
        ln -sf "$BS_SO" "$VS_PLUGIN_PATH/"
    fi

    ldconfig
    cd "$ORIG_DIR"

    log_success "VapourSynth, FFMS2, and BestSource installed with native optimizations."
}

uninstall_vapoursynth() {
    log_info "Uninstalling VapourSynth, FFMS2, and BestSource..."

    # Binaries
    rm -vf /usr/local/bin/vspipe

    # Libraries
    rm -vf /usr/local/lib/libvapoursynth*
    rm -vf /usr/local/lib/libffms2*
    rm -vf /usr/local/lib/libbestsource*

    # Headers
    rm -rf /usr/local/include/vapoursynth
    rm -rf /usr/local/include/ffms2

    # Plugins
    rm -rf /usr/local/lib/vapoursynth

    # Python link
    local SITE_PKG_DIR
    SITE_PKG_DIR="$(get_python_site_packages)"
    rm -vf "$SITE_PKG_DIR/vapoursynth.so"

    # PkgConfig
    rm -vf /usr/local/lib/pkgconfig/vapoursynth.pc
    rm -vf /usr/local/lib/pkgconfig/ffms2.pc

    ldconfig
    log_success "VapourSynth, FFMS2, and BestSource uninstalled."
}
