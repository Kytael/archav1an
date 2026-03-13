#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_vapoursynth() {
    if command -v vspipe &> /dev/null; then
        log_info "VapourSynth is already installed."
        return 0
    fi

    log_info "Compiling VapourSynth from Source..."

    mkdir -p build_tmp
    cd build_tmp || exit 1

    # 1. VapourSynth
    if [ -d "vapoursynth" ]; then rm -rf vapoursynth; fi
    git clone --branch R73 --depth 1 https://github.com/vapoursynth/vapoursynth.git || { log_error "Failed to clone VapourSynth"; cd ..; return 1; }
    cd vapoursynth || { log_error "Failed to cd into vapoursynth"; cd ..; cd ..; return 1; }
    ./autogen.sh || { log_error "VapourSynth autogen failed"; cd ..; cd ..; return 1; }
    ./configure || { log_error "VapourSynth configure failed"; cd ..; cd ..; return 1; }
    make -j "$(nproc)" || { log_error "VapourSynth make failed"; cd ..; cd ..; return 1; }
    make install || { log_error "VapourSynth make install failed"; cd ..; cd ..; return 1; }
    cd ..

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
    log_info "Compiling FFMS2..."
    if [ -d "ffms2" ]; then rm -rf ffms2; fi
    # Use tag 5.0 (compatible with FFmpeg 6.x)
    git clone --branch 5.0 --depth 1 https://github.com/FFMS/ffms2.git || { log_error "Failed to clone FFMS2"; cd ..; return 1; }
    cd ffms2 || { log_error "Failed to cd into ffms2"; cd ..; cd ..; return 1; }
    ./autogen.sh || { log_error "FFMS2 autogen failed"; cd ..; cd ..; return 1; }
    ./configure --enable-shared || { log_error "FFMS2 configure failed"; cd ..; cd ..; return 1; }
    make -j "$(nproc)" || { log_error "FFMS2 make failed"; cd ..; cd ..; return 1; }
    make install || { log_error "FFMS2 make install failed"; cd ..; cd ..; return 1; }
    cd ..

    # Symlink FFMS2 to VapourSynth Autoload Plugin Path
    local VS_PLUGIN_PATH="/usr/local/lib/vapoursynth"
    mkdir -p "$VS_PLUGIN_PATH"

    if [ -f "/usr/local/lib/libffms2.so" ]; then
        log_info "Linking FFMS2 to VapourSynth plugin folder..."
        ln -sf "/usr/local/lib/libffms2.so" "$VS_PLUGIN_PATH/libffms2.so"
    fi

    # 3. BestSource (Alternative and preferred VapourSynth Source plugin)
    log_info "Compiling BestSource..."
    if [ -d "bestsource" ]; then rm -rf bestsource; fi
    git clone --depth 1 --recurse-submodules https://github.com/vapoursynth/bestsource.git || { log_error "Failed to clone BestSource"; cd ..; return 1; }
    cd bestsource || { log_error "Failed to cd into bestsource"; cd ..; cd ..; return 1; }

    # Export explicitly so meson finds the custom FFmpeg master build
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
    # Ensure meson is available (installed via pacman on Arch, or use venv pip)
    if ! command -v meson &> /dev/null && [ -d "$VENV_DIR" ]; then
        "$VENV_DIR/bin/pip" install meson || true
    fi
    meson setup build || { log_error "BestSource meson setup failed"; cd ..; cd ..; return 1; }
    ninja -C build || { log_error "BestSource ninja build failed"; cd ..; cd ..; return 1; }
    ninja -C build install || { log_error "BestSource ninja install failed"; cd ..; cd ..; return 1; }

    # Find where BestSource was installed and link it
    local BS_SO
    BS_SO="$(find /usr/local/lib -name 'libbestsource*' -type f 2>/dev/null | head -1)"
    if [ -n "$BS_SO" ]; then
        log_info "Linking BestSource to VapourSynth plugin folder..."
        ln -sf "$BS_SO" "$VS_PLUGIN_PATH/"
    fi
    cd ..

    ldconfig
    cd .. # Exit build_tmp

    log_success "VapourSynth, FFMS2, and BestSource installed."
}

uninstall_vapoursynth() {
    log_info "Uninstalling VapourSynth and FFMS2..."

    # 1. Binaries
    rm -vf /usr/local/bin/vspipe

    # 2. Libraries
    rm -vf /usr/local/lib/libvapoursynth*
    rm -vf /usr/local/lib/libffms2*
    rm -vf /usr/local/lib/vapoursynth.so

    # 3. Headers
    rm -rf /usr/local/include/vapoursynth
    rm -rf /usr/local/include/ffms2

    # 4. Plugins
    rm -rf /usr/local/lib/vapoursynth

    # 5. Python Link
    local SITE_PKG_DIR
    SITE_PKG_DIR="$(get_python_site_packages)"
    rm -vf "$SITE_PKG_DIR/vapoursynth.so"

    # 6. PkgConfig
    rm -vf /usr/local/lib/pkgconfig/vapoursynth.pc
    rm -vf /usr/local/lib/pkgconfig/ffms2.pc

    # Refresh libs
    ldconfig

    log_success "VapourSynth and FFMS2 uninstalled."
}
