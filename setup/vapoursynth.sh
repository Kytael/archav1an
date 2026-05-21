#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_vapoursynth() {
    if [ -f "$VS_PREFIX/bin/vspipe" ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "VapourSynth (source-built) is already installed at $VS_PREFIX."
        return 0
    fi

    log_info "Compiling VapourSynth from source with native optimizations into $VS_PREFIX..."
    set_native_build_flags

    # Ensure the venv exists — VS configure binds against the venv's Python
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        log_error "Venv missing at $VENV_DIR. Run setup.sh --install python_libs first."
        return 1
    fi

    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    # 1. VapourSynth R76 (meson build, against the uv-managed Python in $VENV_DIR).
    # R74 migrated the build system from autotools to meson; libvapoursynth.so
    # now has a SONAME (libvapoursynth.so.4 from soversion derived from
    # VAPOURSYNTH_API_MAJOR). Isolation continues to work because activate-venv.sh
    # puts $VS_PREFIX/lib on LD_LIBRARY_PATH, and ld.so checks LD_LIBRARY_PATH
    # before the ldconfig cache. Outside the activated env, the system v75
    # remains the default.
    if [ -d "vapoursynth" ]; then rm -rf vapoursynth; fi
    git clone --branch R76 --depth 1 https://github.com/vapoursynth/vapoursynth.git \
        || { cd "$ORIG_DIR"; log_error "Failed to clone VapourSynth"; return 1; }
    cd vapoursynth || { cd "$ORIG_DIR"; log_error "Failed to cd into vapoursynth"; return 1; }

    # Install meson + ninja INTO the venv so that meson runs on the
    # venv's Python interpreter. meson.python.find_installation() returns
    # the interpreter meson itself runs on (per meson docs), so using
    # /usr/sbin/meson — which is shebanged to system /usr/bin/python (3.14
    # on current Arch) — would build VS against the wrong Python.
    VIRTUAL_ENV="$VENV_DIR" uv pip install --quiet meson ninja \
        || { cd "$ORIG_DIR"; log_error "Failed to install meson/ninja into venv"; return 1; }

    local _vs_py_ver
    _vs_py_ver="$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    log_info "VS R76 build: targeting Python $_vs_py_ver from $VENV_DIR (via $VENV_DIR/bin/meson)"

    "$VENV_DIR/bin/meson" setup build \
        --prefix="$VS_PREFIX" \
        --buildtype=release \
        -Dpython.platlibdir="lib/python${_vs_py_ver}/site-packages" \
        -Dpython.purelibdir="lib/python${_vs_py_ver}/site-packages" \
        || { cd "$ORIG_DIR"; log_error "VapourSynth meson setup failed"; return 1; }
    "$VENV_DIR/bin/meson" compile -C build \
        || { cd "$ORIG_DIR"; log_error "VapourSynth meson compile failed"; return 1; }
    "$VENV_DIR/bin/meson" install -C build \
        || { cd "$ORIG_DIR"; log_error "VapourSynth meson install failed"; return 1; }
    cd "$BUILD_DIR"

    # R76 installs everything (libvapoursynth.so.4, libvsscript.so, vspipe,
    # headers, vapoursynth.pc) inside the Python package directory because
    # upstream's install_dir = py.get_install_dir() / 'vapoursynth'. Bridge
    # to the traditional bin/lib/include layout via symlinks so plugin
    # builds (wwxd, vszip, subtext, denoiser) and consumers (activate-venv.sh,
    # ffmpeg detection) keep finding things where they expect.
    local VS_PKG_DIR
    VS_PKG_DIR="$(find "$VS_PREFIX/lib" -name 'vapoursynth.abi*.so' -path '*site-packages*' 2>/dev/null | head -1)"
    if [ -n "$VS_PKG_DIR" ]; then
        VS_PKG_DIR="$(dirname "$VS_PKG_DIR")"
        log_info "Bridging R76 package layout to traditional prefix from $VS_PKG_DIR..."
        mkdir -p "$VS_PREFIX/bin" "$VS_PREFIX/lib" "$VS_PREFIX/include" "$VS_PREFIX/lib/pkgconfig"

        # vspipe binary
        ln -sf "$VS_PKG_DIR/vspipe" "$VS_PREFIX/bin/vspipe"

        # core libraries — symlink the SONAME-versioned file + the unversioned alias.
        # Do NOT symlink libvsscript.so: vsscript.cpp uses dladdr() to find itself
        # and looks up that path in ~/.config/vapoursynth/vapoursynth.toml. With
        # LD_LIBRARY_PATH=$VS_PREFIX/lib set by activate-venv.sh, a symlink at
        # $VS_PREFIX/lib/libvsscript.so would be loaded first, dladdr would
        # return the symlink path, and the toml lookup (keyed by the real path)
        # would miss — vspipe then fails with "Python executable and library
        # path couldn't be determined". vspipe finds libvsscript via its own
        # $ORIGIN rpath, so the symlink wasn't pulling weight anyway.
        ln -sf "$VS_PKG_DIR/libvapoursynth.so.4" "$VS_PREFIX/lib/libvapoursynth.so.4"
        ln -sf "libvapoursynth.so.4"             "$VS_PREFIX/lib/libvapoursynth.so"
        rm -f "$VS_PREFIX/lib/libvsscript.so"

        # headers — point a directory symlink at the package's include/ subtree
        rm -rf "$VS_PREFIX/include/vapoursynth"
        ln -sfn "$VS_PKG_DIR/include"            "$VS_PREFIX/include/vapoursynth"

        # pkg-config: the upstream .pc uses ${pcfiledir}/.. so we can't symlink
        # it (the relative path would resolve to $VS_PREFIX/lib instead of the
        # package dir). Generate an absolute-path .pc inside lib/pkgconfig.
        cat > "$VS_PREFIX/lib/pkgconfig/vapoursynth.pc" <<EOF
prefix=$VS_PKG_DIR
includedir=\${prefix}/include
libdir=\${prefix}

Name: vapoursynth
Description: A frameserver for the 21st century
Version: 76
Cflags: -I\${includedir}
Libs: -L\${libdir} -lvapoursynth
EOF
        log_success "R76 layout bridged: bin/vspipe, lib/libvapoursynth.so.4, include/vapoursynth/, lib/pkgconfig/vapoursynth.pc all wired."
    else
        log_warn "Source-built vapoursynth abi*.so not found under $VS_PREFIX/lib — bridge symlinks NOT created."
    fi

    # Make the Python module visible in the venv via .pth.
    if [ -n "$VS_PKG_DIR" ]; then
        local VENV_SP
        VENV_SP="$("$VENV_DIR/bin/python" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')"
        dirname "$VS_PKG_DIR" > "$VENV_SP/_vapoursynth_native.pth"
        log_info "Wrote $VENV_SP/_vapoursynth_native.pth pointing at $(dirname "$VS_PKG_DIR")"
    fi

    # 2. FFMS2
    log_info "Compiling FFMS2 with native optimizations..."
    if [ -d "ffms2" ]; then rm -rf ffms2; fi
    git clone --branch 5.0 --depth 1 https://github.com/FFMS/ffms2.git \
        || { cd "$ORIG_DIR"; log_error "Failed to clone FFMS2"; return 1; }
    cd ffms2 || { cd "$ORIG_DIR"; log_error "Failed to cd into ffms2"; return 1; }
    ./autogen.sh || { cd "$ORIG_DIR"; log_error "FFMS2 autogen failed"; return 1; }
    ./configure --prefix="$VS_PREFIX" --enable-shared \
        || { cd "$ORIG_DIR"; log_error "FFMS2 configure failed"; return 1; }
    make -j "$(nproc)" || { cd "$ORIG_DIR"; log_error "FFMS2 make failed"; return 1; }
    make install || { cd "$ORIG_DIR"; log_error "FFMS2 make install failed"; return 1; }
    cd "$BUILD_DIR"

    # Symlink FFMS2 to VapourSynth plugin path
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"
    if [ -f "$VS_PREFIX/lib/libffms2.so" ]; then
        log_info "Linking FFMS2 to VapourSynth plugin folder..."
        ln -sf "$VS_PREFIX/lib/libffms2.so" "$VS_PLUGIN_PATH/libffms2.so"
    fi

    # 3. BestSource (meson uses --prefix via meson setup --prefix)
    log_info "Compiling BestSource with native optimizations..."
    if [ -d "bestsource" ]; then rm -rf bestsource; fi
    git clone --depth 1 --recurse-submodules https://github.com/vapoursynth/bestsource.git \
        || { cd "$ORIG_DIR"; log_error "Failed to clone BestSource"; return 1; }
    cd bestsource || { cd "$ORIG_DIR"; log_error "Failed to cd into bestsource"; return 1; }

    if ! command -v meson &> /dev/null && [ -d "$VENV_DIR" ]; then
        VIRTUAL_ENV="$VENV_DIR" uv pip install meson || true
    fi
    meson setup build --prefix="$VS_PREFIX" --buildtype=release \
        -Dc_args="-march=native -O3" \
        -Dcpp_args="-march=native -O3" \
        -Db_lto=true \
        || { cd "$ORIG_DIR"; log_error "BestSource meson setup failed"; return 1; }
    ninja -C build || { cd "$ORIG_DIR"; log_error "BestSource ninja build failed"; return 1; }
    ninja -C build install || { cd "$ORIG_DIR"; log_error "BestSource ninja install failed"; return 1; }

    local BS_SO
    BS_SO="$(find "$VS_PREFIX/lib" -name 'libbestsource*' -type f 2>/dev/null | head -1)"
    if [ -n "$BS_SO" ]; then
        log_info "Linking BestSource to VapourSynth plugin folder..."
        ln -sf "$BS_SO" "$VS_PLUGIN_PATH/"
    fi

    # NOTE: NOT running ldconfig. The R73 .so has no SONAME by design (so it
    # cannot leak into the global ldconfig cache and stomp on pacman v75).
    # Discovery happens via LD_LIBRARY_PATH in activate-venv.sh.
    cd "$ORIG_DIR"

    log_success "VapourSynth, FFMS2, and BestSource installed under $VS_PREFIX with native optimizations."
}

uninstall_vapoursynth() {
    log_info "Uninstalling VapourSynth, FFMS2, and BestSource from $VS_PREFIX..."

    # Native install lives entirely under $VS_PREFIX — single targeted removal.
    # Preserve the venv unless explicitly asked: only remove native components
    # so re-installing doesn't force a full Python rebuild.
    if [ -d "$VS_PREFIX" ]; then
        rm -vrf "$VS_PREFIX/bin/vspipe" \
                "$VS_PREFIX/lib/libvapoursynth"* \
                "$VS_PREFIX/lib/libffms2"* \
                "$VS_PREFIX/lib/libbestsource"* \
                "$VS_PREFIX/lib/vapoursynth" \
                "$VS_PREFIX/lib/python3."*/site-packages/vapoursynth* \
                "$VS_PREFIX/include/vapoursynth" \
                "$VS_PREFIX/include/ffms2" \
                "$VS_PREFIX/lib/pkgconfig/vapoursynth.pc" \
                "$VS_PREFIX/lib/pkgconfig/ffms2.pc" \
                2>/dev/null
    fi

    # Remove the venv .pth so the venv stops trying to import a non-existent module
    if [ -d "$VENV_DIR" ]; then
        local VENV_SP
        VENV_SP="$("$VENV_DIR/bin/python" -c 'import sysconfig;print(sysconfig.get_path("purelib"))' 2>/dev/null)"
        [ -n "$VENV_SP" ] && rm -vf "$VENV_SP/_vapoursynth_native.pth"
    fi

    # Defensive ldconfig (no-op since we never registered SONAMEs, but tidies
    # any leftover cache entry from a prior /usr/local install).
    ldconfig 2>/dev/null || true

    log_success "VapourSynth, FFMS2, and BestSource removed from $VS_PREFIX."
}
