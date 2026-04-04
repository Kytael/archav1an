#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_dav1d() {
    if [ -f /usr/local/lib/libdav1d.so ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "dav1d (source-built) is already installed."
        return 0
    fi

    log_info "Compiling dav1d from source with native optimizations..."
    set_native_build_flags

    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    if [ -d "dav1d" ]; then rm -rf dav1d; fi
    git clone --branch 1.5.1 --depth 1 https://code.videolan.org/videolan/dav1d.git || { cd "$ORIG_DIR"; log_error "Failed to clone dav1d"; return 1; }
    cd dav1d || { cd "$ORIG_DIR"; log_error "Failed to cd into dav1d"; return 1; }

    CC=clang CXX=clang++ meson setup build --buildtype=release \
        --prefix=/usr/local \
        -Dc_args="-march=native -O3" \
        -Db_lto=true || { cd "$ORIG_DIR"; log_error "dav1d meson setup failed"; return 1; }
    ninja -C build || { cd "$ORIG_DIR"; log_error "dav1d build failed"; return 1; }
    ninja -C build install || { cd "$ORIG_DIR"; log_error "dav1d install failed"; return 1; }
    ldconfig
    cd "$ORIG_DIR"

    log_success "dav1d installed with LTO and -march=native."
}

# Common FFmpeg configure flags
_ffmpeg_configure() {
    local extra_cflags="$1"
    local extra_ldflags="$2"
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    ./configure \
      --prefix="/usr/local" \
      --cc=clang \
      --cxx=clang++ \
      --enable-shared \
      --enable-gpl \
      --enable-version3 \
      --enable-libx264 \
      --enable-libx265 \
      --enable-libsvtav1 \
      --enable-libdav1d \
      --enable-libvpx \
      --enable-libass \
      --enable-libfreetype \
      --enable-libfribidi \
      --enable-libfontconfig \
      --enable-libopus \
      --enable-libmp3lame \
      --enable-libvorbis \
      --enable-libwebp \
      --enable-libzimg \
      --enable-libsoxr \
      --enable-libsrt \
      --enable-libvidstab \
      --enable-libbluray \
      --enable-gnutls \
      --enable-vaapi \
      --enable-vulkan \
      --disable-doc \
      --extra-cflags="$extra_cflags -Wno-pass-failed -flto=thin" \
      --extra-ldflags="$extra_ldflags -flto=thin"
}

install_ffmpeg() {
    if [ -f /usr/local/bin/ffmpeg ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "FFmpeg (source-built) is already installed."
        return 0
    fi

    # Build dav1d from source first
    install_dav1d

    log_info "Compiling FFmpeg from source with PGO + LTO + native optimizations..."
    set_native_build_flags

    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    if [ -d "ffmpeg" ]; then rm -rf ffmpeg; fi
    git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git ffmpeg || { cd "$ORIG_DIR"; log_error "Failed to clone FFmpeg"; return 1; }
    cd ffmpeg || { cd "$ORIG_DIR"; log_error "Failed to cd into ffmpeg"; return 1; }

    # --- PGO Pass 1: Build with profiling instrumentation ---
    log_info "FFmpeg PGO pass 1: building instrumented binary..."
    local PROFILE_DIR="$BUILD_DIR/ffmpeg-pgo-profiles"
    mkdir -p "$PROFILE_DIR"

    _ffmpeg_configure \
        "-march=native -O3 -fprofile-generate=$PROFILE_DIR" \
        "-fuse-ld=lld -fprofile-generate=$PROFILE_DIR" \
    || { cd "$ORIG_DIR"; log_error "FFmpeg PGO pass 1 configure failed"; return 1; }

    make -j"$(nproc)" || { cd "$ORIG_DIR"; log_error "FFmpeg PGO pass 1 build failed"; return 1; }

    # --- PGO: Run representative workload to generate profile data ---
    log_info "FFmpeg PGO: generating profile data with representative workload..."

    # Install instrumented build so workloads can run against it
    make install || { cd "$ORIG_DIR"; log_error "FFmpeg PGO pass 1 install failed"; return 1; }
    ldconfig

    # Generate a synthetic test source and exercise common decode/encode paths
    ffmpeg -y -f lavfi -i "testsrc2=duration=10:size=1920x1080:rate=24" \
        -f lavfi -i "sine=frequency=440:duration=10" \
        -c:v libx264 -preset ultrafast -crf 23 \
        -c:a aac -b:a 128k \
        "$BUILD_DIR/pgo_test_h264.mkv" 2>/dev/null || log_warn "PGO h264 encode workload failed"

    # Note: SVT-AV1 and dav1d PGO workloads skipped — Clang's -fprofile-generate corrupts
    # the SVT-AV1 encoder config struct (zeroed width/height/CRF). This is a Clang bug,
    # not fixable without patching FFmpeg source. Acceptable because FFmpeg's SVT-AV1 code
    # is a thin wrapper, and SVT-AV1 itself has its own PGO via -DSVT_AV1_PGO=ON.
    # The h264 encode + filter workloads still profile the decode/demux/filter hot paths.

    # Transcode with filters (exercises zimg, scaling, pixel format conversion)
    ffmpeg -y -i "$BUILD_DIR/pgo_test_h264.mkv" \
        -vf "scale=1280:720,format=yuv420p10le" \
        -c:v libx264 -preset ultrafast -crf 28 \
        -f null - 2>/dev/null || log_warn "PGO filter workload failed"

    rm -f "$BUILD_DIR"/pgo_test_*.mkv

    # Check that profile data was generated
    local profile_count=$(find "$PROFILE_DIR" -name "*.profraw" 2>/dev/null | wc -l)
    if [ "$profile_count" -eq 0 ]; then
        log_warn "No PGO profile data generated. Falling back to non-PGO build."
        make clean
        _ffmpeg_configure \
            "-march=native -O3" \
            "-fuse-ld=lld" \
        || { cd "$ORIG_DIR"; log_error "FFmpeg configure failed"; return 1; }
        make -j"$(nproc)" || { cd "$ORIG_DIR"; log_error "FFmpeg make failed"; return 1; }
    else
        log_info "FFmpeg PGO: collected $profile_count profile files."

        # Merge profiles
        llvm-profdata merge -output="$PROFILE_DIR/default.profdata" \
            "$PROFILE_DIR"/*.profraw || { cd "$ORIG_DIR"; log_error "llvm-profdata merge failed"; return 1; }

        # --- PGO Pass 2: Rebuild with profile data + LTO ---
        log_info "FFmpeg PGO pass 2: rebuilding with profile data + LTO..."
        make clean

        _ffmpeg_configure \
            "-march=native -O3 -fprofile-use=$PROFILE_DIR/default.profdata" \
            "-fuse-ld=lld -fprofile-use=$PROFILE_DIR/default.profdata" \
        || { cd "$ORIG_DIR"; log_error "FFmpeg PGO pass 2 configure failed"; return 1; }

        make -j"$(nproc)" || { cd "$ORIG_DIR"; log_error "FFmpeg PGO pass 2 build failed"; return 1; }
    fi

    make install || { cd "$ORIG_DIR"; log_error "FFmpeg make install failed"; return 1; }
    ldconfig
    rm -rf "$PROFILE_DIR"
    cd "$ORIG_DIR"

    log_success "FFmpeg installed with PGO + LTO + -march=native."
}

uninstall_ffmpeg() {
    log_info "Uninstalling source-built FFmpeg and dav1d..."
    rm -vf /usr/local/bin/ff{mpeg,probe,play}
    rm -vf /usr/local/lib/libav{codec,format,util,device,filter}*
    rm -vf /usr/local/lib/libsw{scale,resample}*
    rm -vf /usr/local/lib/libpostproc*
    rm -vf /usr/local/lib/libdav1d*
    rm -rf /usr/local/include/libav{codec,format,util,device,filter}
    rm -rf /usr/local/include/libsw{scale,resample}
    rm -rf /usr/local/include/libpostproc
    rm -rf /usr/local/include/dav1d
    rm -vf /usr/local/lib/pkgconfig/libav*.pc
    rm -vf /usr/local/lib/pkgconfig/libsw*.pc
    rm -vf /usr/local/lib/pkgconfig/libpostproc.pc
    rm -vf /usr/local/lib/pkgconfig/dav1d.pc
    ldconfig
    log_success "FFmpeg and dav1d uninstalled."
}
