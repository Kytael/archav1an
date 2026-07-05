#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_dav1d() {
    if [ -f "$VS_PREFIX/lib/libdav1d.so" ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
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
    git clone --branch 1.5.3 --depth 1 https://code.videolan.org/videolan/dav1d.git || { cd "$ORIG_DIR"; log_error "Failed to clone dav1d"; return 1; }
    cd dav1d || { cd "$ORIG_DIR"; log_error "Failed to cd into dav1d"; return 1; }

    CC=clang CXX=clang++ meson setup build --buildtype=release \
        --prefix="$VS_PREFIX" \
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
    export PKG_CONFIG_PATH="$VS_PREFIX/lib/pkgconfig:/usr/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

    # NVIDIA GPU accel: enable NVDEC+NVENC+CUDA if ffnvcodec headers and
    # a CUDA toolkit are present. Detected at configure time so CPU-only
    # machines still build.
    local cuda_flags=()
    local cuda_toolkit=""
    for d in /opt/cuda /usr/local/cuda; do
        [ -x "$d/bin/nvcc" ] && { cuda_toolkit="$d"; break; }
    done
    if [ -n "$cuda_toolkit" ] && pkg-config --exists ffnvcodec; then
        # --enable-cuda-llvm uses clang to compile CUDA kernels (free license).
        # --enable-cuda-nvcc is rejected without --enable-nonfree.
        log_info "FFmpeg: enabling CUDA/NVDEC/NVENC (toolkit=$cuda_toolkit, llvm compiler)"
        cuda_flags+=(
            --enable-cuda-llvm
            --enable-cuvid
            --enable-nvdec
            --enable-nvenc
            --enable-ffnvcodec
        )
        extra_cflags="$extra_cflags -I$cuda_toolkit/include"
        extra_ldflags="$extra_ldflags -L$cuda_toolkit/lib64"
    else
        log_warn "FFmpeg: skipping CUDA (ffnvcodec headers or CUDA toolkit missing)"
    fi

    ./configure \
      --prefix="$VS_PREFIX" \
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
      "${cuda_flags[@]}" \
      --disable-doc \
      --extra-cflags="$extra_cflags -Wno-pass-failed -flto=thin" \
      --extra-ldflags="$extra_ldflags -flto=thin"
}

install_nv_codec_headers() {
    # Headers for NVDEC/NVENC in ffmpeg. Tiny header-only repo, safe to always refresh.
    if pkg-config --exists ffnvcodec && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "nv-codec-headers already installed ($(pkg-config --modversion ffnvcodec))."
        return 0
    fi
    log_info "Installing FFmpeg nv-codec-headers..."
    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR" || return 1
    if [ -d "nv-codec-headers" ]; then rm -rf nv-codec-headers; fi
    git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git || { cd "$ORIG_DIR"; log_error "nv-codec-headers clone failed"; return 1; }
    cd nv-codec-headers || { cd "$ORIG_DIR"; return 1; }
    make install PREFIX="$VS_PREFIX" || { cd "$ORIG_DIR"; log_error "nv-codec-headers install failed"; return 1; }
    cd "$ORIG_DIR"
    log_success "nv-codec-headers installed."
}

install_ffmpeg() {
    if [ -f "$VS_PREFIX/bin/ffmpeg" ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "FFmpeg (source-built) is already installed."
        return 0
    fi

    # Build dav1d from source first — a failure would silently produce an
    # FFmpeg without (or with system) dav1d, so treat it as fatal.
    install_dav1d || { log_error "dav1d install failed; aborting FFmpeg build."; return 1; }

    # Install ffnvcodec headers so --enable-cuda-llvm can see them
    install_nv_codec_headers || { log_error "nv-codec-headers install failed; aborting FFmpeg build."; return 1; }

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

    # Generate a synthetic test source and exercise common decode/encode paths.
    # Must run the just-installed instrumented binary, not whatever PATH's
    # ffmpeg is — otherwise no profile data is written and PGO never applies.
    "$VS_PREFIX/bin/ffmpeg" -y -f lavfi -i "testsrc2=duration=10:size=1920x1080:rate=24" \
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
    "$VS_PREFIX/bin/ffmpeg" -y -i "$BUILD_DIR/pgo_test_h264.mkv" \
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
    rm -vf "${VS_PREFIX}/bin/ff"{mpeg,probe,play}
    rm -vf "${VS_PREFIX}/lib/libav"{codec,format,util,device,filter}*
    rm -vf "${VS_PREFIX}/lib/libsw"{scale,resample}*
    rm -vf "${VS_PREFIX}/lib/libpostproc"*
    rm -vf "${VS_PREFIX}/lib/libdav1d"*
    rm -rf "${VS_PREFIX}/include/libav"{codec,format,util,device,filter}
    rm -rf "${VS_PREFIX}/include/libsw"{scale,resample}
    rm -rf "${VS_PREFIX}/include/libpostproc"
    rm -rf "${VS_PREFIX}/include/dav1d"
    rm -vf "${VS_PREFIX}/lib/pkgconfig/libav"*.pc
    rm -vf "${VS_PREFIX}/lib/pkgconfig/libsw"*.pc
    rm -vf "${VS_PREFIX}/lib/pkgconfig/libpostproc.pc"
    rm -vf "${VS_PREFIX}/lib/pkgconfig/dav1d.pc"
    ldconfig
    log_success "FFmpeg and dav1d uninstalled."
}
