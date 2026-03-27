#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_vs_plugins() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    mkdir -p build_tmp
    cd build_tmp || exit 1

    # 1. WWXD
    log_info "Compiling VapourSynth-WWXD..."
    if [ -d "vapoursynth-wwxd" ]; then rm -rf vapoursynth-wwxd; fi
    git clone --branch v1.0 --depth 1 https://github.com/dubhater/vapoursynth-wwxd.git
    cd vapoursynth-wwxd

    # Find VapourSynth headers dynamically
    local VS_INCLUDE=""
    if command -v pkg-config &> /dev/null && pkg-config --exists vapoursynth 2>/dev/null; then
        VS_INCLUDE="$(pkg-config --cflags vapoursynth)"
    elif [ -d "/usr/local/include/vapoursynth" ]; then
        VS_INCLUDE="-I/usr/local/include/vapoursynth"
    elif [ -d "/usr/include/vapoursynth" ]; then
        VS_INCLUDE="-I/usr/include/vapoursynth"
    fi

    gcc -o libwwxd.so -fPIC -shared -O3 -Wall -Wextra -I. $VS_INCLUDE src/*.c -lm

    cp libwwxd.so "$VS_PLUGIN_PATH/"
    cd ..

    # 2. VSZIP
    log_info "Compiling VSZIP..."
    if [ -d "vszip" ]; then rm -rf vszip; fi
    git clone --branch R13 --depth 1 https://github.com/dnjulek/vapoursynth-zip.git vszip
    cd vszip

    # Using existing build script that handles Zig
    cd build-help
    chmod +x build.sh
    ./build.sh

    if [ -f "../zig-out/lib/libvszip.so" ]; then
        cp "../zig-out/lib/libvszip.so" "$VS_PLUGIN_PATH/libvszip.so"
    else
        log_error "VSZIP Compilation failed!"
    fi
    cd ../..

    ldconfig

    # 3. KNLMeansCL (OpenCL denoiser for VapourSynth)
    log_info "Compiling KNLMeansCL..."
    if [ -d "KNLMeansCL" ]; then rm -rf KNLMeansCL; fi
    git clone --depth 1 https://github.com/Khanattila/KNLMeansCL.git
    cd KNLMeansCL
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"$(nproc)"

    if [ -f "libknlmeanscl.so" ]; then
        cp "libknlmeanscl.so" "$VS_PLUGIN_PATH/"
    else
        log_error "KNLMeansCL compilation failed!"
    fi
    cd ../..

    # 4. SubText
    log_info "Compiling SubText..."
    if [ -d "subtext" ]; then rm -rf subtext; fi
    git clone --branch R5 --depth 1 https://github.com/vapoursynth/subtext.git
    cd subtext
    mkdir build && cd build
    meson setup .. --buildtype=release
    ninja

    if [ -f "libsubtext.so" ]; then
        cp "libsubtext.so" "$VS_PLUGIN_PATH/"
    else
        log_error "SubText compilation failed!"
    fi

    cd ../../..

    log_success "VapourSynth plugins installed."
}
