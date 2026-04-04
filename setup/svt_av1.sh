#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_svt_av1() {
    # Check for PSY fork specifically — the standard svt-av1 from pacman won't have PSY flags
    local need_build=true
    if [ -f /usr/local/bin/SvtAv1EncApp ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        need_build=false
        log_info "SVT-AV1-PSY already installed at /usr/local/bin/SvtAv1EncApp."
    fi

    if $need_build; then
        log_info "Compiling SVT-AV1-PSY (5fish Fork)..."

        # Ensure llvm-profdata for PGO
        if ! command -v llvm-profdata &> /dev/null; then
            # On Debian, try to find a versioned binary
            local LLVM_PROFDATA=$(find /usr/bin -name "llvm-profdata-*" 2>/dev/null | sort -V | tail -n 1)
            if [ -n "$LLVM_PROFDATA" ]; then
                log_info "Found $LLVM_PROFDATA. Linking..."
                ln -sf "$LLVM_PROFDATA" /usr/local/bin/llvm-profdata
            else
                log_warn "llvm-profdata not found. PGO might fail."
            fi
        fi

        local ORIG_DIR="$(pwd)"
        local BUILD_DIR="$ORIG_DIR/build_tmp"
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR" || exit 1

        if [ -d "svt-av1-psy" ]; then rm -rf svt-av1-psy; fi
        git clone --branch v2.3.0-C --depth 1 https://github.com/5fish/svt-av1-psy.git || { cd "$ORIG_DIR"; log_error "Failed to clone SVT-AV1-PSY"; return 1; }
        cd svt-av1-psy || { cd "$ORIG_DIR"; log_error "Failed to cd into svt-av1-psy"; return 1; }

        # Patch CMakeLists to fix 'target_link_libraries' PRIVATE/keyword mismatch error
        sed -i 's/\r$//' Source/App/CMakeLists.txt
        sed -i 's/target_link_libraries(SvtAv1EncApp ${PLATFORM_LIBS})/target_link_libraries(SvtAv1EncApp PRIVATE ${PLATFORM_LIBS})/' Source/App/CMakeLists.txt
        sed -i 's/target_link_libraries(SvtAv1EncApp$/target_link_libraries(SvtAv1EncApp PRIVATE/' Source/App/CMakeLists.txt

        mkdir -p Build/linux
        cd Build/linux || { cd "$ORIG_DIR"; log_error "Failed to cd into Build/linux"; return 1; }

        cmake ../.. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
            -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
            -DENABLE_AVX512=ON -DNATIVE=ON \
            -DSVT_AV1_PGO=ON -DSVT_AV1_LTO=ON || { cd "$ORIG_DIR"; log_error "SVT-AV1 cmake failed"; return 1; }

        make -j "$(nproc)" || { cd "$ORIG_DIR"; log_error "SVT-AV1 make failed"; return 1; }
        make install || { cd "$ORIG_DIR"; log_error "SVT-AV1 make install failed"; return 1; }
        cd "$ORIG_DIR"

        log_success "SVT-AV1-PSY installed."
    fi
}

uninstall_svt_av1() {
    log_info "Uninstalling SVT-AV1-PSY..."
    rm -vf /usr/local/bin/SvtAv1EncApp
    rm -vf /usr/local/lib/libSvtAv1Enc*
    rm -rf /usr/local/include/svt-av1
    rm -vf /usr/local/lib/pkgconfig/SvtAv1Enc.pc
    log_success "SVT-AV1-PSY uninstalled."
}
