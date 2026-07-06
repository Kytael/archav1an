#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

SOURCES["vszip:vapoursynth-zip"]="https://github.com/dnjulek/vapoursynth-zip.git|R13"
ARTIFACTS["vszip"]="lib/vapoursynth/libvszip.so"

install_vszip() {
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    mkdir -p "$VS_PLUGIN_PATH"

    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    log_info "Compiling VSZIP..."
    clone_src vszip vapoursynth-zip vszip || { cd "$ORIG_DIR"; return 1; }
    cd vszip || { cd "$ORIG_DIR"; log_error "Failed to cd into vszip"; return 1; }

    # Build with our own zig at $VS_PREFIX/bin (same pattern as fssimu2.sh)
    # instead of upstream build-help/build.sh, which downloads its own zig into
    # the source tree and sudo-installs into the pacman-owned /usr/lib/vapoursynth.
    # R13's build.zig hard-enforces zig >= 0.15.2.
    local ZIG_VERSION="0.15.2"
    local ARCH=$(uname -m)
    local ZIG_ARCH="x86_64"
    if [ "$ARCH" = "aarch64" ]; then
        ZIG_ARCH="aarch64"
    fi
    local ZIG_TARBALL="zig-${ZIG_ARCH}-linux-${ZIG_VERSION}.tar.xz"
    local ZIG_URL="https://ziglang.org/download/${ZIG_VERSION}/${ZIG_TARBALL}"
    local ZIG_DIR="zig-${ZIG_ARCH}-linux-${ZIG_VERSION}"
    if ! "$VS_PREFIX/bin/zig" version 2>/dev/null | grep -q "${ZIG_VERSION}"; then
        log_info "Downloading Zig ${ZIG_VERSION}..."
        wget -q "$ZIG_URL" -O "/tmp/${ZIG_TARBALL}" || { cd "$ORIG_DIR"; log_error "Failed to download Zig"; return 1; }
        tar -xf "/tmp/${ZIG_TARBALL}" -C /tmp || { cd "$ORIG_DIR"; log_error "Failed to extract Zig"; return 1; }
        cp "/tmp/${ZIG_DIR}/zig" "$VS_PREFIX/bin/" || { cd "$ORIG_DIR"; log_error "Failed to copy Zig binary"; return 1; }
        # rm first: re-running cp -r into an existing dir nests lib/ inside it
        rm -rf "$VS_PREFIX/lib/zig"
        cp -r "/tmp/${ZIG_DIR}/lib" "$VS_PREFIX/lib/zig" || { cd "$ORIG_DIR"; log_error "Failed to copy Zig lib"; return 1; }
        rm -rf "/tmp/${ZIG_TARBALL}" "/tmp/${ZIG_DIR}"
    fi

    "$VS_PREFIX/bin/zig" build -Doptimize=ReleaseFast || { cd "$ORIG_DIR"; log_error "VSZIP zig build failed"; return 1; }

    if [ -f "zig-out/lib/libvszip.so" ]; then
        cp "zig-out/lib/libvszip.so" "$VS_PLUGIN_PATH/libvszip.so" || { cd "$ORIG_DIR"; log_error "Failed to copy libvszip.so"; return 1; }
    else
        cd "$ORIG_DIR"; log_error "VSZIP Compilation failed!"; return 1
    fi

    cd "$ORIG_DIR"

    log_success "VSZIP installed."
}

uninstall_vszip() {
    log_info "Uninstalling VSZIP..."
    local VS_PLUGIN_PATH
    VS_PLUGIN_PATH="$(get_vs_plugin_path)"
    find "$VS_PLUGIN_PATH" "$VS_PREFIX/lib/vapoursynth" -name "libvszip.so" -delete 2>/dev/null
    rm -f "$MANIFEST_DIR/vszip.src"
    log_success "VSZIP uninstalled."
}
