# VapourSynth Isolation + uv-managed venv — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move archav1an's source-built native stack (VapourSynth R73, FFMS2, BestSource, plugins, ffmpeg, SVT-AV1, plus the Python venv) out of `/usr/local` + `/opt/auto-boost-av1an` and into a single self-contained `/opt/archav1an` prefix, with the venv created by `uv` against a pinned Python 3.13. Fixes the two real defects documented in `encoder-host:~/Public/vapoursynth-isolation.md`: (1) header symlinks collide with pacman's `python3.14/site-packages/vapoursynth/`, (2) `/usr/local/lib/libvapoursynth.so` has no SONAME so the global `vspipe` silently loads pacman's v75 core into the R73 binary.

**Architecture:** Single prefix `VS_PREFIX=/opt/archav1an` exported from `setup/common.sh` is the root for `bin/`, `lib/`, `lib/vapoursynth/`, `lib/python3/site-packages/`, `include/`, `share/`, and `venv/`. The venv is a clean uv-created venv (no `--system-site-packages`) running the latest available Python by default (overridable via `PYTHON_VERSION` env var), with a `.pth` file pointing at the prefix's site-packages so the R73 vapoursynth module is in the venv's import path. Source-built libs are found at runtime via `LD_LIBRARY_PATH=$VS_PREFIX/lib` (set in `activate-venv.sh`); since the no-SONAME R73 lib is resolved by directory rather than name, it never enters the ldconfig cache and cannot leak into other processes. pacman's system `vapoursynth` package is left entirely alone.

**Tech Stack:** bash setup scripts, uv (Python tooling — venv + pip replacement; *not* uvx, which is for ephemeral tool runs), autotools (VapourSynth/FFMS2), meson+ninja (BestSource/SubText), cmake (vs-mlrt/vsmigx), make (ffmpeg), pacman/paru (system deps), pkg-config, ldconfig.

**Constraints / preconditions:**
- Branch `denoise-server` is 1 commit behind `origin/denoise-server` (`76d06d8` — overlaps `setup/denoiser.sh`). Rebase before editing.
- Four files have uncommitted local changes (`Auto-Boost-Av1an.py`, `setup/denoiser.sh`, `setup/ffmpeg.sh`, `tools/svtav1-dispatch.py`) — stashed at Task 1, restored at Task 14 with explicit conflict resolution for `setup/denoiser.sh`.
- One-time `sudo` is needed in Task 10 (mkdir + chown of `/opt/archav1an`); every other step runs as `$USER`.
- **Python version is flexible.** Default = whatever `python3` resolves to on PATH (latest pacman). Override via `PYTHON_VERSION=<x.y>` env var (uv will download if missing). Upgrade detection in `install_python_libs` rebuilds the venv when the target version differs from what's on disk; if a Python bump breaks a Python dep, `uv pip install` errors out and the script prints the rollback recipe. The AGENTS.md learning about Python 3.14 breaking PyO3 applied to *headroom-ai* specifically, not archav1an's deps — they're free to track the latest.
- Pacman `vapoursynth` (currently v75) stays installed; it is intentionally untouched.

**Out of scope (deferred to follow-up plans):**
- `.gitignore` policy for `models/`, `plans/`, `.claude/`, `prefilter/*.MOV`, session reports.
- Decisions about merging `denoise-server` → `main` or deleting the stale `native-build` remote branch.
- Containerization. Skeleton `Containerfile` is unchanged.

---

## File Map

**Modified:**
- `setup/common.sh` — add `VS_PREFIX`, rewrite `get_vs_plugin_path()`, fix `set_native_build_flags()` paths. Task 2.
- `setup/python_libs.sh` — switch from `python -m venv --system-site-packages` to `uv venv --python 3.13`; switch `pip` → `uv pip`; venv path becomes `$VS_PREFIX/venv`. Task 3.
- `setup/vapoursynth.sh` — pass `--prefix=$VS_PREFIX PYTHON=$VS_PREFIX/venv/bin/python` to configure; pin `pythondir`/`pyexecdir` on `make install`; delete the system-site-packages symlink block; write a `.pth` into the venv; rewrite `uninstall_vapoursynth()` to clear `$VS_PREFIX` only; update FFMS2 + BestSource lib-path references. Task 4.
- `activate-venv.sh` — define `VS_PREFIX`; switch `LD_LIBRARY_PATH`, `VAPOURSYNTH_PLUGIN_PATH`, `PATH`, and the venv source path to the prefix. Task 5.
- `setup/wwxd.sh`, `setup/subtext.sh`, `setup/vszip.sh` — substitute hardcoded `/usr/local/include/vapoursynth` → `$VS_PREFIX/include/vapoursynth` and the uninstall `find` paths. Task 6.
- `setup/ffmpeg.sh` — change `--prefix=/usr/local` → `--prefix="$VS_PREFIX"`, `nv-codec-headers` `PREFIX=/usr/local` → `PREFIX="$VS_PREFIX"`, and update all `/usr/local/*` paths in the uninstall block. Task 7.
- `setup/denoiser.sh` — replace `/usr/local/include` fallback with `$VS_PREFIX/include` and any other prefix-relative path. (All other paths in this file already use `VS_PLUGIN_PATH` / `VS_INCLUDE_DIR` which derive from `common.sh`, so this is small.) Task 8.
- `setup/svt_av1.sh`, `setup/av1an.sh`, `setup/ffvship.sh`, `setup/oxipng.sh`, `setup/fssimu2.sh`, `setup/system_deps.sh`, `setup.sh` (the `is_installed` dispatch) — completeness sweep for any remaining `/usr/local` references. Task 9.

**Created at runtime (not committed):**
- `$VS_PREFIX/` directory tree (Task 10).
- `$VS_PREFIX/venv/lib/python3.13/site-packages/_vapoursynth_native.pth` (Task 4 install step).

**Not touched / deliberately left:**
- `setup/vs_plugins.sh` — legacy file, not sourced by `setup.sh`. Noted in Task 6 but not edited (dead code; cleanup is a separate decision).
- `Containerfile` — out of scope.

---

## Task 1: Prepare working tree

**Files:**
- Modify: working-tree state on branch `denoise-server`.

- [ ] **Step 1: Confirm current state.**

Run from repo root (`/home/user/archav1an`):
```
git status -s
git rev-list --left-right --count origin/denoise-server...HEAD
```
Expected: `M Auto-Boost-Av1an.py`, `M setup/denoiser.sh`, `M setup/ffmpeg.sh`, `M tools/svtav1-dispatch.py` plus several untracked entries; rev-list shows `1\t0` (one commit on origin not on HEAD, none ahead).

- [ ] **Step 2: Stash all uncommitted state including untracked.**

```
git stash push -u -m "vs-isolation: parked feature work + untracked"
git status -s
```
Expected after: no `M` entries; stash created. (Untracked items like `models/`, `plans/`, `.claude/` go into the stash and will be restored at Task 14.)

- [ ] **Step 3: Rebase onto origin/denoise-server to pick up commit 76d06d8.**

```
git pull --rebase origin denoise-server
```
Expected: fast-forward applies `76d06d8 fix(denoiser): CachyOS compat, FORCE_REINSTALL, SMDegrain deps`. Clean rebase, no conflicts (working tree was clean after Step 2).

- [ ] **Step 4: Create topic branch.**

```
git checkout -b vs-isolation
git log --oneline -3
```
Expected: branch `vs-isolation` at the new tip (which is now `76d06d8`).

- [ ] **Step 5: Verify upstream commit landed.**

```
grep -n "havsfunc/refs/tags/r33" setup/denoiser.sh
grep -n "vapoursynth-plugin-ctmf-git" setup/denoiser.sh
```
Expected: both grep matches found (proves `76d06d8` is applied).

---

## Task 2: Add VS_PREFIX to setup/common.sh

**Files:**
- Modify: `setup/common.sh:71-100` (the `get_vs_plugin_path` function and `set_native_build_flags` block).

- [ ] **Step 1: Insert `VS_PREFIX` definition above `get_vs_plugin_path`.**

Edit `setup/common.sh`. Replace the block at lines 71-83:
```
# Helper: get the VapourSynth plugin path
get_vs_plugin_path() {
    # Prefer source-built VapourSynth at /usr/local
    if [ -f /usr/local/lib/pkgconfig/vapoursynth.pc ]; then
        echo "/usr/local/lib/vapoursynth"
    elif command -v pkg-config &> /dev/null && pkg-config --exists vapoursynth 2>/dev/null; then
        echo "$(pkg-config --variable=libdir vapoursynth)/vapoursynth"
    elif [ "$DISTRO_FAMILY" = "arch" ]; then
        echo "/usr/local/lib/vapoursynth"
    else
        echo "/usr/lib/x86_64-linux-gnu/vapoursynth"
    fi
}
```
with:
```
# Install prefix for the source-built native stack (VapourSynth + plugins +
# ffmpeg + SVT-AV1 + venv). Chosen under /opt so pacman never owns anything
# inside it; deliberately *not* /usr/local. See plans/2026-05-20-vapoursynth-isolation.md.
VS_PREFIX="${VS_PREFIX:-/opt/archav1an}"
export VS_PREFIX

# Helper: get the VapourSynth plugin path. Single source of truth: $VS_PREFIX.
get_vs_plugin_path() {
    echo "$VS_PREFIX/lib/vapoursynth"
}
```

- [ ] **Step 2: Point the venv at the prefix.**

In `setup/common.sh`, change line 86 from:
```
VENV_DIR="${VENV_DIR:-/opt/auto-boost-av1an/venv}"
```
to:
```
VENV_DIR="${VENV_DIR:-$VS_PREFIX/venv}"
```

- [ ] **Step 3: Update `set_native_build_flags()` to use the prefix.**

In `setup/common.sh`, replace lines 97-99:
```
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="/usr/local/lib:${LIBRARY_PATH:-}"
```
with:
```
    export PKG_CONFIG_PATH="$VS_PREFIX/lib/pkgconfig:/usr/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export LD_LIBRARY_PATH="$VS_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$VS_PREFIX/lib:${LIBRARY_PATH:-}"
```

- [ ] **Step 4: Smoke-test by sourcing common.sh.**

```
bash -c 'source setup/common.sh && echo VS_PREFIX=$VS_PREFIX && echo VENV_DIR=$VENV_DIR && echo plugin_path=$(get_vs_plugin_path)'
```
Expected output:
```
VS_PREFIX=/opt/archav1an
VENV_DIR=/opt/archav1an/venv
plugin_path=/opt/archav1an/lib/vapoursynth
```

- [ ] **Step 5: Commit.**

```
git add setup/common.sh
git commit -m "setup(common): introduce VS_PREFIX, derive venv + plugin path + build flags from it"
```

---

## Task 3: Convert setup/python_libs.sh to uv + Python 3.13

**Files:**
- Modify: `setup/python_libs.sh` (whole file; only 41 lines).

- [ ] **Step 1: Verify uv is installed.**

```
command -v uv && uv --version
```
Expected: `/home/user/.local/bin/uv` (or similar) and a version string. If absent, install via `curl -LsSf https://astral.sh/uv/install.sh | sh` and re-source PATH before proceeding.

- [ ] **Step 2: Rewrite `setup/python_libs.sh`.**

Replace the entire body of `install_python_libs()` and `uninstall_python_libs()`. New content:

```sh
#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

install_python_libs() {
    log_info "Installing Python Libraries into venv ($VENV_DIR)..."

    if ! command -v uv &> /dev/null; then
        log_error "uv not found on PATH. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi

    # Target Python version:
    #   - default: whatever `python3` resolves to on PATH (latest pacman).
    #   - override: PYTHON_VERSION=<x.y> (e.g. 3.13) — uv downloads if missing.
    local TARGET_PY="${PYTHON_VERSION:-python3}"

    # Detect current venv's Python version (if venv exists). When it differs
    # from the target, rebuild — the venv is binary-incompatible across
    # Python minors, and the source-built VS module is too.
    local CURRENT_PY=""
    local TARGET_PY_CANON=""
    if [ -x "$VENV_DIR/bin/python" ]; then
        CURRENT_PY="$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    fi
    # Canonicalize target to x.y for comparison (resolves "python3" → actual version)
    if [ "$TARGET_PY" = "python3" ] && command -v python3 &> /dev/null; then
        TARGET_PY_CANON="$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    else
        TARGET_PY_CANON="$TARGET_PY"
    fi

    if [ -n "$CURRENT_PY" ] && [ -n "$TARGET_PY_CANON" ] && [ "$CURRENT_PY" != "$TARGET_PY_CANON" ]; then
        log_warn "Python version change detected: venv has $CURRENT_PY, target is $TARGET_PY_CANON. Rebuilding venv at $VENV_DIR."
        log_warn "VapourSynth source-build is binary-linked to the venv's Python; rerun '--install vapoursynth' after this completes."
        rm -rf "$VENV_DIR"
    fi

    # Create venv if it doesn't exist (or just removed for upgrade).
    # Clean venv (no --system-site-packages): the source-built VapourSynth
    # module is exposed via a .pth written by setup/vapoursynth.sh, not by
    # inheriting system site-packages.
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating uv-managed venv (Python $TARGET_PY) at $VENV_DIR..."
        mkdir -p "$(dirname "$VENV_DIR")"
        uv venv --python "$TARGET_PY" "$VENV_DIR" \
            || { log_error "Failed to create venv with uv at Python $TARGET_PY. To pin a specific version: PYTHON_VERSION=3.13 ./setup.sh --install python_libs"; return 1; }
    fi

    VIRTUAL_ENV="$VENV_DIR" uv pip install --upgrade pip \
        || log_warn "pip upgrade failed, continuing..."

    if ! VIRTUAL_ENV="$VENV_DIR" uv pip install \
            vsjetpack numpy rich vstools psutil anitopy pyperclip requests \
            requests_toolbelt natsort colorama Cython; then
        log_error "uv pip install failed under Python $TARGET_PY_CANON."
        log_error "If a binary dep lags behind the latest Python release (e.g. PyO3-based packages on Python bumps),"
        log_error "pin to a known-good version with: PYTHON_VERSION=3.13 ./setup.sh --install python_libs"
        return 1
    fi

    # Remove the pip-published vapoursynth stub which would shadow the source-built
    # R73 module that vapoursynth.sh wires via .pth.
    log_info "Removing pip-installed vapoursynth stub (avoid R73 module shadow)..."
    VIRTUAL_ENV="$VENV_DIR" uv pip uninstall vapoursynth || true

    log_success "Python libraries installed in venv (Python $TARGET_PY_CANON)."
}

uninstall_python_libs() {
    log_info "Uninstalling Python Libraries..."

    if [ -d "$VENV_DIR" ]; then
        VIRTUAL_ENV="$VENV_DIR" uv pip uninstall \
            vsjetpack numpy rich vstools psutil anitopy pyperclip \
            requests requests_toolbelt natsort colorama Cython \
            || true
        log_success "Python libraries uninstalled from venv."
    else
        log_warn "Venv not found at $VENV_DIR, nothing to uninstall."
    fi
}
```

Rationale: by default the installer tracks whatever Python `python3` points at (latest pacman). The override (`PYTHON_VERSION=3.13`) lets you roll back when a Python bump breaks a Python dep — the `uv pip install` error path tells you how. The upgrade-detection block rebuilds the venv when versions change *and* warns that VS source-build needs rerunning (its module .so is binary-linked to the venv's Python interpreter).

- [ ] **Step 3: Lint-check the rewritten script.**

```
bash -n setup/python_libs.sh
```
Expected: no output (syntax OK).

- [ ] **Step 4: Commit.**

```
git add setup/python_libs.sh
git commit -m "setup(python_libs): create venv via uv; flexible Python with PYTHON_VERSION override + upgrade detection"
```

---

## Task 4: Rewrite setup/vapoursynth.sh for prefixed configure + .pth + new uninstall

**Files:**
- Modify: `setup/vapoursynth.sh` (whole file; 122 lines).

- [ ] **Step 1: Replace `install_vapoursynth()`.**

Replace lines 8-91 (the entire function body) with:

```sh
install_vapoursynth() {
    if [ -f "$VS_PREFIX/bin/vspipe" ] && [ "${FORCE_REINSTALL:-0}" != "1" ]; then
        log_info "VapourSynth (source-built) is already installed at $VS_PREFIX."
        return 0
    fi

    log_info "Compiling VapourSynth from source with native optimizations into $VS_PREFIX..."
    set_native_build_flags

    # Ensure the venv exists — VS configure binds against the venv's Python 3.13
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        log_error "Venv missing at $VENV_DIR. Run setup.sh --install python_libs first."
        return 1
    fi

    local ORIG_DIR="$(pwd)"
    local BUILD_DIR="$ORIG_DIR/build_tmp"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    # 1. VapourSynth (R73, built against the uv-managed Python 3.13)
    if [ -d "vapoursynth" ]; then rm -rf vapoursynth; fi
    git clone --branch R73 --depth 1 https://github.com/vapoursynth/vapoursynth.git \
        || { cd "$ORIG_DIR"; log_error "Failed to clone VapourSynth"; return 1; }
    cd vapoursynth || { cd "$ORIG_DIR"; log_error "Failed to cd into vapoursynth"; return 1; }
    ./autogen.sh || { cd "$ORIG_DIR"; log_error "VapourSynth autogen failed"; return 1; }
    ./configure --prefix="$VS_PREFIX" PYTHON="$VENV_DIR/bin/python" \
        || { cd "$ORIG_DIR"; log_error "VapourSynth configure failed"; return 1; }
    make -j "$(nproc)" \
        || { cd "$ORIG_DIR"; log_error "VapourSynth make failed"; return 1; }
    # Pin pythondir/pyexecdir into the prefix so make install never writes into
    # the pacman-owned /usr/lib/python3.14/site-packages/vapoursynth/ tree.
    make install \
        pythondir="$VS_PREFIX/lib/python3/site-packages" \
        pyexecdir="$VS_PREFIX/lib/python3/site-packages" \
        || { cd "$ORIG_DIR"; log_error "VapourSynth make install failed"; return 1; }
    cd "$BUILD_DIR"

    # Make the R73 Python module visible in the venv via .pth (NOT a symlink
    # into system site-packages, which is what triggered the pacman conflict).
    local VS_PY
    VS_PY="$(find "$VS_PREFIX/lib" -name 'vapoursynth*.so' -path '*site-packages*' 2>/dev/null | head -1)"
    if [ -n "$VS_PY" ]; then
        local VENV_SP
        VENV_SP="$("$VENV_DIR/bin/python" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')"
        dirname "$VS_PY" > "$VENV_SP/_vapoursynth_native.pth"
        log_info "Wrote $VENV_SP/_vapoursynth_native.pth pointing at $(dirname "$VS_PY")"
    else
        log_warn "Source-built vapoursynth Python module not found under $VS_PREFIX/lib — .pth not written."
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

    # 3. BestSource (meson uses --prefix via -Dprefix)
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
```

- [ ] **Step 2: Replace `uninstall_vapoursynth()`.**

Replace lines 93-122 with:

```sh
uninstall_vapoursynth() {
    log_info "Uninstalling VapourSynth, FFMS2, and BestSource from $VS_PREFIX..."

    # Native install lives entirely under $VS_PREFIX — single rm is enough.
    if [ -d "$VS_PREFIX" ]; then
        # Preserve the venv unless explicitly asked: only remove native components
        # so re-installing doesn't force a full Python rebuild.
        rm -vrf "$VS_PREFIX/bin/vspipe" \
                "$VS_PREFIX/lib/libvapoursynth"* \
                "$VS_PREFIX/lib/libffms2"* \
                "$VS_PREFIX/lib/libbestsource"* \
                "$VS_PREFIX/lib/vapoursynth" \
                "$VS_PREFIX/lib/python3/site-packages/vapoursynth"* \
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

    # Defensive: also tidy any leftover ldconfig cache entry (no-op if pacman owns the symlinks)
    ldconfig 2>/dev/null

    log_success "VapourSynth, FFMS2, and BestSource removed from $VS_PREFIX."
}
```

- [ ] **Step 3: Lint-check.**

```
bash -n setup/vapoursynth.sh
```
Expected: no output.

- [ ] **Step 4: Commit.**

```
git add setup/vapoursynth.sh
git commit -m "setup(vapoursynth): build into VS_PREFIX, pin pythondir, wire venv via .pth"
```

---

## Task 5: Point activate-venv.sh at the prefix

**Files:**
- Modify: `activate-venv.sh` (58 lines, at repo root).

- [ ] **Step 1: Replace lines 5-17 (the venv path + runtime exports).**

Old:
```sh
VENV_DIR="${VENV_DIR:-/opt/auto-boost-av1an/venv}"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "[WARN] Python venv not found at $VENV_DIR. Run setup.sh first."
    echo "       Falling back to system python3."
fi

# Ensure source-built libraries and VapourSynth plugins are found at runtime
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
export VAPOURSYNTH_PLUGIN_PATH="/usr/local/lib/vapoursynth"
export PATH="/usr/local/bin:$PATH"
```

New:
```sh
# Install prefix for the isolated archav1an native stack. Sourced standalone
# (this file does NOT source setup/common.sh), so we define VS_PREFIX inline.
VS_PREFIX="${VS_PREFIX:-/opt/archav1an}"
VENV_DIR="${VENV_DIR:-$VS_PREFIX/venv}"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "[WARN] Python venv not found at $VENV_DIR. Run setup.sh first."
    echo "       Falling back to system python3."
fi

# Resolve the R73 .so by directory (it has no SONAME) so it overrides the
# pacman v75 library only inside this activated environment.
export LD_LIBRARY_PATH="$VS_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export VAPOURSYNTH_PLUGIN_PATH="$VS_PREFIX/lib/vapoursynth"
export PATH="$VS_PREFIX/bin:$PATH"
```

(Leave lines 19+ — WSL2 CUDA symlink block, AMD ROCm gfx detection, mimalloc preload — untouched.)

- [ ] **Step 2: Lint-check.**

```
bash -n activate-venv.sh
```
Expected: no output.

- [ ] **Step 3: Commit.**

```
git add activate-venv.sh
git commit -m "activate-venv: PATH/LD_LIBRARY_PATH/VS plugin dir keyed off VS_PREFIX"
```

---

## Task 6: Update plugin builders (wwxd, subtext, vszip)

**Files:**
- Modify: `setup/wwxd.sh:29-30`, `setup/wwxd.sh:50`.
- Modify: `setup/subtext.sh:50`.
- Modify: `setup/vszip.sh:43`.

Pattern: every `/usr/local/include/vapoursynth` → `$VS_PREFIX/include/vapoursynth`; every `/usr/local/lib/vapoursynth` in a `find` command → `$VS_PREFIX/lib/vapoursynth`. The `/usr/include/vapoursynth` system fallback stays as a last resort.

- [ ] **Step 1: Edit `setup/wwxd.sh`.**

In `install_wwxd()`, replace line 29-30:
```
    elif [ -d "/usr/local/include/vapoursynth" ]; then
        VS_INCLUDE="-I/usr/local/include/vapoursynth"
```
with:
```
    elif [ -d "$VS_PREFIX/include/vapoursynth" ]; then
        VS_INCLUDE="-I$VS_PREFIX/include/vapoursynth"
```

In `uninstall_wwxd()`, replace line 50:
```
    find "$VS_PLUGIN_PATH" /usr/local/lib/vapoursynth -name "libwwxd.so" -delete 2>/dev/null
```
with:
```
    find "$VS_PLUGIN_PATH" "$VS_PREFIX/lib/vapoursynth" -name "libwwxd.so" -delete 2>/dev/null
```

- [ ] **Step 2: Edit `setup/subtext.sh`.**

In `uninstall_subtext()`, replace line 50:
```
    find "$VS_PLUGIN_PATH" /usr/local/lib/vapoursynth -name "libsubtext.so" -delete 2>/dev/null
```
with:
```
    find "$VS_PLUGIN_PATH" "$VS_PREFIX/lib/vapoursynth" -name "libsubtext.so" -delete 2>/dev/null
```

- [ ] **Step 3: Edit `setup/vszip.sh`.**

In `uninstall_vszip()`, replace line 43:
```
    find "$VS_PLUGIN_PATH" /usr/local/lib/vapoursynth -name "libvszip.so" -delete 2>/dev/null
```
with:
```
    find "$VS_PLUGIN_PATH" "$VS_PREFIX/lib/vapoursynth" -name "libvszip.so" -delete 2>/dev/null
```

- [ ] **Step 4: Note (do not edit) `setup/vs_plugins.sh`.**

This file is not sourced by `setup.sh` (verify with `grep vs_plugins setup.sh` → no match). It is dead code carried from an older layout. Do **not** edit it in this plan; cleaning it up is a separate `.gitignore`/dead-code decision deferred to a follow-up.

- [ ] **Step 5: Lint + commit.**

```
bash -n setup/wwxd.sh setup/subtext.sh setup/vszip.sh
git add setup/wwxd.sh setup/subtext.sh setup/vszip.sh
git commit -m "setup(plugins): wwxd/subtext/vszip include + uninstall paths via VS_PREFIX"
```

---

## Task 7: Update setup/ffmpeg.sh paths

**Files:**
- Modify: `setup/ffmpeg.sh:70` (the active `--prefix=/usr/local`), line 116 (`PREFIX=/usr/local`), and the uninstall block at lines 224-236.

(Note: line 27 in `_ffmpeg_configure` references `/usr/local` inside a commented/unused stanza per the local diff. Confirm during edit — only replace active `--prefix` flags.)

- [ ] **Step 1: Replace the active `--prefix` flag.**

In `setup/ffmpeg.sh:70`, change:
```
      --prefix="/usr/local" \
```
to:
```
      --prefix="$VS_PREFIX" \
```

- [ ] **Step 2: Update nv-codec-headers install prefix.**

In `setup/ffmpeg.sh:116`, change:
```
    make install PREFIX=/usr/local || { cd "$ORIG_DIR"; log_error "nv-codec-headers install failed"; return 1; }
```
to:
```
    make install PREFIX="$VS_PREFIX" || { cd "$ORIG_DIR"; log_error "nv-codec-headers install failed"; return 1; }
```

- [ ] **Step 3: Update the uninstall block.**

In `setup/ffmpeg.sh`, replace lines 224-236:
```
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
```
with:
```
    rm -vf "$VS_PREFIX"/bin/ff{mpeg,probe,play}
    rm -vf "$VS_PREFIX"/lib/libav{codec,format,util,device,filter}*
    rm -vf "$VS_PREFIX"/lib/libsw{scale,resample}*
    rm -vf "$VS_PREFIX"/lib/libpostproc*
    rm -vf "$VS_PREFIX"/lib/libdav1d*
    rm -rf "$VS_PREFIX"/include/libav{codec,format,util,device,filter}
    rm -rf "$VS_PREFIX"/include/libsw{scale,resample}
    rm -rf "$VS_PREFIX"/include/libpostproc
    rm -rf "$VS_PREFIX"/include/dav1d
    rm -vf "$VS_PREFIX"/lib/pkgconfig/libav*.pc
    rm -vf "$VS_PREFIX"/lib/pkgconfig/libsw*.pc
    rm -vf "$VS_PREFIX"/lib/pkgconfig/libpostproc.pc
    rm -vf "$VS_PREFIX"/lib/pkgconfig/dav1d.pc
```

- [ ] **Step 4: Sweep for any remaining `/usr/local` literals in this file.**

```
grep -n "/usr/local" setup/ffmpeg.sh
```
Expected: only matches inside comments or fallbacks where `/usr/local/cuda` is a CUDA toolkit search path (lines 49-50, the `for d in /opt/cuda /usr/local/cuda` loop) — that one is **correct as-is** (it's looking for the CUDA toolkit, not our prefix).

- [ ] **Step 5: Lint + commit.**

```
bash -n setup/ffmpeg.sh
git add setup/ffmpeg.sh
git commit -m "setup(ffmpeg): build into VS_PREFIX; nv-codec-headers + uninstall paths follow"
```

---

## Task 8: Update setup/denoiser.sh include fallback

**Files:**
- Modify: `setup/denoiser.sh:19` (the include-dir fallback).

(The rest of `setup/denoiser.sh` already uses `VS_PLUGIN_PATH`/`VS_INCLUDE_DIR` derived from `common.sh`, so no other substitutions are needed.)

- [ ] **Step 1: Edit the include fallback.**

In `setup/denoiser.sh`, find the line:
```
VS_INCLUDE_DIR="$(pkg-config --variable=includedir vapoursynth 2>/dev/null || echo /usr/local/include)"
```
Change to:
```
VS_INCLUDE_DIR="$(pkg-config --variable=includedir vapoursynth 2>/dev/null || echo "$VS_PREFIX/include")"
```

- [ ] **Step 2: Sweep for any remaining `/usr/local` literals.**

```
grep -n "/usr/local" setup/denoiser.sh
```
Expected: only matches in CUDA toolkit search (`for _d in /opt/cuda/bin /usr/local/cuda/bin` at line 58) — leave that alone.

- [ ] **Step 3: Lint + commit.**

```
bash -n setup/denoiser.sh
git add setup/denoiser.sh
git commit -m "setup(denoiser): fallback include path follows VS_PREFIX"
```

---

## Task 9: Completeness sweep — remaining /usr/local references

**Files:**
- Inspect (and edit if needed): `setup/svt_av1.sh`, `setup/av1an.sh`, `setup/ffvship.sh`, `setup/oxipng.sh`, `setup/fssimu2.sh`, `setup/system_deps.sh`, `setup.sh`.

- [ ] **Step 1: Find all remaining `/usr/local` literals across the setup tree.**

```
grep -rn "/usr/local" setup/ setup.sh | grep -v -E "/usr/local/cuda|/usr/local/lib/wsl-cuda" | grep -v "^Binary"
```

Expected output is the set of remaining literals. The grep filters out two intentional exclusions; **do not "fix" either one — they are correct as written**:
- `/usr/local/cuda` — This is a *search* path used by `setup/common.sh:detect_gpu()` and `setup/ffmpeg.sh:_ffmpeg_configure()` to *locate* the NVIDIA CUDA toolkit (`for d in /opt/cuda /usr/local/cuda; do ... done`). NVIDIA puts the toolkit there on some installs; we do NOT install anything to it. Substituting `$VS_PREFIX/cuda` would stop the toolkit being found and break ffmpeg's `--enable-cuda-llvm` build.
- `/usr/local/lib/wsl-cuda` — This is a directory `setup/common.sh:setup_wsl2_cuda()` *creates and populates* with clean symlinks that bypass malformed NVIDIA WSL2 stubs at `/usr/lib/wsl/lib/libcuda.so.1`. It is a *system-wide* fix that must be findable by anything CUDA-using on the host, not just the activated archav1an env. Moving it into `$VS_PREFIX` would scope the fix to our env and break system CUDA tooling (nvcc/PyTorch/etc. running outside our venv).

- [ ] **Step 2: For each remaining match, decide and edit.**

For each line returned by Step 1 that is NOT a CUDA/WSL exception:
- If it's a build prefix or install destination → change to `$VS_PREFIX/...`.
- If it's an `is_installed` check in `setup.sh` (`[ -f /usr/local/bin/foo ]`) → change to `[ -f "$VS_PREFIX/bin/foo" ]`.
- If it's a comment or docstring → leave (or update only if misleading).

Specifically expected hits and their fixes (verify exact line content with `grep -n` first; line numbers may shift):

In `setup.sh` (the `is_installed` block, lines ~68-114):
```
        "vapoursynth")
            [ -f /usr/local/bin/vspipe ]
        "ffmpeg")
            [ -f /usr/local/bin/ffmpeg ]
        "av1an")
            [ -f /usr/local/bin/av1an ]
        "svt_av1")
            [ -f /usr/local/bin/SvtAv1EncApp ]
        "oxipng")
            [ -f /usr/local/bin/oxipng ]
```
Change every `/usr/local/bin/<x>` here to `"$VS_PREFIX/bin/<x>"`. Same in the `wwxd`/`vszip`/`subtext` checks: `/usr/local/lib/vapoursynth/lib<x>.so` → `"$VS_PREFIX/lib/vapoursynth/lib<x>.so"`.

For each of `setup/svt_av1.sh`, `setup/av1an.sh`, `setup/ffvship.sh`, `setup/oxipng.sh`, `setup/fssimu2.sh`: open the file, find `/usr/local` in any `--prefix`, `cargo install --root`, `cp`, `cmake -DCMAKE_INSTALL_PREFIX=`, `make install PREFIX=`, or uninstall `rm` command, and substitute `$VS_PREFIX`.

For `setup/system_deps.sh`: `/usr/local` should not appear — it's a pacman/apt wrapper. If grep finds anything, evaluate case-by-case.

- [ ] **Step 3: Re-run the sweep until clean.**

```
grep -rn "/usr/local" setup/ setup.sh | grep -v -E "/usr/local/cuda|/usr/local/lib/wsl-cuda"
```
Expected: empty output (no more substitution candidates).

- [ ] **Step 4: Lint all touched scripts.**

```
for f in setup/*.sh setup.sh; do bash -n "$f" || echo "SYNTAX FAIL: $f"; done
```
Expected: no `SYNTAX FAIL` lines.

- [ ] **Step 5: Commit.**

```
git add setup/svt_av1.sh setup/av1an.sh setup/ffvship.sh setup/oxipng.sh setup/fssimu2.sh setup/system_deps.sh setup.sh
git commit -m "setup: route remaining /usr/local installs and is_installed checks through VS_PREFIX"
```
(Stage only the files that actually changed; `git status -s` will show which to add.)

---

## Task 10: Create /opt/archav1an (one-time sudo bootstrap)

**Files:** No code changes; this is a filesystem operation.

- [ ] **Step 1: Confirm the directory does not exist.**

```
ls -ld /opt/archav1an 2>&1
```
Expected: `ls: cannot access '/opt/archav1an': No such file or directory` (or empty dir if pre-created).

- [ ] **Step 2: Create and chown the prefix.**

```
sudo install -d -o "$USER" -g "$USER" /opt/archav1an
ls -ld /opt/archav1an
```
Expected: directory exists with owner `$USER:$USER`, mode `0755`.

- [ ] **Step 3: Sanity-check it's writeable as the user.**

```
touch /opt/archav1an/.write-test && rm /opt/archav1an/.write-test && echo OK
```
Expected: `OK`. No further `sudo` should be needed for the rest of the plan.

---

## Task 11: Run the new setup end-to-end

**Files:** No code changes; this executes the edited scripts.

- [ ] **Step 1: Source the (edited) activate-venv.sh to confirm exports.**

Note: at this point the venv does not yet exist, so the script will warn and fall back to system python. The exports we care about (PATH, LD_LIBRARY_PATH, VS_PREFIX) are still set.

```
source activate-venv.sh
echo VS_PREFIX=$VS_PREFIX
echo VENV_DIR=$VENV_DIR
echo PATH-head=${PATH%%:*}
echo LD_LIBRARY_PATH-head=${LD_LIBRARY_PATH%%:*}
```
Expected: `VS_PREFIX=/opt/archav1an`, `VENV_DIR=/opt/archav1an/venv`, `PATH-head=/opt/archav1an/bin`, `LD_LIBRARY_PATH-head=/opt/archav1an/lib`.

- [ ] **Step 2: Install Python libs (creates the uv-managed venv).**

From the (deactivated, fresh) shell:
```
./setup.sh --install python_libs
```
Expected: log lines `Creating uv-managed venv (Python python3) at /opt/archav1an/venv...`, `Python libraries installed in venv (Python <x.y>).`. Verify:
```
/opt/archav1an/venv/bin/python --version
```
Expected: `Python <x.y>.<z>` matching whatever `python3 --version` reports on the system (currently pacman's 3.14.x). If you need to pin to 3.13 due to a dep break, re-run as `PYTHON_VERSION=3.13 ./setup.sh --install python_libs`.

- [ ] **Step 3: Build ffmpeg (depends on `system_deps` and `svt_av1`).**

```
./setup.sh --install ffmpeg
```
Expected: ffmpeg builds and lands at `/opt/archav1an/bin/ffmpeg`. Verify:
```
/opt/archav1an/bin/ffmpeg -version | head -1
```
Expected: ffmpeg version string.

- [ ] **Step 4: Build VapourSynth + FFMS2 + BestSource.**

```
./setup.sh --install vapoursynth
```
Expected: builds complete; `/opt/archav1an/bin/vspipe` exists; `/opt/archav1an/venv/lib/python3.13/site-packages/_vapoursynth_native.pth` exists.

- [ ] **Step 5: Build remaining VS plugins.**

```
./setup.sh --install wwxd
./setup.sh --install vszip
./setup.sh --install subtext
./setup.sh --install denoiser
```
Expected: each completes; plugins land under `/opt/archav1an/lib/vapoursynth/`.

- [ ] **Step 6: List the installed plugins.**

```
ls /opt/archav1an/lib/vapoursynth/
```
Expected (varies by host): `libffms2.so`, `libbestsource.so`, `libwwxd.so`, `libvszip.so`, `libsubtext.so`, `libknlmeanscl.so`, `libvstrt.so` (NVIDIA) or `libvsmigx.so` (AMD), and possibly the symlinks to pacman-provided `libmvtools.so`, `libremovegrain.so`, `libctmf.so`, `models/`.

---

## Task 12: Run the proposal's verification block

**Files:** No code changes; this is the validation gate from `encoder-host:~/Public/vapoursynth-isolation.md`.

- [ ] **Step 1: No archav1an file lands in any pacman-owned tree.**

```
test -z "$(find /usr/local /usr/lib/python3.14/site-packages/vapoursynth -newer /opt/archav1an/bin/vspipe 2>/dev/null)" && echo "OK: no newer files under pacman trees"
pacman -Qo /opt/archav1an/bin/vspipe
pacman -Qkk vapoursynth
```
Expected:
- `OK: no newer files under pacman trees`
- `error: No package owns /opt/archav1an/bin/vspipe` (this is the *desired* outcome — pacman has no claim on our binary)
- `vapoursynth: 56 total files, 0 altered files`

- [ ] **Step 2: R73 binary uses the R73 core inside the activated env.**

```
source activate-venv.sh
which vspipe
vspipe --version | head -3
ldd "$(command -v vspipe)" | grep vapoursynth
python -c 'import vapoursynth as v; print(v.__file__); print(v.core.version())'
```
Expected:
- `which vspipe` → `/opt/archav1an/bin/vspipe`.
- `vspipe --version` → `Core R73` (NOT R75).
- `ldd` line for vapoursynth → resolves under `/opt/archav1an/lib`.
- Python `v.__file__` → under `/opt/archav1an/lib/python3/site-packages/vapoursynth/...`.
- `v.core.version()` → R73 string.

- [ ] **Step 3: Outside the activated env, archav1an's vspipe is absent.**

Open a fresh shell that has NOT sourced `activate-venv.sh`:
```
env -i PATH=/usr/bin:/bin sh -c 'command -v vspipe || echo "not global (correct)"'
env -i PATH=/usr/bin:/bin sh -c 'vspipe --version 2>&1 | head -1 || true'
```
Expected: either `not global (correct)` (no global vspipe) OR `/usr/bin/vspipe` from pacman → `Core R75`. EITHER is acceptable; what matters is that the bare-env vspipe is NOT `/opt/archav1an/bin/vspipe`.

- [ ] **Step 4: Run a real VPY through the pipeline as the last functional check.**

```
source activate-venv.sh
echo 'import vapoursynth as vs; from vapoursynth import core; clip = core.std.BlankClip(format=vs.YUV420P8, width=320, height=240, length=5); clip.set_output(0)' > /tmp/smoke.vpy
vspipe --info /tmp/smoke.vpy
```
Expected: vspipe prints clip info (5 frames, 320×240, YUV420P8) without errors. This confirms the R73 binary + R73 core + R73 Python module + plugin path all line up.

- [ ] **Step 5: If any verification step fails, stop and diagnose.**

Do NOT proceed to Task 13. Roll back via `git checkout setup/` and re-examine. Common failure modes and pointers:
- `vspipe --version` shows `Core R75` after Step 2 → `LD_LIBRARY_PATH` not set or wrong; check `activate-venv.sh` is sourced and `echo $LD_LIBRARY_PATH | grep /opt/archav1an`.
- `import vapoursynth` ImportError → `.pth` file missing or pointing wrong; check `cat /opt/archav1an/venv/lib/python3.13/site-packages/_vapoursynth_native.pth` and verify the path it contains exists.
- `pacman -Qkk vapoursynth` reports altered files → something wrote into pacman's tree; check `find /usr/lib/python3.14/site-packages/vapoursynth -newer /opt/archav1an/bin/vspipe`.

---

## Task 13: Tear down the old install

**Files:** No code changes; filesystem cleanup of the now-stale install.

- [ ] **Step 1: Confirm new install is fully working (Task 12 all green).**

Do not proceed if any Task 12 step failed.

- [ ] **Step 2: Remove the old `/opt/auto-boost-av1an/venv`.**

```
ls -ld /opt/auto-boost-av1an
sudo rm -rf /opt/auto-boost-av1an
```
Expected: dir removed. (sudo because `/opt` itself is root-owned even if `/opt/auto-boost-av1an` is user-owned in some setups.)

- [ ] **Step 3: Remove the old source-built install under /usr/local.**

```
sudo rm -vf /usr/local/bin/vspipe
sudo rm -vf /usr/local/lib/libvapoursynth*
sudo rm -vf /usr/local/lib/libffms2*
sudo rm -vf /usr/local/lib/libbestsource*
sudo rm -vrf /usr/local/lib/vapoursynth
sudo rm -vrf /usr/local/include/vapoursynth
sudo rm -vrf /usr/local/include/ffms2
sudo rm -vf /usr/local/lib/pkgconfig/vapoursynth.pc
sudo rm -vf /usr/local/lib/pkgconfig/ffms2.pc
sudo rm -vf /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /usr/local/bin/ffplay
sudo rm -vf /usr/local/lib/libav* /usr/local/lib/libsw{scale,resample}* /usr/local/lib/libpostproc* /usr/local/lib/libdav1d*
sudo rm -vrf /usr/local/include/libav* /usr/local/include/libsw* /usr/local/include/libpostproc /usr/local/include/dav1d
sudo rm -vf /usr/local/lib/pkgconfig/libav*.pc /usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/libpostproc.pc /usr/local/lib/pkgconfig/dav1d.pc
sudo rm -vf /usr/local/bin/SvtAv1EncApp /usr/local/bin/av1an /usr/local/bin/oxipng
sudo ldconfig
```
Expected: every `rm -vf` either reports the file removed or "No such file" (acceptable — means it wasn't present). After this, `ls /usr/local/bin/` should be empty of archav1an-installed binaries.

- [ ] **Step 4: Remove stale ~/.local/lib vapoursynth shadow if present.**

(Per CLAUDE.md note — broken `~/.local/bin/vspipe` previously caused init failures.)

```
ls /home/user/.local/lib/python*/site-packages/vapoursynth* 2>/dev/null
rm -rf /home/user/.local/lib/python*/site-packages/vapoursynth*
rm -f /home/user/.local/bin/vspipe
```
Expected: stale shadow removed (if it existed).

- [ ] **Step 5: Re-run the verification block (Task 12 steps 1-4) to confirm nothing broke after the tear-down.**

If any step now fails, the old install was masking a real issue — diagnose before declaring done.

---

## Task 14: Commit isolation work + restore feature stash

**Files:** Working tree state; no further code edits.

- [ ] **Step 1: Confirm all isolation commits are on the topic branch.**

```
git log --oneline vs-isolation ^origin/denoise-server
```
Expected: 5–8 commits from Tasks 2–9 (one per task that committed).

- [ ] **Step 2: Restore the parked stash (the original 4 modified files + untracked).**

```
git stash list
git stash pop
```
Expected: pop succeeds with possible conflict in `setup/denoiser.sh` (the only file that received both upstream changes via `76d06d8` and our local +3-line vsrvrt addition).

- [ ] **Step 3: Resolve conflict in `setup/denoiser.sh` if reported.**

Conflict shape: our stashed change adds `pip install --no-deps vsrvrt` near line 164 of the pre-rebase file; the merge marker will indicate where to re-apply. The 3 added lines are:

```sh
    # RVRT (vsrvrt): --no-deps avoids pulling PyPI vapoursynth-74 stub, which shadows the
    # custom-built VapourSynth and breaks ffms2/other plugins. TODO: revisit RVRT perf before enabling.
    "$VENV_DIR/bin/pip" install --no-deps vsrvrt || log_warn "Failed to install vsrvrt (RVRT denoising unavailable)"
```

Place them inside `install_denoiser()` right before the `local _site` line. Then:
```
git add setup/denoiser.sh
```

Note the stashed change uses `"$VENV_DIR/bin/pip"`; with the uv conversion the line should be updated to:
```sh
    VIRTUAL_ENV="$VENV_DIR" uv pip install --no-deps vsrvrt || log_warn "Failed to install vsrvrt (RVRT denoising unavailable)"
```

- [ ] **Step 4: Confirm the rest of the untracked / modified state was restored cleanly.**

```
git status -s
```
Expected: the original 4 `M` entries (Auto-Boost-Av1an.py, setup/denoiser.sh, setup/ffmpeg.sh, tools/svtav1-dispatch.py) plus the same untracked set you stashed. These remain uncommitted on the `vs-isolation` topic branch — they are the feature work that's deferred to the next plan (gitignore + cleanup + commit organization).

- [ ] **Step 5: Push the topic branch (optional, decided with the user).**

Do NOT push without explicit user OK (per AGENTS.md and project commit-safety policy). When approved:
```
git push -u origin vs-isolation
```

- [ ] **Step 6: Update memory with the completion fact (optional, decided with the user).**

If the install is verified working and committed, consider adding a reference memory pointing at this plan + the active `/opt/archav1an` layout so future sessions know the project is no longer in `/usr/local`. Defer to the user.

---

## Verification checklist

- [ ] Task 1: stash recorded, rebase clean, on branch `vs-isolation`.
- [ ] Task 2: `bash -c 'source setup/common.sh && echo $VS_PREFIX'` prints `/opt/archav1an`.
- [ ] Task 3: `bash -n setup/python_libs.sh` clean; uv command path verified.
- [ ] Task 4: `bash -n setup/vapoursynth.sh` clean; configure/install/uninstall blocks rewritten.
- [ ] Task 5: `activate-venv.sh` exports all reference `$VS_PREFIX`.
- [ ] Task 6: wwxd/subtext/vszip include + uninstall paths use VS_PREFIX.
- [ ] Task 7: `grep '/usr/local' setup/ffmpeg.sh` returns only CUDA-toolkit/WSL2 exceptions.
- [ ] Task 8: `setup/denoiser.sh` include fallback uses `$VS_PREFIX/include`.
- [ ] Task 9: `grep -rn '/usr/local' setup/ setup.sh | grep -v -E '/usr/local/cuda|/usr/local/lib/wsl-cuda'` returns empty.
- [ ] Task 10: `/opt/archav1an` exists, owned by `$USER`.
- [ ] Task 11: `setup.sh --install` succeeds for python_libs, ffmpeg, vapoursynth, wwxd, vszip, subtext, denoiser.
- [ ] Task 12: All five verification commands pass.
- [ ] Task 13: `/usr/local/bin/vspipe` no longer exists; `/opt/auto-boost-av1an` removed; Task 12 still passes after tear-down.
- [ ] Task 14: Conflict in `setup/denoiser.sh` resolved, stash empty, working tree shows the original four `M` + untracked set (the feature work, deferred to the next plan).
