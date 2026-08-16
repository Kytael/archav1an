#!/bin/bash
# If called with sh instead of bash (e.g. via sudo ./setup.sh), re-exec with bash
if [ -z "$BASH_VERSION" ]; then exec bash "$0" "$@"; fi

# Auto-Boost-Av1an Setup Script
# Modularized Installer with Dependency Resolution

BASE_DIR="$(dirname "$(realpath "$0")")"
SETUP_DIR="$BASE_DIR/setup"

# Restore terminal settings on exit (builds/read -n 1 can leave terminal in raw mode)
# On exit, not at the end of the happy path: a run that stops early still has
# to leave the prefix owned by the person who will use it.
trap 'stty sane 2>/dev/null; declare -F restore_ownership >/dev/null && restore_ownership' EXIT

# Source all modules
source "$SETUP_DIR/common.sh"
COMMON_SOURCED=true
source "$SETUP_DIR/system_deps.sh"
source "$SETUP_DIR/nvidia_repo.sh"
source "$SETUP_DIR/python_libs.sh"
source "$SETUP_DIR/ffmpeg.sh"
source "$SETUP_DIR/vapoursynth.sh"
source "$SETUP_DIR/av1an.sh"
source "$SETUP_DIR/svt_av1.sh"
source "$SETUP_DIR/ffvship.sh"
source "$SETUP_DIR/oxipng.sh"
source "$SETUP_DIR/fssimu2.sh"
source "$SETUP_DIR/wwxd.sh"
source "$SETUP_DIR/vszip.sh"
source "$SETUP_DIR/subtext.sh"
source "$SETUP_DIR/denoiser.sh"

# Dependency Graph Definition
# Format: declare -A DEPS
# DEPS[tool]="dep1 dep2"
declare -A DEPENDENCIES
DEPENDENCIES["python_libs"]="system_deps"
DEPENDENCIES["ffmpeg"]="system_deps svt_av1"
DEPENDENCIES["vapoursynth"]="python_libs system_deps ffmpeg"
DEPENDENCIES["av1an"]="vapoursynth ffmpeg system_deps"
DEPENDENCIES["svt_av1"]="system_deps"
DEPENDENCIES["ffvship"]="system_deps ffmpeg"
DEPENDENCIES["oxipng"]="system_deps"
DEPENDENCIES["fssimu2"]="system_deps"
DEPENDENCIES["wwxd"]="vapoursynth"
DEPENDENCIES["vszip"]="vapoursynth"
DEPENDENCIES["subtext"]="vapoursynth ffmpeg"
DEPENDENCIES["denoiser"]="vapoursynth python_libs system_deps"
# Opt-in only: deliberately absent from ALL_TOOLS, so `--install A` never adds
# a third-party apt repo and signing key on the user's behalf.
DEPENDENCIES["nvidia_repo"]=""

# Helper: Check if a tool is installed
is_installed() {
    local tool=$1
    case "$tool" in
        "system_deps")
            # Ask the package list itself rather than testing a hand-picked
            # sample. Sampling reports the component as done on any host that
            # happens to have the sampled packages, so a dependency added later
            # never gets installed. The Arch side used to sample six packages
            # for this reason and had the same blind spot the Debian comment
            # describes; both families now ask the same question.
            [ -z "$(system_deps_missing)" ]
            ;;
        "python_libs")
            [ -d "$VENV_DIR" ] && "$VENV_DIR/bin/pip" show vsjetpack &> /dev/null
            ;;
        "vapoursynth")
            [ -f "$VS_PREFIX/bin/vspipe" ]
            ;;
        "ffmpeg")
            [ -f "$VS_PREFIX/bin/ffmpeg" ]
            ;;
        "av1an")
            [ -f "$VS_PREFIX/bin/av1an" ]
            ;;
        "svt_av1")
            # Check for PSY fork at $VS_PREFIX/bin, not the standard pacman svt-av1
            [ -f "$VS_PREFIX/bin/SvtAv1EncApp" ]
            ;;
        "ffvship")
            # Installed to $VS_PREFIX/bin, which is never on setup's PATH
            [ -x "$VS_PREFIX/bin/FFVship" ] || command -v FFVship &> /dev/null
            ;;
        "oxipng")
            [ -f "$VS_PREFIX/bin/oxipng" ]
            ;;
        "fssimu2")
            # Installed to $VS_PREFIX/bin, which is never on setup's PATH
            [ -x "$VS_PREFIX/bin/fssimu2" ] || command -v fssimu2 &> /dev/null
            ;;
        "wwxd")
            local wwxd_path
            wwxd_path="$(get_vs_plugin_path)"
            [ -f "$wwxd_path/libwwxd.so" ] || \
            [ -f "$VS_PREFIX/lib/vapoursynth/libwwxd.so" ]
            ;;
        "vszip")
            local vszip_path
            vszip_path="$(get_vs_plugin_path)"
            [ -f "$vszip_path/libvszip.so" ] || \
            [ -f "$VS_PREFIX/lib/vapoursynth/libvszip.so" ]
            ;;
        "subtext")
            local subtext_path
            subtext_path="$(get_vs_plugin_path)"
            [ -f "$subtext_path/libsubtext.so" ] || \
            [ -f "$VS_PREFIX/lib/vapoursynth/libsubtext.so" ]
            ;;
        "denoiser")
            local knlm_path
            knlm_path="$(get_vs_plugin_path)"
            # The three SMDegrain plugins are checked too. They are in
            # ARTIFACTS but were not in this test, so a host that had
            # KNLMeansCL, vsscunet and libvstrt.so reported the component as
            # done and skipped it -- which is exactly what happened after the
            # mvtools/removegrain/ctmf builds were fixed: setup declared the
            # denoiser installed and never built them. Same sampling flaw as
            # the old system_deps check.
            [ -f "$knlm_path/libknlmeanscl.so" ] && \
            [ -f "$knlm_path/libmvtools.so" ] && \
            [ -f "$knlm_path/libremovegrain.so" ] && \
            [ -f "$knlm_path/libctmf.so" ] && \
            "$VENV_DIR/bin/pip" show vsscunet &> /dev/null && \
            { [ -f "$knlm_path/libvstrt.so" ] || [ -f "$knlm_path/libvsmigx.so" ]; }
            ;;
        "nvidia_repo")
            nvidia_cuda_repo_present
            ;;
        *) return 1 ;;
    esac
}

get_status_icon() {
    if is_installed "$1"; then
        # ldd-only health check (local + fast; the vspipe load probe and the
        # network update check run in --update, not on every menu render).
        if check_linkage "$1" >/dev/null; then
            echo -e "${GREEN}[INSTALLED]${NC}"
        else
            echo -e "${RED}[BROKEN]${NC}"
        fi
    else
        echo -e "${RED}[MISSING]${NC}"
    fi
}

# update_check <component> [network:0|1]
# One line on stdout: "component|OK/UPDATE/BROKEN/UNKNOWN/MISSING|detail".
# BROKEN (health) takes precedence over staleness; UPDATE over UNKNOWN.
update_check() {
    local component=$1 network=${2:-1}
    if ! is_installed "$component"; then
        # "Not installed" and "installed but no longer complete" are different
        # answers, and only the first one is none of a sweep's business.
        # system_deps is a package list: a host holding 196 of 202 specs, or
        # holding a pinned package at the wrong version, is out of date -- which
        # is exactly what --update exists to fix. Reported as MISSING it was
        # printed and skipped, so a TensorRT pin that had drifted sat there
        # sweep after sweep while the denoiser built against the wrong headers.
        if [ "$component" = "system_deps" ]; then
            local _short _n _total
            _short=$(system_deps_missing)
            _n=$(printf '%s\n' "$_short" | grep -c .)
            _total=$( { [ "$DISTRO_FAMILY" = "arch" ] && arch_system_deps || debian_system_deps; } | grep -c .)
            if [ "$_n" -gt 0 ] && [ "$_n" -lt "$_total" ]; then
                echo "$component|UPDATE|$_n of $_total package specs unsatisfied: $(printf '%s' "$_short" | tr '\n' ' ' | cut -c1-70)"
                return 0
            fi
        fi
        echo "$component|MISSING|not installed"
        return 0
    fi
    local out reasons
    if ! out=$(component_health "$component"); then
        reasons=$(echo "$out" | awk -F'|' '$2=="BROKEN"{printf "%s (%s); ", $1, $3}')
        echo "$component|BROKEN|${reasons%; }"
        return 0
    fi
    if [ "$network" != "1" ]; then
        echo "$component|OK|healthy (update check skipped)"
        return 0
    fi
    out=$(src_update_status "$component")
    if echo "$out" | grep -q "|UPDATE|"; then
        reasons=$(echo "$out" | awk -F'|' '$2=="UPDATE"{printf "%s: %s; ", $1, $3}')
        echo "$component|UPDATE|${reasons%; }"
    elif echo "$out" | grep -q "|UNKNOWN|"; then
        echo "$component|UNKNOWN|no install record (pre-manifest install)"
    else
        echo "$component|OK|up to date"
    fi
    return 0
}

# Global variables for dependency resolution
declare -A RESOLVE_VISITED
declare -a RESOLVE_PLAN

# Components already built by THIS invocation. RESOLVE_VISITED dedupes within
# one install_tool call and is reset on every call, so it cannot see across
# targets -- and install_all_tools calls install_tool once per component. Under
# FORCE_REINSTALL=1 nothing else stops a shared dependency rebuilding once per
# dependent: for --install A that is ffmpeg 8 times, svt_av1 9, system_deps 13.
# A component built moments ago in the same run is never stale, so this guard
# holds in both forced and unforced mode.
declare -A BUILT_THIS_RUN

resolve_deps_recursive() {
    local tool=$1
    if [[ -n "${RESOLVE_VISITED[$tool]}" ]]; then return; fi
    RESOLVE_VISITED[$tool]=1
    local deps="${DEPENDENCIES[$tool]}"
    for dep in $deps; do
        resolve_deps_recursive "$dep"
    done
    RESOLVE_PLAN+=("$tool")
}

run_installer() {
    case "$1" in
        "system_deps") install_system_deps ;;
        "python_libs") install_python_libs ;;
        "ffmpeg") install_ffmpeg ;;
        "vapoursynth") install_vapoursynth ;;
        "av1an") install_av1an ;;
        "svt_av1") install_svt_av1 ;;
        "ffvship") install_ffvship ;;
        "oxipng") install_oxipng ;;
        "fssimu2") install_fssimu2 ;;
        "wwxd") install_wwxd ;;
        "vszip") install_vszip ;;
        "subtext") install_subtext ;;
        "denoiser") install_denoiser ;;
        "nvidia_repo") install_nvidia_repo ;;
        *) log_error "Unknown module: $1"; exit 1 ;;
    esac
}

# Wrapper to install a tool with deps
install_tool() {
    local target_tool=$1
    
    RESOLVE_PLAN=()
    RESOLVE_VISITED=()
    resolve_deps_recursive "$target_tool"
    
    echo ""
    log_info "Target: $target_tool"
    echo "Dependency Tree Calculation:"
    local install_queue=()

    # FORCE_REINSTALL=1 in the environment overrides the is_installed
    # short-circuit so every node in the resolved plan rebuilds without
    # the interactive Y/n prompt. Inside install_tool we re-export it
    # so the called installer functions (which honor it independently
    # to clear caches, re-clone, etc.) see the same value.
    local _env_force="${FORCE_REINSTALL:-0}"

    for item in "${RESOLVE_PLAN[@]}"; do
        if [ -n "${BUILT_THIS_RUN[$item]}" ]; then
            echo -e "  - $item: ${GREEN}Already built in this run${NC}"
        elif [ "$_env_force" = "1" ]; then
            echo -e "  - $item: ${RED}To be reinstalled (FORCE_REINSTALL=1)${NC}"
            install_queue+=("$item")
        elif is_installed "$item"; then
            # Not a blind skip: check staleness (network) + linkage/load health.
            local _chk _verdict _detail
            _chk=$(update_check "$item" 1)
            IFS='|' read -r _ _verdict _detail <<< "$_chk"
            case "$_verdict" in
                UPDATE)
                    echo -e "  - $item: ${YELLOW}Update available${NC} ($_detail)"
                    install_queue+=("$item")
                    ;;
                BROKEN)
                    echo -e "  - $item: ${RED}Broken${NC} ($_detail)"
                    install_queue+=("$item")
                    ;;
                UNKNOWN)
                    echo -e "  - $item: ${YELLOW}Installed, no install record${NC}"
                    if ask_yes_no "    Rebuild $item to establish an update baseline?" "N"; then
                        install_queue+=("$item")
                    fi
                    ;;
                *)
                    echo -e "  - $item: ${GREEN}Already Installed (up to date)${NC}"
                    ;;
            esac
        else
            echo -e "  - $item: ${RED}To be installed${NC}"
            install_queue+=("$item")
        fi
    done

    if [ ${#install_queue[@]} -eq 0 ]; then
        log_success "All dependencies for $target_tool are already installed."
        if ! ask_yes_no "Re-install/Update $target_tool anyway?" "N"; then
            return 0
        fi
        install_queue+=("$target_tool")
        FORCE_REINSTALL=1
    fi

    echo ""
    log_info "The following will be installed: ${install_queue[*]}"
    if [ "$_env_force" != "1" ] && ! ask_yes_no "Proceed?" "Y"; then
        log_warn "Installation cancelled."
        return 0
    fi
    
    for item in "${install_queue[@]}"; do
        log_info "Installing: $item"
        local status=0 _item_force="$_env_force"
        # Items queued while still installed (update/broken/baseline rebuilds)
        # need the module-level "already installed" guard bypassed.
        if [ "$_item_force" != "1" ] && is_installed "$item"; then
            _item_force=1
        fi
        FORCE_REINSTALL="$_item_force"
        run_installer "$item"
        status=$?
        FORCE_REINSTALL="$_env_force"
        if [ $status -ne 0 ]; then
            log_error "Failed to install $item (Exit code: $status). Stopping."
            # Restore to the entry value on failure too: don't leak the
            # 'reinstall anyway' =1, don't clobber an env FORCE_REINSTALL=1.
            FORCE_REINSTALL="$_env_force"
            return 1
        fi
        # Only a successful build counts: a failure must not mask a later retry.
        BUILT_THIS_RUN[$item]=1
    done

    # Restore to the entry value (not 0): env FORCE_REINSTALL=1 must survive
    # for the remaining components of an install_all_tools / multi-pick batch.
    FORCE_REINSTALL="$_env_force"
    log_success "Operation completed."
}

uninstall_tool() {
    local target_tool=$1
    echo ""
    log_warn "You are about to UNINSTALL: $target_tool"
    if ! ask_yes_no "Are you sure?" "N"; then
         return 0
    fi
    
    case "$target_tool" in
        "system_deps") uninstall_system_deps ;;
        "nvidia_repo") uninstall_nvidia_repo ;;
        "python_libs") uninstall_python_libs ;;
        "ffmpeg") uninstall_ffmpeg ;;
        "vapoursynth") uninstall_vapoursynth ;;
        "av1an") uninstall_av1an ;;
        "svt_av1") uninstall_svt_av1 ;;
        "ffvship") uninstall_ffvship ;;
        "oxipng") uninstall_oxipng ;;
        "fssimu2") uninstall_fssimu2 ;;
        "wwxd") uninstall_wwxd ;;
        "vszip") uninstall_vszip ;;
        "subtext") uninstall_subtext ;;
        "denoiser") uninstall_denoiser ;;
        *) log_error "Unknown module: $target_tool"; exit 1 ;;
    esac
}

ALL_TOOLS=("system_deps" "python_libs" "svt_av1" "ffmpeg" "vapoursynth" "av1an" "ffvship" "oxipng" "fssimu2" "wwxd" "vszip" "subtext" "denoiser")

install_all_tools() {
    local all_tools=("${ALL_TOOLS[@]}")
    log_info "Starting Full Installation..."
    setup_build_tmpfs "$(pwd)/build_tmp"
    for t in "${all_tools[@]}"; do
        install_tool "$t" || { log_error "Failed to install $t. Aborting batch."; return 1; }
    done
    # Unmount tmpfs if it was used
    if mountpoint -q "$(pwd)/build_tmp" 2>/dev/null; then
        umount "$(pwd)/build_tmp" 2>/dev/null
        log_info "build_tmp tmpfs unmounted."
    fi
}

# update_tool <component...>
# pacman-style sweep: check every named component (health + upstream), print a
# summary, confirm once, rebuild only what needs it, then cascade-recheck
# linkage (a rebuilt dependency can newly break dependents' ABI).
update_tool() {
    local comps=("$@")
    local c line verdict
    local lines=() rebuild=() unknown_list=()

    log_info "Checking ${#comps[@]} component(s) — linkage, load probes, upstream refs..."
    for c in "${comps[@]}"; do
        line=$(update_check "$c" 1)
        lines+=("$line")
        verdict=$(echo "$line" | cut -d'|' -f2)
        case "$verdict" in
            UPDATE|BROKEN) rebuild+=("$c") ;;
            UNKNOWN) rebuild+=("$c"); unknown_list+=("$c") ;;
        esac
    done

    echo ""
    printf "%-13s %-9s %s\n" "COMPONENT" "STATUS" "DETAIL"
    printf "%-13s %-9s %s\n" "---------" "------" "------"
    local v d
    for line in "${lines[@]}"; do
        IFS='|' read -r c v d <<< "$line"
        case "$v" in
            OK)      printf "%-13s ${GREEN}%-9s${NC} %s\n" "$c" "$v" "$d" ;;
            MISSING) printf "%-13s ${YELLOW}%-9s${NC} %s\n" "$c" "$v" "$d" ;;
            *)       printf "%-13s ${RED}%-9s${NC} %s\n" "$c" "$v" "$d" ;;
        esac
    done
    echo ""

    # python_libs: pip is its own resolver — offer the -U pass whenever the
    # component is installed and part of the sweep.
    local do_pip_update=0
    for c in "${comps[@]}"; do
        if [ "$c" = "python_libs" ] && is_installed python_libs; then
            do_pip_update=1
        fi
    done

    if [ ${#rebuild[@]} -eq 0 ]; then
        log_success "Everything is up to date and healthy."
        if [ "$do_pip_update" = "1" ] && ask_yes_no "Upgrade python_libs pip packages anyway?" "N"; then
            update_python_libs || return 1
        fi
        return 0
    fi

    if [ ${#unknown_list[@]} -gt 0 ]; then
        log_warn "No install record for: ${unknown_list[*]} (pre-manifest installs)."
        log_warn "Rebuilding them establishes the baseline for future update checks."
    fi
    if ! ask_yes_no "Rebuild ${#rebuild[@]} component(s): ${rebuild[*]} ?" "Y"; then
        log_warn "Update cancelled."
        return 0
    fi

    local _env_force="${FORCE_REINSTALL:-0}"
    for c in "${rebuild[@]}"; do
        log_info "Updating: $c"
        FORCE_REINSTALL=1
        if ! run_installer "$c"; then
            FORCE_REINSTALL="$_env_force"
            log_error "Failed to update $c. Stopping."
            return 1
        fi
        FORCE_REINSTALL="$_env_force"
    done

    # Cascade: local health-only recheck (no network) over the whole sweep set.
    local pass
    local broken_now=()
    for pass in 1 2 3; do
        broken_now=()
        for c in "${comps[@]}"; do
            is_installed "$c" || continue
            component_health "$c" >/dev/null || broken_now+=("$c")
        done
        [ ${#broken_now[@]} -eq 0 ] && break
        log_warn "Cascade pass $pass: broken after rebuild: ${broken_now[*]} — rebuilding those too."
        for c in "${broken_now[@]}"; do
            FORCE_REINSTALL=1
            if ! run_installer "$c"; then
                FORCE_REINSTALL="$_env_force"
                log_error "Failed cascade rebuild of $c. Stopping."
                return 1
            fi
            FORCE_REINSTALL="$_env_force"
        done
    done
    if [ ${#broken_now[@]} -gt 0 ]; then
        log_error "Still broken after 3 cascade passes: ${broken_now[*]}"
        return 1
    fi

    if [ "$do_pip_update" = "1" ]; then
        update_python_libs || return 1
    fi
    log_success "Update complete."
}

uninstall_all_tools() {
    local all_tools=("system_deps" "python_libs" "svt_av1" "ffmpeg" "vapoursynth" "av1an" "ffvship" "oxipng" "fssimu2" "wwxd" "vszip" "subtext")
    log_warn "Starting Full Uninstallation..."
    # Reverse order for uninstall
    for (( i=${#all_tools[@]}-1; i>=0; i-- )); do
        uninstall_tool "${all_tools[$i]}"
    done
}

show_menu() {
    local mode=$1
    if [ -z "$mode" ]; then mode="INSTALL"; fi
    
    clear
    echo "=========================================================="
    echo "   Auto-Boost-Av1an Setup Menu"
    echo "=========================================================="
    echo "Mode: $mode"
    echo "----------------------------------------------------------"
    printf "%-3s %-30s %s\n" "ID" "Module" "Status"
    printf "%-3s %-30s %s\n" "--" "------" "------"
    printf "%-3s %-30s %s\n" "1" "Av1an" "$(get_status_icon av1an)"
    printf "%-3s %-30s %s\n" "2" "VapourSynth (+FFMS2)" "$(get_status_icon vapoursynth)"
    printf "%-3s %-30s %s\n" "3" "SVT-AV1-PSY" "$(get_status_icon svt_av1)"
    printf "%-3s %-30s %s\n" "4" "FFmpeg" "$(get_status_icon ffmpeg)"
    printf "%-3s %-30s %s\n" "5" "FFVship" "$(get_status_icon ffvship)"
    printf "%-3s %-30s %s\n" "6" "oxipng" "$(get_status_icon oxipng)"
    printf "%-3s %-30s %s\n" "7" "fssimu2" "$(get_status_icon fssimu2)"
    printf "%-3s %-30s %s\n" "8" "WWXD" "$(get_status_icon wwxd)"
    printf "%-3s %-30s %s\n" "9" "VSZIP" "$(get_status_icon vszip)"
    printf "%-3s %-30s %s\n" "10" "SubText" "$(get_status_icon subtext)"
    printf "%-3s %-30s %s\n" "13" "Denoiser (SCUnet/KNLMeansCL)" "$(get_status_icon denoiser)"
    echo "----------------------------------------------------------"
    echo "11. System Deps Only"
    echo "12. Python Libs Only"
    if [ "$mode" == "INSTALL" ]; then
        echo "A. Install Everything (Full Setup)"
    else
        echo "A. Uninstall Everything"
    fi
    
    if [ "$mode" == "INSTALL" ]; then
        echo "T. Toggle to UNINSTALL Mode"
    else
        echo "T. Toggle to INSTALL Mode"
    fi
    echo "Q. Quit"
    echo "=========================================================="
    stty sane 2>/dev/null
    read -p "Select option(s) (e.g., 1 3 5): " choice_input
    
    # Toggle Mode
    if [[ "$choice_input" =~ ^[Tt]$ ]]; then
        if [ "$mode" == "INSTALL" ]; then show_menu "UNINSTALL"; else show_menu "INSTALL"; fi
        return
    fi
    
    # Quit
    if [[ "$choice_input" =~ ^[Qq]$ ]]; then exit 0; fi
    
    # Install/Uninstall All
    if [[ "$choice_input" =~ ^[Aa]$ ]]; then
        if [ "$mode" == "INSTALL" ]; then
            install_all_tools
        else
            if ask_yes_no "Are you sure you want to uninstall EVERYTHING?" "N"; then
                uninstall_all_tools
            fi
        fi
        
        stty sane 2>/dev/null
        echo "Press Enter to return to menu..."
        read
        show_menu "$mode"
        return
    fi
    
    # Split input by space or comma
    # Replace commas with spaces
    choice_input="${choice_input//,/ }"
    
    for choice in $choice_input; do
        local tool=""
        case "$choice" in
            1) tool="av1an" ;;
            2) tool="vapoursynth" ;;
            3) tool="svt_av1" ;;
            4) tool="ffmpeg" ;;
            5) tool="ffvship" ;;
            6) tool="oxipng" ;;
            7) tool="fssimu2" ;;
            8) tool="wwxd" ;;
            9) tool="vszip" ;;
            10) tool="subtext" ;;
            11) tool="system_deps" ;;
            12) tool="python_libs" ;;
            13) tool="denoiser" ;;
            *) log_warn "Invalid option: $choice (skipping)"; continue ;;
        esac
        
        if [ -n "$tool" ]; then
            if [ "$mode" == "INSTALL" ]; then
                install_tool "$tool" || { log_error "Failed to install $tool. Stopping batch."; break; }
            else
                uninstall_tool "$tool"
            fi
        fi
    done
    
    echo "Press Enter to return to menu..."
    read
    show_menu "$mode"
}

# Main Execution Flow
# Parse -y/--yes flag from any position
for arg in "$@"; do
    if [ "$arg" == "-y" ] || [ "$arg" == "--yes" ]; then
        AUTO_YES=true
    fi
done

check_distro
detect_gpu

# Only the commands that build something need the prefix or root. check_root
# used to run unconditionally here, so `./setup.sh --help` asked for a password
# on any host whose prefix was not already user-writable.
#
# Order matters within the branch: preflight_sudo asks what still needs
# installing, which needs the distro family and (on Arch) the GPU vendor, and it
# must authenticate before check_root spends the first sudo. One prompt, at
# second zero, or an immediate abort.
case "${1:-}" in
    -h|--help|--uninstall) : ;;
    *)
        preflight_sudo || exit 1
        check_root
        ;;
esac

setup_wsl2_cuda

# Filter out -y/--yes from positional args
ARGS=()
for arg in "$@"; do
    if [ "$arg" != "-y" ] && [ "$arg" != "--yes" ]; then
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]}"

usage() {
    cat <<EOF
Usage: ./setup.sh [-y|--yes] [command]

Commands:
  (no command)                interactive menu
  --install <component...|A>  install components (A = everything);
                              already-installed components are checked for
                              upstream updates and broken linkage first
  --update [component...|A]   pacman-style sweep (default: all): rebuild only
                              components with upstream changes, broken linkage,
                              or plugins that fail to load
  --uninstall <component|A>   uninstall a component (A = everything)
  -h, --help                  this text

Components: ${ALL_TOOLS[*]}
Env: FORCE_REINSTALL=1 forces rebuilds; PYTHON_VERSION pins the venv Python.
EOF
}

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
    exit 0
elif [ "$1" == "--update" ]; then
    shift
    if [ $# -eq 0 ] || [[ "$1" =~ ^[Aa]$ ]]; then
        update_tool "${ALL_TOOLS[@]}"
    else
        for _component in "$@"; do
            case " ${ALL_TOOLS[*]} " in
                *" $_component "*) ;;
                *) log_error "Unknown component: $_component"; usage; exit 2 ;;
            esac
        done
        update_tool "$@"
    fi
elif [ "$1" == "--install" ] && [ -n "$2" ]; then
    if [[ "$2" =~ ^[Aa]$ ]]; then
        install_all_tools
    else
        # Accept multiple components: ./setup.sh --install ffmpeg av1an denoiser
        shift
        for _component in "$@"; do
            install_tool "$_component" || exit 1
        done
    fi
elif [ "$1" == "--uninstall" ] && [ -n "$2" ]; then
    if [[ "$2" =~ ^[Aa]$ ]]; then
        if ask_yes_no "Are you sure you want to uninstall EVERYTHING?" "N"; then
            uninstall_all_tools
        fi
    else
        uninstall_tool "$2"
    fi
elif [ -n "$1" ]; then
    log_error "Unknown argument: $1"
    usage
    exit 2
else
    show_menu "INSTALL"
fi
