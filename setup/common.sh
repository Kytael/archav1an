#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then 
        log_error "Please run as root (sudo)."
        exit 1
    fi
}

check_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_LIKE=$ID_LIKE
    else
        log_error "Cannot detect Linux distribution."
        exit 1
    fi

    log_info "Detected Distribution: $DISTRO"

    if command -v pacman &> /dev/null; then
        DISTRO_FAMILY="arch"
        log_success "Detected Arch-based system ($DISTRO)."
    elif [ -f /etc/debian_version ]; then
        DISTRO_FAMILY="debian"
        log_success "Detected Debian/Ubuntu-based system."
    else
        log_error "Error: This script supports Arch-based (CachyOS, Manjaro, etc.) and Debian/Ubuntu systems."
        log_info "Detected: $DISTRO"
        exit 1
    fi

    export DISTRO_FAMILY
}

# Helper: get the Python site-packages directory dynamically
get_python_site_packages() {
    python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/usr/lib/python3/dist-packages"
}

# Helper: get the VapourSynth plugin path
get_vs_plugin_path() {
    if command -v pkg-config &> /dev/null && pkg-config --exists vapoursynth 2>/dev/null; then
        echo "$(pkg-config --variable=libdir vapoursynth)/vapoursynth"
    elif [ "$DISTRO_FAMILY" = "arch" ]; then
        echo "/usr/lib/vapoursynth"
    else
        echo "/usr/lib/x86_64-linux-gnu/vapoursynth"
    fi
}

ask_yes_no() {
    local prompt="$1"
    local default="$2" # Y or N
    
    local yn_prompt="[y/n]"
    if [ "$default" == "Y" ]; then yn_prompt="[Y/n]"; fi
    if [ "$default" == "N" ]; then yn_prompt="[y/N]"; fi
    
    read -p "$prompt $yn_prompt " -n 1 -r
    echo ""
    
    if [ -z "$REPLY" ]; then
        if [ "$default" == "Y" ]; then return 0; fi
        if [ "$default" == "N" ]; then return 1; fi
    fi
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}
