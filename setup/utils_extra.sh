#!/bin/bash

# Source common functions if not already sourced
if [ -z "$COMMON_SOURCED" ]; then
    source "$(dirname "$0")/common.sh"
fi

# Legacy wrapper — individual tools are now installed via their own modules
# (oxipng.sh, fssimu2.sh) and managed through setup.sh's dependency system.
install_utils_extra() {
    install_oxipng
    install_fssimu2
}
