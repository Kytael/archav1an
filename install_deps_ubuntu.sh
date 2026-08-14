#!/bin/bash
# Wrapper for new modular setup
#
# Two steps, not one. Only system_deps needs root: it is the target that
# installs distro packages through apt. Everything after it builds into
# /opt/archav1an, which setup.sh creates owned by you with a single sudo
# prompt of its own.
#
# Running the whole install under sudo looks equivalent and is not.
# check_root() returns early when EUID is 0, so the prefix never gets
# chowned and every artifact lands root-owned -- which then blocks the
# next ordinary run. AUR helpers also refuse to build as root outright.
set -e
echo "Launching new modular setup (Full Install)..."
chmod +x setup.sh
sudo ./setup.sh --install system_deps
./setup.sh --install A
