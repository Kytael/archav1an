#!/bin/bash
# Wrapper for new modular setup
#
# Do not add sudo here. setup.sh runs as your user on purpose: it takes one
# sudo to create /opt/archav1an owned by you, and system_deps escalates itself
# for apt. Running the whole thing as root leaves the prefix and venv
# root-owned, and makepkg/paru refuse to run as root at all.
set -e
echo "Launching new modular setup (Full Install)..."
chmod +x setup.sh
./setup.sh --install A
