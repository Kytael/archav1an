#!/bin/bash
# Activate the Auto-Boost-Av1an Python virtual environment
# Source this from any script that needs to call python3

VENV_DIR="/opt/auto-boost-av1an/venv"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "[WARN] Python venv not found at $VENV_DIR. Run setup.sh first."
    echo "       Falling back to system python3."
fi
