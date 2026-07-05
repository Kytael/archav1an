#!/bin/bash
# Auto-Boost-Av1an: Comparison Script
# Runs VapourSynth comparison via tools/comp.py


# Activate Python venv
source "$(dirname "$(realpath "$0")")/../activate-venv.sh"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"
TOOL_SCRIPT="$ROOT_DIR/tools/comp.py"

if [ ! -f "$TOOL_SCRIPT" ]; then
    echo "Error: tools/comp.py not found!"
    exit 1
fi

echo "Launching comparison script..."
python3 "$TOOL_SCRIPT"

echo "Comparisons complete."
echo "Cleaning up index cache..."

# Cleanup: only source-index artifacts. Keep Comparisons/ (slow.pics .url
# shortcuts) and generated.compframes (frame-analysis cache for reruns);
# comp.py manages its own screens dir.
rm -f *.lwi *.ffindex 2>/dev/null
