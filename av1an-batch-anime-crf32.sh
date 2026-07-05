#!/bin/bash

# av1an-batch-anime-crf32.sh
# Direct Av1an encode — Anime CRF 32, single pass (no Auto-Boost).
# Place source files in Input/, encoded output goes to Output/.

cd "$(dirname "$0")"


# Activate Python venv
source "$(dirname "$(realpath "$0")")/activate-venv.sh"
touch "tools/sh-used-$(basename "$0").txt"

WORKER_COUNT=4

# --- STEP 1A: WORKER COUNT CHECK ---
CONFIG_FILE="tools/workercount-config.txt"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "First Run Detected: Calculating optimal encode worker count..."
    python3 "tools/workercount.py"
fi

if [ -f "$CONFIG_FILE" ]; then
    WORKER_COUNT=$(grep "^workers=" "$CONFIG_FILE" | cut -d= -f2 | tr -d '\r')
fi

echo "Starting Av1an Batch (Anime CRF 32) with $WORKER_COUNT workers..."

rm -f "tools/tag-manifest.txt"
mkdir -p Input Output

while IFS= read -r -d '' f <&3; do
    filename=$(basename -- "$f")
    stem="${filename%.*}"
    rel_dir=$(dirname -- "$f")
    rel_dir="${rel_dir#Input}"
    rel_dir="${rel_dir#/}"

    if [ -n "$rel_dir" ]; then
        mkdir -p "Output/${rel_dir}"
        OUTPUT_FILE="Output/${rel_dir}/${stem}-av1.mkv"
    else
        OUTPUT_FILE="Output/${stem}-av1.mkv"
    fi

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping \"$f\" — output already exists."
        continue
    fi

    echo "==============================================================================="
    echo "Processing \"$f\"..."
    echo "-------------------------------------------------------------------------------"

    # Anime Standard (CRF 32) — v1.66 5fish svt-av1-psy, single pass
    python3 tools/av1an-dispatch.py \
        -i "$f" \
        -o "$OUTPUT_FILE" \
        --quality 32 \
        --photon-noise 2 \
        --workers "$WORKER_COUNT" \
        --final-speed 4 \
        --final-params "--lp 3 --tune 0 --hbd-mds 1 --keyint 305 --filtering-noise-detection 4"

done 3< <(find Input -type f \( -iname "*.mkv" -o -iname "*.mp4" -o -iname "*.mov" -o -iname "*.m2ts" \) -print0 | sort -z)

# --- TAGGING & CLEANUP ---
echo "Tagging output files..."
python3 tools/tag.py

echo "Cleaning up temporary files and folders..."
python3 tools/cleanup.py

echo "All tasks finished."
