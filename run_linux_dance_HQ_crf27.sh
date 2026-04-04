#!/bin/bash

# run_linux_dance_HQ_crf27.sh
# Single-pass SvtAv1EncApp encode — Dance / Performance CRF 27, full temporal context.
# Bypasses av1an chunking so encoder sees the entire clip without forced chunk resets.
# Place source files in Input/, encoded output goes to Output/.

cd "$(dirname "$0")"

# Activate Python venv
source "$(dirname "$(realpath "$0")")/activate-venv.sh"
touch "tools/sh-used-$(basename "$0").txt"

WORKER_COUNT=4

# --- WORKER COUNT CHECK ---
CONFIG_FILE="tools/workercount-config.txt"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "First Run Detected: Calculating optimal encode worker count..."
    python3 "tools/workercount.py"
fi

if [ -f "$CONFIG_FILE" ]; then
    WORKER_COUNT=$(grep "^workers=" "$CONFIG_FILE" | cut -d= -f2 | tr -d '\r')
fi

LP=$(nproc)
[ "$LP" -gt 64 ] && LP=64
echo "Starting SvtAv1EncApp Batch (Dance HQ CRF 27) — single-pass, ${LP} threads..."
# Extra args passed to this script are forwarded to svtav1-dispatch.py (e.g. --denoise-scunet)
EXTRA_ARGS=("$@")

trap 'trap "" INT TERM; echo "Interrupted."; kill 0; exit 130' INT TERM

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

    # Dance / Performance HQ (CRF 27) — single-pass SvtAv1EncApp
    python3 tools/svtav1-dispatch.py \
        -i "$f" \
        -o "$OUTPUT_FILE" \
        --quality 27 \
        --photon-noise 6 \
        --lp "$LP" \
        --speed 4 \
        --encoder-params "--tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.8 --sharp-tx 1 --sharpness 1 --tf-strength 2 --variance-boost-strength 1 --variance-octile 7 --enable-dlf 2" \
        "${EXTRA_ARGS[@]}"

done 3< <(find Input -type f \( -iname "*.mkv" -o -iname "*.mp4" -o -iname "*.mov" -o -iname "*.m2ts" \) -print0 | sort -z)

# --- TAGGING & CLEANUP ---
echo "Tagging output files..."
python3 tools/tag.py

echo "Cleaning up temporary files and folders..."
python3 tools/cleanup.py

echo "All tasks finished."
