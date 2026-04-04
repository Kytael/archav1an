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

echo "Starting SvtAv1EncApp Batch (Dance HQ CRF 27) — single-pass, 16 threads..."
# Extra args passed to this script are forwarded to svtav1-dispatch.py (e.g. --denoise-scunet)
EXTRA_ARGS=("$@")

rm -f "tools/tag-manifest.txt"
mkdir -p Input Output
shopt -s nullglob

for f in Input/*.[Mm][Kk][Vv] Input/*.[Mm][Pp]4 Input/*.[Mm][Oo][Vv] Input/*.[Mm]2[Tt][Ss]; do
    [ -f "$f" ] || continue
    filename=$(basename -- "$f")
    stem="${filename%.*}"

    OUTPUT_FILE="Output/${stem}-av1.mkv"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping \"$f\" — output already exists."
        continue
    fi

    echo "==============================================================================="
    echo "Processing \"$f\"..."
    echo "-------------------------------------------------------------------------------"

    # Dance / Performance HQ (CRF 27) — single-pass SvtAv1EncApp, 16 threads
    python3 tools/svtav1-dispatch.py \
        -i "$f" \
        -o "$OUTPUT_FILE" \
        --quality 27 \
        --photon-noise 6 \
        --lp 16 \
        --speed 4 \
        --encoder-params "--tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.8 --sharp-tx 1 --sharpness 1 --tf-strength 2 --variance-boost-strength 1 --variance-octile 7 --enable-dlf 2" \
        "${EXTRA_ARGS[@]}"

done

# --- TAGGING & CLEANUP ---
echo "Tagging output files..."
python3 tools/tag.py

echo "Cleaning up temporary files and folders..."
python3 tools/cleanup.py

echo "All tasks finished."
