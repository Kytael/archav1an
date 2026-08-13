#!/bin/bash

# run_linux_dance_HQ_crf27.sh
# Single-pass SvtAv1EncApp encode — Dance / Performance CRF 27, full temporal context.
# Bypasses av1an chunking so encoder sees the entire clip without forced chunk resets.
# Place source files in Input/, encoded output goes to Output/.

cd "$(dirname "$0")"

# Activate Python venv
source "$(dirname "$(realpath "$0")")/activate-venv.sh"
touch "tools/sh-used-$(basename "$0").txt"

# --lp is a parallelism LEVEL in [0, 6], not a thread count, so the old
# LP=$(nproc) here only ever clamped to 6 with a warning. 0 lets the encoder
# choose from the core count: level 5 on a 16-thread host, 6 at 24 or more.
# See docs/lp-and-encoder-parallelism.md.
echo "Starting SvtAv1EncApp Batch (Dance HQ CRF 27) — single-pass..."
# Extra args passed to this script are forwarded to svtav1-dispatch.py (e.g. --denoise-scunet)
EXTRA_ARGS=("$@")

trap 'trap "" INT TERM; echo "Interrupted."; kill 0; exit 130' INT TERM

rm -f "tools/tag-manifest.txt"
mkdir -p Input Output

FAILED_FILES=()

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
    if ! python3 tools/svtav1-dispatch.py \
        -i "$f" \
        -o "$OUTPUT_FILE" \
        --quality 27 \
        --photon-noise 6 \
        --lp 0 \
        --speed 4 \
        --encoder-params "--tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.8 --sharp-tx 1 --sharpness 1 --tf-strength 2 --variance-boost-strength 1 --variance-octile 7 --enable-dlf 2" \
        "${EXTRA_ARGS[@]}"; then
        echo "FAILED: \"$f\""
        FAILED_FILES+=("$f")
    fi

done 3< <(find Input -type f \( -iname "*.mkv" -o -iname "*.mp4" -o -iname "*.mov" -o -iname "*.m2ts" \) -print0 | sort -z)

# --- CLEANUP ---
echo "Cleaning up temporary files and folders..."
python3 tools/cleanup.py

if [ ${#FAILED_FILES[@]} -gt 0 ]; then
    echo "WARNING: ${#FAILED_FILES[@]} file(s) FAILED:"
    printf '  %s\n' "${FAILED_FILES[@]}"
    exit 1
fi
echo "All tasks finished."
