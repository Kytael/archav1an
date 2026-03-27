#!/bin/bash
cd "$(dirname "$0")"
source "$(dirname "$(realpath "$0")")/activate-venv.sh"

# Live Action Standard (CRF 32) Params — v1.66 5fish svt-av1-psy
python3 tools/pipeline.py \
    --tag-script "$(basename "$0")" \
    --quality 32 \
    --autocrop \
    --photon-noise 4 \
    --workers 4 \
    --fast-speed 8 \
    --final-speed 4 \
    --fast-params "--lp 3 --tune 3 --hbd-mds 0 --keyint 305 --ac-bias 0.8 --filtering-noise-detection 4" \
    --final-params "--lp 3 --tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.8 --filtering-noise-detection 4" \
    "$@"
