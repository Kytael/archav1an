#!/bin/bash
cd "$(dirname "$0")"
source "$(dirname "$(realpath "$0")")/activate-venv.sh"

# Live Action Higher Quality (CRF 18) Params — v1.66 5fish svt-av1-psy
python3 tools/pipeline.py \
    --tag-script "$(basename "$0")" \
    --quality 18 \
    --autocrop \
    --photon-noise 4 \
    --workers 4 \
    --fast-speed 8 \
    --final-speed 4 \
    --fast-params "--lp 3 --tune 3 --hbd-mds 0 --keyint 305 --ac-bias 1.2 --filtering-noise-detection 1" \
    --final-params "--lp 3 --tune 3 --hbd-mds 1 --keyint 305 --ac-bias 1.2 --filtering-noise-detection 1"
