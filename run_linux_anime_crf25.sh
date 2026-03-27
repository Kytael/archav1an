#!/bin/bash
cd "$(dirname "$0")"
source "$(dirname "$(realpath "$0")")/activate-venv.sh"

# Anime High (CRF 25) Params — v1.66 5fish svt-av1-psy
python3 tools/pipeline.py \
    --tag-script "$(basename "$0")" \
    --quality 25 \
    --photon-noise 2 \
    --workers 4 \
    --fast-speed 8 \
    --final-speed 4 \
    --fast-params "--lp 3 --tune 0 --hbd-mds 0 --keyint 305 --noise-level-thr 16000 --lineart-psy-bias 4 --texture-psy-bias 2 --filtering-noise-detection 4" \
    --final-params "--lp 3 --tune 0 --hbd-mds 1 --keyint 305 --noise-level-thr 16000 --lineart-psy-bias 4 --texture-psy-bias 2 --filtering-noise-detection 4"
