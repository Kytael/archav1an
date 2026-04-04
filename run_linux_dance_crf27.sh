#!/bin/bash
cd "$(dirname "$0")"
source "$(dirname "$(realpath "$0")")/activate-venv.sh"

# Dance / Performance (CRF 27) Params — v1.66 5fish svt-av1-psy
python3 tools/pipeline.py \
    --tag-script "$(basename "$0")" \
    --quality 27 \
    --autocrop \
    --photon-noise 6 \
    --workers 4 \
    --fast-speed 8 \
    --final-speed 4 \
    --fast-params "--lp 3 --tune 3 --hbd-mds 0 --keyint 305 --ac-bias 0.8 --sharp-tx 1 --sharpness 1 --tf-strength 2 --variance-boost-strength 1 --variance-octile 7 --enable-dlf 2" \
    --final-params "--lp 3 --tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.8 --sharp-tx 1 --sharpness 1 --tf-strength 2 --variance-boost-strength 1 --variance-octile 7 --enable-dlf 2" \
    "$@"
