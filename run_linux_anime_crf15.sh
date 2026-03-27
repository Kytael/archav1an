#!/bin/bash
cd "$(dirname "$0")"
source "$(dirname "$(realpath "$0")")/activate-venv.sh"

# Anime Higher (CRF 15) Params
python3 tools/pipeline.py \
    --tag-script "$(basename "$0")" \
    --quality 15 \
    --aggressive \
    --photon-noise 2 \
    --workers 4 \
    --fast-speed 8 \
    --final-speed 4 \
    --fast-params "--ac-bias 1.0 --complex-hvs 1 --keyint -1 --variance-boost-strength 1 --enable-dlf 2 --noise-level-thr 16000 --variance-md-bias 1 --cdef-bias 1 --luminance-qp-bias 20 --qm-min 8 --keyint -1 --tune 0 --balancing-q-bias 1 --chroma-qmc-bias 2 --filtering-noise-detection 4 --balancing-r0-based-layer-offset 0 --chroma-qm-min 10" \
    --final-params "--ac-bias 1.0 --complex-hvs 1 --keyint -1 --variance-boost-strength 1 --enable-dlf 2 --noise-level-thr 16000 --variance-md-bias 1 --cdef-bias 1 --luminance-qp-bias 20 --qm-min 8 --keyint -1 --tune 0 --balancing-q-bias 1 --chroma-qmc-bias 2 --filtering-noise-detection 4 --balancing-r0-based-layer-offset 0 --chroma-qm-min 10 --lp 3"
