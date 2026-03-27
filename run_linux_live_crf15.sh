#!/bin/bash
cd "$(dirname "$0")"
source "$(dirname "$(realpath "$0")")/activate-venv.sh"

# Live Action Higher (CRF 15) Params — v1.66 5fish svt-av1-psy
python3 tools/pipeline.py \
    --tag-script "$(basename "$0")" \
    --quality 15 \
    --autocrop \
    --aggressive \
    --photon-noise 4 \
    --workers 4 \
    --fast-speed 8 \
    --final-speed 4 \
    --fast-params "--ac-bias 1.0 --complex-hvs 1 --keyint -1 --variance-boost-strength 2 --enable-dlf 2 --luminance-qp-bias 20 --tune 3" \
    --final-params "--ac-bias 1.0 --complex-hvs 1 --keyint -1 --variance-boost-strength 2 --enable-dlf 2 --luminance-qp-bias 20 --tune 3 --lp 3"
