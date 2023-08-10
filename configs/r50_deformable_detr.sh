#!/usr/bin/env bash

set -x

EXP_DIR=weights
PY_ARGS=${@:1}

# python -u main.py \
python -u make_pickle.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
