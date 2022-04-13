#!/usr/bin/env bash

set -x

EXP_DIR=exps/refcoco
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_config configs/refcoco.json \
    --batch_size 1 \
    --lr 1e-5 \
    --save_freq 1 \
    --eval_skip 1 \
    --eval \
    --resume /data/beauty_detr/bdetr_arxiv/refcoco_85.9.pth
