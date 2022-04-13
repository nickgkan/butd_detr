#!/usr/bin/env bash

set -x

EXP_DIR=exps/refcoco_plus
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_config configs/refcoco+.json \
    --batch_size 1 \
    --lr 1e-5 \
    --save_freq 1 \
    --eval_skip 1 \
    --resume /projects/katefgroup/language_grounding/nai/butd_finetune_refcoco_plus_ckpt12/checkpoint0017.pth \
    --eval