#!/usr/bin/env bash

set -x

EXP_DIR=exps/pretrain
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dataset_config configs/pretrain.json \
    --batch_size 1 \
    --lr 1e-4 --lr_backbone 1e-5 --text_encoder_lr 5e-5 \
    --weight_decay 1e-4 \
    --large_scale \
    --save_freq 1 \
    --eval_skip 1 \
    --ema \
    --eval \
    --resume /data/beauty_detr/bdetr_arxiv/pretrain_2d.pth
