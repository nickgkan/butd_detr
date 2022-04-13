#!/usr/bin/env bash


set -x

EXP_DIR=exps/butd_finetune_flickr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_config configs/flickr.json \
    --batch_size 1 \
    --lr 1e-5 \
    --weight_decay 1e-4 \
    --save_freq 1 \
    --eval_skip 1 \
    --ema \
    --eval \
    --resume /data/beauty_detr/bdetr_arxiv/pretrain_2d.pth
