#!/usr/bin/env bash


set -x

EXP_DIR=exps/butd_finetune_flickr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_config configs/flickr.json \
    --ema \
    --resume /projects/katefgroup/language_grounding/bdetr_arxiv/pretrain_2d.pth \
    --visualize_custom_image \
    --img_path img.png \
    --custom_text "a woman"
