TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root DATA_ROOT \
    --val_freq 5 --batch_size 24 --save_freq 5 --print_freq 1000 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset sr3d --test_dataset sr3d \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --lr_decay_epochs 30 35 \
    --pp_checkpoint PATH/TO/gf_detector_l6o256.pth \
    --butd_cls --self_attend
