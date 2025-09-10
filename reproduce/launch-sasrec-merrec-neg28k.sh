#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# please use LLM_deepspeed, it is considerable faster than DDP.
# consider reduce num_workers from 11 to some smaller value

python3 main.py \
    --config_file hllm/IDNet/sasrec.yaml hllm/overall/ID_deepspeed.yaml \
    --optim_args.learning_rate 1e-3 \
    --loss nce \
    --train_batch_size 8 \
    --item_start_position 0 \
    --item_end_position 0 \
    --train_feat_max_length 512 \
    --MAX_ITEM_LIST_LENGTH "$2" \
    --epochs 201 \
    --data_path interaction \
    --dataset merrec_2000 \
    --hidden_dropout_prob 0.5 \
    --attn_dropout_prob 0.5 \
    --n_layers 4 \
    --n_heads 4 \
    --embedding_size 64 \
    --inner_size 1 \
    --show_progress True \
    --update_interval 100 \
    --num_workers 4 \
    --data_split False \
    --suppress_history False \
    --eval_batch_size 8 \
    --checkpoint_dir hllm/checkpoint \
    --seed 42 \
    --optim_args.weight_decay 0.0 \
    --stopping_step 10
