#!/bin/bash

python3 main.py \
    --config_file hllm/overall/LLM_deepspeed.yaml hllm/HLLM/HLLM.yaml \
    --MAX_ITEM_LIST_LENGTH "$2" \
    --epochs 20 \
    --optim_args.learning_rate 1e-4 \
    --checkpoint_dir hllm/checkpoint \
    --loss nce \
    --MAX_TEXT_LENGTH 256 \
    --scheduler_args.warmup 0.15 \
    --data_path interaction \
    --dataset eb_nerd_512 \
    --text_keys '["title","subtitle", "topics"]' \
    --text_path information \
    --item_pretrain_dir hllm/Llama-3.2-1B \
    --user_pretrain_dir hllm/Llama-3.2-1B \
    --train_batch_size 2 \
    --data_split False \
    --suppress_history False \
    --num_workers 4 \
    --eval_batch_size 16 \
    --gradient_checkpointing True \
    --auto_resume True \
    --seed 42 \
    --stage 3
