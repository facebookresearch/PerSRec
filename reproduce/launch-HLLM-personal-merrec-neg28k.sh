#!/bin/bash
# please use LLM_deepspeed, it is considerable faster than DDP.
# consider reduce num_workers from 11 to some smaller value

python3 main.py \
    --config_file hllm/overall/LLM_deepspeed.yaml hllm/HLLM/HLLM.yaml \
    --epochs 20 \
    --optim_args.learning_rate 1e-4 \
    --checkpoint_dir hllm/checkpoint \
    --loss nce \
    --model PersonalizedHLLM \
    --number_of_user_tokens "$2" \
    --predict_with_full_item_seq true \
    --loss_on_all_items true \
    --pretrain_item_num 400 \
    --recent_item_num 100 \
    --item_start_position 0 \
    --item_end_position 0 \
    --train_feat_max_length 512 \
    --MAX_ITEM_LIST_LENGTH 500 \
    --MAX_ITEM_LIST_LENGTH_TEST 500 \
    --MAX_TEXT_LENGTH 256 \
    --scheduler_args.warmup 0.15 \
    --dataset eb_nerd_512 \
    --text_keys '["title","subtitle", "topics"]' \
    --data_path interaction \
    --text_path information \
    --item_pretrain_dir hllm/Llama-3.2-1B \
    --user_pretrain_dir hllm/Llama-3.2-1B \
    --train_batch_size 3 \
    --data_split False \
    --suppress_history False \
    --num_workers 4 \
    --eval_batch_size 12 \
    --auto_resume True \
    --gradient_checkpointing True \
    --seed 42 \
    --stage 3
