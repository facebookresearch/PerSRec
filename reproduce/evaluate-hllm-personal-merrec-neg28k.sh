#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# please use LLM_deepspeed, it is considerable faster than DDP.
# consider reduce num_workers from 11 to some smaller value
# evaluate the decay of compressed pretrain for different recent UIH by changing last_k_eval_test.
# parameters:
#   MAX_ITEM_LIST_LENGTH_TEST: total sequence length for test.
#       if not set, will fallback to MAX_ITEM_LIST_LENGTH if not set
#   MAX_ITEM_LIST_LENGTH: total sequence length for train and test.
#       Please don't change it after training due to impact to position embedding.
#   pretrain_item_num: number of leading items to be used as pretrain UIH.
#   recent_item_num: number of trailing items to be used as recent/evaluation UIH.
#       if not set, will fallback to MAX_ITEM_LIST_LENGTH - pretrain_item_num
#   number_of_user_tokens: number of learnable tokens.
# examples:
#   MAX_ITEM_LIST_LENGTH=1792, pretrain_item_num=1536, number_of_user_tokens=4. It will
#       compress first 1536 items to 4 learnable tokens and then combine with
#       remaining 1792-1536=256 items to predict the last item
#   MAX_ITEM_LIST_LENGTH=1792, MAX_ITEM_LIST_LENGTH_TEST=1792, pretrain_item_num=1536,
#       recent_item_num=256, number_of_user_tokens=4. Same as above.
#   MAX_ITEM_LIST_LENGTH=1792, MAX_ITEM_LIST_LENGTH_TEST=2000, pretrain_item_num=1536,
#       recent_item_num=256, number_of_user_tokens=4. It will
#       compress first 1536 items to 4 learnable tokens and then combine with
#       last 256 items (of 2000 items) to predict the last time

python3 main.py \
    --val_only true \
    --config_file manifold://post_ray/tree/qiangzhang/hllm/overall/LLM_deepspeed.yaml manifold://post_ray/tree/qiangzhang/hllm/HLLM/HLLM.yaml \
    --optim_args.learning_rate 1e-4 \
    --checkpoint_dir manifold://post_ray/tree/qiangzhang/hllm/checkpoint/torchx-hllm_shared_1024_4_256-c1xldhp02xkk2c/PersonalizedHLLM-4.pth \
    --loss nce \
    --model PersonalizedHLLM \
    --number_of_user_tokens 4 \
    --predict_with_full_item_seq true \
    --loss_on_all_items true \
    --pretrain_item_num 1024 \
    --recent_item_num 256 \
    --item_start_position -2000 \
    --item_end_position -720 \
    --train_feat_max_length 2000 \
    --MAX_ITEM_LIST_LENGTH 1280 \
    --MAX_ITEM_LIST_LENGTH_TEST 1280 \
    --MAX_TEXT_LENGTH 256 \
    --scheduler_args.warmup 0.15 \
    --data_path manifold://deep_retrieval/tree/byzhang/interaction \
    --dataset merrec_2000 \
    --text_keys '["brand_name","category_name"]' \
    --text_path manifold://deep_retrieval/tree/byzhang/information \
    --item_pretrain_dir manifold://post_ray/tree/qiangzhang/hllm/Llama-3.2-1B \
    --user_pretrain_dir manifold://post_ray/tree/qiangzhang/hllm/Llama-3.2-1B \
    --train_batch_size 3 \
    --data_split False \
    --suppress_history False \
    --num_workers 4 \
    --eval_batch_size 12 \
    --auto_resume True \
    --gradient_checkpointing True \
    --seed 42 \
    --stage 3
