a#!/bin/bash
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
    --config_file PerSRechllm/IDNet/hstu.yaml hllm/overall/ID_deepspeed.yaml \
    --optim_args.learning_rate 1e-3 \
    --optim_args.weight_decay 0.0 \
    --loss nce \
    --model PersonalizedHSTU \
    --number_of_user_tokens 4 \
    --predict_with_full_item_seq true \
    --loss_on_all_items true \
    --pretrain_item_num 1024 \
    --recent_item_num 256 \
    --train_batch_size 8 \
    --item_start_position -2000 \
    --item_end_position -720 \
    --train_feat_max_length 2000 \
    --MAX_ITEM_LIST_LENGTH 1280 \
    --MAX_ITEM_LIST_LENGTH_TEST 1280 \
    --epochs 201 \
    --data_path interaction \
    --dataset merrec_2000 \
    --hidden_dropout_prob 0.5 \
    --attn_dropout_prob 0.5 \
    --n_layers 16 \
    --n_heads 8 \
    --item_embedding_size 64 \
    --hstu_embedding_size 64 \
    --fix_temp True \
    --show_progress True \
    --update_interval 100 \
    --num_workers 4 \
    --data_split False \
    --suppress_history False \
    --eval_batch_size 8 \
    --checkpoint_dir hllm/checkpoint/torchx-hstu_shared_merrec_train_full_1024_4_256_deepspeed_1x-wvxm2f9b/PersonalizedHSTU-31.pth \
    --seed 42 \
    --stopping_step 10
fi
