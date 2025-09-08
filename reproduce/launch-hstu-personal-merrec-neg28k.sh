#!/bin/bash
# please use LLM_deepspeed, it is considerable faster than DDP.
# consider reduce num_workers from 11 to some smaller value
# some key args:
#   number_of_user_tokens: number of learnable tokens to be added.
#   insert_method: how to apply the learnable tokens. Use to 0 to insert all after pretrain_item_num
#       or List[Tuple[offset, k]] in json encoded string to insert k user embedding at offset of pos_emb.
#       sum(k) === number_of_user_tokens
#   pretrain_item_num: if set, we will use the full item sequence for prediction.
#   model: HSTU or PersonalizedHSTU
#   pretrain_item_num: number of items to be used for longer history.
#   recent_item_num: number of items to be used for recent history.
#   train_feat_max_length: if not provided, will use MAX_ITEM_LIST_LENGTH + 1. The number of items from the
#       sequence loaded to be used for training and inference.
#   item_start_position. default 0. The index of the first item (inclusive) from train_feat_max_length to be used for training and inference.
#   item_end_position: default 0 or None. The index of the last item (inclusive) from train_feat_max_length to be used for training and inference.
#   MAX_ITEM_LIST_LENGTH: max length of item sequence. This is also the max length of transformer could support.
#       MAX_ITEM_LIST_LENGTH = pretrain_item_num + recent_item_num
python3 main.py \
    --config_file hllm/IDNet/hstu.yaml hllm/overall/ID_deepspeed.yaml \
    --optim_args.learning_rate 1e-3 \
    --optim_args.weight_decay 0.0 \
    --loss nce \
    --model PersonalizedHSTU \
    --number_of_user_tokens "$2" \
    --insert_method 0 \
    --predict_with_full_item_seq true \
    --loss_on_all_items true \
    --pretrain_item_num 400 \
    --recent_item_num 100 \
    --train_batch_size 8 \
    --item_start_position 0 \
    --item_end_position 0 \
    --train_feat_max_length 512 \
    --MAX_ITEM_LIST_LENGTH 500 \
    --epochs 201 \
    --data_path interaction \
    --dataset eb_nerd_512 \
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
    --checkpoint_dir hllm/checkpoint \
    --seed 42 \
    --stopping_step 10
