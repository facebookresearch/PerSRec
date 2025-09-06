# [Efficient Sequential Recommendation for Long Term User Interest Via Personalization]()


## Changes to [Original HLLM CodeBase](https://github.com/bytedance/HLLM)
- rewrite the dataloader to support larger dataset and make the data loading more efficient;
- deprecated csv support but using parquet format instead;
- add support of efficient sequential modeling via attention masking.

## Installation

This code could run in devgpu without buck2. To do that, please take those additional steps:
0. set up the proxy according to this [wiki](https://www.internalfb.com/intern/wiki/Development_Environment/Internet_Proxy/#python-pip)
1. `conda create -n hllm`
2. `conda activate hllm`
3. `conda install pip`. conda install could be very slow.
4. $$if you are using python 3.13, please change to python 3.10 instead.$$ `conda install python=3.10`
5. `pip install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
6. `pip install ninja` to speed up the flash_atten build
7. `pip install -r requirements.txt`. flash_atten could be very slow to build
8. `pip install xformers` for merrec specific
9. for LLama 3.1+, you need to `pip install transformers==4.43.1`

## Dataset

To support larger dataset and make the data loading more efficient, we have deprecated csv support but using parquet format instead. You can use the following command to convert csv to parquet:
```python
import pandas as pd

# Read the CSV file
df = pd.read_csv("your_file.csv")

# Write the DataFrame to a Parquet file
df.to_parquet("your_file.parquet")
```

TODO: how to process Merrec and NB-NERD dataset.

Please make sure you have configured `text_path`, `data_path`, `item_pretrain_dir` and `user_pretrain_dir` accordingly.

## Training
We are using torchx as the helper to launch the training and evaluation job. You could find the scripts under reproduce/. To train HLLM on PixelRec / Amazon Book Reviews, you can run the following command.

> Set `master_addr`, `master_port`, `nproc_per_node`, `nnodes` and `node_rank` in environment variables for multinodes training.

> All hyper-parameters (except model's config) can be found in code/REC/utils/argument_list.py and passed through CLI. More model's hyper-parameters are in `IDNet/*` or `HLLM/*`.

```python
# Item and User LLM are initialized by specific pretrain_dir.
python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \ # We use deepspeed for training by default.
--loss nce \
--epochs 5 \
--dataset {Pixel200K / Pixel1M / Pixel8M / amazon_books} \
--train_batch_size 16 \
--MAX_TEXT_LENGTH 256 \
--MAX_ITEM_LIST_LENGTH 10 \
--checkpoint_dir saved_path \
--optim_args.learning_rate 1e-4 \
--item_pretrain_dir item_pretrain_dir \ # Set to LLM dir.
--user_pretrain_dir user_pretrain_dir \ # Set to LLM dir.
--text_path text_path \ # Use absolute path to text files.
--text_keys '[\"title\", \"tag\", \"description\"]' # Please remove tag in books dataset.
```
> You can use `--gradient_checkpointing True` and `--stage 3` with deepspeed to save memory.

You can also train ID-based models by the following command.
```python
python3 main.py \
--config_file overall/ID.yaml IDNet/{hstu / sasrec / llama_id}.yaml \
--loss nce \
--epochs 201 \
--dataset {Pixel200K / Pixel1M / Pixel8M / amazon_books} \
--train_batch_size 64 \
--MAX_ITEM_LIST_LENGTH 10 \
--optim_args.learning_rate 1e-4
```


Some key args:
- number_of_user_tokens: number of learnable tokens to be added.
- insert_method: how to apply the learnable tokens. Use to 0 to insert all after pretrain_item_num or List[Tuple[offset, k]] in json encoded string to insert k user embedding at offset of pos_emb. sum(k) === number_of_user_tokens
- pretrain_item_num: if set, we will use the full item sequence for prediction.
- model: HSTU or PersonalizedHSTU
- pretrain_item_num: number of items to be used for longer history.
- recent_item_num: number of items to be used for recent history.
- train_feat_max_length: if not provided, will use MAX_ITEM_LIST_LENGTH + 1. The number of items from the sequence loaded to be used for training and inference.
- item_start_position. default 0. The index of the first item (inclusive) from train_feat_max_length to be used for training and inference.
- item_end_position: default 0 or None. The index of the last item (inclusive) from train_feat_max_length to be used for training and inference.
- MAX_ITEM_LIST_LENGTH: max length of item sequence. This is also the max length of transformer could support. MAX_ITEM_LIST_LENGTH === pretrain_item_num + recent_item_num

## Inference
We couldn't release the model's weight due to company's policy as of now but we are working on it.

Then you can evaluate models by the following command (the same as training but val_only).
```python
python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \ # We use deepspeed for training by default.
--loss nce \
--epochs 5 \
--dataset {Pixel200K / Pixel1M / Pixel8M / amazon_books} \
--train_batch_size 16 \
--MAX_TEXT_LENGTH 256 \
--MAX_ITEM_LIST_LENGTH 10 \
--checkpoint_dir saved_path \
--optim_args.learning_rate 1e-4 \
--item_pretrain_dir item_pretrain_dir \ # Set to LLM dir.
--user_pretrain_dir user_pretrain_dir \ # Set to LLM dir.
--text_path text_path \ # Use absolute path to text files.
--text_keys '[\"title\", \"tag\", \"description\"]' \ # Please remove tag in books dataset.
--val_only True # Add this for evaluation
```



## Citation

If our work has been of assistance to your work, feel free to give us a star ⭐ or cite us using :

```
TODO
```

> Thanks to the excellent code repository [HLLM](https://github.com/bytedance/HLLM), [RecBole](https://github.com/RUCAIBox/RecBole), [VisRec](https://github.com/ialab-puc/VisualRecSys-Tutorial-IUI2021), [PixelRec](https://github.com/westlake-repl/PixelRec) and [HSTU](https://github.com/facebookresearch/generative-recommenders/tree/main) !
> This repository is released under the Apache License 2.0, some codes are modified from HLLM, which are released under the Apache License 2.0, respectively.
