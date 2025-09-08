# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.
import gc
import logging
from typing import Dict, List, Optional
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import logging
import REC.data.comm as comm
from REC.data.shareables import SharedList

from .trainset import TextSEQTrainDataset

class BatchTextDataset(Dataset):
    def __init__(self, config, item_text: Optional[List[Dict[str, str]]] = None):
        self.max_text_length = config['MAX_TEXT_LENGTH']

        self.text_path = config['text_path']
        self.text_keys = config['text_keys']
        self.tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True)
        # self.pad_id = self.tokenizer.pad_token_id
        # assert self.pad_id is not None, f"pad_token_id can't be {self.pad_id}"
        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.local_rank = comm.get_local_rank()
        if item_text:
            self.env = item_text
            self.logger.info(f"Get env from train_dataset and skip loading text")
        else:
            self.logger.info(f"Create new env for BatchTextDataset. Consider to provide train_dataset to avoid this")
            if self.local_rank == 0:
                env = TextSEQTrainDataset.load_content(self.text_path, self.text_keys)
            else:
                self.logger.info(f"skip load_content on rank {self.local_rank}")
                env = []
            self.env = SharedList(env)
            del env
            gc.collect()
        self.item_num = len(self.env)

    def __len__(self):
        return self.item_num

    def __getitem__(self, index: int):
        def process_item(item: int):
            # note item would be a str instead of int.
            # it contains element like [pad], some id but str format
            item_i = None
            if item != "[PAD]":
                item = int(item)
                # starts from 1
                if not 1 <= item < len(self.env):
                    self.logger.warning(f"item {item} out of range of {1}-{len(self.env)}")
                else:
                    # could be an empty dictionary {}
                    item_i = self.env[item]
            text_str = ""
            if item_i:
                text_str = f"{self.item_prompt}"
                for key in self.text_keys:
                    value = item_i[key]
                    if value and str(value) != 'nan':
                        text_str += f"{key}: {value}"

            ids = self.tokenizer.encode(text_str)
            ids = ids[:self.max_text_length]
            mask = [1] * len(ids)
            return ids, mask

        pos_input_ids, pos_cu_input_lens, pos_position_ids = [], [], []
        ids, _ = process_item(index)
        pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
        pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
        pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())
        outputs = {
            "pos_item_ids": torch.as_tensor(index, dtype=torch.int64),
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64)
        }
        return outputs
