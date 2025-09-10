# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import datetime
from typing import List

import pytz
import torch
from torch.utils.data import Dataset

class SeqEvalDataset(Dataset):
    def __init__(self, config, interaction_data, phase='valid'):
        self.interaction_data = interaction_data
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH_TEST'] if config['MAX_ITEM_LIST_LENGTH_TEST'] else config['MAX_ITEM_LIST_LENGTH']
        self.user_seq = interaction_data.user_seq
        self.time_seq = interaction_data.time_seq
        self.user_id = interaction_data.train_feat['user_id']
        self.use_time = config['use_time']
        self.phase = phase
        self.length = len(self.user_seq)
        self.item_num = interaction_data.item_num
        self.last_k_eval_test = config['last_k_eval_test'] or 2
        assert config.get("fix_pretrain_seq", None) is None, "fix_pretrain_seq is not supported, if you need it use hllm:ba49aad"

    def __len__(self):
        return self.length

    def _padding_sequence(self, sequence, max_length):
        sequence = list(sequence)
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return sequence

    def _padding_time_sequence(self, sequence, max_length):
        sequence = list(sequence)
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        vq_time = []
        for time in sequence:
            dt = datetime.datetime.fromtimestamp(time, pytz.timezone('UTC'))
            vq_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])
        return vq_time

    def __getitem__(self, index):
        last_num = self.last_k_eval_test if self.phase == 'valid' else self.last_k_eval_test // 2
        history_seq = self.user_seq[index][:-last_num]
        item_seq = self._padding_sequence(history_seq, self.max_item_list_length)
        item_target = self.user_seq[index][-last_num]
        user_id = self.user_id[index]
        if self.use_time:
            history_time_seq = self.time_seq[index][:-last_num]
        else:
            history_time_seq = []
        time_seq = self._padding_time_sequence(history_time_seq, self.max_item_list_length)

        return torch.tensor(history_seq), item_seq, item_target, time_seq, torch.tensor(user_id)
