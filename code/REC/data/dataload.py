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
import copy
import os
from collections import Counter
from logging import getLogger, INFO
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
import torch

from REC.utils import set_color
from REC.utils.enum_type import InputType
from torch_geometric.utils import degree
from REC.data.shareables import SharedList
import REC.data.comm as comm


class InteractionData:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config['data_path']
        self.dataset_name = config['dataset']
        self.data_split = config['data_split']
        self.last_k_eval_test = config['last_k_eval_test'] or 2
        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)
        self.local_rank = comm.get_local_rank()
        self.uid_field = 'user_id'
        self.iid_field = 'item_id'
        # TODO: make this configurable
        # decopule max_item_list_len from MAX_ITEM_LIST_LENGTH
        # MAX_ITEM_LIST_LENGTH is also used by the model
        self.max_item_list_len = config.get("train_feat_max_length", config["MAX_ITEM_LIST_LENGTH"] + 1)
        assert self.max_item_list_len >= config["MAX_ITEM_LIST_LENGTH"] + 1, f"max_item_list_len should be >= {config['MAX_ITEM_LIST_LENGTH'] + 1} to support last_k_eval"
        # slice the sequence to the start and end position with self.max_item_list_len
        # by default it selects the whole sequence
        self.item_start_position = config.get("item_start_position", 0)
        self.item_end_position = config.get("item_end_position", None)
        if self.item_end_position == 0:
            self.item_end_position = None
        print(f"slice sequence {self.max_item_list_len} to {self.item_start_position}:{self.item_end_position}")
        # defer all the actual loading to build()

    def _from_scratch(self):
        self.logger.info(set_color(f'Loading {self.__class__} from scratch with {self.data_split = }.', 'green'))
        inter_feat_path = os.path.join(self.dataset_path, self.dataset_name)
        if os.path.isfile(inter_feat_path + ".csv"):
            self._data_processing(inter_feat_path)
        elif os.path.isfile(inter_feat_path + ".parquet"):
            self._data_processing_fast(inter_feat_path)
        else:
            raise ValueError(f'File {inter_feat_path} not exist.')

    def _data_processing_fast(self, inter_feat_path):
        self.logger.info(f'Loading interaction feature from parquet [{inter_feat_path}].')
        df = pl.read_parquet(
            inter_feat_path + ".parquet", columns=['object_id', 'user_id', 'event_time']
        ).rename({'event_time': 'timestamp', "object_id": "item_id"})
        self.logger.info(f'Interaction feature loaded successfully from [{inter_feat_path}].')
        self.logger.info(f'Interaction feature preview: {df.head()}')

        self.id2token: Dict[str, List[str]] = {}
        self.user_num = len(df) + 1  # user_id starts from 1
            # this will convert mp to a list[str] from list[Any]
        self.id2token['user_id'] = np.array(['[PAD]'] + list(range(1, self.user_num + 1)))

        self.item_num = df.select(pl.col("item_id").list.explode().max()).item() + 1
        self.id2token['item_id'] = np.array(['[PAD]'] + list(range(1, self.item_num + 1)))

        self.inter_num = df.select(pl.col("item_id").list.len().sum()).item()

        # used for evaluate and only need list form
        self.user_seq: List[List[int]] = []
        self.time_seq: List[List[int]] = []
        # polar use iter_rows and pandas uses iterrows
        for row in df.sort(by=["user_id"], descending=False).iter_rows(named=True):
            self.user_seq.append(row['item_id'] if isinstance(row['item_id'], list) else row['item_id'].to_list())
            self.time_seq.append(row['timestamp'] if isinstance(row['timestamp'], list) else row['timestamp'].to_list())

        df_flat = df.with_columns([
            pl.col("item_id").list.slice(0, pl.col("item_id").list.len() - self.last_k_eval_test).alias("item_id"),
            pl.col("timestamp").list.slice(0, pl.col("timestamp").list.len() - self.last_k_eval_test).alias("timestamp")
        ]).with_columns([
            pl.col("user_id").repeat_by(pl.col("item_id").list.len()).alias("user_id"),
        ]).explode(["user_id", "item_id", "timestamp"]
        ).sort(by=["user_id", "timestamp"], descending=False)
        self.train_feat: Dict[str, List[int]] = df_flat.to_dict(as_series = False)
        for k in self.train_feat:
            self.logger.info(f"{k} head: {self.train_feat[k][:5]}")


    def _data_processing(self, inter_feat_path):
        df = pd.read_csv(
            inter_feat_path + ".csv", delimiter=',', dtype={'item_id': str, 'user_id': str, 'timestamp': int}, header=0, names=['item_id', 'user_id', 'timestamp']
        )
        self.logger.info(f'Interaction feature loaded successfully from [{inter_feat_path}].')
        inter_feat = df
        self.logger.info(f'Interaction feature preview: {inter_feat.head()}')
        self.id2token = {}
        remap_list = ['user_id', 'item_id']
        for feature in remap_list:
            feats = inter_feat[feature]
            new_ids_list, mp = pd.factorize(feats)
            # this will convert mp to a list[str] from list[Any]
            mp = ['[PAD]'] + list(mp)
            token_id = {t: i for i, t in enumerate(mp)}
            mp = np.array(mp)

            self.id2token[feature] = mp
            inter_feat[feature] = inter_feat[feature].map(token_id)

        self.user_num = len(self.id2token['user_id'])
        self.item_num = len(self.id2token['item_id'])
        self.logger.info(f"{self.user_num = } {self.item_num = }")
        self.logger.info(f"{inter_feat['item_id'].isna().any() = } {inter_feat['user_id'].isna().any() = }")
        self.inter_num = len(inter_feat)

        self.sort(inter_feat, by='timestamp')
        user_list = inter_feat['user_id'].values
        item_list = inter_feat['item_id'].values
        timestamp_list = inter_feat['timestamp'].values
        grouped_index = self._grouped_index(user_list)

        self.user_seq: List[List[int]] = []
        self.time_seq: List[List[int]] = []
        for _, index in grouped_index.items():
            self.user_seq.append(item_list[index])
            self.time_seq.append(timestamp_list[index])

        train_feat = dict()
        indices = []

        for index in grouped_index.values():
            indices.extend(list(index)[:-2])
        for k in inter_feat:
            train_feat[k] = inter_feat[k].values[indices]
        self.train_feat = train_feat
        del inter_feat


    def build(self):
        # only perform the actual operation in local rank 0 and broadcast to other ranks
        if self.local_rank == 0:
            self.logger.info(f"build {self.dataset_name} dataload on rank {self.local_rank}")
            self._from_scratch()

            if self.config['MODEL_INPUT_TYPE'] == InputType.AUGSEQ:
                train_feat = self._build_aug_seq(self.train_feat)
            elif self.config['MODEL_INPUT_TYPE'] == InputType.SEQ:
                train_feat = self._build_seq(self.train_feat)
            id2token = self.id2token

            # slicing
            # user_seq and time_seq is only for evaluation and may not need slicing.
            self.user_seq = [s[self.item_start_position:self.item_end_position] for s in self.user_seq]
            self.time_seq = [s[self.item_start_position:self.item_end_position] for s in self.time_seq]
            for k in ["item_seq", "time_seq"]:
                train_feat[k] = [s[self.item_start_position:self.item_end_position] for s in train_feat[k]]
        else:
            # load from local rank 0
            self.logger.info(f"skip load {self.dataset_name} dataload on rank {self.local_rank}")
            train_feat = {k: [] for k in ["user_id", "item_seq", "time_seq"]}
            self.user_seq = []
            self.time_seq = []
            id2token = {k: [] for k in ['user_id', 'item_id']}
        self.logger.info(f"broadcast train_feat dataload")
        self.train_feat = {k: SharedList(v) for k, v in train_feat.items()}
        self.logger.info(f"broadcast id2token dataload")
        self.id2token = {k: SharedList(v) for k, v in id2token.items()}
        self.logger.info(f"broadcast user_seq dataload")
        self.user_seq = SharedList(self.user_seq)
        self.logger.info(f"broadcast time_seq dataload")
        self.time_seq = SharedList(self.time_seq)
        self.user_num = len(self.id2token["user_id"])
        self.item_num = len(self.id2token["item_id"])
        self.inter_num = sum(len(values) for values in self.user_seq)
        self.counter = {
            "user_id": Counter({key: len(value) for key, value in enumerate(self.user_seq)}),
            "item_id": Counter(item for values in self.user_seq for item in values),
        }
        self.logger.info(f"{self.user_num = }, {self.item_num = }, {self.inter_num = }")
        self.logger.info(f"{len(self.user_seq) = }, {len(self.time_seq) = }, train_feat { {k: len(v) for k, v in self.train_feat.items()} }")
        del train_feat, id2token
        gc.collect()


    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index

    def _build_seq(self, train_feat):
        max_item_list_len = self.max_item_list_len

        uid_list, item_list_index = [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid:
                save = True
            if save:
                if (self.data_split is None or self.data_split == True) and i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset 
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0], sub[-1]+1))
                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(max(i - max_item_list_len, seq_start), i))  # keep the last max_item_list_len items

                save = False
                last_uid = uid
                seq_start = i

        seq_train_feat = {}
        seq_train_feat['user_id'] = np.array(uid_list)
        seq_train_feat['item_seq'] = []
        seq_train_feat['time_seq'] = []
        for index in item_list_index:
            seq_train_feat['item_seq'].append(train_feat['item_id'][index])
            seq_train_feat['time_seq'].append(train_feat['timestamp'][index])

        return seq_train_feat

    def _build_aug_seq(self, train_feat):
        # TODO: fix like _build_seq. But not used so far.
        max_item_list_len = self.max_item_list_len

        # by = ['user_id', 'timestamp']
        # ascending = [True, True]
        # for b, a in zip(by[::-1], ascending[::-1]):
        #     index = np.argsort(train_feat[b], kind='stable')
        #     if not a:
        #         index = index[::-1]
        #     for k in train_feat:
        #         train_feat[k] = train_feat[k][index]

        uid_list, item_list_index = [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid:
                save = True
            if save:
                if i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0], sub[-1]+1))
                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start, i))
                save = False
                last_uid = uid
                seq_start = i

        seq_train_feat = {}
        aug_uid_list = []
        aug_item_list = []
        for uid, item_index in zip(uid_list, item_list_index):
            st = item_index.start
            ed = item_index.stop
            lens = ed - st
            for sub_idx in range(1, lens):
                aug_item_list.append(train_feat['item_id'][slice(st, st+sub_idx+1)])
                aug_uid_list.append(uid)

        seq_train_feat['user_id'] = np.array(aug_uid_list)
        seq_train_feat['item_seq'] = aug_item_list

        return seq_train_feat

    @staticmethod
    def sort(inter_feat, by, ascending=True):
        """Perform sort in place"""

        if isinstance(inter_feat, pd.DataFrame):
            inter_feat.sort_values(by=by, ascending=ascending, inplace=True)

        else:
            if isinstance(by, str):
                by = [by]

            if isinstance(ascending, bool):
                ascending = [ascending]

            if len(by) != len(ascending):
                if len(ascending) == 1:
                    ascending = ascending * len(by)
                else:
                    raise ValueError(f'by [{by}] and ascending [{ascending}] should have same length.')
            for b, a in zip(by[::-1], ascending[::-1]):
                index = np.argsort(inter_feat[b], kind='stable')
                if not a:
                    index = index[::-1]
                for k in inter_feat:
                    inter_feat[k] = inter_feat[k][index]

    @property
    def avg_actions_of_users(self):
        """Get the average number of users' interaction records.

        Returns:
            numpy.float64: Average number of users' interaction records.
        """
        return self.inter_num / self.user_num

    @property
    def avg_actions_of_items(self):
        """Get the average number of items' interaction records.

        Returns:
            numpy.float64: Average number of items' interaction records.
        """
        return self.inter_num / self.item_num

    @property
    def sparsity(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.inter_num / self.user_num / self.item_num

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [set_color(self.dataset_name, 'pink')]
        if self.uid_field:
            info.extend([
                set_color('The number of users', 'blue') + f': {self.user_num}',
                set_color('Average actions of users', 'blue') + f': {self.avg_actions_of_users}'
            ])
        if self.iid_field:
            info.extend([
                set_color('The number of items', 'blue') + f': {self.item_num}',
                set_color('Average actions of items', 'blue') + f': {self.avg_actions_of_items}'
            ])
        info.append(set_color('The number of interactions', 'blue') + f': {self.inter_num}')
        if self.uid_field and self.iid_field:
            info.append(set_color('The sparsity of the dataset', 'blue') + f': {self.sparsity * 100}%')

        return '\n'.join(info)

    def copy(self, new_inter_feat):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.

        Args:
            new_inter_feat (Interaction): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    @property
    def user_counter(self):
        return self.counter('user_id')

    @property
    def item_counter(self):
        return self.counter('item_id')

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = torch.tensor(self.train_feat[self.uid_field])
        col = torch.tensor(self.train_feat[self.iid_field]) + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num + self.item_num)

        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight
