# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implements HSTU (Hierarchical Sequential Transduction Unit) in
Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
(https://arxiv.org/abs/2402.17152).
"""

import abc
import copy
from logging import getLogger, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from REC.model.basemodel import all_gather, BaseModel
from REC.utils.enum_type import InputType


def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x


TIMESTAMPS_KEY = "timestamps"


class RelativeAttentionBiasModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias


HSTUCacheState = Tuple[torch.Tensor, torch.Tensor]


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,  # [bs, 1, n, n]
):
    B, _, n, _ = attention_mask.size()

    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        q.view(B, n, num_heads, attention_dim),
        k.view(B, -1, num_heads, attention_dim),
    )
    qk_attn = F.silu(qk_attn) / n
    qk_attn = qk_attn * attention_mask
    # print(f"{qk_attn.size() = } {v.size() = }")
    attn_output = torch.einsum(
        "bhnm,bmhd->bnhd",
        qk_attn,
        v.reshape(B, -1, num_heads, linear_dim),
    ).reshape(B, n, num_heads * linear_dim)
    return attn_output


class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 2 * num_heads
                        + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads * (3 if concat_ua else 1),
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_kv: Optional[HSTUCacheState] = None,
    ) -> Tuple[torch.Tensor, HSTUCacheState]:
        """
        Args:
            x: batch x seq x dim
            attention_mask: batch x head x seq x seq. head could be 1
            past_kv: optional cache for k and v. If present, it will append to existing k and v.
                Be careful about the position embedding!
        Returns:
            x': batch x seq x dim. attended x.
            past_kv: a tuple of k and v.
        """

        normed_x = self._norm_input(x)
        if self._linear_config == "uvqk":
            batched_mm_output = torch.matmul(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        if past_kv:
            # batch x seq x dim
            k = torch.concat([past_kv[0], k], dim=1)
            v = torch.concat([past_kv[1], v], dim=1)

        B: int = attention_mask.size(0)
        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            attn_output = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                attention_mask=attention_mask,
            )

        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + x
        )

        return [new_outputs, [k, v]]


class HSTUJagged(torch.nn.Module):
    def __init__(
        self,
        modules: List[SequentialTransductionUnitJagged],
        autocast_dtype: torch.dtype,
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: torch.dtype = autocast_dtype

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_kv: Optional[List[HSTUCacheState]] = None,
        output_kv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
            output_kv: if true, return the k and v.
        Returns:
            x' = f(x), (B, N, D) x float
        """
        cache = []
        if past_kv is not None:
            assert (
                len(past_kv) == len(self._attention_layers)
            ), f"kv cache {len(past_kv)} and attention layers {len(self._attention_layers)} mismatch"
        for i, layer in enumerate(self._attention_layers):
            x, kv = layer(
                x=x,
                attention_mask=attention_mask,
                past_kv=past_kv[i] if past_kv else None,
            )
            if output_kv:
                cache.append(kv)

        if output_kv:
            return x, cache
        return x


class HSTU(BaseModel):
    """
    Implements HSTU (Hierarchical Sequential Transduction Unit) in
    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,
    https://arxiv.org/abs/2402.17152.

    Note that this implementation is intended for reproducing experiments in
    the traditional sequential recommender setting (Section 4.1.1), and does
    not yet use optimized kernels discussed in the paper.
    """

    input_type = InputType.SEQ

    def __init__(self, config, item_num: int, user_num: int):
        super().__init__()
        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)
        self.item_num = item_num
        self._item_embedding_dim: int = config["item_embedding_size"]
        self._hstu_embedding_dim: int = config["hstu_embedding_size"]
        self._max_sequence_length: int = config["MAX_ITEM_LIST_LENGTH"]
        self._num_blocks: int = config["n_layers"]
        self._num_heads: int = config["n_heads"]
        self._dqk: int = config["hstu_embedding_size"] // config["n_heads"]
        self._dv: int = config["hstu_embedding_size"] // config["n_heads"]
        self._linear_activation: str = (
            config["hidden_act"] if config["hidden_act"] else "silu"
        )
        self._linear_dropout_rate: float = config["hidden_dropout_prob"]
        self._attn_dropout_rate: float = config["attn_dropout_prob"]
        self._enable_relative_attention_bias: bool = (
            config["enable_relative_attention_bias"]
            if config["enable_relative_attention_bias"]
            else False
        )
        self._linear_config = "uvqk"
        self._normalization = "rel_bias"
        self.position_embedding = nn.Embedding(
            self._max_sequence_length + 1, self._hstu_embedding_dim
        )
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._hstu_embedding_dim,
                    linear_hidden_dim=self._dv,
                    attention_dim=self._dqk,
                    normalization=self._normalization,
                    linear_config=self._linear_config,
                    linear_activation=self._linear_activation,
                    num_heads=self._num_heads,
                    # TODO: change to lambda x.
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=self._max_sequence_length
                            + self._max_sequence_length,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if self._enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=self._linear_dropout_rate,
                    attn_dropout_ratio=self._attn_dropout_rate,
                    concat_ua=False,
                )
                for _ in range(self._num_blocks)
            ],
            autocast_dtype=None,
        )

        self.item_embedding = nn.Embedding(
            self.item_num, self._item_embedding_dim, padding_idx=0
        )
        self.item_id_proj_tower = (
            nn.Identity()
            if config["item_embedding_size"] == config["hstu_embedding_size"]
            else nn.Linear(
                config["item_embedding_size"], config["hstu_embedding_size"], bias=False
            )
        )
        self.loss = config["loss"]
        if self.loss == "nce":
            if config["fix_temp"]:
                self.logger.info(f"Fixed logit_scale 20")
                self.logit_scale = nn.Parameter(
                    torch.ones([]) * np.log(1 / 0.05), requires_grad=False
                )
            else:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.nce_thres = config["nce_thres"] if config["nce_thres"] else 0.99
            self.num_negatives = config["num_negatives"]
            self.logger.info(f"nce thres setting to {self.nce_thres}")
        else:
            raise NotImplementedError(f"Only nce is supported")

        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length,
                        self._max_sequence_length,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = True
        self.reset_params()

    def reset_params(self):
        for name, params in self.named_parameters():
            if (
                ("_hstu" in name)
                or ("_embedding_module" in name)
                or ("logit_scale" in name)
            ):
                if self._verbose:
                    print(f"Skipping init for {name}")
                continue
            try:
                truncated_normal(params.data, mean=0.0, std=0.02)
                if self._verbose:
                    print(
                        f"Initialize {name} as trunc normal: {params.data.size()} params"
                    )
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        debug_str = (
            f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str + super().__repr__()

    def forward(self, interaction):
        items, neg_items, masked_index, user_id = (
            interaction  # [batch, 2, seq_len]    #[batch, max_seq_len-1]
        )
        if self.num_negatives:
            neg_items = torch.randint(
                low=1,
                high=self.item_num,
                size=(items.size(0), items.size(1) - 1, self.num_negatives),
                dtype=items.dtype,
                device=items.device,
            )

        pos_items_embs = self.item_id_proj_tower(
            self.item_embedding(items)
        )  # [batch, 2, max_seq_len+1, dim]
        neg_items_embs = self.item_id_proj_tower(
            self.item_embedding(neg_items)
        )  # [128, 200, 1024, 50]
        input_emb = pos_items_embs[:, :-1, :]  # [batch, max_seq_len, dim]

        position_ids = torch.arange(
            masked_index.size(1), dtype=torch.long, device=masked_index.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(masked_index)
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_emb + position_embedding

        attention_mask = self.get_attention_mask(masked_index)
        output_embs = self._hstu(x=input_emb, attention_mask=attention_mask)

        target_pos_embs = pos_items_embs[:, 1:, :]  # [batch, max_seq_len, dim]

        return self._compute_loss(
            output_embs, masked_index, target_pos_embs, neg_items_embs
        )

    def _compute_loss(
        self,
        output_embs: torch.Tensor,
        masked_index: torch.Tensor,
        target_pos_embs: torch.Tensor,
        neg_items_embs: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute the loss.

        Args:
            output_embs: output embedding sequence from HSTU. [batch, max_seq_len, dim]
            masked_index: if True, the item is invalid for computing loss. [batch, max_seq_len]
            target_pos_embs: target item embeddings. [batch, max_seq_len, dim]
            target_neg_embs: negative item embeddings. [batch, max_seq_len, negative_num, dim]
        """
        neg_embedding_all = neg_items_embs  # [batch, max_seq_len, dim]

        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))
        logit_scale = self.logit_scale.exp()
        output_embs = output_embs / output_embs.norm(dim=-1, keepdim=True)
        target_pos_embs = target_pos_embs / target_pos_embs.norm(dim=-1, keepdim=True)
        neg_embedding_all = neg_embedding_all / neg_embedding_all.norm(
            dim=-1, keepdim=True
        )
        pos_logits = F.cosine_similarity(
            output_embs, target_pos_embs, dim=-1
        ).unsqueeze(-1)
        if self.num_negatives:
            neg_logits = F.cosine_similarity(
                output_embs.unsqueeze(2), neg_embedding_all, dim=-1
            )
            fix_logits = F.cosine_similarity(
                target_pos_embs.unsqueeze(2), neg_embedding_all, dim=-1
            )
        else:
            D = neg_embedding_all.size(-1)
            neg_embedding_all = all_gather(neg_embedding_all, sync_grads=True).reshape(
                -1, D
            )  # [num, dim]
            neg_embedding_all = neg_embedding_all.transpose(-1, -2)
            neg_logits = torch.matmul(output_embs, neg_embedding_all)
            fix_logits = torch.matmul(target_pos_embs, neg_embedding_all)

        neg_logits[fix_logits > self.nce_thres] = torch.finfo(neg_logits.dtype).min
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[masked_index.bool()] * logit_scale
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        model_out = {}
        model_out["loss"] = F.cross_entropy(logits, labels)
        model_out["nce_samples"] = (
            (logits > torch.finfo(logits.dtype).min / 100).sum(dim=1).float().mean()
        )
        for k in [1, 5, 10, 50, 100]:
            if k > logits.size(1):
                break
            indices = logits.topk(k, dim=1).indices
            model_out[f"nce_top{k}_acc"] = (
                labels.view(-1, 1).eq(indices).any(dim=1).float().mean()
            )
        return model_out

    @torch.no_grad()
    def predict(self, item_seq, time_seq, item_feature, user_id):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_id_proj_tower(self.item_embedding(item_seq))
        item_emb = item_emb + position_embedding
        attention_mask = self.get_attention_mask(item_seq)
        output_embs = self._hstu(x=item_emb, attention_mask=attention_mask)
        seq_output = output_embs[:, -1]
        seq_output = seq_output / seq_output.norm(dim=-1, keepdim=True)

        scores = torch.matmul(seq_output, item_feature.t())
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        weight = self.item_id_proj_tower(self.item_embedding.weight)
        return weight / weight.norm(dim=-1, keepdim=True)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        # extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask


class PersonalizedHSTU(HSTU):
    """
    HSTU with learnable tokens as personalized UIH.
    """

    input_type = InputType.SEQ

    def __init__(self, config, item_num: int, user_num: int):
        """HSTU with learnable tokens as personalized UIH.

        Args:
            item_num: vocab size of items.
            user_num: vocab size of users.
        Args from config:
            number_of_user_tokens: number of learnable tokens per user.
            loss_on_all_items: if True, loss is applied to all items, except the personalized tokens;
                otherwise, loss is only applied to the items after the personalized tokens.
                Default is True.
            pretrain_item_num: number of items for pretrain. Must > 0,
                the first pretrain_item_num items are used for pretrain.
            recent_item_num: number of items for recent. Must > 0. If not present,
                will be calculated as MAX_ITEM_LIST_LENGTH - pretrain_item_num.
        """
        self.user_num = user_num
        self.number_of_user_tokens = config.get("number_of_user_tokens", None)
        assert (
            self.number_of_user_tokens is not None and self.number_of_user_tokens > 0
        ), f"number_of_user_tokens must be set in config and nonngeative, but got {self.number_of_user_tokens}"
        self.loss_on_all_items = config.get("loss_on_all_items", True)
        assert self.loss_on_all_items is True, "loss_on_all_items must be True"
        self.pretrain_item_num = config.get("pretrain_item_num", None)
        assert (
            self.pretrain_item_num is not None and self.pretrain_item_num > 0
        ), f"pretrain_item_num must be set in config and nonngeative, but got {self.pretrain_item_num}"
        # the length of recent UIH is MAX_ITEM_LIST_LENGTH - number_of_user_tokens
        self.recent_item_num = config.get(
            "recent_item_num", config["MAX_ITEM_LIST_LENGTH"] - self.pretrain_item_num
        )
        assert (
            self.recent_item_num is not None and self.recent_item_num > 0
        ), f"if set recent_item_num must be nonngeative, but got {self.recent_item_num}"
        self.predict_with_full_item_seq = config.get("predict_with_full_item_seq", True)
        self.insert_method = config.get("insert_method", None)
        if self.insert_method is None:
            # derive one from pretrain_item_num for backward compatibility
            self.insert_method = [(self.pretrain_item_num, self.number_of_user_tokens)]
        else:
            # assume it is a json encoded list of (offset, k)
            assert sum([k for _, k in self.insert_method]) == self.number_of_user_tokens, f"sum of k must be {self.number_of_user_tokens=}, but got {self.insert_method}"
        config = copy.deepcopy(config)
        # override MAX_ITEM_LIST_LENGTH in config to count for additional personalized tokens
        config["MAX_ITEM_LIST_LENGTH"] += self.number_of_user_tokens
        # drop out the items in recent UIH for training
        self.drop_out = config.get("item_dropout", 0.0)
        assert 0 <= self.drop_out <= 1, f"item_dropout must be in [0, 1], but got {self.drop_out}"
        super().__init__(config, item_num, user_num)
        # we use embedding table to store the learnable tokens
        # need to reshape the embedding dim to match the HSTU embedding dim
        # TODO: _hstu_embedding_dim vs item_embedding_size
        # self.user_embedding = nn.Embedding(
            # self.user_num,
            # self._hstu_embedding_dim * self.number_of_user_tokens,
            # padding_idx=0,
        # )
        # all users share the same learnable tokens
        self.user_embedding = nn.Parameter(torch.randn(1, self.number_of_user_tokens, self._hstu_embedding_dim))
        self.reset_params()
        if not self.predict_with_full_item_seq:
            self.logger.info("will only use recent UIH and ignore learnable tokens")
        self.logger.info(f"will insert learnable tokens according to {self.insert_method=}")

    def __repr__(self) -> str:
        debug_str = "Personalized" + super().__repr__() + f"{self.pretrain_item_num=}-{self.number_of_user_tokens=}-{self.loss_on_all_items=}-{self.predict_with_full_item_seq=}-{self.recent_item_num=}"
        return debug_str

    def insert_user_embedding(
        self, 
        pos_emb: torch.Tensor, 
        masked_index: torch.Tensor, 
        user_embedding: torch.Tensor, 
        neg_emb: Optional[torch.Tensor], 
        method: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[Tuple[int, int]]]:
        """insert user embedding into UIH.

        Args:
            pos_emb: batch x seq_len x dim. item embedding
            masked_index: batch x seq_len. binary tensor indicates availablity of item
            user_embedding: batch x k x dim. user embedding
            neg_emb: batch x seq_len x num_neg x dim. optional negative item embedding
            method: list of (offset, k) to insert k user embedding at offset of pos_emb
        """
        assert sum([k for _, k in method]) == self.number_of_user_tokens, f"sum of k must be {self.number_of_user_tokens=}, but got {method}"
        new_pos_emb = []
        new_masked_index = []
        new_neg_emb = []
        cumulative_offset = 0
        cumulative_k = 0
        item_segments = []
        for (offset, k) in method:
            item_segments.append((cumulative_offset + cumulative_k, cumulative_k + offset - 1))
            new_pos_emb.append(pos_emb[:, cumulative_offset:offset])
            new_pos_emb.append(user_embedding[:, cumulative_k: (cumulative_k + k)])

            new_masked_index.append(masked_index[:, cumulative_offset:offset])
            new_masked_index.append(torch.ones(
                    (pos_emb.shape[0], k),
                    dtype=masked_index.dtype,
                    device=masked_index.device,
                )
            )

            if neg_emb is not None:
                new_neg_emb.append(neg_emb[:, cumulative_offset:offset])
                new_neg_emb.append(neg_emb[:, cumulative_offset:(cumulative_offset + k)])
            
            cumulative_offset = offset
            cumulative_k += k
        assert cumulative_k == self.number_of_user_tokens, f"{cumulative_k=} must be {self.number_of_user_tokens=}"
        # for last piece
        item_segments.append((cumulative_offset + cumulative_k, pos_emb.shape[1] + self.number_of_user_tokens - 1))
        self.logger.info(f"{item_segments=}")
        new_pos_emb.append(pos_emb[:, cumulative_offset:])
        new_masked_index.append(masked_index[:, cumulative_offset:])
        if neg_emb is not None:
            new_neg_emb.append(neg_emb[:, cumulative_offset:])

        new_pos_emb = torch.concat(new_pos_emb, dim=1)
        new_masked_index = torch.concat(new_masked_index, dim=1)
        if neg_emb is not None:
            new_neg_emb = torch.concat(new_neg_emb, dim=1)
        else:
            new_neg_emb = None
        assert new_pos_emb.shape[1] == pos_emb.shape[1] + self.number_of_user_tokens, f"{new_pos_emb.shape=} mismatch {pos_emb.shape=} + {self.number_of_user_tokens=}"
        assert new_masked_index.shape[1] == masked_index.shape[1] + self.number_of_user_tokens, f"{new_masked_index.shape=} mismatch {masked_index.shape=} + {self.number_of_user_tokens=}"
        return new_pos_emb, new_masked_index, new_neg_emb, item_segments


    def forward(self, interaction):
        """forward with [items[:pretrain], learnable_tokens, items[-pretrain:]]"""
        items, neg_items, masked_index, user_id = (
            interaction  # [batch, 2, seq_len]    #[batch, max_seq_len-1]
        )
        if self.num_negatives:
            neg_items = torch.randint(
                low=1,
                high=self.item_num,
                size=(items.size(0), items.size(1) - 1, self.num_negatives),
                dtype=items.dtype,
                device=items.device,
            )

        pos_items_embs = self.item_id_proj_tower(
            self.item_embedding(items)
        )  # [batch, 2, max_seq_len+1, dim]
        neg_items_embs = self.item_id_proj_tower(
            self.item_embedding(neg_items)
        )  # [128, 200, 1024, 50]
        # user_embedding = self.user_embedding(user_id).reshape(
        #     user_id.shape[0], self.number_of_user_tokens, self._hstu_embedding_dim
        # )
        user_embedding = self.user_embedding.expand(user_id.shape[0], -1, -1)

        # TODO: try other ways to insert the personalized tokens
        pos_items_embs, masked_index, neg_items_embs, item_segments = self.insert_user_embedding(
            pos_items_embs, 
            masked_index, 
            user_embedding, 
            neg_items_embs, 
            self.insert_method,
        )

        input_emb = pos_items_embs[:, :-1, :]  # [batch, max_seq_len, dim]
        position_ids = torch.arange(
            masked_index.size(1), dtype=torch.long, device=masked_index.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(masked_index)
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_emb + position_embedding

        attention_mask = self.get_attention_mask(
            masked_index, item_segments=item_segments
        )
        output_embs = self._hstu(x=input_emb, attention_mask=attention_mask)

        target_pos_embs = pos_items_embs[:, 1:, :]  # [batch, max_seq_len, dim]
        # override masked_index to exclude the personalized tokens for computing the loss
        loss_mask = torch.zeros_like(masked_index)
        for (start, end) in item_segments:
            loss_mask[:, start:end + 1] = masked_index[:, start:end + 1]
        return self._compute_loss(
            output_embs, loss_mask, target_pos_embs, neg_items_embs
        )

    @torch.no_grad()
    def predict(
        self,
        item_seq,
        time_seq,
        item_feature,
        user_id,
        pretrain_item_seq: Optional[torch.Tensor] = None,
        past_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """predict without kv cache"""
        assert past_kv is None, "use predict2 for predict with kv cache"
        assert pretrain_item_seq is None, "use predict2 for predict with kv cache"
        position_ids = torch.arange(
            item_seq.size(1) + self.number_of_user_tokens, dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand((item_seq.shape[0], -1))
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_id_proj_tower(self.item_embedding(item_seq))
        user_embedding = self.user_embedding.expand(user_id.shape[0], -1, -1)
        item_emb, item_seq, _, item_segments = self.insert_user_embedding(
            item_emb, 
            item_seq, 
            user_embedding, 
            None, 
            self.insert_method,
        )        
        item_emb = item_emb + position_embedding
        attention_mask = self.get_attention_mask(item_seq, item_segments=item_segments)
        output_embs = self._hstu(x=item_emb, attention_mask=attention_mask)
        seq_output = output_embs[:, -1]
        seq_output = seq_output / seq_output.norm(dim=-1, keepdim=True)

        scores = torch.matmul(seq_output, item_feature.t())
        return scores

    @torch.no_grad()
    def compress_pretrain(
        self, item_seq: torch.Tensor, user_id: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """generate kv cache of the personalized tokens by using pretrain uih and learnable tokens.

        Args:
            item_seq: batch x seq. the pretrain UIH to compress to learnable tokens
            user_id: batch. user id of the uih
        Returns:
            list of kv cache of the personalized tokens. Each kv cache is a tuple of (key, value)
        """
        assert (
            item_seq.shape[1] == self.pretrain_item_num
        ), f"item_seq must have {self.pretrain_item_num} items, but got {item_seq.shape[1]}"
        # use the full item seq and insert learnable tokens after pretrain
        position_ids = torch.arange(
            self.pretrain_item_num + self.number_of_user_tokens,
            dtype=torch.long,
            device=item_seq.device,
        )
        position_ids = position_ids.unsqueeze(0).expand((item_seq.shape[0], -1))
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_id_proj_tower(self.item_embedding(item_seq))
        # user_embedding = self.user_embedding(user_id).reshape(
        #     user_id.shape[0], self.number_of_user_tokens, self._hstu_embedding_dim
        # )
        user_embedding = self.user_embedding.expand(user_id.shape[0], -1, -1)
        item_emb = torch.concat([item_emb, user_embedding], dim=1) + position_embedding
        # use a regular causal attention mask
        attention_mask = self.get_attention_mask(
            torch.concat(
                [
                    item_seq,
                    torch.ones(
                        (item_seq.shape[0], self.number_of_user_tokens),
                        dtype=item_seq.dtype,
                        device=item_seq.device,
                    ),
                ],
                dim=1,
            )
        )
        _, past_kv = self._hstu(
            x=item_emb,
            attention_mask=attention_mask,
            output_kv=True,
        )
        # only need the k, v for learnable tokens
        return [
            (k[:, self.pretrain_item_num :], v[:, self.pretrain_item_num :])
            for k, v in past_kv
        ]

    @torch.no_grad()
    def predict2(
        self,
        item_seq,
        time_seq,
        item_feature,
        user_id,
        pretrain_item_seq: Optional[torch.Tensor] = None,
        past_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """predict next item embedding from given item_seq with kv_cache of personalized tokens.

        Args:
            item_seq: batch x seq. the item sequence to predict next item
            item_feature: num x dim. embedding of candidate items
            user_id: batch. user id of the item_seq
            pretrain_item_seq: optional pretrain item sequence. It would be used to generate past_kv.
            past_kv: optional kv cache of the personalized tokens.
        """
        # TODO: handle multiple segments
        # asume the item_seq is the full item sequence, including pretrain UIH and recent UIH
        assert item_seq.shape[1] >= self.pretrain_item_num + self.recent_item_num, f"item_seq must have at least {self.pretrain_item_num + self.recent_item_num} items, but got {item_seq.shape[1]}"
        pretrain_item_seq, item_seq = (
            item_seq[:, : self.pretrain_item_num],
            item_seq[:, -self.recent_item_num:],
        )
        if not self.predict_with_full_item_seq:
            pretrain_item_seq = None

        # if pretrain_item_seq present, generate past_kv first
        if pretrain_item_seq is not None:
            assert (
                past_kv is None
            ), f"past_kv must be None when pretrain_item_seq is present"
            past_kv = self.compress_pretrain(pretrain_item_seq, user_id)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        attention_mask = self.get_attention_mask(item_seq)
        if past_kv is not None:
            # past_kv should cotains the k, v for all learnable tokens.
            cache_size = past_kv[0][0].shape[1]
            assert (
                cache_size == self.number_of_user_tokens
            ), f"past_kv must have {self.number_of_user_tokens} items, but got {cache_size}"
            # if kv present, we need to update the position embeddings to accomendate that
            # those kv are generated with [pretrain, learnable]
            position_ids += self.pretrain_item_num + self.number_of_user_tokens
            # extend attention mask: batch x 1 x seq x seq -> batch x head x seq x (cache + seq)
            attention_mask = torch.cat(
                [
                    torch.ones(
                        attention_mask.size(0),
                        attention_mask.size(1),
                        attention_mask.size(2),
                        cache_size,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=-1,
            )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_id_proj_tower(self.item_embedding(item_seq))
        item_emb = item_emb + position_embedding
        output_embs = self._hstu(
            x=item_emb, attention_mask=attention_mask, past_kv=past_kv
        )
        seq_output = output_embs[:, -1]
        seq_output = seq_output / seq_output.norm(dim=-1, keepdim=True)

        scores = torch.matmul(seq_output, item_feature.t())
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        weight = self.item_id_proj_tower(self.item_embedding.weight)
        return weight / weight.norm(dim=-1, keepdim=True)

    def get_attention_mask(
        self, item_seq, bidirectional=False, item_segments: List[Tuple[int, int]] = []
    ):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention.

        Args:
            item_segments: start and end location of each segment in item_seq. If provided,
                each segments won't be able to attend to others.
        """
        # B x seq_len
        attention_mask = item_seq != 0
        # B x 1 x 1 x seq_len (for head)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            # B x 1 x seq_len x seq_len
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
            # disallow attention from recent items to pretrain items
            # TODO: this assumes item_segments is the same across item_seq
            for seg in range(len(item_segments) - 1):
                start1, end1 = item_segments[seg]
                start2, end2 = item_segments[seg + 1]
                extended_attention_mask[
                    :, :, start2 :, start1: (end1 + 1)
                ] = False
        else:
            assert (
                len(item_segments) == 0
            ), f"item_segments is not supported for bidirectional attention mask"
        # extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask
