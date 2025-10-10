# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from syntra.modeling.transformer import RoPEAttention

from syntra.modeling.syntra_utils import get_activation_fn, get_clones


class NotionAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        pos_enc_at_cross_attn_keys: bool = True,
        pos_enc_at_cross_attn_queries: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.cross_attn = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def forward(
        self,
        prompt,
        notions,
        prompt_pos: Optional[Tensor] = None,
        notion_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt = self.norm1(notions)
        tgt = self.cross_attn(
            q=tgt + notion_pos if self.pos_enc_at_cross_attn_queries else tgt,
            k=prompt + prompt_pos if self.pos_enc_at_cross_attn_keys else prompt,
            v=prompt,
            **kwds,
        )
        notions = notions + self.dropout1(tgt)

        # MLP
        tgt = self.norm2(notions)
        tgt = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        notions = notions + self.dropout3(tgt)
        return notions


class PromptAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        d_model: int,
        dim_feedforward: int,
        self_attention: nn.Module,
        dropout: float = 0.1,
        pos_enc_at_attn: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn

    def forward(
        self,
        prompt,
        prompt_pos: Optional[Tensor] = None,
    ) -> torch.Tensor:

        # Self-Attn
        tgt = self.norm1(prompt)
        q = k = tgt + prompt_pos if self.pos_enc_at_attn else tgt
        tgt = self.self_attn(q, k, v=tgt)
        prompt = prompt + self.dropout1(tgt)

        # MLP
        tgt = self.norm2(prompt)
        tgt = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        prompt = prompt + self.dropout3(tgt)
        return prompt


class NotionAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        cross_attention: nn.Module,
        self_attention: nn.Module,
        dim_feedforward: int,
        num_sa_layers: int,
        num_ca_layers: int,
        activation: str,
        pos_enc_at_attn: bool = True,  # Add pos enc at self-attn?
        pos_enc_at_cross_attn_keys: bool = True,  # Add pos enc at cross-attn keys?
        pos_enc_at_cross_attn_queries: bool = False,  # Add pos enc at cross-attn queries?
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.sa_layers = get_clones(PromptAttentionLayer(
            activation=activation,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            self_attention=self_attention,
            pos_enc_at_attn=pos_enc_at_attn,
        ), num_sa_layers)
        self.ca_layers = get_clones(NotionAttentionLayer(
            activation=activation,
            cross_attention=cross_attention,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            pos_enc_at_cross_attn_keys=pos_enc_at_cross_attn_keys,
            pos_enc_at_cross_attn_queries=pos_enc_at_cross_attn_queries,
        ), num_ca_layers)
        
        self.num_ca_layers = num_ca_layers
        self.num_ca_layers = num_ca_layers
        self.norm = nn.LayerNorm(d_model)
        self.batch_first = batch_first

    def forward(
        self,
        prompt_emb: torch.Tensor,  # self-attention inputs
        notion_emb: torch.Tensor,  # cross-attention inputs
        prompt_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        notion_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
    ):
        if not self.batch_first:
            # Convert to seq first
            prompt_emb = prompt_emb.transpose(0, 1)
            notion_emb = notion_emb.transpose(0, 1)
            if prompt_pos is not None:
                prompt_pos = prompt_pos.transpose(0, 1)
            if notion_pos is not None:
                notion_pos = notion_pos.transpose(0, 1)

        # Self-Attention layers on prompt embeddings
        # for layer in self.sa_layers:
        #     prompt_emb = layer(prompt=prompt_emb, prompt_pos=prompt_pos)
        # Cross-Attention layers on notion embeddings
        for layer in self.ca_layers:
            kwds = {}
            if isinstance(layer.cross_attn, RoPEAttention):
                kwds = {"num_k_exclude_rope": 0}

            notion_emb = layer(
                prompt=prompt_emb,
                notions=notion_emb,
                prompt_pos=prompt_pos,
                notion_pos=notion_pos,
                **kwds,
            )
        normed_output = self.norm(notion_emb)

        if not self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
