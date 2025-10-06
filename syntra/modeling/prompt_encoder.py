# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type

import torch
from torch import nn

from syntra.modeling.position_encoding import PositionEmbeddingRandom

from syntra.modeling.syntra_utils import LayerNorm2d, get_activation_fn


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        notion_attention: nn.Module,
        num_notion_embeddings: int = 4,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.num_notion_embeddings = num_notion_embeddings 

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.notion_embeddings = nn.Embedding(num_notion_embeddings, embed_dim) 

        # downscale the input mask to the image embedding size
        mask_downscaling = []
        for i in range(4):
            in_dim = 1 if i == 0 else 2**(i+1)
            out_dim = 2**(i+2)
            mask_downscaling.extend([
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                LayerNorm2d(out_dim),
                activation(),
            ])
        mask_downscaling.append(nn.Conv2d(out_dim, embed_dim, kernel_size=1))
        self.mask_downscaling = nn.Sequential(*mask_downscaling)

        self.notion_attention = notion_attention

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _get_device(self) -> torch.device:
        return self.notion_embeddings[0].weight.device

    def forward(
        self,
        src_emb: torch.Tensor,
        src_pos_emb: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Learn notion features from the source embeddings and masks.
        Arguments:
          src_emb (torch.Tensor): source image embeddings, in the shape
            BxCxHxW
          masks (torch.Tensor): binary masks of the notions, in the shape
            BxMxHxW, where M is the number of masks for each src image.
        Returns:
          torch.Tensor: notion features, in the shape BxNxC, where N is
            the number of notions.
        """

        B, T, N = masks.shape[:3]
        H, W = src_emb.shape[-2:]
        L = T * H * W
        mask_embeddings = self.mask_downscaling(masks.flatten(0, 2).unsqueeze(1))  # (B*T*N)xCxhxw
        mask_embeddings = mask_embeddings.view(B, T, N, self.embed_dim, *self.image_embedding_size)

        # merge src_emb and mask_embeddings 
        prompt_embeddings = src_emb.unsqueeze(2) * mask_embeddings
        prompt_embeddings = prompt_embeddings.permute(0, 2, 1, 4, 5, 3) # BxNxTxHxWxC
        prompt_embeddings = prompt_embeddings.flatten(0, 1).flatten(1, 3) # (B*N)x(L)xC, L=T*H*W

        src_pos_emb = src_pos_emb.unsqueeze(2).repeat(1, 1, N, 1, 1, 1) # BxTxNxHxWxC
        src_pos_emb = src_pos_emb.permute(0, 2, 1, 4, 5, 3) # BxNxTxHxWxC
        src_pos_emb = src_pos_emb.flatten(0, 1).flatten(1, 3) # (B*N)x(L)xC, L=T*H*W

        # process notion embeddings with notion attention
        notion_embeddings = self.notion_embeddings.weight.unsqueeze(0).repeat(B, 1, 1).unsqueeze(2).flatten(0, 1)  # (B*N)x1xC
        notions = self.notion_attention(
            prompt_embeddings,
            notion_embeddings,
            src_pos_emb,
        )

        notions = notions.squeeze(1).view(B, N, self.embed_dim)

        return notions
    
  
