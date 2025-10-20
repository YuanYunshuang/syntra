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
        num_tokens_per_notion: int,
        notion_attention: nn.Module,
        num_notion_embeddings: int = 4,
        use_dense_embeddings: bool = True,
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
          num_tokens_per_notion (int): The number of tokens to prepresent each notion
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.num_notion_embeddings = num_notion_embeddings 
        self.num_tokens_per_notion = num_tokens_per_notion
        self.use_dense_embeddings = use_dense_embeddings

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.notion_embeddings = nn.Embedding(num_tokens_per_notion, embed_dim) 

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
        
        self.merge_image_mask_pair = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            LayerNorm2d(embed_dim),
            activation(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

        self.notion_attention = None
        if num_tokens_per_notion > 0:
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
            BxTxCxHxW
          masks (torch.Tensor): binary masks of the notions, in the shape
            BxTxNxHxW, where M is the number of masks for each src image.
        Returns:
          torch.Tensor: notion features, in the shape BxNxC, where N is
            the number of notions.
        """

        B, T, N = masks.shape[:3]
        H, W = src_emb.shape[-2:]
        L = H * W
        mask_embeddings = self.mask_downscaling(masks.flatten(0, 2).unsqueeze(1))  # (B*T*N)xCxhxw
        mask_embeddings = mask_embeddings.view(B, T, N, self.embed_dim, H, W)

        # from PIL import Image
        # for b in range(B):
        #     for t in range(T):
        #         cur_masks = masks[b, t]
        #         cur_src = src_emb[b, t].detach().max(dim=0).values
        #         cur_src = torch.nn.functional.interpolate(
        #             cur_src.unsqueeze(0).unsqueeze(0), size=self.input_image_size, mode="bilinear", align_corners=False
        #         ).squeeze()
        #         cur_src = (cur_src - cur_src.min()) / (cur_src.max() - cur_src.min())
        #         Image.fromarray((cur_masks[:3].permute(1, 2, 0).cpu().numpy()*255).astype('uint8')).save('/home/yuan/Downloads/mask.png')
        #         Image.fromarray((cur_src * 255).cpu().numpy().astype('uint8')).save('/home/yuan/Downloads/src.png')
        #         print('save mask and src image')

        # merge src_emb and mask_embeddings 
        prompt_embeddings = src_emb.unsqueeze(2) * mask_embeddings
        prompt_embeddings = self.merge_image_mask_pair(prompt_embeddings.flatten(0, 2)) # (B*T*N)xCxHxW
        prompt_embeddings = prompt_embeddings.view(B, T, N, self.embed_dim, H, W) # BxTxNxCxHxW
        dense_embeddings = prompt_embeddings.permute(0, 2, 1, 3, 4, 5) if self.use_dense_embeddings else None # BxNxTxCxHxW

        prompt_embeddings = prompt_embeddings.permute(0, 2, 1, 4, 5, 3) # BxNxTxHxWxC
        prompt_embeddings = prompt_embeddings.flatten(0, 2).flatten(1, 2) # (B*N*T)x(L)xC, L=H*W

        # process notion embeddings with notion attention
        # Nt, C -> 1xNtxC -> (B*N*T)xNtxC
        notion_embeddings = self.notion_embeddings.weight.unsqueeze(0).repeat(B*N*T, 1, 1)
        if self.num_tokens_per_notion == 0:
            return notion_embeddings.view(B, N, T, 0, self.embed_dim), dense_embeddings
        
        # 1xCxHxW -> (B*N*T)x(L)xC, L=H*W
        pos_emb = self.get_dense_pe().flatten(2).permute(0, 2, 1)
        pos_emb = pos_emb.repeat(B*N*T, 1, 1)
        notions = self.notion_attention(
            prompt_embeddings,
            notion_embeddings,
            pos_emb,
        ) # (B*N*T)xNtxC

        notions = notions.view(B, N, T, self.num_tokens_per_notion, self.embed_dim) # BxNxTxNtxC

        return notions, dense_embeddings
    
  
