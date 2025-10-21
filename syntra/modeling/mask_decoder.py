# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type
from copy import deepcopy

import torch
from torch import nn

from syntra.modeling.syntra_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        pre_max_pool_dense_prompt: bool = True,
        use_global_tokens_as_dense_prompt_embeddings: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.pre_max_pool_dense_prompt = pre_max_pool_dense_prompt
        self.use_global_tokens_as_dense_prompt_embeddings = (
            use_global_tokens_as_dense_prompt_embeddings
        )

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlp = MLP(
            transformer_dim, transformer_dim, transformer_dim // 8, 3
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        notion_embeddings: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor] = None,
        high_res_features: Optional[List[torch.Tensor]] = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): image embeddings from
            the image encoder, of shape [b, c, h, w]
          image_pe (torch.Tensor): positional encoding for the image
          notion_embeddings (torch.Tensor): prompt embeddings from
            the prompt encoder, of shape [b, n, c]
          high_res_features (list(torch.Tensor), optional): high
            resolution feature maps from the image encoder of shape
            [b, c, h', w'] and [b, c, h'', w''], where h' > h and
            w' > w, h'' > h' and w'' > w'. These are used to
            produce higher-resolution masks.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            notion_embeddings=notion_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            high_res_features=high_res_features,
        )

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor, # BxNxCxHxW
        image_pe: torch.Tensor,
        notion_embeddings: torch.Tensor, # BxNxTxNtxC
        dense_prompt_embeddings: Optional[torch.Tensor] = None, # BxNxTxCxHxW
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        
        B, N, T, Nt, C = notion_embeddings.shape # BxNxTxNtxC
        output_tokens, s = self._get_output_tokens()

        if not self.pre_max_pool_dense_prompt:
            tokens = torch.cat((output_tokens.unsqueeze(0).expand(B*N*T, -1, -1), # (B*N*T)x2x256 or (B*N)xTx3x256
                                notion_embeddings.flatten(0, 2)), 
                                dim=1) # (B*N*T)x(2+Nt)x256 or (B*N*T)x(3+Nt)x256
            # repeat target image embeddings and pos emb
            tgt = torch.repeat_interleave(image_embeddings.unsqueeze(1), N*T, dim=1).flatten(0, 1) # (B*N*T)xCxHxW
            if dense_prompt_embeddings is not None:
                dense_prompt_embeddings = dense_prompt_embeddings.flatten(0, 2)  # (B*N*T)xCxHxW
        else:
            tokens = torch.cat((output_tokens.unsqueeze(0).expand(B*N, -1, -1), # (B*N)x2x256 or (B*N)x3x256
                                notion_embeddings.flatten(0, 1).flatten(1, 2)), 
                               dim=1) # (B*N)x(2+Nt*T)x256 or (B*N)x(3+Nt*T)x256
            # repeat target image embeddings and pos emb
            tgt = torch.repeat_interleave(image_embeddings.unsqueeze(1), N, dim=1).flatten(0, 1) # (B*N)xCxHxW
            if dense_prompt_embeddings is not None:
                dense_prompt_embeddings = dense_prompt_embeddings.flatten(0, 1).max(dim=1).values  # (B*N)xCxHxW
        
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"       
        pos_tgt = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) # (B*N)x256xHxW or (B*N*T)x256xHxW
        b, c, h, w = tgt.shape # b=(B*N) or (B*N*T)

        if dense_prompt_embeddings is not None:
            if self.use_global_tokens_as_dense_prompt_embeddings:
                # process dense prompt embeddings to get global notion features
                tgt = self.deepen_image_feature(tgt) # (B*N*T)x(4C)xHxW
                global_notion_features = self.global_notion_feature_layer(dense_prompt_embeddings) # (B*N*T)x(4C)x1x1
                tgt = tgt + global_notion_features
                tgt = self.recover_image_feature_depth(tgt) # (B*N*T)xCxHxW
            else:
                tgt = tgt + dense_prompt_embeddings
        # Run the transformer
        hs, tgt = self.transformer(tgt, pos_tgt, tokens)


        # Upscale mask embeddings and predict masks using the mask tokens
        tgt = tgt.transpose(1, 2).view(b, c, h, w) # bx(HW)xC -> bxCxHxW

        # upscaled_embedding = self.output_upscaling(tgt) # bxC'xH'xW'
        if not self.use_high_res_features:
            upscaled_embedding = self._get_upscaled_embedding(tgt)
        else:
            upscaled_embedding = self._get_upscaled_embedding(tgt, high_res_features)

        res = self._generate_output(hs, upscaled_embedding, s, B, N, T)

        return res

    def _get_upscaled_embedding(self, tgt, high_res_features: Optional[List[torch.Tensor]] = None):
        if high_res_features is None:
            upscaled_embedding = self.output_upscaling(tgt)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            repeat_n = tgt.size(0) // feat_s0.size(0)
            feat_s0 = torch.repeat_interleave(feat_s0.unsqueeze(1), repeat_n, dim=1).flatten(0, 1) # bxCxHxW
            feat_s1 = torch.repeat_interleave(feat_s1.unsqueeze(1), repeat_n, dim=1).flatten(0, 1)
            upscaled_embedding = act1(ln1(dc1(tgt) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)
        return upscaled_embedding
    
    def _get_output_tokens(self):
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight, # 1x256
                    self.iou_token.weight, # 1x256
                    self.mask_tokens.weight, # 1x256
                ],
                dim=0,
            ) # 3x256
            s = 1
        else:
            output_tokens = torch.cat(
                [
                    self.iou_token.weight, # 1x256
                    self.mask_tokens.weight, # 1x256
                ], dim=0
            ) # 2x256
        return output_tokens, s
    
    def _generate_output(self, hs, upscaled_embedding, s, B, N, T):
        iou_token_out = hs[:, s, :] # (B*N)x256 or (B*N*T)x256
        mask_token_out = hs[:, s+1, :] # (B*N)x256 or (B*N*T)x256

        hyper_in = self.output_hypernetworks_mlp(mask_token_out).unsqueeze(1) # bx1x32
        b, c, h, w = upscaled_embedding.shape
        # bx1x32 @ bx32x(HW) -> bx1x(HW) -> BxNxHxW
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w))
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, s, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(*iou_pred.shape)

        if self.pre_max_pool_dense_prompt:
            masks = masks.view(B, N, h, w)
            iou_pred = iou_pred.view(B, N) # BxN
            object_score_logits = object_score_logits.view(B, N) # BxN
        else:
            masks = masks.view(B, N, T, h, w).max(dim=2).values # BxNxHxW   
            iou_pred = iou_pred.view(B, N, T).max(dim=2).values # BxN
            object_score_logits = object_score_logits.view(B, N, T).max(dim=2).values # BxN
        
        return masks, iou_pred, mask_token_out, object_score_logits



class MaskRefineDecoder(MaskDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        transformer_dim = self.transformer_dim
        activation = kwargs.get("activation", nn.GELU)
        self.mask_downscale = nn.Sequential(
            nn.Conv2d(1, transformer_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 8),
            activation(),
            nn.Conv2d(transformer_dim // 8, transformer_dim, kernel_size=2, stride=2),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        masks: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dense_prompt_embeddings = self.mask_downscale(masks)  # Bx1xHxW -> BxCxH'xW'
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            dense_prompt_embeddings=dense_prompt_embeddings,
            high_res_features=high_res_features,
        )

        return masks, iou_pred, mask_tokens_out, object_score_logits
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor, # BxNxCxHxW
        image_pe: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor] = None, # BxNxTxCxHxW
        high_res_features: Optional[List[torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        B = image_embeddings.size(0)
        N = dense_prompt_embeddings.size(0) // B
        output_tokens, s = self._get_output_tokens()
        tokens = output_tokens.unsqueeze(0).expand(B*N, -1, -1) # (B*N)x2x256 or (B*N)x3x256
        # repeat target image embeddings and pos emb
        tgt = torch.repeat_interleave(image_embeddings.unsqueeze(1), N, dim=1).flatten(0, 1) # (B*N)xCxHxW
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"       
        pos_tgt = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) # (B*N)x256xHxW or (B*N*T)x256xHxW
        b, c, h, w = tgt.shape # b=(B*N) or (B*N*T)

        tgt = tgt + dense_prompt_embeddings
        # Run the transformer
        hs, tgt = self.transformer(tgt, pos_tgt, tokens)

        # Upscale mask embeddings and predict masks using the mask tokens
        tgt = tgt.transpose(1, 2).view(b, c, h, w) # bx(HW)xC -> bxCxHxW

        # upscaled_embedding = self.output_upscaling(tgt) # bxC'xH'xW'
        upscaled_embedding = self._get_upscaled_embedding(tgt, high_res_features)

        res = self._generate_output(hs, upscaled_embedding, s, B, N)

        return res
    
    def _generate_output(self, hs, upscaled_embedding, s, B, N):
        iou_token_out = hs[:, s, :] # (B*N)x256 or (B*N*T)x256
        mask_token_out = hs[:, s+1, :] # (B*N)x256 or (B*N*T)x256

        hyper_in = self.output_hypernetworks_mlp(mask_token_out).unsqueeze(1) # bx1x32
        b, c, h, w = upscaled_embedding.shape
        # bx1x32 @ bx32x(HW) -> bx1x(HW) -> BxNxHxW
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w))

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, s, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(*iou_pred.shape)

        masks = masks.view(B, N, h, w)
        iou_pred = iou_pred.view(B, N) # BxN
        object_score_logits = object_score_logits.view(B, N) # BxN
        
        return masks, iou_pred, mask_token_out, object_score_logits