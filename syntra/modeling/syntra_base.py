# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from syntra.modeling.mask_decoder import MaskDecoder
from syntra.modeling.mask_decoder_v2 import MaskDecoder as MaskDecoderV2
from syntra.modeling.prompt_encoder import PromptEncoder
from syntra.modeling.transformer import TwoWayTransformer
from syntra.modeling.syntra_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SynTraBase(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        notion_attention,
        num_notions=4,  # default 1 input frame + 6 previous frames
        num_tokens_per_notion=4,  # default 4 tokens to represent each notion
        image_size=384,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_notion_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_notion_enc=0.0,  # bias factor for mask sigmoid prob
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features=False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Whether to use self-attention on the key features at the output of the Mask decoder
        self_attention_at_decoder_output: bool = False,
        use_dense_prompt_embeddings: bool = True,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features = use_high_res_features
        self.num_feature_levels = 3 if use_high_res_features else 1

        # Part 2: notion attention to condition current frame's visual features
        # with notions from retrieved frames
        self.notion_attention = notion_attention
        self.hidden_dim = image_encoder.neck.d_model
        self.num_notions = num_notions  # Number of notions
        self.num_tokens_per_notion = num_tokens_per_notion
        self.use_dense_prompt_embeddings = use_dense_prompt_embeddings
       
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_notion_enc = sigmoid_scale_for_notion_enc
        self.sigmoid_bias_for_notion_enc = sigmoid_bias_for_notion_enc

        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 3: SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.self_attend_key_at_output = self_attention_at_decoder_output

        self._build_heads()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )
    

    def _build_heads(self):
        """Build prompt encoder and mask decoder."""
        self.prompt_embed_dim = self.hidden_dim
        self.image_embedding_size = self.image_size // self.backbone_stride

        self.prompt_encoder = PromptEncoder(
            embed_dim=self.prompt_embed_dim,
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            notion_attention=self.notion_attention,
            num_notion_embeddings=self.num_notions,
            num_tokens_per_notion=self.num_tokens_per_notion
            use_dense_embeddings=self.use_dense_prompt_embeddings,
        )

        self.mask_decoder = MaskDecoder(
                transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.hidden_dim,
                mlp_dim=2048,
                num_heads=8,
                self_attend_key_at_output=self.self_attend_key_at_output,
            ),
            transformer_dim=self.hidden_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please define the forward method in the subclass."
        )

    def _forward_heads(self, tgt_features, tgt_pos_embeds, 
                       src_features, src_pos_embeds, src_mask, 
                       high_res_features=None):
        # embed prompt image-label paris
        notions, dense_prompt_embeddings = self.prompt_encoder(
            src_features, src_pos_embeds, src_mask
        ) # B x N_notion x N_token x C
        tgt_pos = self.prompt_encoder.get_dense_pe()
        # Cross attention between tgt features and notion embeddings
        low_res_masks, iou_pred, mask_tokens_out, object_score_logits = self.mask_decoder(
            tgt_features, tgt_pos, notions, dense_prompt_embeddings, high_res_features
        )
        # if self.pred_obj_scores:
        #     is_obj_appearing = object_score_logits > 0 # BxN

        #     # Mask used for spatial memories is always a *hard* choice between obj and no obj,
        #     # consistent with the actual mask prediction
        #     low_res_masks = torch.where(
        #         is_obj_appearing[..., None, None],
        #         low_res_masks,
        #         NO_OBJ_SCORE,
        #     ) # BxNxHxW
        
        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_masks = low_res_masks.float()
        high_res_masks = F.interpolate(
            low_res_masks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "pred_ious": iou_pred,
            "object_score_logits": object_score_logits,
        }

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        B, T = img_batch.shape[:2]
        backbone_out = self.image_encoder(img_batch.flatten(0, 1))
        # convert NxCxHxW to BxTxCxHxW
        for k, v in backbone_out.items():
            if k in ["backbone_fpn", "vision_pos_enc"]:
                backbone_out[k] = [x.view(B, T, *x.shape[1:]) for x in v]
            else:
                backbone_out[k] = v.view(B, T, *v.shape[1:])
        if self.use_high_res_features:
            # precompute projected level 0 and level 1 features in Mask decoder
            # to avoid running it again on every click
            # only target features are needed in mask decoder
            backbone_out["backbone_fpn"][0] = self.mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0][:, 0]
            )
            backbone_out["backbone_fpn"][1] = self.mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1][:, 0]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

