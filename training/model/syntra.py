# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random

import numpy as np
import torch
import torch.distributed

from syntra.modeling.backbones.dora import DoRAWrapper
from syntra.modeling.syntra_base import SynTraBase
from syntra.modeling.syntra_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)

from syntra.utils.misc import concat_points

from training.utils.data_utils import BatchedSrcTgtDatapoint


class SynTraTrain(SynTraBase):
    def __init__(
        self,
        image_encoder,
        notion_attention=None,
        freeze_image_encoder=False,
        freeze_stage1=False,
        dora_rank: int=0, # 0 for not using dora
        **kwargs,
    ):
        super().__init__(image_encoder, notion_attention, **kwargs)
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)
        self.dora_rank = dora_rank
        self.freeze_stage1 = freeze_stage1

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if freeze_stage1:
            self._freeze_stage1()
        
        self.visual_dict = {}
    
    def _freeze_stage1(self):
        for name, p in self.named_parameters():
            if "refine" not in name:
                p.requires_grad = False
        
    def dora_adapt(self):
        r = self.dora_rank
        if r > 0:
            self.image_encoder.trunk.blocks = DoRAWrapper(self.image_encoder.trunk, r)
            
            num_trainable_params = sum(
                p.numel() for p in self.image_encoder.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.image_encoder.parameters())
            assert num_trainable_params < total_params, "No parameters are frozen!"
            
            logging.info(f"DoRA applied with rank {r} for image encoder: \n" +
                         f"Trainable: {num_trainable_params}," +
                         f"Total: {total_params}, " + 
                         f"Ratio: {num_trainable_params / total_params * 100: .02f}%.")
        if self.freeze_stage1:
            self._freeze_stage1()

    def forward(self, input: BatchedSrcTgtDatapoint, **kwargs):
        # precompute image features on all frames 
        backbone_out = self.forward_image(input.img_batch)
        
        # forward heads
        src_features, src_pos_embeds = self._prepare_prompt_input(
            backbone_out
        )
        tgt_features, tgt_pos_embeds, high_res_features = self._prepare_decoder_input(
            backbone_out
        )

        output_dict = self._forward_heads(
            tgt_features, tgt_pos_embeds, 
            src_features, src_pos_embeds, 
            input.src_mask_batch.float(),
            high_res_features,
            **kwargs
        )

        self.visual_dict = {
            "img": input.img_batch,
            "pred_masks": output_dict["pred_masks_high_res"],
            "refined_pred_masks": output_dict.get("refined_pred_masks_high_res", None),
            "pred_ious": output_dict["pred_ious"],
            "src_mask": input.src_mask_batch,
            "tgt_mask": input.tgt_mask_batch,
        }

        return output_dict

    def _prepare_prompt_input(self, backbone_out):
        img_features = backbone_out['vision_features']
        src_pos_embeds = backbone_out['vision_pos_enc'][-1]
        src_features = img_features[:, 1:]
        src_pos_embeds = src_pos_embeds[:, 1:]
        return src_features, src_pos_embeds
    
    def _prepare_decoder_input(self, backbone_out):
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        fpn_feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        # feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        # take the last level as the target feature
        # NxCxHxW => BxTxCxHxW => BxCxHxW
        tgt_vision_feats = backbone_out['vision_features'][:, 0]
        tgt_pos_embeds = vision_pos_embeds[-1][:, 0]

        # High-resolution feature maps for the SAM head, NxCxHxW => BxTxCxHxW => BxCxHxW
        if len(fpn_feature_maps) > 1:
            high_res_features = fpn_feature_maps[:-1]
        else:
            high_res_features = None

        return tgt_vision_feats, tgt_pos_embeds, high_res_features
    
    def log_visuals(self, logger, global_step, phase):
        if not self.visual_dict:
            return
        # we only visualize one random batch
        batch_idx = random.randint(0, len(self.visual_dict["img"]) - 1)
        imgs = self.visual_dict["img"][batch_idx] # (T, 3, H, W)
        gt_masks = self.visual_dict["tgt_mask"][batch_idx] # (N, H, W)
        pred_masks = self.visual_dict["pred_masks"][batch_idx].sigmoid() # (N, H, W)
        # pred_ious = self.visual_dict["pred_ious"][0] # (T)
        src_mask = self.visual_dict["src_mask"][batch_idx] # (T-1, N, H, W)
        masks = torch.cat([gt_masks.unsqueeze(0), src_mask], dim=0) # (T, N, H, W)

        # unnormalize the image
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
        imgs = imgs * std + mean

        logger.log_images(
            f"visual/input", imgs[:, :, ::2, ::2], global_step, dataformats="NCHW"
        )
        for i in range(self.num_notions):
            logger.log_images(
                f"visual/gt/notion_{i}", masks[:, i:i+1][:, :, ::2, ::2], global_step, dataformats="NCHW"
            )
            logger.log_images(
                f"visual/pred/notion_{i}", pred_masks[i:i+1, ::2, ::2], global_step, dataformats="CHW"
            )

        # all target images in a batch
        logger.log_images(
            f"visual/all_targets", self.visual_dict["img"][:, 0, :, ::2, ::2], global_step, dataformats="NCHW"
        )

        refined_pred_masks = self.visual_dict.get("refined_pred_masks", None)
        if refined_pred_masks is not None:
            refined_pred_masks = refined_pred_masks[batch_idx].sigmoid()
            for i in range(self.num_notions):
                logger.log_images(
                    f"visual/refined_pred/notion_{i}", refined_pred_masks[i:i+1, ::2, ::2], global_step, dataformats="CHW"
                )


