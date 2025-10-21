# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import random
from copy import deepcopy

import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.syntra_raw_dataset import SynTraRawDataset
from training.dataset.syntra_sampler import SynTraSampler
from training.dataset.syntra_segment_loader import JSONSegmentLoader

from training.utils.data_utils import Frame, Notion, SrcTgtDatapoint

MAX_RETRIES = 100


class SynTraDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        syntra_dataset: SynTraRawDataset,
        sampler: SynTraSampler,
        multiplier: int,
        notion_size: int = 3,
        pos_sample_prob: float = 0.75,
        target_segments_available=True,
    ):
        self._transforms = transforms
        self.training = training
        self.sampler = sampler
        self.syntra_dataset = syntra_dataset
        self.notion_size = notion_size
        self.pos_sample_prob = pos_sample_prob

        self.repeat_factors = torch.ones(len(self.syntra_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier
        print(f"Raw dataset length = {len(self.syntra_dataset)}")

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.target_segments_available = target_segments_available
            

    def _get_datapoint(self, idx):

        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                # sample a target image
                target, segment_loader = self.syntra_dataset.get_target(idx)
                # sample frames (src+tgt) to be used in a datapoint
                sampled_frames = self.sampler.sample(
                    target, segment_loader, epoch=self.curr_epoch
                )
                break  # Succesfully loaded syntra datapoint
            except Exception as e:
                if self.training:
                    # logging.warning(
                    #     f"Loading failed (id={idx}); Retry {retry} with exception: {e}"
                    # )
                    idx = random.randrange(0, len(self.syntra_dataset))
                else:
                    # Shouldn't fail to load a val sample
                    raise e

        datapoint = self.construct(target, sampled_frames, segment_loader)
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def construct(self, target, sampled_frames, segment_loader):
        """
        Constructs a SrcTgtDatapoint sample to pass to transforms
        """

        images = []
        rgb_images = load_images(sampled_frames)
        sampled_colors = []
        sampled_notion_ids = []
        target_notion_ids = []

        # Iterate over the sampled frames and store their rgb data and segment data 
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size
            images.append(
                Frame(
                    sample_name=frame.frame_id,
                    data=rgb_images[frame_idx],
                    notions=[],
                )
            )
            # We load the gt segments (notions) associated with the current frame
            segments, cls_id_to_colors = segment_loader.load(frame.frame_id)
            
            for cls_id, cls_color in cls_id_to_colors.items():
                # update visible colors in all frames
                if cls_color not in sampled_colors:
                    sampled_colors.append(cls_color)
                # Extract the segment
                segment = segments[cls_id].to(torch.uint8)
                # ignore the src segment if the current foregaround area is too small
                if frame_idx > 0 and segment.sum() < 0.005 * (h * w):
                    continue
                # segment = torch.zeros(h, w, dtype=torch.uint8)

                mapped_cls_id = sampled_colors.index(cls_color)
                images[frame_idx].notions.append(
                    Notion(
                        cls_id=mapped_cls_id,
                        frame_id=frame.frame_id,
                        segment=segment,
                        color=cls_color,
                    )
                )
                if frame_idx > 0:
                    # only sample notions from source frames
                    sampled_notion_ids.append(mapped_cls_id)
                else:
                    target_notion_ids.append(mapped_cls_id)
        
        sampled_notion_ids = list(set(sampled_notion_ids))
        if len(sampled_notion_ids) > self.notion_size:
            # positive ids are these that are both in the source and target image
            # negative ids are these that are only in the source image
            pos_ids = list(set(sampled_notion_ids) & set(target_notion_ids))
            # assign a sampling weight to each id
            sampled_notion_id_weights = [self.pos_sample_prob if x in pos_ids else (1 - self.pos_sample_prob) for x in sampled_notion_ids]
            # Randomly sample a subset of notion ids to fit into notion_size according to the sampling weights
            sampled_notion_ids = random.choices(sampled_notion_ids, weights=sampled_notion_id_weights, k=self.notion_size)
            # Filter out notions in each frame that are not in the sampled_notion_ids
            for frame in images:
                frame.notions = [n for n in frame.notions if n.cls_id in sampled_notion_ids]

        # shuffle the notion ids and colors
        random.shuffle(sampled_notion_ids)
        sampled_colors = [sampled_colors[i] for i in sampled_notion_ids]

        return SrcTgtDatapoint(
            frames=images,
            target_id=target.target_id,
            valid_src_notion_ids=sampled_notion_ids,
            valid_src_notion_colors=sampled_colors,
            notion_size=self.notion_size,
            size=(h, w),
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.syntra_dataset)


def load_images(frames):
    all_images = []
    cache = {}
    for frame in frames:
        if frame.data is None:
            # Load the frame rgb data from file
            path = frame.image_path
            if path in cache:
                all_images.append(deepcopy(all_images[cache[path]]))
                continue
            with g_pathmgr.open(path, "rb") as fopen:
                all_images.append(PILImage.open(fopen).convert("RGB"))
            cache[path] = len(all_images) - 1
        else:
            # The frame rgb data has already been loaded
            # Convert it to a PILImage
            all_images.append(tensor_2_PIL(frame.data))

    return all_images


def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)
