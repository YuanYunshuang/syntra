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
        always_target=True,
        target_segments_available=True,
    ):
        self._transforms = transforms
        self.training = training
        self.sampler = sampler
        self.syntra_dataset = syntra_dataset

        self.repeat_factors = torch.ones(len(self.syntra_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier
        print(f"Raw dataset length = {len(self.syntra_dataset)}")

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target
        self.target_segments_available = target_segments_available
            

    def _get_datapoint(self, idx):

        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                # sample a target image
                target, segment_loader = self.syntra_dataset.get_target(idx)
                # sample frames and object indices to be used in a datapoint
                sampled_frames_and_classes = self.sampler.sample(
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

        datapoint = self.construct(target, sampled_frames_and_classes, segment_loader)
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def construct(self, target, sampled_frames_and_classes, segment_loader):
        """
        Constructs a SrcTgtDatapoint sample to pass to transforms
        """
        sampled_frames = sampled_frames_and_classes.franmes
        sampled_classes = sampled_frames_and_classes.class_ids
        sampled_cls_id_to_color = sampled_frames_and_classes.cls_id_to_color
        sampled_colors = [x for x in sampled_cls_id_to_color.values()]

        images = []
        rgb_images = load_images(sampled_frames)
        # Iterate over the sampled frames and store their rgb data and segment data 
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size
            images.append(
                Frame(
                    data=rgb_images[frame_idx],
                    notions=[],
                )
            )
            # We load the gt segments associated with the current frame
            segments, cls_id_to_colors = segment_loader.load(frame.frame_id)
            cur_colors = [x for x in cls_id_to_colors.values()]
            for cls_id, cls_color in sampled_cls_id_to_color.items():
                # Extract the segment
                if cls_color in cur_colors:
                    assert (
                        segments[cls_id] is not None
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = segments[cls_id].to(torch.uint8)
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.uint8)

                images[frame_idx].notions.append(
                    Notion(
                        cls_id=cls_id,
                        frame_id=frame.frame_id,
                        segment=segment,
                    )
                )
        return SrcTgtDatapoint(
            frames=images,
            target_id=0,
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
