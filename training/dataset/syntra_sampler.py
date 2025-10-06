# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List

from training.dataset.syntra_segment_loader import LazySegments

MAX_RETRIES = 1000


@dataclass
class SampledFramesAndClasses:
    frames: List
    class_ids: List[int]
    cls_id_to_color: dict


class SynTraSampler:
    def __init__(self, sort_sources=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_sources = sort_sources

    def sample(self, target):
        raise NotImplementedError()


class RandomUniformSampler(SynTraSampler):
    def __init__(
        self,
        num_src: int,
        max_num_cls: int,
        sort_sources: bool = True,
    ):
        super().__init__(sort_sources)
        self.num_src = num_src
        self.max_num_cls = max_num_cls

    def sample(self, target, segment_loader, epoch=None):

        if len(target.frames) - 1 < self.num_src:
            raise Exception(
                f"Cannot sample {self.num_src} sources for target {target.target.frame_id} " +
                f"as it only has {len(target)} selected neighborhood sources."
            )
        # sample source frames
        if self.sort_sources:
            # ordered by frame id
            sources = sorted(target.frames[1:], key=lambda x: x.frame_id)
        else:
            # use the original order
            sources = target.frames[1:]
        sampled_sources = random.sample(sources, self.num_src)
        frames = [target.frames[0]] + sampled_sources

        return frames


class EvalSampler(SynTraSampler):
    """
    VOS Sampler for evaluation: sampling all the frames and all the objects in a video
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(self, target, segment_loader, epoch=None):
        """
        Sampling all the frames and all the objects
        """
        if self.sort_sources:
            # ordered by frame id
            sources = sorted(target.sources, key=lambda x: x.frame_id)
        else:
            # use the original order
            sources = target.sources
        cls_ids = segment_loader.load(target.target.frame_id).keys()
        if len(cls_ids) == 0:
            raise Exception("Target frame has no objects")

        return SampledSourcesAndClasses(sources=target.sources, class_ids=cls_ids)
