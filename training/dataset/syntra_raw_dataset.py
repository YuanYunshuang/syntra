# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import logging
import os
from dataclasses import dataclass

from typing import List, Optional, Iterable

import pandas as pd

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset.syntra_segment_loader import (
    SynTraSegmentLoader,
)


@dataclass
class SynTraFrame:
    frame_id: int
    image_path: str
    mask_path: Optional[str] = None
    visible_cls_colors: Optional[List[Iterable[int]]] = None
    data: Optional[torch.Tensor] = None


@dataclass
class SynTraMatches:
    target_id: int
    frames: List[SynTraFrame]

    def __len__(self):
        return len(self.frames)


class SynTraRawDataset:
    def __init__(self):
        pass

    def get_target(self, idx):
        raise NotImplementedError()


class PNGRawDataset(SynTraRawDataset):
    def __init__(
        self,
        root_folder,
        data_info_file,
        is_palette=True,
    ):
        self.root_folder = root_folder
        self.dataset_name = os.path.basename(root_folder)
        self.is_palette = is_palette

        # Read the subset defined in data_info_file
        with g_pathmgr.open(os.path.join(root_folder, data_info_file), "r") as f:
            self.dinfo = json.load(f)
            self.target_names = [x for x in self.dinfo.keys()]
        
        self.img_dir = os.path.join(self.root_folder, 'imgs')
        self.lbl_dir = os.path.join(self.root_folder, 'lbls')

    def get_target(self, idx):
        """
        Given a SynTraTarget object index, return the mask tensors.
        """
        target_name = self.target_names[idx]
        dataset, name = target_name.rsplit(".", 1)
        assert dataset == self.dataset_name, f"Dataset name mismatch: {dataset} vs {self.dataset_name}"

        if self.is_palette:
            segment_loader = SynTraSegmentLoader(self.lbl_dir)
        else:
            raise NotImplementedError(
                "Only palettised png segment is implemented."
            )

        all_srcs = sorted(self.dinfo[target_name]["src_samples"])

        frames = [
            SynTraFrame(
                target_name,
                image_path=os.path.join(self.img_dir, f"{name}.png"),
                mask_path=os.path.join(self.lbl_dir, f"{name}.png"),
        )
        ]
        for _, fid in enumerate(all_srcs):
            cur_name = fid.rsplit(".", 1)[-1]
            frames.append(SynTraFrame(
                fid, 
                image_path=os.path.join(self.img_dir, f"{cur_name}.png"),
                mask_path=os.path.join(self.lbl_dir, f"{cur_name}.png"),
            ))

        target = SynTraMatches(target_id=0, frames=frames)
        return target, segment_loader

    def __len__(self):
        return len(self.target_names)


