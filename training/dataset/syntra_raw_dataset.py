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
    JSONSegmentLoader,
    MultiplePNGSegmentLoader,
    SynTraSegmentLoader,
    SA1BSegmentLoader,
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


class SA1BRawDataset(SynTraRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_targets_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # no filtering by default
        uncertain_iou=-1,  # no filtering by default
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou  # stability score

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [
                path.split(".")[0] for path in subset if path.endswith(".jpg")
            ]  # remove extension

        # Read and process excluded files if provided
        if excluded_targets_list_txt is not None:
            with g_pathmgr.open(excluded_targets_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files and it exists
        self.target_names = [
            target_name for target_name in subset if target_name not in excluded_files
        ]

    def get_target(self, idx):
        """
        Given a SynTratarget object, return the mask tensors.
        """
        target_name = self.target_names[idx]

        target_frame_path = os.path.join(self.img_folder, target_name + ".jpg")
        target_mask_path = os.path.join(self.gt_folder, target_name + ".json")

        segment_loader = SA1BSegmentLoader(
            target_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            target_frame_path=target_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(SynTraFrame(frame_idx, image_path=target_frame_path))
        target_name = target_name.split("_")[-1]  # filename is sa_{int}
        # target id needs to be image_id to be able to load correct annotation file during eval
        target = SynTraMatches(target_name, int(target_name), frames)
        return target, segment_loader

    def __len__(self):
        return len(self.target_names)


class JSONRawDataset(SynTraRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_targets_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.sample_rate = sample_rate
        self.rm_unannotated = rm_unannotated
        self.ann_every = ann_every
        self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_targets_list_txt is not None:
            if isinstance(excluded_targets_list_txt, str):
                excluded_targets_lists = [excluded_targets_list_txt]
            elif isinstance(excluded_targets_list_txt, ListConfig):
                excluded_targets_lists = list(excluded_targets_list_txt)
            else:
                raise NotImplementedError

            for excluded_targets_list_txt in excluded_targets_lists:
                with open(excluded_targets_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.target_names = sorted(
            [target_name for target_name in subset if target_name not in excluded_files]
        )

    def get_target(self, target_idx):
        """
        Given a SynTratarget object, return the mask tensors.
        """
        target_name = self.target_names[target_idx]
        target_json_path = os.path.join(self.gt_folder, target_name + "_manual.json")
        segment_loader = JSONSegmentLoader(
            target_json_path=target_json_path,
            ann_every=self.ann_every,
            frames_fps=self.frames_fps,
        )

        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, target_name))
            )
        ]

        frames = [
            SynTraFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{target_name}/%05d.jpg" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        target = SynTraMatches(target_name, target_idx, frames)
        return target, segment_loader

    def __len__(self):
        return len(self.target_names)
