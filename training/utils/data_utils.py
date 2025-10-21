# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
import random
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedSrcTgtMetaData:
    """
    This class represents metadata about a batch of target images with paired source images.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedSrcTgtDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    src_mask_batch: torch.BoolTensor
    tgt_mask_batch: torch.BoolTensor
    metadata: Optional[BatchedSrcTgtMetaData]
    sample_names: Optional[List[List[str]]]
    notion_colors: Optional[List[List[Tuple[int, int, int]]]]

    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of images per tgt-src bundle.
        """
        return self.batch_size[0]

    @property
    def num_tgt(self) -> int:
        """
        Returns the number of tgt images in the batch.
        """
        return self.tgt_mask_batch.shape[1]
    

    @property
    def num_src(self) -> int:
        """
        Returns the number of tgt images in the batch.
        """
        return self.src_mask_batch.shape[1]

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.flatten(0, 1)
    
    @property
    def flat_src_mask_batch(self) -> torch.BoolTensor:
        """
        Returns a flattened src_mask_batch of shape [(B*(T-1))*Nx1xHxW]
        """

        return self.src_mask_batch.flatten(0, 2).unsqueeze(1)
    
    @property
    def input_size(self) -> Tuple[int, int]:
        """
        Returns the height and width of the input images.
        """
        return self.img_batch.shape[-2], self.img_batch.shape[-1]


@dataclass
class Notion:
    cls_id: int
    frame_id: str
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask
    color: Tuple[int, int, int]


@dataclass
class Frame:
    sample_name: str
    data: Union[torch.Tensor, PILImage.Image]
    notions: List[Notion]


@dataclass
class SrcTgtDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    target_id: int
    notion_size: int
    valid_src_notion_ids: List[int]
    valid_src_notion_colors: List[Tuple[int, int, int]]
    size: Tuple[int, int]


def visualize_batch(batch: BatchedSrcTgtDatapoint):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    # colors for different classes
    colors = [
    (34, 139, 34),    # 'wald' (forest): Forest Green
    (124, 252, 0),    # 'grünland' (grassland): Lawn Green
    (220, 20, 60),    # 'siedlung' (settlement): Crimson
    (30, 144, 255),   # 'fließgewässer' (flowing water): Dodger Blue
    (0, 191, 255)     # 'stillgewässer' (still water): Deep Sky Blue
    ]

    B = len(batch)
    T = len(batch[0].frames)
    # each batch contains T images and T label masks, the 1st in T is the target frame, the rest are source frames
    fig, axs = plt.subplots(B * 2, T, figsize=(T * 5, B * 5 * 2))
    for b in range(B):
       frames = batch[b].frames
       # plot images
       for t in range(T):
           img = np.array(frames[t].data.permute(1, 2, 0))
           axs[b * 2, t].imshow(img)
           axs[b * 2, t].axis('off')
       # plot masks
       for t in range(T):
           rgb_mask = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
           for n in frames[t].notions:
               # convert binary mask to rgb mask
               mask = n.segment.bool().numpy()
               color = colors[n.cls_id % len(colors)]
               rgb_mask[mask] = list(color)
            
           axs[b * 2 + 1, t].imshow(rgb_mask)
           axs[b * 2 + 1, t].axis('off')
    plt.tight_layout()
    plt.savefig(f"{os.environ.get('HOME')}/Downloads/batch_visualization.png")
    plt.close()


def collate_fn(
    batch: List[SrcTgtDatapoint],
    dict_key,
) -> BatchedSrcTgtDatapoint:
    """
    Args:
        batch: A list of SrcTgtDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    # visualize_batch(batch)
    img_batch = []
    tgt_msk_batch = []
    src_msk_batch = []
    names_batch = []
    notion_colors_batch = []
    for tgt in batch:
        img_batch += [torch.stack([frame.data for frame in tgt.frames], dim=0)]
        names_batch.append([frame.sample_name for frame in tgt.frames])
        valid_notion_ids = tgt.valid_src_notion_ids
        h, w = tgt.size
        # generate shuffle index for notions
        notion_idxs = random.sample(range(tgt.notion_size), tgt.notion_size)
        # collect target masks
        tgt_masks = torch.zeros((tgt.notion_size, h, w), dtype=torch.bool)
        notion_colors = [[0, 0, 0]] * tgt.notion_size
        for n in tgt.frames[0].notions:
            if n.cls_id in valid_notion_ids:
                tgt_masks[valid_notion_ids.index(n.cls_id)] = n.segment.to(torch.bool)
                notion_colors[valid_notion_ids.index(n.cls_id)] = list(n.color)
        tgt_msk_batch.append(tgt_masks[notion_idxs])  # shuffle the target masks
        notion_colors = [notion_colors[i] for i in notion_idxs]
        # collect source masks
        src_masks = torch.zeros((tgt.notion_size, (len(tgt.frames) - 1), h, w), dtype=torch.bool)
        for i, frame in enumerate(tgt.frames[1:]):
            for n in frame.notions:
                if n.cls_id in valid_notion_ids:
                    src_masks[valid_notion_ids.index(n.cls_id), i] = n.segment.to(torch.bool)
        src_msk_batch.append(src_masks[notion_idxs].permute(1, 0, 2, 3))  # shuffle the source masks
        notion_colors_batch.append(notion_colors)


    img_batch = torch.stack(img_batch, dim=0) # BxTxCxHxW
    tgt_msk_batch = torch.stack(tgt_msk_batch, dim=0) # BxNxHxW
    src_msk_batch = torch.stack(src_msk_batch, dim=0) # Bx(T-1)xNxHxW
    B = len(batch)
    
    return BatchedSrcTgtDatapoint(
        img_batch=img_batch,
        src_mask_batch=src_msk_batch,
        tgt_mask_batch=tgt_msk_batch,
        metadata=None,
        sample_names=names_batch,
        dict_key=dict_key,
        batch_size=[B],
        notion_colors=notion_colors_batch,
    )
