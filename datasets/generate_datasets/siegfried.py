# Description: Script to generate datasets from Siegfried dataset.
# Author: Yunshuang Yuan
# Date: 2025-10-09

import os
import json
import numpy as np
import sys

from PIL import Image

colors = {
    'railway': (128, 0, 0),
    'vineyard': (0, 128, 0),
}

def generate_colormap(data_root, dataset_name):
    color_map = {
        dataset_name: colors[dataset_name],
        'nonlabeled': (255, 255, 255),
        'background': (0, 0, 0),
    }
    with open(os.path.join(data_root, f'color_map.json'), 'w') as f:
        json.dump(color_map, f, indent=4)



def generate_siegfrid(data_root_in, data_root_out, dataset_name):
    # prepare output directory
    img_out_dir = os.path.join(data_root_out, "imgs")
    lbl_out_dir = os.path.join(data_root_out, "lbls")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    # copy to output directory, generate split list files and data info files
    split_info = {}
    dinfo = {}
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(data_root_in, split)
        lbl_dir = os.path.join(data_root_in, 'annotation', split)
        samples = []
        split_info[split] = {dataset_name: 0.0, 'background': 0.0, 'nonlabeled': 0.0, 'n_samples': 0}
        for filename in os.listdir(img_dir):
            name = os.path.splitext(filename)[0]
            samples.append(f"siegfried.{dataset_name}.{name}")
            img_path = os.path.join(img_dir, filename)
            lbl_path = os.path.join(lbl_dir, filename)
            assert os.path.exists(lbl_path), f"Label file {lbl_path} does not exist."

            img = Image.open(img_path).convert("RGB")
            lbl = Image.open(lbl_path)
            lbl_rgb = np.zeros((*lbl.size[:2], 3), dtype=np.uint8)
            lbl = np.array(lbl)
            lbl_rgb[lbl > 0] = colors[dataset_name]
            # get label ratios
            fg_ratio = np.sum(lbl > 0) / (lbl.shape[0] * lbl.shape[1])
            bg_ratio = 1 - fg_ratio
            split_info[split][dataset_name] += fg_ratio
            split_info[split]['background'] += bg_ratio
            split_info[split]['n_samples'] += 1
            dinfo[f"siegfried.{dataset_name}.{name}"] = {
                dataset_name: fg_ratio,
                'background': bg_ratio,
                'nonlabeled': 0.0,
            }

            img.save(os.path.join(img_out_dir, filename.replace('.tif', '.png')))
            Image.fromarray(lbl_rgb).save(os.path.join(lbl_out_dir, filename.replace('.tif', '.png')))
        
        # save split list
        with open(os.path.join(data_root_out, f"{split}_samples.txt"), 'w') as f:
            f.write("\n".join(samples))

    # save data info
    with open(os.path.join(data_root_out, f"siegfried.{dataset_name}.json"), 'w') as f:
        json.dump(dinfo, f, indent=4)
    
    # save split summary to txt
    summary_file = os.path.join(data_root_out, f"split_summary.txt")
    n_total = sum([v['n_samples'] for v in split_info.values()])
    with open(summary_file, 'w') as f:
        for split, info in split_info.items():   
            f.write(f"########################## {split.upper()} SPLIT ##########################\n")
            f.write(f"Num. samples : {info['n_samples']}/{n_total}\n")
            f.write(f"Ratio summary (npix=summed_ratio*224²):\n")
            for k, v in info.items():
                if k != 'n_samples':                    
                    f.write(f"{k}: {v:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    data_root_in = "/home/yuan/data/HisMap/Siegfried_Railway_Vineyard/dataset_"
    data_root_out = "/home/yuan/data/HisMap/syntra384/siegfried."

    for subset in ['railway', 'vineyard']:
        # generate_siegfrid(data_root_in + subset, data_root_out + subset, subset)
        generate_colormap(data_root_out + subset, subset)