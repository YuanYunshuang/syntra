# Description: Script to generate meta info:
#   (train_samples.txt) train/val/test split, 
#   (dataset_name.json) cls ratio for each sample, 
#   (selected_n.json) fewshot split, selected n samples per class.
#   (paring_n_intercls.json) paired src and tgt samples by Faiss Indexing.
# Author: Yunshuang Yuan
# Date: 2025-10-09

import os
import json
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
from PIL import Image

from data_utils import read_tiff


train_img_size = 384
npix = train_img_size * train_img_size


# step 1
def generate_data_info(root_dir, dataset_list=None):
    if dataset_list is None:
        dataset_list = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for dataset_name in dataset_list:
        print(f"Generating data info for: {dataset_name}")
        # load color map
        with open(os.path.join(root_dir, dataset_name, 'color_map.json'), 'r') as f:
            color_map = json.load(f)
            
        lbl_dir = os.path.join(root_dir, dataset_name, 'lbls')
        dinfo_filename = os.path.join(root_dir, dataset_name, f'{dataset_name}.json')
        sample_list_txt = os.path.join(root_dir, dataset_name, f'{dataset_name}.txt')
        lbl_files = [x for x in os.listdir(lbl_dir) if x.endswith(".png")]
        dinfo = {}
        
        num_cls = len(color_map)
        for lbl_file in lbl_files:
            name = lbl_file.split('.')[0]
            lbl = np.array(Image.open(os.path.join(lbl_dir, lbl_file)))
            dinfo[f'{dataset_name}.{name}'] = {}
            for semantic, color in color_map.items():
                color = np.array(color).reshape(1, 1, 3)
                cur_mask = (color == lbl).all(axis=-1) 
                dinfo[f'{dataset_name}.{name}'][semantic] = cur_mask.sum() / npix
        with open(dinfo_filename, 'w') as f:
            json.dump(dinfo, f, indent=4)
        with open(sample_list_txt, 'w') as f:
            f.write('\n'.join(dinfo.keys()))


# step 2
def train_test_val_split(root_dir, ratios={'train':0.7, 'test':0.2, 'val': 0.1}, dataset_list=None):
    if dataset_list is None:
        dataset_list = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for d in dataset_list:
        print(f"Splitting dataset: {d}")
        # delete existing split files
        for split in ratios.keys():
            split_file = os.path.join(root_dir, d, f"{split}_samples.txt")
            if os.path.exists(split_file):
                os.remove(split_file)
        summary_file = os.path.join(root_dir, d, f"split_summary.txt")
        if os.path.exists(summary_file):
            os.remove(summary_file)
        # load samples and data info
        with open(os.path.join(root_dir, d, f"{d}.txt"), 'r') as f:
            samples = [line.strip() for line in f.readlines()]
        with open(os.path.join(root_dir, d, f"{d}.json"), 'r') as f:
            dinfo = json.load(f)
        with open(os.path.join(root_dir, d, f"color_map.json"), 'r') as f:
            color_map = json.load(f)

        n_total = len(samples)
        for split, ratio in ratios.items():
            num_split = int(n_total * ratio) 
            split_samples = np.random.choice(samples, num_split, replace=False).tolist()
            samples = [s for s in samples if s not in split_samples]
            # save split samples
            split_file = os.path.join(root_dir, d, f"{split}_samples.txt")
            with open(split_file, 'w') as f:
                for s in split_samples:
                    f.write(f"{s}\n")

            # parse split summary
            summary = {k: 0 for k in color_map.keys()}
            for sample in split_samples:
                for k, r in dinfo[sample].items():
                    summary[k] += r
            summary_file = os.path.join(root_dir, d, f"split_summary.txt")
            with open(summary_file, 'a') as f:
                f.write(f"########################## {split.upper()} SPLIT ##########################\n")
                f.write(f"Num. samples : {num_split}/{n_total}\n")
                f.write(f"Ratio summary (npix=summed_ratio*{train_img_size}²):\n")
                for k, v in summary.items():
                    f.write(f"{k}: {v:.4f}\n")
                f.write("\n")


def select_roi_samples(dinfo_file, num_samples_per_cls_per_sheet=1, min_ratio=0.1, max_ratio=0.8, plan="A", save_path=None):
    """
    Select regions of interest (ROI) samples for labeling. 
    Normally, for real use case we should select them manually.
    But for the experiments, we use a simple automatic way to select them from the training set.
    Args:
        dinfo_file (str): Path to the data info JSON file.
        num_samples_per_cls_per_sheet (int): Number of samples to select per class per map sheet.
        min_ratio (float): Minimum ratio of pixels for a class to be considered valid.
        max_ratio (float): Maximum ratio of pixels for a class to be considered valid.
        save_path (str): Path to save the selected samples JSON file.
    Returns:
        dict: A dictionary containing selected ROI samples.
    """
    # load data info file of all samples
    with open(dinfo_file, 'r') as f:
        dinfo = json.load(f)
    with open(os.path.join(os.path.dirname(dinfo_file), 'color_map.json'), 'r') as f:
        color_map = json.load(f)
    # filter to get only training samples
    with open(os.path.dirname(dinfo_file) + '/train_samples.txt', 'r') as f:
        train_samples = [line.strip() for line in f.readlines()]
    dinfo = {k: v for k, v in dinfo.items() if k in train_samples}

    def parse_sheet(sample_name):
        if plan == "A":
            return sample_name.split('.')[-1].rsplit('_', 2)[0]
        elif plan == "B":
            return sample_name.split('.')[1]
        else:
            raise ValueError(f"Unknown plan: {plan}")

    if 'siegfried' in dinfo_file:
        map_sheets = [os.path.basename(dinfo_file).rsplit('.', 1)[0]]
    else:
        map_sheets = list(set([parse_sheet(x) for x in dinfo.keys()]))

    # select samples for each class
    fg_classes = [k for k in color_map.keys() if k not in ['background', 'nonlabeled']]
    selected_samples = {k: [] for k in fg_classes}
    for cls in fg_classes:
        for sheet in map_sheets:
            if 'siegfried' in sheet:
                samples_in_sheet = dinfo
                replace = False
                if 'railway' in sheet:
                    min_ratio = 0.02
            else:
                samples_in_sheet = {name: info for name, info in dinfo.items() if parse_sheet(name) == sheet}
                replace = True
            cls_samples_in_ratio_range = [
                name for name, info in samples_in_sheet.items() 
                if info[cls] >= min_ratio and info[cls] <= max_ratio
            ]
            if len(cls_samples_in_ratio_range) == 0:
                print(f"No samples found for class {cls} within the specified ratio range.")
                continue
            if len(cls_samples_in_ratio_range) < num_samples_per_cls_per_sheet:
                replace = True
                print(f"Only {len(cls_samples_in_ratio_range)} samples found for class {cls}," + \
                      f"less than the requested {num_samples_per_cls_per_sheet}. Selecting all available samples with replacement.")
            selected_samples[cls].extend(
                np.random.choice(cls_samples_in_ratio_range,
                                 num_samples_per_cls_per_sheet, replace=replace).tolist()
            )
    
    # save selected samples
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving selected samples to {save_path}")
        with open(save_path, 'w') as f:
            json.dump(selected_samples, f, indent=4)
    
    # save summary to txt
    summary_file = os.path.join(os.path.dirname(dinfo_file), 
                                f"selected_{num_samples_per_cls_per_sheet}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Selected samples from training set:\n")
        for cls, samples in selected_samples.items():
            f.write(f"{cls}: {len(samples)} samples\n")
        f.write("\n")
        f.write(f"Total selected samples: {sum([len(set(v)) for v in selected_samples.values()])}\n")
        f.write(f"Total samples before selection: {len(train_samples)}\n")

    return selected_samples


def visualize_selected_samples(selected_samples, root_dir):
    img_dir = os.path.join(root_dir, 'imgs')
    lbl_dir = os.path.join(root_dir, 'lbls')

    for cls, samples in selected_samples.items():
        for sample in samples:
            sample_name = sample.rsplit('.', 1)[1]
            img_file = os.path.join(img_dir, f"{sample_name}.png")
            lbl_file = os.path.join(lbl_dir, f"{sample_name}.png")
            if os.path.exists(img_file) and os.path.exists(lbl_file):
                img = Image.open(img_file)
                lbl = Image.open(lbl_file)
                # convert lbl to binary mask for the current class
                lbl = np.array(lbl)
                lbl = (lbl == 2**(8 - cls)) & (lbl != 255)
                lbl = Image.fromarray(lbl.astype(np.uint8) * 255)

                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(f"Class {cls} - Sample {sample_name}", fontsize=16)
                ax[0].imshow(img)
                ax[0].set_title(f"Image: {sample_name}")
                ax[0].axis('off')

                ax[1].imshow(lbl, cmap='jet', alpha=0.5)
                ax[1].set_title(f"Label: {sample_name} (Class {cls})")
                ax[1].axis('off')

                # show legend
                handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {cls}', 
                                      markerfacecolor='r', markersize=10, alpha=0.5)]
                ax[1].legend(handles=handles, loc='upper right')

                plt.show()
            else:
                print(f"Image or label file for sample {sample} not found.")


def select_fewshot_samples(data_root, dataset_list=None, nshot=10, plan="A"):
    if dataset_list is None:
        dataset_list = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    selected_for_map_collections = {}
    for d in dataset_list:
        print(f"Selecting fewshot samples for: {d}")
        selected_samples = select_roi_samples(
            dinfo_file=os.path.join(data_root, d, f"{d}.json"),
            num_samples_per_cls_per_sheet=nshot, 
            min_ratio=0.05, 
            max_ratio=0.8, 
            plan=plan,
            save_path=os.path.join(data_root, d, f"selected_{nshot}.json")
        )

        collection = d.split('.')[0]
        for k, v in selected_samples.items():
            if collection not in selected_for_map_collections:
                selected_for_map_collections[collection] = {}
            if k not in selected_for_map_collections[collection]:
                selected_for_map_collections[collection][k] = []
            selected_for_map_collections[collection][k].extend(v)
        # visualize_selected_samples(selected_samples, os.path.join(data_root, d))
    # save selected samples for all map collections
    save_path = os.path.join(data_root, f"nshot{nshot}_summary.txt")
    for mc, samples in selected_for_map_collections.items():
        with open(save_path, 'a') as f:
            f.write(f"########################## Map Collection: {mc} ##########################\n")
            for cls, s_list in samples.items():
                f.write(f"Class {cls} (non-repeatitive): {len(set(s_list))} samples\n")
            total_unique_samples = len(set([s for sl in samples.values() for s in sl]))
            f.write(f"Total unique samples selected: {total_unique_samples}\n")
            f.write("\n")

if __name__ == "__main__":
    data_root = "/home/yuan/data/HisMap/syntra384_sheets"
    dataset_list = [x for x in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, x))] # ['donauwoerth.a', 'donauwoerth.b']
    # generate_data_info(data_root, dataset_list=dataset_list)
    # train_test_val_split(data_root, dataset_list=dataset_list)
    select_fewshot_samples(data_root, nshot=100, dataset_list=dataset_list, plan="B")
