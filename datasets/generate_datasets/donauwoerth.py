# Description: Script to generate datasets from Donauwörth maps for training and testing.
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
data_root_in = "/home/yuan/data/HisMap/Bayern/processed"
colors = [
(34, 139, 34),    # 'wald' (forest): Forest Green
(124, 252, 0),    # 'grünland' (grassland): Lawn Green
(30, 144, 255),   # 'wasser' (water): Dodger Blue
]
semantics = ['forest', 'grass', 'water']

def generate_colormap(data_root):
    color_map = {}
    for i, color in enumerate(colors):
        color_map[semantics[i]] = color
    color_map['nonlabeled'] = (255, 255, 255)
    color_map['background'] = (0, 0, 0)
    with open(os.path.join(data_root, 'color_map.json'), 'w') as f:
        json.dump(color_map, f, indent=4)


def generate_donauwoerth(data_root_in, data_root_out, sheet_list=None):
    print(f"Generating Donauwörth dataset from {data_root_in} to {data_root_out}...")
    # prepare output directory
    img_out_dir = os.path.join(data_root_out, "imgs")
    lbl_out_dir = os.path.join(data_root_out, "lbls")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    map_dir = os.path.join(data_root_in, "maps")
    lbl_dir = os.path.join(data_root_in, "lbls")
    if sheet_list is None:
        sheet_list = os.listdir(map_dir)
    for filename in sheet_list:
        name = os.path.splitext(filename)[0]
        map_path = os.path.join(map_dir, filename)
        lbl_path = os.path.join(lbl_dir, filename)
        assert os.path.exists(lbl_path), f"Label file {lbl_path} does not exist."

        print(f"Processing {filename}...")
        map_rgb = np.array(Image.open(map_path).convert("RGB"))
        
        # convert lbl to color image
        lbl = Image.open(lbl_path)
        lbl_np = np.array(lbl)
        unique_lbls = np.unique(lbl_np)
        print(f"Unique labels in {filename}: {unique_lbls}")
        # remap labels
        lbl_np[lbl_np == 4] = 3  # water
        lbl_np[lbl_np == 5] = 3

        lbl_color = np.zeros((lbl_np.shape[0], lbl_np.shape[1], 3), dtype=np.uint8)
        for i, color in enumerate(colors):
            lbl_color[lbl_np == i + 1] = color
        
        # crop map and lbl into patches of size 'train_img_size x train_img_size'
        w, h = map_rgb.shape[:2]

        for x in range(0, w, train_img_size):
            for y in range(0, h, train_img_size):
                map_patch = map_rgb[x:x+train_img_size, y:y+train_img_size]
                lbl_patch = lbl_color[x:x+train_img_size, y:y+train_img_size]
                # if the patch is smaller than train_img_size, pad it with zeros
                if map_patch.shape[0] < train_img_size or map_patch.shape[1] < train_img_size:
                    continue
                    # map_patch_padded = np.zeros((train_img_size, train_img_size, 3), dtype=np.uint8)
                    # lbl_patch_padded = np.zeros((train_img_size, train_img_size, 3), dtype=np.uint8)
                    # map_patch_padded[:map_patch.shape[0], :map_patch.shape[1]] = map_patch
                    # lbl_patch_padded[:lbl_patch.shape[0], :lbl_patch.shape[1]] = lbl_patch
                    # map_patch = map_patch_padded
                    # lbl_patch = lbl_patch_padded
                # save the patch
                map_patch_pil = Image.fromarray(map_patch)
                lbl_patch_pil = Image.fromarray(lbl_patch)
                map_patch_pil.save(os.path.join(img_out_dir, f"{name}_{x}_{y}.png"))
                lbl_patch_pil.save(os.path.join(lbl_out_dir, f"{name}_{x}_{y}.png"))

def conver_rgb_channels(root_out):
    img_files = os.listdir(os.path.join(root_out + '.b', "imgs"))
    os.makedirs(os.path.join(root_out + '.b', "images"), exist_ok=True)
    for f in img_files:
        img = Image.open(os.path.join(root_out + '.b', "imgs", f))
        img = np.array(img)[:, :, ::-1]  # convert RGB to BGR
        img = Image.fromarray(img)
        img.save(os.path.join(root_out + '.b', "imgs", f))


def planA():
    root_out = "/home/yuan/data/HisMap/syntra384/donauwoerth"
    # sheet in year 1959 has very different styles than other sheets, split it into a new dataset
    list_a = [x for x in os.listdir(os.path.join(data_root_in, "maps")) if not '1959' in x]
    list_b = [x for x in os.listdir(os.path.join(data_root_in, "maps")) if '1959' in x]
    generate_donauwoerth(data_root_in, root_out + '.a', sheet_list=list_a)
    generate_donauwoerth(data_root_in, root_out + '.b', sheet_list=list_b)
    generate_colormap(root_out + '.a')
    generate_colormap(root_out + '.b')


def planB():
    root_out = "/home/yuan/data/HisMap/syntra384_sheets/donauwoerth"
    lists = os.listdir(os.path.join(data_root_in, "maps"))
    lists.sort()
    datasets = {}
    for l in lists:
        year = l.split('.')[0].split('_')[-1]
        if not year in datasets:
            datasets[year] = []
        datasets[year].append(l)
    
    for d, files in datasets.items():
        sub_root_out = f"{root_out}.{d}" 
        generate_donauwoerth(data_root_in, sub_root_out, sheet_list=files)
        generate_colormap(sub_root_out)
        

if __name__ == "__main__":
    
    planB()
