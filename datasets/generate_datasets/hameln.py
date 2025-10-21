# Description: Script to generate datasets from high-seafiles (ikg_ml_dataset) for training and testing.
# Author: Yunshuang Yuan
# Date: 2025-09-29

import os
import json
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
from PIL import Image

from data_utils import read_tiff


train_img_size = 384
data_root_hameln = "/home/yuan/data/HisMap/ikg_ml_dataset"
colors = [
(34, 139, 34),    # 'wald' (forest): Forest Green
(124, 252, 0),    # 'grünland' (grassland): Lawn Green
(220, 20, 60),    # 'siedlung' (settlement): Crimson
(30, 144, 255),   # 'wasser' (flowing water): Dodger Blue
]
semantics = ['forest', 'grass', 'settlement', 'water']

def generate_colormap(data_root):
    color_map = {}
    for i, color in enumerate(colors):
        color_map[semantics[i]] = color
    color_map['nonlabeled'] = (255, 255, 255)
    color_map['background'] = (0, 0, 0)
    with open(os.path.join(data_root, 'color_map.json'), 'w') as f:
        json.dump(color_map, f, indent=4)


def rename_files_to_lowercase(folder_path):
    """
    Rename all files in the specified folder to lowercase.

    Args:
        folder_path (str): Path to the folder containing the files.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path '{folder_path}' is not a valid directory.")

    for filename in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, filename.lower())

        # Rename only if the file name changes
        if old_file_path != new_file_path:
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")


def decode_labels(encoded_labels):
    """
    Decode a (H, W) uint8 array into a (H, W, 8) boolean array using NumPy vectorization.

    Args:
        encoded_labels (np.ndarray): Encoded uint8 array of shape (H, W).

    Returns:
        np.ndarray: Decoded boolean array of shape (H, W, 8).
    """
    # Ensure the input is a uint8 array
    assert encoded_labels.dtype == np.uint8, "The input array must be of uint8 type."

    # Create an array of bit weights for the 8 channels: [128, 64, 32, 16, 8, 4, 2, 1]
    bit_weights = np.array([1 << (7 - bit) for bit in range(8)], dtype=np.uint8)

    # Use broadcasting to compare each bit weight with the encoded labels
    decoded_labels = (encoded_labels[..., None] & bit_weights) > 0

    return decoded_labels


def encode_labels(boolean_array):
    """
    Encode a (H, W, 8) boolean array into a (H, W) uint8 array using NumPy vectorization.

    Args:
        boolean_array (np.ndarray): Boolean array of shape (H, W, 8).

    Returns:
        np.ndarray: Encoded uint8 array of shape (H, W).
    """
    # Ensure the input is a boolean array with shape (H, W, 8)
    assert boolean_array.shape[-1] == 8, "The input array must have 8 channels."
    assert boolean_array.dtype == np.bool_, "The input array must be of boolean type."

    # Create an array of bit weights for the 8 channels: [128, 64, 32, 16, 8, 4, 2, 1]
    bit_weights = np.array([1 << (7 - bit) for bit in range(8)], dtype=np.uint8)

    # Use broadcasting and summation to encode the boolean array
    encoded_array = np.sum(boolean_array * bit_weights, axis=-1, dtype=np.uint8)

    return encoded_array


def generate_hameln_encoded_lbl(output_root, seperate_years=True):
    """
    Generate dataset from Hameln tk_image and label_tif folders.
    Args:
        output_root (str): Root directory to save the generated dataset.
        seperate_19xx_20xx (bool): Whether to separate datasets for 19xx and 20xx tk images. 
            If set False, datasets will be splited according to the year of tk images.
    """
    ROI_min = [7434, 10371]
    ROI_max = [15113, 20459]
    
    tk_image_files = [x for x in os.listdir(os.path.join(data_root_hameln, 'tk_image')) \
                      if x.endswith(".tif") and 'area' not in x]
    tk_image_files = sorted(tk_image_files)
    non_tk19xx_area = np.logical_not(read_tiff(os.path.join(data_root_hameln, 'tk_image', 'tk19xx_area.tif')))
    non_tk20xx_area = np.logical_not(read_tiff(os.path.join(data_root_hameln, 'tk_image', 'tk20xx_area.tif')))
    tk_labels = ['wald', 'grünland', 'siedlung', 'fließgewässer', 'stillgewässer']

    def read_ROI(tif_file):
        img = read_tiff(tif_file)
        if img is None:
            return None
        return img[ROI_min[0]:ROI_max[0], ROI_min[1]:ROI_max[1]]

    for img_file in tk_image_files:
        print(img_file)
        year = img_file.split('.')[0]
        if int(year) < 2000:
            tk_area = non_tk19xx_area
            dataset_split = "a"
        else:
            tk_area = non_tk20xx_area
            dataset_split = "b"
        # create output folders
        img_dir = os.path.join(output_root, f'hameln.{dataset_split}', 'imgs')
        lbl_dir = os.path.join(output_root, f'hameln.{dataset_split}', 'lbls')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        tk_img = read_tiff(os.path.join(data_root_hameln, 'tk_image', img_file))
        # make sure 3 channels for tk_img
        if tk_img.ndim==2:
            tk_img = tk_img[..., None].repeat(3, 2)
        elif tk_img.shape[2]>3:
            tk_img = tk_img[..., :3]
        
        # 0: background, 1: foreground, 255: not labeled
        mask_labeled_area = read_tiff(os.path.join(data_root_hameln, 'label_tif', img_file))
        # get crop parameters
        xs, ys = np.where(mask_labeled_area == 1)
        min_x, min_y = xs.min(), ys.min()
        max_x, max_y = xs.max(), ys.max()
        # crop to valid area for generating dataset
        mask_unlabeled_area = np.logical_not(mask_labeled_area[min_x:max_x, min_y:max_y])
        tk_img = tk_img[min_x:max_x, min_y:max_y]

        all_labels = np.zeros(mask_unlabeled_area.shape + (8,), dtype=bool)
        all_labels[mask_unlabeled_area] = True
        for i, l in enumerate(tk_labels):
            # 0: background, 1: foreground, 255: not labeled
            lbl_mask = read_tiff(os.path.join(data_root_hameln, 'label_tif', img_file.replace('.tif', f'_{l}.tif')))
            if lbl_mask is None:
                lbl_mask = np.zeros(tk_area.shape, dtype=np.uint8)
                        # some tk image do not have valid mapping pixels, set the corresponding mask to background (0)
            xs, ys = np.where(tk_area)
            lbl_mask[xs, ys] = 0

            # crop label to the valid area
            lbl_mask = lbl_mask[min_x:max_x, min_y:max_y]
            xs, ys = np.where(lbl_mask==1)
            all_labels[xs, ys, np.zeros(xs.shape, dtype=int) + i] = True
        all_labels = encode_labels(all_labels)
        
        for x in range(0, max_x - min_x, train_img_size):
            for y in range(0, max_y - min_y, train_img_size):
                # naming accordint to the original tk_image coordinates before cropping
                filename = f"{year}_{x + min_x}_{y + min_y}"
                if x + train_img_size <= tk_img.shape[0] and y + train_img_size <= tk_img.shape[1]:
                    imgc = tk_img[x:x+train_img_size, y:y+train_img_size]
                    lblc = all_labels[x:x+train_img_size, y:y+train_img_size]
                else:
                    imgc = np.full((train_img_size, train_img_size, 3), 255, dtype=tk_img.dtype)
                    lblc = np.full((train_img_size, train_img_size), 255, dtype=all_labels.dtype)
                    imgc[:min(train_img_size, tk_img.shape[0] - x), :min(train_img_size, tk_img.shape[1] - y)] = \
                        tk_img[x:min(x + train_img_size, tk_img.shape[0]), y:min(y + train_img_size, tk_img.shape[1])]
                    lblc[:min(train_img_size, tk_img.shape[0] - x), :min(train_img_size, tk_img.shape[1] - y)] = \
                        all_labels[x:min(x + train_img_size, tk_img.shape[0]), y:min(y + train_img_size, tk_img.shape[1])]

                # for i in range(1, 9):
                #     print(2**i, np.sum(lblc==2**i))
                Image.fromarray(imgc).save(os.path.join(img_dir, f"{filename}.png"))
                Image.fromarray(lblc).save(os.path.join(lbl_dir, f"{filename}.png"))


def generate_hameln_rgb_lbl(output_root, seperate_years=True):
    """
    Generate dataset from Hameln tk_image and label_tif folders.
    Args:
        output_root (str): Root directory to save the generated dataset.
        seperate_years (bool): Whether to separate datasets according to their mapping year. 
            If set False, the datasets will be separated into 19xx and 20xx datasets.
    """
    ROI_min = [7434, 10371]
    ROI_max = [15113, 20459]
    
    tk_image_files = [x for x in os.listdir(os.path.join(data_root_hameln, 'tk_image')) \
                      if x.endswith(".tif") and 'area' not in x]
    tk_image_files = sorted(tk_image_files)
    non_tk19xx_area = np.logical_not(read_tiff(os.path.join(data_root_hameln, 'tk_image', 'tk19xx_area.tif')))
    non_tk20xx_area = np.logical_not(read_tiff(os.path.join(data_root_hameln, 'tk_image', 'tk20xx_area.tif')))
    tk_labels = ['wald', 'grünland', 'siedlung', 'fließgewässer', 'stillgewässer']

    for img_file in tk_image_files:
        print(img_file)
        year = img_file.split('.')[0]
        if int(year) < 2000:
            continue
        if int(year) < 2000:
            tk_area = non_tk19xx_area
            dataset_split = year if seperate_years else "a" 
        else:
            tk_area = non_tk20xx_area
            dataset_split = year if seperate_years else "b"
        
        # create output folders
        img_dir = os.path.join(output_root, f'hameln.{dataset_split}', 'imgs')
        lbl_dir = os.path.join(output_root, f'hameln.{dataset_split}', 'lbls')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        generate_colormap(os.path.join(output_root, f'hameln.{dataset_split}'))

        tk_img = read_tiff(os.path.join(data_root_hameln, 'tk_image', img_file))
        # make sure 3 channels for tk_img
        if tk_img.ndim==2:
            tk_img = tk_img[..., None].repeat(3, 2)
        elif tk_img.shape[2]>3:
            tk_img = tk_img[..., :3]
        
        # 0: background, 1: foreground, 255: not labeled
        mask_labeled_area = read_tiff(os.path.join(data_root_hameln, 'label_tif', img_file))
        # get crop parameters
        xs, ys = np.where(mask_labeled_area == 1)
        min_x, min_y = xs.min(), ys.min()
        max_x, max_y = xs.max(), ys.max()
        # crop to valid area for generating dataset
        mask_unlabeled_area = np.logical_not(mask_labeled_area[min_x:max_x, min_y:max_y])
        tk_img = tk_img[min_x:max_x, min_y:max_y]

        all_labels = np.zeros(mask_unlabeled_area.shape, dtype=np.uint8)
        all_labels[mask_unlabeled_area] = 255
        for i, l in enumerate(tk_labels):
            # 0: background, 1: foreground, 255: not labeled
            lbl_mask = read_tiff(os.path.join(data_root_hameln, 'label_tif', img_file.replace('.tif', f'_{l}.tif')))
            if lbl_mask is None:
                lbl_mask = np.zeros(tk_area.shape, dtype=np.uint8)
            # some tk image do not have valid mapping pixels, set the corresponding mask to background (0)
            xs, ys = np.where(tk_area)
            lbl_mask[xs, ys] = 0

            # crop label to the valid area
            lbl_mask = lbl_mask[min_x:max_x, min_y:max_y]
            xs, ys = np.where(lbl_mask==1)
            all_labels[xs, ys] = i + 1

            del lbl_mask
            
        # merge llb=5 and lbl=4 (stillgewässer and fließgewässer into wasser)
        all_labels[all_labels == 5] = 4
        # convert to rgb label
        all_labels_color = np.zeros(all_labels.shape + (3,), dtype=np.uint8)
        for i, color in enumerate(colors):
            all_labels_color[all_labels == i + 1] = color
        
        for x in range(0, max_x - min_x, train_img_size):
            for y in range(0, max_y - min_y, train_img_size):
                # naming accordint to the original tk_image coordinates before cropping
                filename = f"{x + min_x}_{y + min_y}" if seperate_years else f"{year}_{x + min_x}_{y + min_y}"
                if x + train_img_size <= tk_img.shape[0] and y + train_img_size <= tk_img.shape[1]:
                    imgc = tk_img[x:x+train_img_size, y:y+train_img_size]
                    lblc = all_labels_color[x:x+train_img_size, y:y+train_img_size]
                # else:
                #     continue
                    # imgc = np.zeros((train_img_size, train_img_size, 3), dtype=tk_img.dtype)
                    # lblc = np.zeros((train_img_size, train_img_size, 3), dtype=all_labels.dtype)
                    # imgc[:min(train_img_size, tk_img.shape[0] - x), :min(train_img_size, tk_img.shape[1] - y)] = \
                    #     tk_img[x:min(x + train_img_size, tk_img.shape[0]), y:min(y + train_img_size, tk_img.shape[1])]
                    # lblc[:min(train_img_size, tk_img.shape[0] - x), :min(train_img_size, tk_img.shape[1] - y)] = \
                    #     all_labels_color[x:min(x + train_img_size, tk_img.shape[0]), y:min(y + train_img_size, tk_img.shape[1])]

                    Image.fromarray(imgc).save(os.path.join(img_dir, f"{filename}.png"))
                    Image.fromarray(lblc).save(os.path.join(lbl_dir, f"{filename}.png"))


def generate_data_info(root_dir, dataset_name):
    lbl_dir = os.path.join(root_dir, dataset_name, 'lbls')
    dinfo_filename = os.path.join(root_dir, dataset_name, f'{dataset_name}.json')
    sample_list_txt = os.path.join(root_dir, dataset_name, f'{dataset_name}.txt')
    lbl_files = [x for x in os.listdir(lbl_dir) if x.endswith(".png")]
    dinfo = {}
    npix = 448 * 448
    num_cls = 5
    for lbl_file in lbl_files:
        name = lbl_file.split('.')[0]
        lbl = np.array(Image.open(os.path.join(lbl_dir, lbl_file)))
        lbl = decode_labels(lbl).transpose(2, 0, 1)[:num_cls]  # (num_cls, H, W)
        npix_non_valid = np.all(lbl, axis=0).sum()
        dinfo[f'{dataset_name}.{name}'] = {i+1: (l.sum() - npix_non_valid) / npix for i, l in enumerate(lbl)}
        # print(f'{dataset_name}.{name}:', dinfo[f'{dataset_name}.{name}'])
    with open(dinfo_filename, 'w') as f:
        json.dump(dinfo, f, indent=4)
    with open(sample_list_txt, 'w') as f:
        f.write('\n'.join(dinfo.keys()))


def train_test_val_split(root_dir, ratios={'train':0.7, 'test':0.2, 'val': 0.1}, num_cls=5):
    for d in os.listdir(root_dir):
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
            summary = {str(i): 0 for i in range(1, num_cls + 1)}
            for sample in split_samples:
                for cls, r in dinfo[sample].items():
                    summary[cls] += r
            summary_file = os.path.join(root_dir, d, f"split_summary.txt")
            with open(summary_file, 'a') as f:
                f.write(f"########################## {split.upper()} SPLIT ##########################\n")
                f.write(f"Test samples : {num_split}/{n_total}\n")
                f.write(f"Test split summary (npix=summed_ratio*448²):\n")
                for k, v in summary.items():
                    f.write(f"Class {k}: {v:.4f}\n")
                f.write("\n")


def select_roi_samples(dinfo_file, num_cls, num_samples_per_cls_per_sheet=1, min_ratio=0.1, max_ratio=0.8, save_path=None):
    """
    Select regions of interest (ROI) samples for labeling. 
    Normally, for real use case we should select them manually.
    But for the experiments, we use a simple automatic way to select them from the training set.
    Args:
        dinfo_file (str): Path to the data info JSON file.
        num_cls (int): Number of classes in the dataset.
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
    # filter to get only training samples
    with open(os.path.dirname(dinfo_file) + '/train_samples.txt', 'r') as f:
        train_samples = [line.strip() for line in f.readlines()]
    dinfo = {k: v for k, v in dinfo.items() if k in train_samples}

    map_sheets = list(set([x.split('.')[2].split('_')[0] for x in dinfo.keys()]))

    # select samples for each class
    selected_samples = {i: [] for i in range(1, num_cls + 1)}
    for cls in range(1, num_cls + 1):
        for sheet in map_sheets:
            samples_in_sheet = {name: info for name, info in dinfo.items() if name.split('.')[2].split('_')[0] == sheet}
            cls_samples_in_ratio_range = [
                name for name, info in samples_in_sheet.items() 
                if info[str(cls)] >= min_ratio and info[str(cls)] <= max_ratio
            ]
            if len(cls_samples_in_ratio_range) == 0:
                print(f"No samples found for class {cls} within the specified ratio range.")
                continue
            if len(cls_samples_in_ratio_range) < num_samples_per_cls_per_sheet:
                print(f"Only {len(cls_samples_in_ratio_range)} samples found for class {cls}," + \
                      f"less than the requested {num_samples_per_cls_per_sheet}. Selecting all available samples.")
                selected_samples[cls].extend(cls_samples_in_ratio_range)
            else:
                selected_samples[cls].extend(
                    np.random.choice(cls_samples_in_ratio_range, 
                                     num_samples_per_cls_per_sheet, replace=False).tolist()
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
            f.write(f"Class {cls}: {len(samples)} samples\n")
        f.write("\n")
        f.write(f"Total selected samples: {sum([len(v) for v in selected_samples.values()])}\n")

    return selected_samples


def copy_selected_samples(selected_samples, root_dir):
    img_dir = os.path.join(root_dir, 'imgs')
    lbl_dir = os.path.join(root_dir, 'lbls')
    save_img_dir = os.path.join(root_dir, 'selected_imgs')
    save_lbl_dir = os.path.join(root_dir, 'selected_lbls')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)
    for cls, samples in selected_samples.items():
        for sample in samples:
            sample_name = sample.rsplit('.', 1)[1]
            img_file = os.path.join(img_dir, f"{sample_name}.png")
            lbl_file = os.path.join(lbl_dir, f"{sample_name}.png")
            if os.path.exists(img_file) and os.path.exists(lbl_file):
                os.system(f"cp {img_file} {save_img_dir}")
                os.system(f"cp {lbl_file} {save_lbl_dir}")
            else:
                print(f"Image or label file for sample {sample} not found.")


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


if __name__ == "__main__":
    seperate_years = True
    data_root = f"/home/yuan/data/HisMap/syntra{train_img_size}" + f"{'_sheets' if seperate_years else ''}"
    generate_hameln_rgb_lbl(data_root, seperate_years=seperate_years)




