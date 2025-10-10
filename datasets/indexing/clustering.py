import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from indexdb import IndexDatabase





def select_roi_samples(dinfo_file, num_cls, num_samples_per_cls=1, min_ratio=0.1, max_ratio=0.8, save_path=None):
    """
    Select regions of interest (ROI) samples for labeling. 
    Normally, for real use case we should select them manually.
    But for the experiments, we use a simple automatic way to select them from the training set.
    Args:
        dinfo_file (str): Path to the data info JSON file.
        num_cls (int): Number of classes in the dataset.
        num_samples_per_cls (int): Number of samples to select per class.
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

    # select samples for each class
    selected_samples = {}
    for cls in range(1, num_cls + 1):
        cls_samples_in_ratio_range = [
            name for name, info in dinfo.items() 
            if info[str(cls)] >= min_ratio and info[str(cls)] <= max_ratio
        ]
        if len(cls_samples_in_ratio_range) == 0:
            print(f"No samples found for class {cls} within the specified ratio range.")
            continue
        if len(cls_samples_in_ratio_range) < num_samples_per_cls:
            print(f"Only {len(cls_samples_in_ratio_range)} samples found for class {cls}, less than the requested {num_samples_per_cls}. Selecting all available samples.")
            selected_samples[cls] = cls_samples_in_ratio_range
        else:
            selected_samples[cls] = np.random.choice(cls_samples_in_ratio_range, num_samples_per_cls, replace=False).tolist()
    
    # save selected samples
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving selected samples to {save_path}")
        with open(save_path, 'w') as f:
            json.dump(selected_samples, f, indent=4)
    
    # save summary to txt
    summary_file = os.path.join(os.path.dirname(dinfo_file), f"selected_{num_samples_per_cls}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Selected {num_samples_per_cls} samples per class from training set:\n")
        for cls, samples in selected_samples.items():
            f.write(f"Class {cls}: {len(samples)} samples\n")
            for s in samples:
                f.write(f"  - {s}\n")
        f.write("\n")

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
    data_root = "/home/yuan/data/HisMap/syntra"
    # train_test_val_split(data_root)

    nshot = 50
    for d in os.listdir(data_root):
        print(f"Processing dataset: {d}")
        selected_samples = select_roi_samples(
            dinfo_file=os.path.join(data_root, d, f"{d}.json"),
            num_cls=5, 
            num_samples_per_cls=nshot, 
            min_ratio=0.05, 
            max_ratio=0.8, 
            save_path=os.path.join(data_root, d, f"selected_{nshot}.json")
        )

        # visualize_selected_samples(
        #     selected_samples, 
        #     root_dir=os.path.join(data_root, d)
        # )