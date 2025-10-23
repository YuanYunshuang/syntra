import os
from PIL import Image
import numpy as np

from datasets.generate_datasets.hameln import generate_colormap as gen_cmap_hameln
from datasets.generate_datasets.donauwoerth import generate_colormap as gen_cmap_bayern
from datasets.generate_datasets.siegfried import generate_colormap as gen_cmap_siegfried


color_map = {
    0: (0, 0, 0),        # background
    1: (34, 139, 34),    # 'wald' (forest): Forest Green
    2: (124, 252, 0),    # 'grünland' (grassland): Lawn Green
    3: (220, 20, 60),    # 'siedlung' (settlement): Crimson
    4: (30, 144, 255),   # 'fließgewässer' (water): Dodger Blue
    5: (128, 0, 0),      # 'railway': Maroon
    6: (0, 128, 0),      # 'vineyard': Green
    255: (255, 255, 255) # nonlabeled: White
}


palette = [0, 0, 0] * 256
for class_value, color in color_map.items():
    palette[class_value * 3: class_value * 3 + 3] = color

def copy_img(src_path, dst_path):
    subdataset = os.listdir(src_path)
    for item in subdataset:
        dataset, sheet = item.split(".")
        img_path = os.path.join(src_path, item, "imgs")
        out_img_path = os.path.join(dst_path, dataset, "imgs")
        for img_file in os.listdir(img_path):
            img = Image.open(os.path.join(img_path, img_file))
            img.save(os.path.join(out_img_path, f"{sheet}_{img_file}"))

            print(f"Saved image to {os.path.join(out_img_path, f'{sheet}_{img_file}')}")


def generate_smol_dataset(input_dir, output_dir):
    # split dataset according to mapping years and sheet ids
    # datasets = ['hameln', 'bayern', 'siegfried']
    datasets = ['siegfried']
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        dataset_path = os.path.join(input_dir, dataset)

        img_dir = os.path.join(dataset_path, "imgs")
        lbl_dir = os.path.join(dataset_path, "lbls")

        for img_file in os.listdir(img_dir):
            subdir, x, y = img_file.rsplit('_', 2)
            new_filename = f"{x}_{y}"

            output_subdir_img = os.path.join(output_dir, f"{dataset}.{subdir}", "imgs")
            output_subdir_lbl = os.path.join(output_dir, f"{dataset}.{subdir}", "lbls")
            os.makedirs(output_subdir_img, exist_ok=True)
            os.makedirs(output_subdir_lbl, exist_ok=True)

            src_img_path = os.path.join(img_dir, img_file)
            src_lbl_path = os.path.join(lbl_dir, img_file)

            dst_img_path = os.path.join(output_subdir_img, new_filename)
            dst_lbl_path = os.path.join(output_subdir_lbl, new_filename)

            # copy img
            img = Image.open(src_img_path)
            img.save(dst_img_path)

            # convert lbl to rgb paletted and save
            lbl_image = Image.open(src_lbl_path).convert("P")
            # merge class 'fließgewässer' and 'standing water' into one class
            lbl_array = np.array(lbl_image)
            # print(np.unique(lbl_array))
            if "railway" in subdir:
                lbl_array[lbl_array > 0] = 5  
            elif "vineyard" in subdir:
                lbl_array[lbl_array > 0] = 6
            else:
                lbl_array[lbl_array == 5] = 4  # merge class 5 into class 4
            lbl_image = Image.fromarray(lbl_array, mode="P")
            lbl_image.putpalette(palette)
            lbl_image.save(dst_lbl_path)

            # # load and check
            # loaded_lbl_image = Image.open(dst_lbl_path)
            # loaded_lbl_image = loaded_lbl_image.convert("P")
            # loaded_unique_labels = np.unique(np.array(loaded_lbl_image))
            # color_map_check = loaded_lbl_image.getpalette()
            # idx_to_color = {i: tuple(color_map_check[i*3:i*3+3]) for i in range(256)}
            # print({idx: idx_to_color[idx] for idx in loaded_unique_labels})
            # print(loaded_unique_labels)


def generate_color_maps(output_dir):
    subdatasets = os.listdir(output_dir)
    for subdataset in subdatasets:
        if "hameln" in subdataset:
            subdataset_path = os.path.join(output_dir, subdataset)
            gen_cmap_hameln(subdataset_path)
        elif "bayern" in subdataset:
            subdataset_path = os.path.join(output_dir, subdataset)
            gen_cmap_bayern(subdataset_path)
        elif "siegfried" in subdataset:
            subdataset_path = os.path.join(output_dir, subdataset)
            gen_cmap_siegfried(subdataset_path, subdataset.split('.')[1].split('_')[0])


def generate_split(input_dir, output_dir, split):
    def parse_samples_and_group(sample_list):
        sample_dict = {}
        for sample in sample_list:
            dataset, filename = sample.split('.')
            sheet, k1, k2 = filename.rsplit('_', 2)
            subdir = f"{dataset}.{sheet}"
            if subdir not in sample_dict:
                sample_dict[subdir] = []
            sample_dict[subdir].append(f"{k1}_{k2}")
        return sample_dict

    def save_split_files_to_txt(sample_dict, filename):
        for subdir, samples in sample_dict.items():
            split_file = os.path.join(output_dir, subdir, filename)
            with open(split_file, 'w') as f:
                for sample in samples:
                    f.write(f"{sample}\n")

    if split == "train":
        train_split_file = os.path.join(input_dir, "train_all.txt")
        with open(train_split_file, 'r') as f:
            train_samples = f.read().splitlines()
        train_syntra_dict = parse_samples_and_group(train_samples)
        save_split_files_to_txt(train_syntra_dict, "train_samples.txt")
    elif "val" in split or ("train" in split and "shot" in split):
        val_split_file = os.path.join(input_dir, f"{split}.txt")
        with open(val_split_file, 'r') as f:
            val_samples = f.read().splitlines()
        val_syntra_dict = parse_samples_and_group(val_samples)
        save_split_files_to_txt(val_syntra_dict, f"{split}.txt")
    elif split == "test":
        test_split_files = ["test_hameln.txt", "test_bayern.txt", "test_siegfried.txt"]
        test_samples = []
        for test_split_file in test_split_files:
            test_split_file = os.path.join(input_dir, test_split_file)
            with open(test_split_file, 'r') as f:
                test_samples.extend(f.read().splitlines())
        test_syntra_dict = parse_samples_and_group(test_samples)
        save_split_files_to_txt(test_syntra_dict, "test_samples.txt")

if __name__ == "__main__":
    input_directory = "/koko/datasets/SMOL"
    output_directory = "/koko/datasets/SMOL_syntra"
    # generate_smol_dataset(input_directory, output_directory)
    # generate_color_maps(output_directory)

    generate_split(input_directory, output_directory, split="test")
