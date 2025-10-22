import os
from geotiff import GeoTiff
import numpy as np
from PIL import Image


def read_tiff(filename):
    if not os.path.exists(filename):
        print('File not found: {}'.format(filename))
        return None
    geo_tiff = GeoTiff(filename, crs_code=25832)
    tiff_array = np.array(geo_tiff.read())
    return tiff_array


def convert_encoded_label_to_rgb(label_path, color_map):
    label = Image.open(label_path)
    label = np.array(label)
    h, w = label.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_value, color in color_map.items():
        mask = label == class_value
        rgb_image[mask] = color
    rgb_image = Image.fromarray(rgb_image).convert("RGB")
    rgb_image.save(label_path)

    return rgb_image


def update_pil_palette(pil_img, color_map):
    # Build 256*3 palette
    palette = [0] * (256 * 3)
    for class_value, (r, g, b) in color_map.items():
        palette[3*class_value : 3*class_value+3] = [r, g, b]
    pil_img.putpalette(palette)
    return pil_img

def update_lbl_palette(root_dir):
    colors = [
        [0, 0, 0],        # background
        [34, 139, 34],    # 'wald' (forest): Forest Green
        [124, 252, 0],    # 'grünland' (grassland): Lawn Green
        [220, 20, 60],    # 'siedlung' (settlement): Crimson
        [30, 144, 255],   # 'fließgewässer' (flowing water): Dodger Blue
        [128, 0, 0],      # 'railway': Maroon
        [0, 128, 0],      # 'vineyear': Green
    ]
    nonlabeled_color = np.array([255, 255, 255])  # white for nonlabeled
    colors = np.array(colors).reshape(-1)
    pil_palette = np.array([0] * (256 * 3))
    pil_palette[:len(colors)] = colors
    pil_palette[-3:] = nonlabeled_color
    pil_palette = pil_palette.astype(np.uint8).tolist()

    for dataset in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, dataset)):
            continue
        dataset_dir = os.path.join(root_dir, dataset, 'lbls')
        for filename in os.listdir(dataset_dir):
            mask_path = os.path.join(dataset_dir, filename)
            masks = Image.open(mask_path)
            masks_rgb = np.array(masks.convert("RGB"))
            # uniq_colors = np.unique(np.array(masks_rgb).reshape(-1, 3), axis=0)
            # print(filename, uniq_colors)

            new_masks = np.zeros_like(masks_rgb[..., 0])
            for color in colors.reshape(-1, 3):
                if (color == [0, 0, 0]).all():
                    continue
                cls_id = np.where((colors.reshape(-1, 3) == color).all(axis=1))[0][0]
                new_masks[(masks_rgb == color.reshape(1, 1, 3)).all(axis=2)] = cls_id
            new_masks[(masks_rgb == nonlabeled_color.reshape(1, 1, 3)).all(axis=2)] = 255

            new_masks = Image.fromarray(new_masks, mode="P")
            new_masks.putpalette(pil_palette)
            new_masks.save(mask_path, format="PNG")


if __name__=="__main__":
    update_lbl_palette('/home/yuan/data/HisMap/syntra384')
