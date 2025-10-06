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


if __name__=="__main__":
    root_dir = '/home/yuan/data/HisMap/syntra/hameln.b/lbls'
    colors = [
    (34, 139, 34),    # 'wald' (forest): Forest Green
    (124, 252, 0),    # 'grünland' (grassland): Lawn Green
    (220, 20, 60),    # 'siedlung' (settlement): Crimson
    (30, 144, 255),   # 'fließgewässer' (flowing water): Dodger Blue
    (0, 191, 255)     # 'stillgewässer' (still water): Deep Sky Blue
    ]
    color_map = {2**(8 - i): colors[i - 1] for i in range(1, 6)}
    for f in os.listdir(root_dir):
        filename = os.path.join(root_dir, f)
        convert_encoded_label_to_rgb(filename, color_map)