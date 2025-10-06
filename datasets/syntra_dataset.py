import os
import numpy as np
import json
from PIL import Image
import torch

from torch.utils.data import Dataset


class SyntraDataset(Dataset):
    def __init__(self, root_dir, selection, transform=None, selected_datasets=None):
        """
        A PyTorch Dataset class for the Syntra dataset.
        Args:
            root_dir (str): Root directory of the dataset.
            selection(str): Data info JSON file name without extention.
            transform (callable, optional): Optional transform to be applied on a sample.
            selected_datasets (list, optional): List of selected dataset names to include. If None, include all.
        """
        # list all datasets folders
        self.datasets = sorted(os.listdir(root_dir))
        if selected_datasets:
            self.datasets = [d for d in self.datasets if d in selected_datasets]
        # load data info
        self.data_info = {}
        for dset in self.datasets:
            data_info_file = os.path.join(root_dir, dset, f'{selection}.json')
            with open(data_info_file, 'r') as f:
                self.data_info[dset] = json.load(f)
        
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'imgs')
        self.lbl_dir = os.path.join(root_dir, 'lbls')
        self.transform = transform
        self.img_files = sorted(os.listdir(self.img_dir))