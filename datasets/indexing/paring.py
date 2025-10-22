# Decription: Generate src-tgt pairing info to json file. Given a target image, find its k nearest source images 
# Author: Yunshuang Yuan
# Date: 2025-10-01
# All rights reserved.

import os
import json

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from indexdb import IndexDatabase, DinoV2Encoder, BaseDataloaders


def generate_src_tgt_intra_cls(data_root, nshot, k=8):
    img_encoder = DinoV2Encoder()
    for dataset in os.listdir(data_root):
        with open(os.path.join(data_root, dataset, f'selected_{nshot}.json'), 'r') as f:
            data_info = json.load(f)
        
        pairing_info = {}
    
        for c, cls_samples in data_info.items():
            # build index db for current cls_samples
            map_dataloader = BaseDataloaders(data_root, cls_samples, batch_size=4)
            index_db = IndexDatabase(map_dataloader=map_dataloader, img_encoder=img_encoder)
            for item in cls_samples:
                sample_name = item.rsplit('.', 1)[1] 
                distances, indices =  index_db.internel_search(
                    dataset, sample_name, k=k) 
                # remove self from the pairing list
                mask = distances>1e-3
                distances = distances[mask]
                indices = indices[mask]
                # update pairing info
                sample_name = f"{dataset}.{c}." + sample_name 
                pairing_info[sample_name] = {
                    'src_samples': [f"{dataset}.{c}." + cls_samples[i].rsplit('.', 1)[1] for i in indices],
                    'distances': distances.tolist()
                }
            del index_db
            del map_dataloader

        # save pairing info
        save_path = os.path.join(data_root, dataset, f'paring_{nshot}_intracls.json')
        with open(save_path, 'w') as f:
            json.dump(pairing_info, f, indent=4)

def load_samples(data_info_file, remove_repeat=True):
    if '.json' in data_info_file:
        with open(data_info_file, 'r') as f:
            data_info = json.load(f)
        all_samples = []
        for cls_samples in data_info.values():
            all_samples += cls_samples
        
    else:
        with open(data_info_file, 'r') as f:
            all_samples = f.read().splitlines()
    if remove_repeat:
        all_samples = list(set(all_samples))
    return all_samples


def generate_src_tgt_inter_cls(data_root, tgt_info_file, src_info_file=None, k=8, visualize=False):
    img_encoder = DinoV2Encoder()
    if src_info_file is None:
        src_info_file = tgt_info_file
    for dataset in os.listdir(data_root):
        if not os.path.isdir(os.path.join(data_root, dataset)):
            continue

        cur_tgt_info_file = os.path.join(data_root, dataset, tgt_info_file)
        cur_src_info_file = os.path.join(data_root, dataset, src_info_file)

        tgt_samples = load_samples(cur_tgt_info_file, remove_repeat=False)
        src_samples_no_repeat = load_samples(cur_src_info_file, remove_repeat=True)
        
        pairing_info = {}
    
        # build index db for all samples
        map_dataloader = BaseDataloaders(data_root, src_samples_no_repeat, batch_size=4)
        index_db = IndexDatabase(map_dataloader=map_dataloader, img_encoder=img_encoder, merge_patch_token=False)

        for sample in tgt_samples:
            sample_name = sample.rsplit('.', 1)[1] 
            sample_img_path = os.path.join(data_root, dataset, 'imgs', f"{sample_name}.png")
            distances, indices =  index_db.external_search(
                dataset, sample_img_path, k=k) 
            # remove self from the pairing list
            mask = distances>1e-3
            distances = distances[mask]
            indices = indices[mask]
            if len(indices) < k - 1:
                print(f"Warning: sample {sample} has less than {k-1} neighbors. Final selected neighbors: {len(indices)}")
            # update pairing info
            pairing_info[sample] = {
                'src_samples': [src_samples_no_repeat[i] for i in indices],
                'distances': distances.tolist()
            }
        del index_db
        del map_dataloader

        # save pairing info
        src_mode = os.path.splitext(src_info_file)[0].replace('selected_', '').replace('_samples', '')
        tgt_mode = os.path.splitext(tgt_info_file)[0].replace('selected_', '').replace('_samples', '')
        save_path = os.path.join(data_root, dataset, f'pairing_{src_mode}_{tgt_mode}_intercls.json')
        with open(save_path, 'w') as f:
            json.dump(pairing_info, f, indent=4)
        
        if visualize:
            visualize_generateion(dataset, pairing_info, mode='inter', n_vis = 5)



def visualize_generateion(dataset, pairing_info, mode='inter', n_vis = 5):

        img_dir = os.path.join(data_root, dataset, 'imgs')
        # select a few samples to visualize 
        if mode == 'intra':
            vis_keys = []
            # select 5 samples per class
            for cls in range(1, 6):
                cls_keys = [k for k in pairing_info.keys() if k.split('.')[2]==str(cls)]
                vis_keys += np.random.choice(cls_keys, size=n_vis, replace=False).tolist()
        else:
            vis_keys = np.random.choice(list(pairing_info.keys()), size=n_vis, replace=False).tolist()

        for tgt in vis_keys:
            info = pairing_info[tgt]
            tgt_name = tgt.rsplit('.', 2)[-1]
            src_names = [s.rsplit('.', 2)[-1] for s in info['src_samples']]
            n_src = len(src_names)
            fig, ax = plt.subplots(1, n_src+1, figsize=(4*(n_src+1), 4))
            fig.suptitle(f"Target: {tgt}", fontsize=16)
            tgt_img = Image.open(os.path.join(img_dir, f"{tgt_name}.png")).convert("RGB")
            ax[0].imshow(tgt_img)
            ax[0].set_title("Target")
            ax[0].axis('off')
            for i, src_name in enumerate(src_names):
                src_img = Image.open(os.path.join(img_dir, f"{src_name}.png")).convert("RGB")
                ax[i+1].imshow(src_img)
                ax[i+1].set_title(f"Source {i+1}")
                ax[i+1].axis('off')
            plt.show()


if __name__=="__main__":
    data_root = "/home/yuan/data/HisMap/syntra384_sheets"
    for tgt_info_file in ['test_samples.txt', 'selected_100.json', 'val_samples.txt']:
        generate_src_tgt_inter_cls(data_root,
                                tgt_info_file=tgt_info_file,
                                src_info_file='selected_100.json',
                                k=8, visualize=False)