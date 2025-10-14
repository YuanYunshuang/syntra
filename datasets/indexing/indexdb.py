import os
import json
import tqdm
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from faiss_utils import FAISSIndexer

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")


class BaseDataset(Dataset):
    def __init__(self, data_root, filenames):
        self.filenames = [f + '.png' for f in filenames]  # currently assuming all images are .png
        self.data_root = os.path.join(data_root, 'imgs') 

                # Define preprocessing for the image
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(           # Normalize using ImageNet mean and std
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        full_path = os.path.join(self.data_root, filename)
        image = Image.open(full_path).convert('RGB')
        image = self.preprocess(image)  
        return full_path, image
    
    def collate_fn(self, batch):
        paths, images = zip(*batch)
        images = torch.stack(images, dim=0)
        return paths, images
    

class BaseDataloaders:
    def __init__(self, data_root, sample_list, batch_size=32, shuffle=False, num_workers=4):
        """
        data_root: Root directory where images are stored.
        sample_list: list of samples or TXT file for the list of samples for loading.
        """
        if isinstance(sample_list, str):
            with open(sample_list, 'r') as f:
                filenames = [line.strip() for line in f.readlines()]
        else:
            filenames = sample_list
        self.datasets = {}
        for fn in filenames:
            dataset, filename = fn.rsplit('.', 1)
            if dataset not in self.datasets:
                self.datasets[dataset] = []
            self.datasets[dataset].append(filename)
        # instantiate dataset and dataloader
        self.dataloaders = {}
        for dataset, filenames in self.datasets.items():
            self.datasets[dataset] = BaseDataset(os.path.join(data_root, dataset), filenames)
            self.dataloaders[dataset] = DataLoader(self.datasets[dataset], 
                                                   batch_size=batch_size, 
                                                   shuffle=shuffle, 
                                                   num_workers=num_workers)

    def get_loader(self, dataset):
        return self.dataloaders[dataset]
    
    def get_datasets(self):
        return list(self.datasets.keys())


class DinoV2Encoder:
    def __init__(self):
        # Load the DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', source='github')
        self.model.cuda().eval()  # Set the model to evaluation mode

    def generate_embedding(self, input_tensor):
        # Generate embeddings
        with torch.no_grad():  # Disable gradient computation
            embeddings = self.model.forward_features(input_tensor)

        return embeddings
    

class IndexDatabase:
    def __init__(self, db_path=None, map_dataloader=None, img_encoder=None, 
                 index_dim=384, cache_original_vec=True, merge_patch_token=True):
        self.db_path = db_path
        self.index_dim = index_dim
        self.cache_original_vec = cache_original_vec
        self.merge_patch_token = merge_patch_token
        self.img_encoder = DinoV2Encoder() if img_encoder is None else img_encoder
        self.img_index = {}
        self.encoded_vectors = {}
        self.patch_index = {}
        self.map_idx_to_path = {}
        self.map_idx_to_name = {}
        self.map_name_to_idx = {}
        # if map dataloader is None, load from db_path
        if map_dataloader is None:
            assert os.path.exists(db_path), f"db_path {db_path} does not exist, please provide a valid path or a map_dataloader to generate the index."
            self.load_index(db_path)
        else:
            self.generate_index(map_dataloader)
    
    @property
    def datasets(self):
        return list(self.img_index.keys())
    
    def generate_index(self, map_dataloader):
        datasets = map_dataloader.get_datasets()
        for dataset in datasets:
            dataloader = map_dataloader.get_loader(dataset)
            faiss_index = FAISSIndexer(dimension=384)  # DINOv2 Vit-S/14 embedding dimension
            cur_paths = []
            cur_img_vectors = []
            pbar = tqdm.tqdm(dataloader)
            pbar.set_postfix({'Processing dataset': dataset})
            for paths, imgs in pbar:
                imgs = imgs.cuda()
                emb_dict = self.img_encoder.generate_embedding(imgs)
                token = emb_dict['x_norm_clstoken']  # (B, C)
                if self.merge_patch_token:
                    patch_token = emb_dict['x_norm_patchtokens']  # (B, 256, C)
                    # merge cls and patch tokens
                    token = token + patch_token.mean(dim=1)  # (B, C)
                cur_paths.extend(paths)
                faiss_index.add(token)
                if self.cache_original_vec:
                    cur_img_vectors.append(token)
            print(f"Processed {len(cur_paths)} images in dataset {dataset}")
            self.img_index[dataset] = faiss_index
            if self.cache_original_vec:
                self.encoded_vectors[dataset] = torch.cat(cur_img_vectors, dim=0)
            self.map_idx_to_path[dataset] = cur_paths
            self.map_idx_to_name[dataset] = {
                i: os.path.basename(p).split('.')[0] for i, p in enumerate(cur_paths)}
            self.map_name_to_idx[dataset] = {
                k: v for v, k in self.map_idx_to_name[dataset].items()}

    
    def save_index(self):
        assert self.db_path is not None, "db_path must be specified to save the index."
        for dataset in self.datasets:
            faiss_index = self.img_index[dataset]
            cur_paths = self.map_idx_to_path[dataset]
            if faiss_index is None or len(cur_paths) == 0:
                print(f"No index or paths found for dataset {dataset}, skipping save.")
                continue
            print(f"Saving index for dataset {dataset} with {len(cur_paths)} entries.")
            # Save the index to disk
            os.makedirs(self.db_path, exist_ok=True)
            index_file = os.path.join(self.db_path, dataset, "index.faiss")
            faiss_index.save(index_file)
            # Save the index map to disk
            map_file = os.path.join(self.db_path, dataset, "index_map.json")
            with open(map_file, 'w') as f:
                json.dump(cur_paths, f)


    def load_index(self, db_path):
        datasets = [x for x in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, x))]
        for dataset in datasets:
            index_file = os.path.join(db_path, dataset, f"index.faiss")
            map_file = os.path.join(db_path, dataset, f"index_map.json")
            if os.path.exists(index_file) and os.path.exists(map_file):
                faiss_index = FAISSIndexer(dimension=384)
                faiss_index.load(index_file)
                with open(map_file, 'r') as f:
                    cur_paths = json.load(f)
                self.img_index[dataset] = faiss_index
                self.map_idx_to_path[dataset] = cur_paths
            else:
                print(f"Index files for dataset {dataset} not found in {self.db_path}. Please generate the index first.")

    def internel_search(self, dataset, sample_name, k=4):
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found in index database.")
        try:
            sample_idx = self.map_name_to_idx[dataset][sample_name]
        except ValueError:
            raise ValueError(f"Sample {sample_name} not found in dataset {dataset}.")
        
        faiss_index = self.img_index[dataset]
        if faiss_index is None or len(self.map_idx_to_path[dataset]) == 0:
            raise ValueError(f"No index found for dataset {dataset}. Please generate the index first.")

        # get sample encoding vector
        if self.cache_original_vec and dataset in self.encoded_vectors:
            cls_token = self.encoded_vectors[dataset][sample_idx].unsqueeze(0)
        else:
            img_path = self.map_idx_to_path[dataset][sample_idx]
            image = Image.open(img_path).convert('RGB')
            image = BaseDataset("", []).preprocess(image).unsqueeze(0).cuda()
            emb_dict = self.img_encoder.generate_embedding(image)
            cls_token = emb_dict['x_norm_clstoken']  # (1, C)
        distances, indices = faiss_index.search(cls_token, k=k)
        return distances[0], indices[0]
    
    def external_search(self, dataset, query_img_path, k=4):
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found in index database.")
        if not os.path.exists(query_img_path):
            raise ValueError(f"Query image {query_img_path} does not exist.")
        image = Image.open(query_img_path).convert('RGB')
        image = BaseDataset("", []).preprocess(image).unsqueeze(0).cuda()
        emb_dict = self.img_encoder.generate_embedding(image)
        cls_token = emb_dict['x_norm_clstoken']  # (1, C)
        faiss_index = self.img_index[dataset]
        if faiss_index is None or len(self.map_idx_to_path[dataset]) == 0:
            raise ValueError(f"No index found for dataset {dataset}. Please generate the index first.")
        distances, indices = faiss_index.search(cls_token, k=k)
        return distances[0], indices[0]

    def load_image_by_index(self, dataset, index):
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found in index database.")
        if index < 0 or index >= len(self.map_idx_to_path[dataset]):
            raise IndexError(f"Index {index} out of bounds for dataset {dataset}.")
        img_path = self.map_idx_to_path[dataset][index]
        image = Image.open(img_path).convert('RGB')
        image = BaseDataset("", []).preprocess(image)
        return img_path, image
    
    def visualize_search_results(self, query_img_path, queried_inds, dataset_name, save_path=None):
        if save_path is None:
            save_path = os.path.join(os.environ.get('HOME'), 'Downloads', "search_result.png")
            print(f"Save path not provided, saving to {save_path}")
        query_img = Image.open(query_img_path).convert('RGB')
        query_img = query_img.resize((224, 224))
        queried_paths = [self.map_idx_to_path[dataset_name][idx] for idx in queried_inds]
        queried_imgs = [Image.open(path).convert('RGB').resize((224, 224)) for path in queried_paths]
        all_imgs = [query_img] + queried_imgs
        widths, heights = zip(*(i.size for i in all_imgs))
        margin = 10   
        total_width = sum(widths) + (len(all_imgs) - 1) * margin
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in all_imgs:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0] + margin
        new_im.save(save_path)

    def __repr__(self):
        return f"IndexDatabase({self.img_index})"
    

def generate_indices(data_root, sample_list_txt, db_path, batch_size=64):
    map_dataloader = BaseDataloaders(data_root, sample_list_txt, batch_size=batch_size)
    index_db = IndexDatabase(db_path, map_dataloader)
    index_db.save_index()
    return index_db


def generate_indices_for_all_datasets(data_root, db_path, batch_size=64):
    datasets = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    img_encoder = DinoV2Encoder()
    for dataset in datasets:
        sample_list_txt = os.path.join(data_root, dataset, f"{dataset}.txt")
        if not os.path.exists(sample_list_txt):
            print(f"Sample list txt {sample_list_txt} does not exist, skipping dataset {dataset}.")
            continue
        print(f"Generating index for dataset {dataset}...")
        map_dataloader = BaseDataloaders(data_root, sample_list_txt, batch_size=batch_size)
        index_db = IndexDatabase(db_path, map_dataloader, img_encoder)
        index_db.save_index()
        print(f"Dataset: {dataset}, Number of images indexed: {len(index_db.img_index[dataset])}")


def search_demo(index_db, query_dataset_name, query_sample_list_txt):
    query_dataloader = BaseDataloaders(data_root, query_sample_list_txt, batch_size=1).get_loader(query_dataset_name)
    faiss_index = index_db.img_index[query_dataset_name]

    for paths, imgs in query_dataloader:
        imgs = imgs.cuda()
        emb_dict = index_db.img_encoder.generate_embedding(imgs)
        cls_token = emb_dict['x_norm_clstoken']
        
        distances, indices = faiss_index.search(cls_token, k=5)
        print(f"Query image: {paths[0]}")
        print(f"Top 5 nearest neighbors in dataset {dataset}:")
        index_db.visualize_search_results(paths[0], indices[0], query_dataset_name, save_path=None)
        print(distances)


if __name__ == "__main__":
    data_root = "/home/yuan/data/HisMap/syntra"
    db_path = "assets/index_db"
    batch_size = 64
    regenerate_index = True

    if regenerate_index:
        index_db = generate_indices_for_all_datasets(data_root, db_path=db_path, batch_size=batch_size)
    else:
        index_db = IndexDatabase(db_path)

    

       