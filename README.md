# SYNTRA – Segmenting Your Notions Through Retrieval-Augmentation

## Installation
Requirement: CUDA11.8
```bash
conda create -n syntra python=3.10
conda activate syntra
# faiss
conda install -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=11.4,<=11.8'
# torch and xformers
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.29 --index-url https://download.pytorch.org/whl/cu118
# install other requirements
pip install tqdm geotiff matplotlib iopath
```
## Prepare data
1. Generate data to the syntra format. Available scripts are in `datasets/generate_datasets`.
2. Generate data index using `modeling/indexdb.py`.
3. Select fewshot samples with `modeling/clustering.py`.

## Map collections
- https://historicalcharts.noaa.gov/
- https://github.com/Archiel19/FRAx4?tab=readme-ov-file#dataset
- https://www.davidrumsey.com/luna/servlet/RUMSEY~8~1
- https://ngmdb.usgs.gov/topoview/viewer/#4/40.00/-100.00


