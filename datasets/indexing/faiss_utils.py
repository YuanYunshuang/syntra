import os
import numpy as np
import faiss  
import faiss.contrib.torch_utils 


class FAISSIndexer:
    def __init__(self, dimension):
        self.dimension = dimension
        # use a single GPU
        self.res = faiss.StandardGpuResources()  
        # build a flat GPU index
        # self.gpu_index_flat = faiss.index_cpu_to_gpu(self.res, 0, faiss.IndexFlatL2(dimension))
        self.gpu_index_flat = faiss.GpuIndexFlatL2(self.res, dimension)

    def add(self, vectors):
        self.gpu_index_flat.add(vectors.contiguous())

    def search(self, query_vectors, k):
        assert k > 0, "k should be a positive integer"
        distances, indices = self.gpu_index_flat.search(query_vectors, k)
        return distances, indices

    def save(self, file_path):
        cpu_index = faiss.index_gpu_to_cpu(self.gpu_index_flat)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        faiss.write_index(cpu_index, file_path)

    def load(self, file_path):
        cpu_index = faiss.read_index(file_path)
        self.gpu_index_flat = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
    
    def __len__(self):
        return self.gpu_index_flat.ntotal
    
    def reset(self):
        self.gpu_index_flat.reset()

    def is_trained(self):
        return self.gpu_index_flat.is_trained
    
    def train(self, training_vectors):
        self.check(training_vectors)
        self.gpu_index_flat.train(training_vectors)