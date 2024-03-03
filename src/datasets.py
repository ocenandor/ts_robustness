import numpy as np
from scipy.io.arff import loadarff
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

class FordDataset(Dataset):
    def __init__(self, path, config) -> None:
        super().__init__()
        raw_data, meta = loadarff(path)
        self.raw_data = np.array(raw_data.tolist(), dtype=float)
        self.column_names = meta.names()
        self.data = self.raw_data[:, :-1]
        # self.data = np.expand_dims(self.data.data, 2)
        self.labels = self.raw_data[:, -1]
        self.labels[np.where(self.labels == -1)] = 0
        self.labels = [self.labels[i] * np.ones(self.get_num_subarrays(self.data[i],
                                                                               window_size=config['seq_length'],
                                                                               step=config['step'])) for i in range(len(self.data))]
        self.data = [self.generate_subarrays(self.data[i], window_size=config['seq_length'], step=config['step']) for i in range(len(self.data))]
        self.data = np.vstack(self.data).astype(np.float32)
        self.labels = np.concatenate(self.labels).astype(np.float32)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)
    
    def get_num_subarrays(self, arr, window_size, step=1):
        return (len(arr) - window_size) // step + 1

    def generate_subarrays(self, arr, window_size, step=1, ):
        num_subarrays = self.get_num_subarrays(arr, window_size, step)
        subarrays = np.zeros((num_subarrays, window_size), dtype=arr.dtype)
        
        for i in range(num_subarrays):
            start = i * step
            end = start + window_size
            subarrays[i] = arr[start:end]
        
        return subarrays


