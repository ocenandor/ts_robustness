import numpy as np
from scipy.io.arff import loadarff
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

class FordDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        raw_data, meta = loadarff(path)
        self.raw_data = np.array(raw_data.tolist(), dtype=float)
        self.column_names = meta.names()
        self.data = self.raw_data[:, :-1].astype(np.float32)
        self.data = np.expand_dims(self.data.data, 2)
        self.labels = self.raw_data[:, -1].astype(np.float32)
        self.labels[np.where(self.labels == -1)] = 0
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)

