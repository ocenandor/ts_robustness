import numpy as np
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
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


def make_dataset(config, dataset_dir='../data/FordA', val=False, return_loader=True):
    
    train_path = dataset_dir + '/FordA_TRAIN.arff'
    test_path = dataset_dir + '/FordA_TEST.arff'
    train_dataset = FordDataset(train_path, config['data'])
    test_dataset = FordDataset(test_path, config['data'])

    idx = np.arange(len(train_dataset))
    idx_train, idx_val = train_test_split(idx, train_size=0.8, stratify=train_dataset.labels, random_state=config['random_state'])

    if val:
        train_sampler = SubsetRandomSampler(idx_train)
    else:
        train_sampler = SubsetRandomSampler(idx)
    val_sampler = SubsetRandomSampler(idx_val)
    
    if not return_loader:
        return train_dataset, test_dataset, train_sampler, val_sampler
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], sampler=train_sampler)
        val_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], sampler=val_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=config['data']['batch_size'])
        return train_dataloader, val_dataloader, test_dataloader