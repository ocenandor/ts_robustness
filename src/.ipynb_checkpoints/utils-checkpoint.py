import numpy as np
import torch


def split_batch(batch, n_splits=1):
    """
    collate function for torch Dataloader
    to efficiently train time series models one has to split them in small sequences
    (instead of training sequence [500, 1] it's better to train [125, 4])

    Args:
        batch (list): list of size (batch size) with elemnets dataset
        n_splits (int, optional): number of chunks after splitting. Defaults to 1.

    Returns:
        list[torch.tensor, torch.tensor]: 
    """
    assert 500 % n_splits == 0
    xs = []
    ys = []
    for x, y in batch:
        xs.append(np.split(x, n_splits))
        ys.append(y) 
    xs = torch.tensor(xs).squeeze(-1).permute(0, 2, 1)
    ys = torch.tensor(ys)
    return [xs, ys]

def build_optimizer(config, model):
    return config['train']['optimizer'](model.parameters(), lr=config['train']['lr'])