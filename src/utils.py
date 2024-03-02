import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from warmup_scheduler import GradualWarmupScheduler
from torch.optim import Adam, AdamW, SGD
from torch.functional import F


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
    xs = torch.tensor(np.array(xs)).squeeze(-1).permute(0, 2, 1)
    ys = torch.tensor(np.array(ys))
    return [xs, ys]

def duplicate_batch(batch, n_splits=4):
    xs = []
    ys = []
    for x, y in batch:
        xs.append(np.tile(x, n_splits).reshape(n_splits, -1))
        ys.append(y) 
    xs = torch.tensor(np.array(xs)).squeeze(-1).permute(0, 2, 1)
    ys = torch.tensor(np.array(ys))
    return [xs, ys]

def str2torch(method: str):
    torch_dict = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD,
        'elu': F.elu,
        'relu': F.relu,
        'sigmoid': F.sigmoid
    }
    return torch_dict[method]

def build_optimizer(config, model):
    return config['train']['optimizer'](model.parameters(), lr=config['train']['lr'])

def build_scheduler(optimizer, scheduler_config):
    warmup = None
    scheduler = None

    scheduler_type = scheduler_config.get("type", None)

    if scheduler_type == "StepLR":
        step_size = scheduler_config.get("step_size", 10)
        gamma = scheduler_config.get("gamma", 0.5)
        scheduler =  lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "ExponentialLR":
        gamma = scheduler_config.get("gamma", 0.95)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == "CosineLR":
        T_max = scheduler_config.get("T_max", 10)
        eta_min = scheduler_config.get("eta_min", 0)
        gamma = scheduler_config.get("gamma", 1)
        def lr_lambda(epoch):
            return max(eta_min, 5 * (1 + np.cos(np.pi * epoch / T_max))) * gamma ** (epoch)        
        scheduler = LambdaLR(optimizer, lr_lambda)

    if scheduler_config["warmup_epochs"] >= 1 and scheduler_config['type'] != 'CosineLR':
        warmup = GradualWarmupScheduler(optimizer, multiplier=1.5, total_epoch=5, after_scheduler=scheduler)
    
    return scheduler, warmup


def transform_config(config: dict):
    """Function to transform config from first experiments to wandb sweep parameters config
    Args:
        config()
    """
    def transform_section(section):
        transformed = {}
        for key, value in section.items():
            if isinstance(value, dict):
                transformed[key] = {"parameters": transform_section(value)}
            else:
                transformed[key] = {"values": [value]}
        return transformed
    
    transformed_config = {"parameters": transform_section(config)}
    return transformed_config