import json
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler #FIXME change imports
from ignite.contrib.handlers import wandb_logger
from ignite.engine import (Engine, Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import ModelCheckpoint
from ignite.handlers.param_scheduler import LRScheduler
from ignite.metrics import Accuracy, Loss
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import wandb

sys.path.append('..')
from train import train

from src.datasets import FordDataset
from src.models import TransformerClassification
from src.utils import build_optimizer, duplicate_batch, split_batch, str2torch

if __name__ == '__main__': #TODO need test
    # Load config 
    with open('configs/sweeps/sweep3_cnn.json') as f:
        config = json.load(f)
    # Init wandb sweep
    sweep_id = wandb.sweep(entity='ts-robustness', sweep=config, project="ml-course")
    wandb.agent(sweep_id, partial(train, dataset_dir='data/FordA'), count=500)