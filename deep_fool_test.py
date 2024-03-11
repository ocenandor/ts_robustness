import argparse
import json
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
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

from attacks.deepfool import deepfool
from src.datasets import FordDataset, make_dataset
from src.models import TransformerClassification
from src.utils import build_optimizer, str2torch


def test(config, weights):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['train']['optimizer'] = str2torch(config['train']['optimizer'])

    _, test_dataset, _, _ = make_dataset(config, args.data, return_loader=False)
    model = TransformerClassification(config).to(device) #TODO from config
    weights = torch.load(weights, map_location='cpu')
    model.load_state_dict(weights)

    iters = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        test_sample = torch.from_numpy(test_dataset[i][0]).unsqueeze(1).to(device)
        r_tot, loop_i, label, k_i, pert_image = deepfool(test_sample, model, 2, max_iter=30, device=device)
        iters.append(loop_i)
    return iters

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="train json config")
    parser.add_argument("weights", type=str, help="weights of model")
    parser.add_argument("--data", type=str, default='data/FordA', help="path to data folder")
    
    args = parser.parse_args()

    with open(args.config) as f:
        config =  json.load(f)

    iters = test(config, args.weights)
    unique, counts = np.unique(iters, return_counts=True)
    print('distribution of number of iterations till inversion of label')
    print(unique)
    print(counts)
