import argparse
import random
import json
import sys

import numpy as np
import torch
import tqdm

# TODO do more elegant import with packages and __init__
sys.path.append('.')
from src.attacks.bim import bim
from src.attacks.deepfool import deepfool
from src.attacks.simba import simba
from src.datasets import make_dataset
from src.models import (CNNClassification, LSTMClassification,
                        TransformerClassification)

MODELS = {
    'cnn': CNNClassification,
    'transformer': TransformerClassification,
    'lstm': LSTMClassification
    }

ATTACKS = {
    'bim': bim,
    'deepfool': deepfool,
    'simba': simba
    }

ATTACK_PARMS = {
    'bim': (0.02, 100),
    'deepfool': (0.5, 50),
    'simba': (2, 500)
    }


def test(config: dict, weights: str, attack='deepfool',
         parameter=0.02, max_iter=50, data_dir='data/FordA',
         verbose=True, scale=1, device=None):
    np.random.seed(config['random_state'])
    random.seed(config['random_state'])
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ### Initialize model and dataset
    _, test_dataset, _, _ = make_dataset(config, data_dir, return_loader=False)
    model = MODELS[config['model']['name']] # TODO move all such dicts to one place
    model = model(config).to(device)
    weights = torch.load(weights, map_location='cpu')
    model.load_state_dict(weights)

    ### Initialize attack
    attack = ATTACKS[attack] 

    ### Test model
    L = len(test_dataset)
    number = int(scale * L)
    iters = np.zeros(L) * np.NaN
    pert_norms = np.zeros(L) * np.NaN
    changed = np.arange(L) * np.NaN
    idx = np.random.choice(np.arange(L), number, replace=False)
    

    for i in tqdm.tqdm(idx, disable=not verbose):
        test_sample = torch.from_numpy(test_dataset[i][0]).to(device)
        _, loop_i, is_changed, pert_norm = attack(test_sample, model, parameter, max_iter=max_iter, device=device)
        iters[i] = loop_i
        changed[i] = is_changed
        pert_norms[i] = pert_norm

    return np.array(iters), changed, np.array(pert_norms) #TODO More explicit names

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="train json config")
    parser.add_argument("weights", type=str, help="weights of model")
    parser.add_argument("attack", type=str, help="type of attack")
    parser.add_argument("-s", "--strength", type=float, default=0.01, help="strength parameter of attack")
    parser.add_argument("--max_iter", type=int, default=30, help="maximum number of iterations to attack")
    parser.add_argument("--data", type=str, default='data/FordA', help="path to data folder")
    parser.add_argument("--scale", type=float, default=1, help="part of dataset to attack")

    
    args = parser.parse_args()

    with open(args.config) as f:
        config =  json.load(f)
    
    iters, changed, pert_norms = test(config, args.weights, args.attack.lower(), args.strength, args.max_iter, data_dir=args.data, scale=args.scale)
    model_name = config['model']['name']
    unique, counts = np.unique(iters, return_counts=True)
    print('mean robustness')
    print(f'{np.nanmean(pert_norms):.4f}')
    changed = changed[~np.isnan(changed)]
    print(f'model changed the decision for {changed.mean() * 100 :.1f}% of data')
