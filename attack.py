import argparse
import json

import joblib
import numpy as np
import torch
import tqdm

from src.attacks.bim import bim # TODO do more elegant import with packages and __init__
from src.attacks.deepfool import deepfool
from src.datasets import make_dataset
from src.models import (CNNClassification, LSTMClassification,
                        TransformerClassification)
from src.utils import str2torch

MODELS = {
    'cnn': CNNClassification,
    'transformer': TransformerClassification,
    'lstm': LSTMClassification
    }

ATTACKS = {
    'bim': bim,
    'deepfool': deepfool
}


def test(config, weights, attack='deepfool', parameter=0.02):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### Initialize model and dataset
    _, test_dataset, _, _ = make_dataset(config, args.data, return_loader=False)
    model = MODELS[config['model']['name']] # TODO move all such dicts to one place
    model = model(config).to(device)
    weights = torch.load(weights, map_location='cpu')
    model.load_state_dict(weights)

    ### Initialize attack
    attack = ATTACKS[attack] 

    ### Test model
    iters = []
    # for i in tqdm.tqdm(range(len(test_dataset))):
    for i in tqdm.tqdm(range(20)):
        test_sample = torch.from_numpy(test_dataset[i][0]).to(device) # TODO unsqueeze for adding "feature" dimension, but should change for more interpretibility
        _, loop_i = attack(test_sample, model, parameter, max_iter=30, device=device)
        iters.append(loop_i)
    return iters

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="train json config")
    parser.add_argument("weights", type=str, help="weights of model")
    parser.add_argument("--attack", type=str, default='deepfool', help="type of attack")
    parser.add_argument("--strength", type=float, default=0.01, help="strength parameter of attack")
    parser.add_argument("--data", type=str, default='data/FordA', help="path to data folder")
    
    args = parser.parse_args()

    with open(args.config) as f:
        config =  json.load(f)

    iters = test(config, args.weights, args.attack)
    model_name = config['model']['name']
    joblib.dump(iters, f'{model_name}_{args.attack}_transformer.pkl')
    unique, counts = np.unique(iters, return_counts=True)
    print('distribution of number of iterations till inversion of label')
    print(unique)
    print(counts)
