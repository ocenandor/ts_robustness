import argparse
import json

import joblib
import numpy as np
import torch
import tqdm

from src.attacks.deepfool import deepfool
from src.datasets import make_dataset
from src.models import TransformerClassification, CNNClassification
from src.utils import str2torch

MODELS = {
    'cnn': CNNClassification,
    'transformer': TransformerClassification
    }


def test(config, weights, overshoot=0.02):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['train']['optimizer'] = str2torch(config['train']['optimizer'])

    _, test_dataset, _, _ = make_dataset(config, args.data, return_loader=False)
    model = MODELS[config['model']['name']] # TODO move all such dicts to one place
    model = model(config).to(device)
    weights = torch.load(weights, map_location='cpu')
    model.load_state_dict(weights)

    iters = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        test_sample = torch.from_numpy(test_dataset[i][0]).unsqueeze(1).to(device)
        r_tot, loop_i, label, k_i, pert_image = deepfool(test_sample, model, 2, max_iter=30, device=device, overshoot=overshoot)
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
    joblib.dump(iters, 'deepfool_transformer.pkl')
    unique, counts = np.unique(iters, return_counts=True)
    print('distribution of number of iterations till inversion of label')
    print(unique)
    print(counts)
