import sys
import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

sys.path.append('../')
from src.datasets import make_dataset
from src.models import LSTMClassification
from src.utils import str2torch
from src.attacks.BIM import BIM

def test(config, weights, epsilon=0.01, max_iteration=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['train']['optimizer'] = str2torch(config['train']['optimizer'])

    _, test_dataset, _, _ = make_dataset(config, args.data, return_loader=False)
    length_dataset = len(test_dataset)
    
    model = LSTMClassification(config).to(device) #TODO from config
    weights = torch.load(weights, map_location='cpu')
    model.load_state_dict(weights)

    number_of_iterations = []
    correct_pure = 0
    correct_advers = 0
    for i in tqdm(range(length_dataset)):
        input = torch.tensor(test_dataset[i][0]).to(device)
        target = torch.tensor(test_dataset[i][1]).to(device)
        
        iter = 0
        predict_is_inverse = False
        
        while((not predict_is_inverse) and (iter<max_iteration)):
            prediction = model.forward(input.unsqueeze(0))
            adversarial_input = BIM(model, input, target, epsilon=epsilon, num_iterations=iter)
            adversarial_prediction = model.forward(adversarial_input)

            if(prediction.round() != adversarial_prediction.round()):
                predict_is_inverse = True
            else:   
                iter += 1
        
        correct_pure += (prediction.to('cpu').round() == target.to('cpu')).sum().item()
        correct_advers += (adversarial_prediction.to('cpu').round() == target.to('cpu')).sum().item()
        
        number_of_iterations.append(iter)
        
    print("___________________________________________")
    print(f'{np.array(number_of_iterations).mean():.3f} +- {np.array(number_of_iterations).std():.3f} iterations for invert predictions')
    print(f'Accuracy for not attacked dataset: {correct_pure/length_dataset}')
    print(f'Accuracy on attacked dataset: {correct_advers/length_dataset}')
    print("___________________________________________")
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="train json config")
    parser.add_argument("weights", type=str, help="weights of model")
    parser.add_argument("--data", type=str, default='data/FordA', help="path to data folder")
    
    args = parser.parse_args()

    with open(args.config) as f:
        config =  json.load(f)

    test(config, args.weights)
