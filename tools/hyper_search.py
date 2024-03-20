import json
import sys
from functools import partial

import wandb

sys.path.append('..')
from train import train

if __name__ == '__main__': #TODO need test
    # Load config 
    with open('configs/sweeps/sweep3_cnn copy.json') as f: #TODO make arg parser
        config = json.load(f)
    # Init wandb sweep
    sweep_id = wandb.sweep(entity='<ENTITY>', sweep=config, project="<PROJECT>")
    wandb.agent(sweep_id, partial(train, dataset_dir='data/FordA'), count=300)