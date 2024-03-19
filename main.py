import json
import sys
import warnings

import joblib

warnings.filterwarnings('ignore')
from glob import glob

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

sys.path.append('.')
from metrics import create_table
from src.utils import open_config
from tools.attack import ATTACKS, test
from tools.train import train

if __name__ == '__main__':

    ATTACK_PARMS = {'bim': (0.002, 200),
                    'deepfool': (0.5, 50),
                    'simba': (10, 1000)}

    configs = glob('configs/*500.json')
    results = {}
    for conf_file in configs: #TODO make with parse args to choose models and attacks
        config = open_config(conf_file)
        model_name = config['model']['name'] #TODO Think about logging, to make it more user-friendly
        print('Train', model_name.upper())
        train(config=config, wandb_log=False, save_dir='demo/', dataset_dir='data/FordA', verbose=False)
        for attack in tqdm(ATTACKS): 
            results.setdefault(attack, {})   
            print('Attack with', attack)  
            iters, _, pert_norms = test(config, f'demo/{model_name}.pt', attack, *ATTACK_PARMS[attack], verbose=False)      
            results[attack][model_name] = {'iters': iters, 'pert_norms': pert_norms}
    
    print('\n                      ==========================')
    print('                                DONE!')
    print('                      ==========================')
    print("Mean robustness models over FordA dataset")
    table = create_table(results)
    print(tabulate(table, headers='keys', tablefmt='psql'))
    table.to_csv('demo/results.csv')      
    print('results saved in results.csv')      
    joblib.dump(results, 'demo/500_results.pkl')
