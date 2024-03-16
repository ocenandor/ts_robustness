import json
import sys
import warnings
warnings.filterwarnings('ignore')
from tabulate import tabulate

from glob import glob
import pandas as pd

sys.path.append('.')
from metrics import calcul_mean_confidence_interval


from tools.train import train, MODELS
from tools.attack import test, ATTACKS

def form_cell(m, low, high, *args, conf_level=95):
    return f'{m:.1f} {conf_level}% CI [{low:.1f}, {high:.1f}]'

if __name__ == '__main__':
    configs = glob('configs/*.json')
    results = pd.DataFrame(columns=MODELS.keys(), index=ATTACKS.keys())

    for conf_file in configs: #TODO make with parse args to choose models and attacks
        with open(conf_file) as f:
            config = json.load(f)
        model_name = config['model']['name'] #TODO Think about logging, to make it more user-friendly
        print('Process', model_name.upper())
        train(config=config, wandb_log=False, save_dir='demo/', dataset_dir='data/FordA')
        for attack in ATTACKS:
            iters = test(config=config, weights=f'demo/{model_name}.pt', attack=attack, verbose=False)
            cell = form_cell(iters.mean(), *calcul_mean_confidence_interval(iters))
            results.loc[attack, model_name] = cell
    print('\n                      ==========================')
    print('                                DONE!           ')
    print('                      ==========================')
    print("Mean number of iterations to change model's decision")
    print(tabulate(results, headers='keys', tablefmt='psql'))
    results.to_csv('results.csv')      
    print('results saved in results.csv')      
