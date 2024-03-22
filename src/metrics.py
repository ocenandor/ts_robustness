import numpy as np
import pandas as pd
import scipy


def calcul_mean_confidence_interval(iters, statistic=np.mean):
    conf_interval = scipy.stats.bootstrap([iters], statistic).confidence_interval
    return conf_interval.low, conf_interval.high

def form_cell(m, low, high, *args, conf_level=95):
    return f'{m:.3f} {conf_level}% CI [{low:.3f}, {high:.3f}]'

def create_table(results, key='pert_norms', func=np.mean):
    table = pd.DataFrame(index=results.keys()) #methods
    for method in results:
        for model in results[method]: #models
            arr = results[method][model][key] #array
            arr = arr[~np.isnan(arr)]
            metric = func(arr)
            low, high = calcul_mean_confidence_interval(arr, func)
            cell = form_cell(metric, low, high)
            table.loc[method, model] = cell
    return table

def calcul_average_rank_single_method(method_results, key='pert_norms'):
    ranks = []
    df = pd.DataFrame(columns=method_results.keys())
    table = pd.DataFrame(columns=method_results.keys())
    # return df
    for model in df.columns:
        df[model] = method_results[model][key]
        
    rangs = np.argsort(df)
    for i, model in enumerate(df.columns): # for every model
        rank = np.where(rangs==i)[1] + 1 
        low, high = calcul_mean_confidence_interval(rank)
        table.loc[0, model] = form_cell(rank.mean(), low, high)
    return table

def calcul_average_rank(results, key='pert_norms'):
    table = pd.DataFrame() #methods
    for i, (method, method_results) in enumerate(results.items()):
        ranks = calcul_average_rank_single_method(method_results, key)
        table = pd.concat([table, ranks], axis=0)
    table.index = list(results.keys())
    return table