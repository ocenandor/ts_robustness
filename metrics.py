import scipy
import numpy as np

def calcul_mean_confidence_interval(iters, statistic=np.mean):
    conf_interval = scipy.stats.bootstrap([iters], statistic).confidence_interval
    return conf_interval.low, conf_interval.high
