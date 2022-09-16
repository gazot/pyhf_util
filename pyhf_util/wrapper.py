import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import json 
import seaborn as sns
from matplotlib import rcParams

from typing import Tuple
from .utils import *


# --------------------------------------------


def load_compile_benchmark(models, prefix, alg='MH'):
    """
    >>> load_compile_benchmark(['2_bin_corr', '2_bin_uncorr'], ['no_jax', 'do_jax'])
    """
    time, memory = list(), list()   # each list contains 4 elements

    for model in models: 
        for p in prefix: 
            path  = "data/{}/{}_{}".format(model, p, alg)
            jdata = load_json(path)

            time.append(jdata['compile_time'])
            memory.append(jdata['compile_bytes'])

    # setup DataFrame
    time_df = pd.DataFrame({'time_1': time[:2], 'time_2': time[2:]}, index = ['numpy', 'jax'])
    mem_df = pd.DataFrame({'mem_1': memory[:2], 'mem_2': memory[2:]}, index = ['numpy', 'jax'])
    
    return time_df, mem_df


def df_stats(df, index, item: str = 'Model'):
    """Compute median and std of a DataFrame (<index>, value, <item>)"""
    
    model_names = df['Model'].unique()
    # get (new) unique indices (nsteps, backend)
    index_entries = df[index].unique()

    mu, std = list(), list()
    for m in model_names:
        mu_i, std_i = list(), list()
        for n in index_entries: 
            tmp = df.loc[(df[item] == m) & (df[index] == n)]
            mu_i.append(tmp.median().value)
            std_i.append(tmp.std().value)
        mu.append(mu_i)
        std.append(std_i)

    return pd.DataFrame(dict(zip(model_names, mu)), index=index_entries), std


def load_run_time_benchmark(models: list, prefix: list, alg='MH'):
    """
    Load run_times for all combinations of models & prefix and returns a pd.DataFrame for each element in prefix.
    """
    # short model names
    model_names = ['m'+str(i) for i, _ in enumerate(models, 1)]

    # makro for stacking 
    stack = lambda arr: np.hstack([np.hstack(arr[0]), np.hstack(arr[1])])

    # one DataFrame for each prefix
    df_list = list()

    for p in prefix: 
        times, steps, model_list = list(), list(), list()

        for i, model in enumerate(models):
            path  = "data/{}/{}_{}".format(model, p, alg)
            data = np.load(path + '.npz')
            arrs = np.array([data[file] for file in data.files])
            times.append(arrs)

            # nsteps from json
            nsteps= load_json(path)['nsteps']
            # create meta data based on arrs dimension
            dim = arrs.shape 
            steps += steps_str(nsteps, dim[-1])
            model_list += [model_names[i]] * np.prod(dim)

        df = pd.DataFrame({'nsteps': steps, 'value': stack(times), 'Model': model_list}) 
        df_list.append(df)

    return df_list


def load_llh_benchmark(models: list, prefix: list, alg=''):
    """
    Load llh benchmarks for models & prefix
    """
    # short model names
    model_names = ['m'+str(i) for i, _ in enumerate(models, 1)]
    back_names = ['numpy', 'jax']

    times, back, model_list = list(), list(), list()

    for (i, model) in enumerate(models):
        for (j, p) in enumerate(prefix): 
            path  = "data/{}/{}_{}".format(model, p, alg)
            data = np.load(path + '.npz')
            file = data.files[0]
            times.append(data[file]*1e3)
                
            dim = times[-1].shape
            model_list += [model_names[i]] * np.prod(dim)
            back += [back_names[j]] * np.prod(dim)

    return pd.DataFrame({'Model': model_list, 'value': np.hstack(times), 'backend': back})