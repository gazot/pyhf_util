import numpy as np 
import psutil           # memory usage
import json
import os 

from time import time
from typing import Callable, Tuple
from datetime import datetime 


def stamp()-> None:  
    """Printout current time stamp"""
    print(datetime.now().strftime("%H:%M:%S"), "Uhr")


#! Note  memory measuring in Python may not be accurate
# measure memory in MiB
memory_usage = lambda: psutil.Process().memory_info().rss / 2**20


# BENCHMARK ROUTINE
def benchmark_function(posterior, ):
    pass 


def timed(func: Callable, *args, **kwargs) -> Tuple[float, float]:
    """ 
    Compute run time and memory consumption of a function func(*args, **kwargs) 
    @returns (run_time, memory consumption)
    """
    t_start = time()
    m_start = memory_usage()
    func(*args, **kwargs)
    m_end = memory_usage()
    t_end = time()
    print('time {:10.5f} s   |  memory {:10.1f} MiB'.format(t_end - t_start, m_end - m_start))
    return t_end - t_start, m_end - m_start



def benchmark(niter: int, func: Callable, *args, **kwargs):
    """
    Benchmark the function 'func(*args, **kwargs)' by running it niter times. (similar to timeit)
    """
    # save times in list
    times = list()
    for _ in range(niter):
        t_start = time()
        func(*args, **kwargs)
        times.append(time() - t_start)
    return times


def save_benchmark(model:str, alg:str, prefix:str, benchs:dict, times:np.ndarray) -> None:
    """
    Save the current benchmark to 'data/<model>/<prefix>_<alg>'

    @param bench:   benchmarks as dict
    @param times:   time results as np.array

    >>> save_benchmark(model, alg, prefix, benchs, times)
    """

    path = 'data/' + model

    if not os.path.exists(path):
        os.mkdir(path)
    
    print("Writing results to '{}'".format(path))
    name = "{}/{}_{}".format(path, prefix, alg)

    # save json
    with open(name + '.json', 'w') as f: 
        json.dump(benchs, f, indent=2)
    
    # save npz
    np.savez(name, *times)