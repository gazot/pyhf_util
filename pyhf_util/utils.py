import json 
import numpy as np 

def load_json(path):
    """Read json file from 'path'."""
    with open(path + '.json') as f:
        return json.load(f)



def steps_str(nsteps, n=1, k=True) -> list[str]:
    """Format nsteps as list of strings"""
    steps = list()
    for s in nsteps:
        steps += [s] * n
    # format numbers 1,000,000
    steps_fmt = "{:}k" if k else "{:,}"
    if k:
        return ["{:}k".format(int(s/1000)) for s in steps]
    else:
        return ["{:,}".format(s) for s in steps]



def array_from_jl_samples(samples):
    """np.array (ndim, n) from jl.DensitySampleVector (n samples) """
    return np.array([s for s in samples.v])