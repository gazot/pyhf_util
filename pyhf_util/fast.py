import numpy as np
import pybat
import pyhf

from pyhf_llh import *
from pyhf_util.bench import stamp
from priorhf import make_prior

from .utils import array_from_jl_samples
from .bench import benchmark
from juliacall import Main as jl



def setup(file: str, quiet: bool, use_jax=True):
    """General setup for all functions, @returns BAT.PosteriorMeassure"""
    if quiet: 
        # disable logging 
        jl.seval('using Logging')
        jl.seval('Logging.disable_logging(Logging.Warn)')

    # likelihood
    if use_jax:
        pyhf.set_backend('jax')
        llh, llh_grad = pyhf_llh_with_grad(file)
        likelihood = jl.PyCallDensityWithGrad(llh, llh_grad)
    else:
        llh = pyhf_llh(file)
        likelihood = jl.PyCallDensity(llh)

    # prior
    prior = make_prior(file)
    
    return jl.BAT.PosteriorMeasure(likelihood, prior)



def finalize(samples, save_path):
    """Handle samples."""
    #print(jl.bat_report(samples))

    if save_path is not None:  
        print("save samples to '{}'".format(save_path))
        arr = array_from_jl_samples(samples)
        np.save(save_path, arr)
    


def run_MH(file: str, nsteps: int=10_000, nchains: int=2, niter=0, use_jax=True,
           quiet: bool=False, burnin_cycles: int=50, save_path=None):
    """
    Run Metropolis Hastings with nsteps and nchains
    @param quiet            disable output from BAT
    @param burnin_cycles    increase burnin
    @param save_path        write results as np.array to path
    """ 
    quiet = quiet or niter > 0
    posterior = setup(file, quiet, use_jax)

    # sampler 
    burnin = jl.BAT.MCMCMultiCycleBurnin(nsteps_per_cycle=10_000, max_ncycles=burnin_cycles)
    method = jl.BAT.MCMCSampling(mcalg = jl.BAT.MetropolisHastings(), burnin=burnin, nsteps=nsteps, nchains=nchains)
    
    stamp() 

    samples = jl.bat_sample(posterior, method).result


    # benchmark if niter
    if niter > 0:
        times = benchmark(niter, jl.bat_sample, posterior, method)
        
    stamp() 
    finalize(samples, save_path)

    if niter > 0: 
        return times 


def run_HMC(file: str, nsteps: int=10_000, nchains: int=2, niter=0, 
           quiet: bool=False, burnin_cycles: int=50, save_path=None):
    """
    Run HamiltonianMC with nsteps and nchains
    @param quiet            disable output from BAT
    @param burnin_cycles    increase burnin
    @param save_path        write results as np.array to path
    """ 
    quiet = quiet or niter > 0
    posterior = setup(file, quiet)

    # sampler 
    burnin = jl.BAT.MCMCMultiCycleBurnin(nsteps_per_cycle=10_000, max_ncycles=burnin_cycles)
    method = jl.BAT.MCMCSampling(mcalg = jl.BAT.HamiltonianMC(), burnin=burnin, nsteps=nsteps, nchains=nchains)
    
    stamp() 
    samples = jl.bat_sample(posterior, method).result
    
    # benchmark if niter
    if niter > 0:
        times = benchmark(niter, jl.bat_sample, posterior, method)
    stamp() 

    finalize(samples, save_path)



def run_MH_no_jax(file: str, nsteps: int=10_000, nchains: int=2, niter=0, 
                  quiet: bool=False, burnin_cycles: int=50, save_path=None):
    """
    Run Metropolis Hastings with nsteps and nchains
    @param quiet            disable output from BAT
    @param burnin_cycles    increase burnin
    @param save_path        write results as np.array to path
    """ 

    pass 
