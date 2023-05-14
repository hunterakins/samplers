"""
Description:
    Test parallel-tempering affine invariant sampler on rosenbrock distribution as in Goodman and Weare

Date:
    4/28/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from samplers.pt_emcee import ParallelTemperedAffineInvariantSampler
from samplers.likelihoods import GaussianLikelihood, MultimodalGaussianLikelihood, RosenbrockTwoD


def f_log_prior(x):
    """
    Two-d x
    Uniform prior
    on x inside square centered at originand
    """
    if max(abs(x)) > 100:
        return -np.inf
    else:
        return 0
    
def rosenbrock_test():
    """
    Test rosenrbcok
    """
    num_params = 2
    seed = None
    likelihood = RosenbrockTwoD()
    likelihood.plot()
    log_p = likelihood.f_log_p

    # create an Sampler object
    sampler = ParallelTemperedAffineInvariantSampler(log_p, f_log_prior)

    # set run parameters
    N = 2000
    a = 2 # stretch parameter
    L = 100 # number of walkers
    S = 4 # number of walkers used for covariance estimation

    # set the chain parameters
    M = 2
    Tmax = 100
    temp_ladder = np.exp(np.linspace(0, np.log(Tmax), M))
    print(temp_ladder) # must be cold to hot
    prop_weights = [10,0] # only walk
    sampler.initialize_sampler(num_params, a, S, L, M, temp_ladder, prop_weights)

    # draw from the prior
    var_prior = 20**2
    sigma0 = var_prior*np.eye(2)
    x0 = np.zeros((num_params, M, L))
    for temp_i in range(M):
        for walker_i in range(L):
            x0[:,temp_i, walker_i] = np.random.multivariate_normal(np.zeros(num_params), sigma0)

    sampler.sample(x0, N)

    N_burn_in = 100
    samples = sampler.samples.copy()
    #samples = samples.transpose((0, 2, 1))
    cold_samples = samples[:,0,...]
    hot_samples = samples[:,-1,...]
    hot_samples = hot_samples.reshape(hot_samples.shape[0], hot_samples.shape[1]*hot_samples.shape[2])
    cold_samples = cold_samples.reshape(cold_samples.shape[0], cold_samples.shape[1]*cold_samples.shape[2])

    # plot the various histograms
    sampler.diagnostic_plot(N_burn_in)
    hot_samples = hot_samples[:,N_burn_in:]
    cold_samples = cold_samples[:,N_burn_in:]
    plt.figure()
    plt.plot(cold_samples[0,:])
    plt.suptitle('Cold chain samples')

    plt.figure()
    plt.suptitle('Hot chain histogram')
    plt.hist2d(hot_samples[0,:], hot_samples[1,:], bins=[100,100])
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.colorbar()

    plt.figure()
    plt.suptitle('Cold chain histogram')
    plt.hist2d(cold_samples[0,:], cold_samples[1,:], bins=[100,100])
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.colorbar()

    def f(x):
        x = x[:,0,...] # pick the cold chain
        #x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) 
        avg = np.mean(x, axis=1) # avg over walker
        return avg
    M_list, tau_list = sampler.calculate_iact(f, N_burn_in)
    plt.figure()
    plt.suptitle('IACT est.')
    plt.ylabel('IACT (tau)')
    plt.xlabel('M (maximum  lag in ACF for IACT estimation)')
    plt.plot(M_list, tau_list)
    #plt.plot(mean_acorr[i,:])


    # also plot some of the swap maps...
    plt.figure()
    plt.suptitle('Some example swap tracks of two different walkers\nNot sure I implemented this correctly')
    plt.plot(sampler.chain_index_array[0,0,::2])
    plt.plot(sampler.chain_index_array[-1,0,::2])
    plt.ylabel('Chain temp.')
    plt.xlabel('Iteration')
    plt.show()

rosenbrock_test()
