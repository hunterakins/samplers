"""
Description:
    Test affine invariant sampler on rosenbrock distribution as in Goodman and Weare

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
from samplers.emcee import AffineInvariantSampler
from samplers.likelihoods import GaussianLikelihood, MultimodalGaussianLikelihood, RosenbrockTwoD


# test rosenbrock
def rosenbrock_test():
    """
    
    Tunable parameters inside this script:
    N - number of samples for each walker
    a - stretch parameter
    L - number of walkers
    S - number of walkers used for covariance estimation
    prop_weights - weights for proposal distribution. 1st element is weight for
        stretch move, 2nd element is weight for reflection move
        they must be integers. they are unnormalized. So if you want to do only 
        stretch moves, set prop_weights = [1,0]
        if you want to do only reflection moves, set prop_weights = [0,1]
        if you want an even mix, do [1,1]
    var_prior sets the variance of the distribution to draw initial samples from
        (not a true prior distribution used to compute a posterior. the rosenbrock
        is the posterior here...


    """
    num_params = 2
    seed = None
    likelihood = RosenbrockTwoD()
    likelihood.plot()
    log_p = likelihood.f_log_p

    # create an Sampler object
    sampler = AffineInvariantSampler(log_p)

    # set run parameters
    N = 10**4
    a = 2 # stretch parameter
    L = 10 # number of walkers
    S = 4 # number of walkers used for covariance estimation
    prop_weights = [1,0]
    sampler.initialize_sampler(num_params, a, S, L, prop_weights)

    # draw from the prior
    var_prior = 20**2
    sigma0 = var_prior*np.eye(2)
    x0 = np.zeros((num_params, L))
    for i in range(L):
        x0[:,i] = np.random.multivariate_normal(np.zeros(num_params), sigma0)

    sampler.sample(x0, N)
    samples = sampler.samples.copy()
    def f(x): # function to calculate iact...avg over walkers
        x = np.mean(x, axis=1) # avg. samples over walkers
        return x

    # in Goodman and Weare, the IACT is 20 *1e3 for x1 and 67*1e3 for x2 with stretch move and 10 walkers
    N_burn_in = int(N/5)
    i = 0
    M_list, tau_list = sampler.calculate_iact(f, N_burn_in)
    print(M_list, tau_list)

    plt.figure()
    plt.suptitle('IACT estimates for the samples')
    plt.plot(M_list, tau_list)
    plt.xlabel('Block size M for int. act')
    plt.ylabel('Int. autocorr. time')

    samples = sampler.samples.copy()
    #samples = samples.transpose((0, 2, 1))
    samples = samples.reshape(samples.shape[0], samples.shape[1]*samples.shape[2])
    sampler.diagnostic_plot()
    print(samples.shape)
    samples = samples[:,1000:]
    plt.figure()
    plt.hist2d(samples[0,:], samples[1,:], bins=[100,100])
    plt.colorbar()

# test rosenbrock using numba
def njit_rosenbrock_test():
    """
    
    Tunable parameters inside this script:
    N - number of samples for each walker
    a - stretch parameter
    L - number of walkers
    S - number of walkers used for covariance estimation
    prop_weights - weights for proposal distribution. 1st element is weight for
        stretch move, 2nd element is weight for reflection move
        they must be integers. they are unnormalized. So if you want to do only 
        stretch moves, set prop_weights = [1,0]
        if you want to do only reflection moves, set prop_weights = [0,1]
        if you want an even mix, do [1,1]
    var_prior sets the variance of the distribution to draw initial samples from
        (not a true prior distribution used to compute a posterior. the rosenbrock
        is the posterior here...


    """
    num_params = 2
    seed = None
    likelihood = RosenbrockTwoD()
    likelihood.plot()
    log_p = likelihood.f_log_p

    # create an Sampler object
    sampler = AffineInvariantSampler(log_p)

    # set run parameters
    N = 10**5
    a = 2 # stretch parameter
    L = 10 # number of walkers
    S = 4 # number of walkers used for covariance estimation
    prop_weights = [1,0]
    sampler.initialize_sampler(num_params, a, S, L, prop_weights)

    # draw from the prior
    var_prior = 20**2
    sigma0 = var_prior*np.eye(2)
    x0 = np.zeros((num_params, L))
    for i in range(L):
        x0[:,i] = np.random.multivariate_normal(np.zeros(num_params), sigma0)

    sampler.njit_sample(x0, N)
    samples = sampler.samples.copy()
    def f(x): # function to calculate iact...avg over walkers
        x = np.mean(x, axis=1) # avg. samples over walkers
        return x

    # in Goodman and Weare, the IACT is 20 *1e3 for x1 and 67*1e3 for x2 with stretch move and 10 walkers
    N_burn_in = int(N/5)
    i = 0
    M_list, tau_list = sampler.calculate_iact(f, N_burn_in)
    print(M_list, tau_list)

    plt.figure()
    plt.suptitle('IACT estimates for the samples')
    plt.plot(M_list, tau_list)
    plt.xlabel('Block size M for int. act')
    plt.ylabel('Int. autocorr. time')

    samples = sampler.samples.copy()
    #samples = samples.transpose((0, 2, 1))
    samples = samples.reshape(samples.shape[0], samples.shape[1]*samples.shape[2])
    sampler.diagnostic_plot()
    print(samples.shape)
    samples = samples[:,1000:]
    plt.figure()
    plt.hist2d(samples[0,:], samples[1,:], bins=[100,100])
    plt.colorbar()

if __name__ == '__main__':
    import time
    now = time.time()
    njit_rosenbrock_test()
    print('njit time', time.time()-now)
    now = time.time()
    plt.show()
    #rosenbrock_test()
    #print('python time', time.time()-now)
