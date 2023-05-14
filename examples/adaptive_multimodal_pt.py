"""
Description:

Date:

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
from samplers.likelihoods import GaussianLikelihood, MultimodalGaussianLikelihood, RosenbrockTwoD
from samplers.pt import ParallelTemperedAdaptiveSampler, AdaptiveChainParams
from samplers.helpers import *

def adaptive_multimodal_gaussian_test():
    """
    Test MHSampler on an correlated gaussian noise
    """
    # define a multimodal Gaussian distribtino
    num_params = 1
    seed = None
    num_peaks = 5
    mu_std = 20 # standard deviation of mean distribution
    sigma_std = .3 # standard deviation of variance distribution
    sigma_mean = 1
    mu_list = [np.random.normal(0, mu_std, num_params) for i in range(num_peaks)]
    sigma_list = [np.random.normal(sigma_mean, sigma_std, num_params) for i in range(num_peaks)]
    likelihood = MultimodalGaussianLikelihood(num_params, mu_list, sigma_list, colored=True, seed=seed)
    log_p = likelihood.f_log_p

    print('Running an adaptive parallel tempering sampler on a multimodal Gaussian likelihood in {0} dimensional space with {1} modes'.format(num_params, num_peaks))

    def log_f_prior(x):
        """ Set some bounds
        """
        for i in range(num_params):
            if (x[i] < -100) or (x[i] > 100):
                return -np.inf
        return 0

    # create an MHSampler object
    sampler = ParallelTemperedAdaptiveSampler(log_p, log_f_prior) 

    # set the chain parameters
    num_chains = 6
    Tmax = 100
    temp_ladder = np.exp(np.linspace(0, np.log(Tmax), num_chains))
    ladder  = 1/temp_ladder # convert to beta
    print('Using {0} chains with the temperature schedule {1}'.format(num_chains, [round(x,1) for x in temp_ladder]))

    sd =(2.4)**2 / num_params # default scale factor # from Haario
    eps = 1e-7
    nu = 1000
    update_after_burn = False
    chain_params_list = [AdaptiveChainParams(ladder[i], num_params, eps, sd, nu, update_after_burn) for i in range(num_chains)]

    # get initial points and jump covariance
    x0 = np.zeros(num_params)
    x0_list = [x0.copy() for i in range(num_chains)]
    sigma0 = np.square(mu_std)*np.eye(num_params)
    sigma0_list = [sigma0.copy() for i in range(num_chains)]

    # run the sampler
    N = 10000
    swap_interval = 1
    chain_list = sampler.sample(x0_list, sigma0_list, chain_params_list, N, swap_interval=swap_interval)
    cold_samples = chain_list[0].samples

    # plot the samples
    chain_fig_list, fig3 = sampler.diagnostic_plot(density=True)

    for chain_i in range(num_chains):
        chain = chain_list[chain_i]
        beta = chain.params.beta
        fig1 = chain_fig_list[chain_i][0]
        fig2 = chain_fig_list[chain_i][1]
        axes = fig2.axes
        num_rows, num_cols = get_subplot_dims(num_params)
        for i in range(num_params):
            ax_row = int(i/num_cols)
            ax_col = i % num_cols
            mu_vals = [x[i] for x in likelihood.mu_list]
            sigma_vals = [x[i] for x in likelihood.sigma_list]
            min_val = chain.samples[i,:].min()
            max_val = chain.samples[i,:].max()
            grid = np.linspace(min_val, max_val, 100)
            vals = np.zeros(grid.size)
            for j in range(num_peaks):
                vals += np.exp(-(grid-mu_vals[j])**2/(2*sigma_vals[j]**2)) / np.sqrt(2*np.pi*sigma_vals[j]**2)
            dx = grid[1] - grid[0]
            vals = np.power(vals, beta)
            vals = vals / (vals.sum()*dx) # approximate normalization
            axes[num_cols*ax_row+ ax_col].plot(grid, vals)

    # plot IACT for each chain
    plt.figure()
    for i in range(num_chains):
        f = lambda x: x
        M_list, tau_list = sampler.calculate_iact(i, f, 0) # I do burn-in in the sampler
        plt.plot(M_list, tau_list, label='chain {0}'.format(i))
    plt.xlabel('M')
    plt.ylabel('Tau')
    plt.grid()
    plt.legend()
    plt.show()

adaptive_multimodal_gaussian_test()
