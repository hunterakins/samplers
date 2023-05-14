"""
Description:
    Test affine invariant sampler on a multimodal distribution
    Distribution is a mixture of N Gaussians in M-dimensional space
    They can be either colored or white.

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


def adaptive_multimodal_gaussian_test():
    """
    Test Affine Inv.Sampler on an correlated gaussian noise
    """
    # define a mulitmodal Gaussian distribtino
    num_params = 6
    seed = None
    num_peaks = 6
    mu_std = 20 # standard deviation of mean distribution
    sigma_mean = 3
    sigma_std = 1 # standard deviation of variance distribution
    likelihood = MultimodalGaussianLikelihood(num_params, mu_std, sigma_mean, sigma_std, num_peaks, colored=True, seed=seed)
    log_p = likelihood.f_log_p

    # create an MHSampler object
    sampler = AffineInvariantSampler(log_p)

    # set run parameters
    N = 3000
    a = 2 # stretch parameter
    #L = num_params + 6 # number of walkers
    L = 100
    S = 4 # number of walkers used for covariance estimation
    sampler.initialize_sampler(num_params, a, S, L)

    # draw from the prior
    var_prior = 2*mu_std**2 # to get samples around possible peaks..
    sigma0 = var_prior*np.eye(num_params)
    x0 = np.zeros((num_params, L))
    for i in range(L):
        x0[:,i] = np.random.multivariate_normal(np.zeros(num_params), sigma0)

    sampler.sample(x0, N)

    # plot the samples
    fig_list = sampler.diagnostic_plot(density=True)
    axes_list = [x[1].axes for x in fig_list]
    num_cols = int(np.sqrt(num_params))+1
    for i in range(num_params):
        ax_row = int(i/num_cols)
        ax_col = i % num_cols
        mu_vals = [x[i] for x in likelihood.mu_list]
        sigma_vals = [x[i] for x in likelihood.sigma_list]
        min_val = min(mu_vals) - 3*max(sigma_vals)
        max_val = max(mu_vals) + 3*max(sigma_vals)
        grid = np.linspace(min_val, max_val, 100)
        vals = np.zeros(grid.size)
        for j in range(num_peaks):
            vals += np.exp(-(grid-mu_vals[j])**2/(2*sigma_vals[j]**2)) / np.sqrt(2*np.pi*sigma_vals[j]**2)

        norm = np.sum(vals) * (grid[1] - grid[0])
        vals /= norm
        for axes in axes_list:
            axes[num_cols*ax_row+ ax_col].plot(grid, vals)

    plt.show()

adaptive_multimodal_gaussian_test()
