"""
Description:
    Example with isovelocity ssp and fixed dimensional bottom
    Multi-frequency, maximum likelihood source, amplitude and phase
    likelihood function

Date:
    3/14/2024

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from pykrak import pykrak_env
from pykrak import pressure_calc as pc
from copy import deepcopy
from samplers.helpers import unif_order_stat
import time
from scipy import special
from samplers.pt import AdaptivePTSampler, Chain, ChainParams, mh_walk
from samplers.geo_pt import FixedDGeoSampler
from optimiz import gi_helpers as gh

def f_proposal(prop_cov):
    """
    Generate a random multivariate Gaussian sample according 
    to the proposal covariance prop_cov (zero mean)
    """
    if prop_cov.size > 1:
        val =  np.random.multivariate_normal(np.zeros(prop_cov.shape[0]), prop_cov)
        return val, None
    else: # dim is 1
        val = np.random.randn()*np.sqrt(prop_cov)
        log_prob = -np.log(2*np.pi*prop_cov)/2 - val**2/(2*prop_cov)
        return val, log_prob

def ex1():
    """
    Setup a geoacoustic fixed d inversion
    """
    freq_list = [50, 70, 90]

    c_range = (1500.0, 2000.0)
    rho_range = (0.8, 2.5)
    attn_range = (0.05, 0.4)
    D = 100
    Z_hs = 140
    interval = (D, Z_hs)
    pos_range = interval
    positions = [105, 112] # interface depths
    c1, rho1, attn1 = 1540.0, 1.2, 0.2
    c2, rho2, attn2 = 1800.0, 2.0, 0.3
    c3, rho3, attn3 = 1900.0, 2.2, 0.4
    k = len(positions) # number of interfaces
    c_hs = 2300.0
    rho_hs = 2.0
    attn_hs = 0.02
    native_values = np.array([c1, rho1, attn1, c2, rho2, attn2, c3, rho3, attn3])[:,None]
    positions = np.array(positions)[:,None]
    dim = 3
    x = np.vstack((positions, native_values))
    x = gh.scale_x(x, k, pos_range, c_range, rho_range, attn_range)
    zw = np.linspace(0, D, 10)
    cw = 1500.0*np.ones(zw.size)
    env = gh.get_env(x, k, zw, cw, c_hs, rho_hs, attn_hs, Z_hs,c_range=c_range, rho_range=rho_range, attn_range=attn_range)
    env.plot_env()
    zs, zr, rr = 90.0, np.linspace(20.0, 80.0, 22), 5*1e3

    sigma_n = 1.0
    K = zr.size
    y_list, R_cov_list = gh.get_R_msmt_list(env, freq_list, zs, zr, rr, K, sigma_n)

    kwargs = {'R_msmt_list': R_cov_list, 'zw': zw, 'cw':cw, 'K': K, 'k': k, 'freq_list': freq_list, 'zs': zs, 'zr': zr, 'rr': rr, 'D': D, 'Z_hs': Z_hs, 'c_hs': c_hs, 'rho_hs': rho_hs, 'attn_hs': attn_hs, 'c_range': c_range, 'rho_range': rho_range, 'attn_range': attn_range}


    #f_log_lh = get_log_lh_func(R_cov_list, K, k, freq_list, zs, zr, rr, D, Z_hs, c_hs, rho_hs, attn_hs)
    f_log_lh = gh.get_log_lh_func(**kwargs)
    f_log_prior = gh.get_log_prior(k, dirichlet=True)

    def f_log_lh_vec(x):
        if np.any(np.abs(x) > 1):
            return -np.inf
        return f_log_lh(x[:,None])

    def f_log_prior_vec(x):
        return f_log_prior(x[:,None])


    def f_prior():
        x = gh.get_prior_sample(k, dirichlet=True)
        return x[:,0]


    move_probs = [1.0]
    T_arr = np.power(10, np.linspace(0, 3, 10))
    T_arr = np.array([1.0])
    beta_arr = 1/T_arr
    dim = k + 3*(k+1)


    sampler = FixedDGeoSampler(move_probs, dim, beta_arr, 
                                        f_log_prior_vec, f_log_lh_vec, f_proposal,
                                        k, pos_range, c_range, rho_range, attn_range)
    N_tune = 250
    variance_ranges = np.logspace(-2, 0, 4)
    N_scm = 2000
    prop_covs = sampler.tune_proposal(N_tune, f_prior, variance_range=variance_ranges, N_scm=N_scm)
    N_samples = int(2*1e4)
    update_after_burn = False
    N_burn_in = 1000
    nu = N_burn_in # number of samples to run before adaptive update
    swap_interval = 10 # propose chain swaps every step

    """ 
    Now run 
    """
    sampler.initialize_chains(N_samples, N_burn_in, nu, f_prior, update_after_burn, swap_interval, prop_covs)
    sampler.sample()

    log_p_ar_fig, fig_ax_list, dim_fig = sampler.single_chain_diagnostic_plot(0)

    sampler.plot_dist(0, N_bins=40)

    sampler.diagnostic_plot()

    x_map, l = sampler.get_map_x(0)
    env = gh.get_env(x_map[:,None], k, zw, cw, c_hs, rho_hs, attn_hs, Z_hs,c_range=c_range, rho_range=rho_range, attn_range=attn_range)
    env.plot_env()

    plt.show()


ex1()
