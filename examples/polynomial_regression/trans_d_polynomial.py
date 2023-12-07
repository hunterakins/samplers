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
from samplers.poly_rj import AdaptiveTransDPTSampler, Chain, ChainParams, mh_walk
from samplers.helpers import *
from samplers.examples.polynomial_regression.gaussian_regression import get_evidence, get_post_mean_cov, get_H

pics_folder = '/home/hunter/research/samplers/notes/pics/'

def f_log_prior(x):
    """
    Return log prior probability of x
    Assume regardless it is a zero mean unit variance Gaussian?
    """
    dim = int(x[0])
    sigma_sq = 1.0
    xvals = x[1:dim+1]
    log_prior = -float(dim)*np.log(2*np.pi*sigma_sq)/2 - np.sum(xvals**2)/(2*sigma_sq)
    return log_prior

def get_log_lh(y, tgrid, noise_std, beta=1.0):
    """
    Fix msmts y
    """
    n = tgrid.size
    def f_log_lh(x):

        """
        x is polynomial coefficients plus the dimesnion
        """
        dim = int(x[0])
        xvals = x[1:dim+1]
        y_pred = np.polyval(xvals, tgrid)
        log_lh = -n/2*np.log(2*np.pi*noise_std**2 / beta) - beta*np.sum((y-y_pred)**2)/(2*noise_std**2)
        return log_lh
    return f_log_lh

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

def f_log_gprime(x, **kwargs):
    """
    Return the log of the probability that scalar x
    was generated by the proposal distribution
    The proposal is gaussian with unit variance unless specified in kwargs
    """
    if kwargs.get('sigma_sq') is None:
        sigma_sq = 1.0
    else:
        sigma_sq = kwargs.get('sigma_sq')
    return -np.log(2*np.pi*sigma_sq)/2 - x**2/(2*sigma_sq)

def f_prior(dim):
    """
    Generate a random sample from the prior
    """
    cov = np.eye(dim)
    return f_proposal(cov)[0]

def compute_evidence(dim_grid, tgrid, y_meas, noise_std, beta):
    """
    Prior sigma is 1
    noise_sigma = noise_std / sqrt(beta)
    """
    prior_sigma = 1.0
    noise_sigma = noise_std / np.sqrt(beta)
    evidence_list = []
    for dim in dim_grid:
        H = get_H(tgrid, dim)
        x_mean, x_cov = get_post_mean_cov(H, y_meas, noise_sigma, prior_sigma)
        evidence = get_evidence(H, y_meas, noise_sigma, prior_sigma)
        evidence_list.append(evidence[0,0])
    return evidence_list

def h_diffeo_birth(x, u):
    xprime = x.copy()
    dim = int(x[0])
    xprime[0] += 1 # update dimension
    xprime[dim+1] = u # update new coefficient
    return xprime

def J_birth(x, u):
    return 1

def h_diffeo_death(x):
    xprime = x.copy()
    xprime[0] -= 1 # update dimension
    uprime = x[-1]
    return xprime, uprime

def J_death(x):
    return 1


def plot_evidence(dim_list, tgrid, y, noise_std, beta_list):
    for beta in beta_list:
        f_log_lh = get_log_lh(y, tgrid, noise_std, beta=beta) # set log lh fun
        #evidence_grid = compute_evidence(np.array(dim_list), f_log_lh, 10000, 1.0)
        evidence_grid = compute_evidence(dim_list, tgrid, y, noise_std, beta)
        plt.figure()
        plt.suptitle(f'Analytic calculation of evidence for beta={beta}')
        plt.plot(dim_list, evidence_grid, 'k', label=f'beta={beta}')
        plt.xlabel('dim')
        plt.ylabel('evidence')
        plt.grid()
        plt.legend()
    return

def plot_lh(coeffs, f_log_lh):
    plt.figure()
    plt.suptitle('log likelihood')
    for i in range(coeffs.size):
        lh_vals = np.zeros(101)
        delta_vals = np.linspace(-2.0, 2.0, 101)
        for j in range(delta_vals.size):
            x = np.zeros(coeffs.size+1)
            delta = delta_vals[j]
            x[1:] = coeffs.copy()
            x[0] = coeffs.size
            x[i+1] += delta
            lh = np.exp(f_log_lh(x))
            lh_vals[j] = lh
            x[i+1] -= delta
        plt.plot(delta_vals, lh_vals, label='coeff {}'.format(i+1))
    plt.xlabel('diff. between cand. and true val.')
    plt.ylabel('likelihood')
    plt.grid()
    plt.legend()

def example_comparison_script():
    """
    Do a transdimensional regression on the polynomial from the other examples 
    I used MH to do
    """
    """
    First generate some data and introduce some noise
    """
    snr_db = 20
    N = 30 # num time points
    tgrid = np.linspace(-1, 2, N)
    num_chains = 10
    Tmax = 100
    update_after_burn = False # don't update covariance matrices after burn in
    eps = 1e-7
    dim_list = [2,3,4,5,6,7]
    move_probs = [[1.0, 0.0]] + [[0.5, 0.5]]*(len(dim_list)-2) + [[0.0, 1.0]]
    temp_ladder = np.exp(np.linspace(0, np.log(Tmax), num_chains))
    beta_arr  = 1/temp_ladder # convert to beta


    coeffs = np.load('m_true.npy')
    print('coeffs', coeffs)
    print('true dim', coeffs.size)
    y = np.polyval(coeffs, tgrid)
    noise_var = (np.var(y)/(10**(snr_db/10)))
    noise_std = np.sqrt(noise_var)
    y_true = np.copy(y)
    y += noise_std*np.random.randn(tgrid.size)

    
    fig, ax = plt.subplots()
    ax.plot(tgrid, y_true, 'r', label='true model')
    ax.plot(tgrid, y, 'o', label='msmt')
    ax.set_xlabel('Time')
    ax.set_ylabel('y')

    f_log_lh = get_log_lh(y, tgrid, noise_std) # set log lh fun

    plot_lh(coeffs, f_log_lh)
    

    tmp_beta_list = [1.0, 1/Tmax]
    plot_evidence(dim_list, tgrid, y, noise_std, tmp_beta_list)

    sampler = AdaptiveTransDPTSampler(move_probs, dim_list, beta_arr, f_log_prior, f_log_lh, f_proposal, f_log_gprime, 
                                        h_diffeo_birth, J_birth, h_diffeo_death, J_death)
    prop_covs = sampler.tune_proposal(100, f_prior)
    N_samples = int(20*1e3)
    nu = N_samples # this means no adaptive update
    N_burn_in = 1000
    swap_interval = 10 # propose chain swaps every step

    """ 
    Now run 
    """
    sampler.initialize_chains(N_samples, N_burn_in, nu, f_prior, update_after_burn, swap_interval, prop_covs)
    sampler.sample()

    plt.figure()
    plt.plot(sampler.death_log_p_ratio, label='death log alpha')
    plt.plot(sampler.birth_log_p_ratio, label='birth log alpha')
    plt.legend()
    cold_samples, log_probs, _, _, _ = sampler.get_chain_info(0) # get cold chain
    map_x = cold_samples[:, np.argmax(log_probs)]
    dim = int(map_x[0])
    map_coeff = map_x[1:dim+1]
    print('MAP coeff: {}'.format(map_coeff))
    cold_samples = cold_samples[:, N_burn_in:]
    vals = np.zeros((tgrid.size, cold_samples.shape[1]))
    for i in range(cold_samples.shape[1]):
        dim = int(cold_samples[0, i])
        vals[:, i] = np.polyval(cold_samples[1:dim+1, i], tgrid)
    mean_val = np.mean(vals, axis=1)
    std_val = np.std(vals, axis=1)
    #ax.plot(tgrid, np.polyval(map_coeff, tgrid), 'k--', alpha=1, label='map')
    ax.plot(tgrid, mean_val, 'k--', alpha=1, label='mean')
    ax.fill_between(tgrid, mean_val-2*std_val, mean_val+2*std_val, alpha=0.2, label='2 std')
    ax.legend()
    ax.grid(True)


    #fig.savefig(pics_folder + 'trans_d_poly_regression_data.pdf')

    log_p_ar_fig, fig_ax_list, dim_fig = sampler.single_chain_diagnostic_plot(0)

    #dim_fig.savefig(pics_folder + 'trans_d_poly_dim_hist.pdf')

    #log_p_ar_fig.savefig(pics_folder + 'trans_d_poly_log_p_ar.pdf')

    log_p_ar_fig, fig_ax_list, dim_fig = sampler.single_chain_diagnostic_plot(-1)
    

    #dim_fig.savefig(pics_folder + 'trans_d_poly_dim_hist_hot_chain.pdf')

    #log_p_ar_fig.savefig(pics_folder + 'trans_d_poly_log_p_ar_hot_chain.pdf')


    swap_fig = plt.figure()
    #plt.imshow(self.swap_mat, aspect='auto', cmap='gray_r', interpolation='none')
    for i in range(len(sampler.chain_list)):
        plt.plot(sampler.swap_mat[i,:])
    plt.show()

    #sampler.diagnostic_plot()


def get_H(t, m):
    """
    Given time samples t and model dimension m
    make the model matrix H
    """
    H = np.zeros((t.size, m))
    for i in range(m):
        H[:, i] = t**i
    H = H[:,::-1]
    return H

def get_A(Gn1, Gn):
    """
    Gn1 is the model matrix for dimension n-1
    subsampled at n-1 points (so it is square
    and hopefully invertible)
    Gn is the model matrix for dimension n
    subsampled at n-1 points
    """
    n = Gn.shape[1]
    mat1 = np.linalg.inv(Gn1)@Gn
    A = np.zeros((n, n))
    A[0,0] = 1
    A[1:, :] = mat1
    A_inv = np.linalg.inv(A)
    return A, A_inv

def get_sub_inds(N, m):
    """
    Get m indices from N
    start at end points and move in
    """
    stride = int((N-1)/(m-1))
    left_over = N -1 - stride*(m-1)
    inds = [0]
    for i in range(m-1):
        ind = stride  + inds[-1]
        if left_over > 0:
            ind += 1
            left_over -= 1
        inds.append(ind)
    return inds

def make_h_diffeo(tgrid, dim_list):
    H_list = []
    sub_inds_list = []
    for m in dim_list:
        H_list.append(get_H(tgrid, m))
        sub_inds_list.append(get_sub_inds(tgrid.size, m))

    A_list, A_inv_list = [], []
    J_birth_list, J_death_list = [], []
    for m_i in range(len(dim_list) - 1):
        Hn1 = H_list[m_i]
        H = H_list[m_i+1]
        sub_inds = sub_inds_list[m_i]
        Gn1 = Hn1[sub_inds, :]
        Gn = H[sub_inds, :]
        A, A_inv = get_A(Gn1, Gn)
        A_list.append(A)
        A_inv_list.append(A_inv)
        J_birth_list.append(np.linalg.det(A))
        J_death_list.append(1/np.linalg.det(A_inv))


    def h_diffeo_birth(x, u):
        dim = int(x[0])
        dim_ind = dim_list.index(dim)
        coeff = x[1:dim+1]
        tmp_vec = np.zeros(dim+1)
        tmp_vec[1:] = coeff
        tmp_vec[0] = u
        A = A_inv_list[dim_ind] 
        H = H_list[dim_ind]

        coeffprime = A@tmp_vec
        H = H_list[dim_ind+1]
        xprime = np.zeros(x.size)
        xprime[0] = dim+1
        xprime[1:dim+2] = coeffprime
        return xprime

    def J_birth(x, u):
        dim = int(x[0])
        dim_ind = dim_list.index(dim)
        return J_birth_list[dim_ind]
    
    def h_diffeo_death(x):
        dim = int(x[0])
        dim_ind = dim_list.index(dim)
        coeff = x[1:dim+1]
        uprime = coeff[0] # leading coeff.
        A = A_list[dim_ind-1] # I end up in dim_ind - 1
        H = H_list[dim_ind]

        coeffprime = A@coeff
        H = H_list[dim_ind-1]
        xprime = np.zeros(x.size)
        xprime[0] = dim-1
        xprime[1:dim] = coeffprime[1:]
        return xprime, uprime

    def J_death(x):
        dim = int(x[0])
        dim_ind = dim_list.index(dim-1) # index by state they end up in
        return J_death_list[dim_ind]

    return h_diffeo_birth, J_birth, h_diffeo_death, J_death

def alternative_proposal_script():
    """
    Try the better proposal 
    """
    """
    First generate some data and introduce some noise
    """
    snr_db = 20
    N = 30 # num time points
    tgrid = np.linspace(-1, 2, N)
    num_chains = 1
    Tmax = 1
    update_after_burn = False # don't update covariance matrices after burn in
    eps = 1e-7
    dim_list = [2,3,4,5,6,7]
    move_probs = [[1.0, 0.0]] + [[0.5, 0.5]]*(len(dim_list)-2) + [[0.0, 1.0]]
    temp_ladder = np.exp(np.linspace(0, np.log(Tmax), num_chains))
    beta_arr  = 1/temp_ladder # convert to beta


    coeffs = np.load('m_true.npy')
    print('coeffs', coeffs)
    print('true dim', coeffs.size)
    y = np.polyval(coeffs, tgrid)
    noise_var = (np.var(y)/(10**(snr_db/10)))
    noise_std = np.sqrt(noise_var)
    y_true = np.copy(y)
    y += noise_std*np.random.randn(tgrid.size)

    
    fig, ax = plt.subplots()
    ax.plot(tgrid, y_true, 'r', label='true model')
    ax.plot(tgrid, y, 'o', label='msmt')
    ax.set_xlabel('Time')
    ax.set_ylabel('y')

    f_log_lh = get_log_lh(y, tgrid, noise_std) # set log lh fun

    plot_lh(coeffs, f_log_lh)
    

    plot_evidence(dim_list, tgrid, y, noise_std, [1.0])

    h_diffeo_birth, J_birth, h_diffeo_death, J_death = make_h_diffeo(tgrid, dim_list)
    sampler = AdaptiveTransDPTSampler(move_probs, dim_list, beta_arr, f_log_prior, f_log_lh, f_proposal, f_log_gprime, 
                                        h_diffeo_birth, J_birth, h_diffeo_death, J_death)
    prop_covs = sampler.tune_proposal(1000, f_prior)
    N_samples = int(20*1e3)
    nu = N_samples # this means no adaptive update
    N_burn_in = 1000
    swap_interval = 10 # propose chain swaps every step

    """ 
    Now run 
    """
    sampler.initialize_chains(N_samples, N_burn_in, nu, f_prior, update_after_burn, swap_interval, prop_covs)
    sampler.sample()

    cold_samples, log_probs, _, _, _ = sampler.get_chain_info(0) # get cold chain
    map_x = cold_samples[:, np.argmax(log_probs)]
    dim = int(map_x[0])
    map_coeff = map_x[1:dim+1]
    print('MAP coeff: {}'.format(map_coeff))
    cold_samples = cold_samples[:, N_burn_in:]
    vals = np.zeros((tgrid.size, cold_samples.shape[1]))
    for i in range(cold_samples.shape[1]):
        dim = int(cold_samples[0, i])
        vals[:, i] = np.polyval(cold_samples[1:dim+1, i], tgrid)
    mean_val = np.mean(vals, axis=1)
    std_val = np.std(vals, axis=1)
    #ax.plot(tgrid, np.polyval(map_coeff, tgrid), 'k--', alpha=1, label='map')
    ax.plot(tgrid, mean_val, 'k--', alpha=1, label='mean')
    ax.fill_between(tgrid, mean_val-2*std_val, mean_val+2*std_val, alpha=0.2, label='2 std')
    ax.legend()
    ax.grid(True)


    #fig.savefig(pics_folder + 'trans_d_poly_regression_data.pdf')

    log_p_ar_fig, fig_ax_list, dim_fig = sampler.single_chain_diagnostic_plot(0)

    #dim_fig.savefig(pics_folder + 'trans_d_poly_dim_hist.pdf')

    #log_p_ar_fig.savefig(pics_folder + 'trans_d_poly_log_p_ar.pdf')
    plt.show()

alternative_proposal_script()
#example_comparison_script()
