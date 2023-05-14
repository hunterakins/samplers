""" Description:
    Parallel tempering class

Date:
    4/24/2023

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
from samplers.proposals import gaussian_proposal
from samplers.online_scm import update_mean, update_scm
from samplers.helpers import *


class Chain:
    def __init__(self, params, x0, sigma0, N):
        """
        Initialize variables for the chains that will have N samples
        """
        self.x0 = x0
        self.sigma0 = sigma0 # starting covariance matrix for jump proposal
        self.params = params
        self.N = N
        self.samples = np.zeros((params.dim, N))
        self.log_probs = np.zeros((N))
        self.samples[:,0] = x0.copy() # initial sample
        self.acceptance_ratios = np.zeros(N)

        # to track
        self.sigma = sigma0 # covariance matrix for jump proposal to update
        self.mu = np.zeros(x0.size)

class AdaptiveChainParams:
    """
    Parameters for adaptive MCMC chain
    """
    def __init__(self, beta, dim, eps=1e-6, sd=None, nu=1000, update_after_burn=False):
        self.dim = dim # dimension of state space
        self.beta=beta # inverse temperature
        self.eps = eps # lower bound for covariance matrix eigenvalues
        if sd is None:
            sd = 2.4**2 / dim # default from Haario 2001
        self.sd = sd # scaling of proposal (may eventually be adaptive)
        self.nu = nu # number of samples to draw from sigm0 before recursively updating cov
        self.update_after_burn = update_after_burn # if true, update covariance after burn in

class ParallelTemperedAdaptiveSampler:
    """
    Parallel tempered sampler
    Use adaptive sampling (Haario) and user specified temperature ladder
    """
    def __init__(self, f_log_lh, f_log_prior, f_log_lh_kwargs={}, f_log_prior_kwargs={}):
        """
        log likelihood function
        log prior function 
        together these define the posterior distribution to sample
        """
        self.f_log_lh = f_log_lh
        self.f_log_prior = f_log_prior
        self.f_log_lh_kwargs = f_log_lh_kwargs
        self.f_log_prior_kwargs = f_log_prior_kwargs

    def _init_chains(self, x0_list, sigma0_list, chain_params_list, N):
        """
        Initialize variables for the chains that will have N samples
        and have initial state vectors in x0_list
        """
        num_chains = len(x0_list)
        chain_list = []
        for i in range(num_chains):
            chain = Chain(chain_params_list[i], x0_list[i], sigma0_list[i], N) 
            chain_list.append(chain)
        return chain_list

    def sample(self, x0_list, sigma0_list, chain_params_list, N, swap_interval=1):
        """
        Parallel tempering for chains in chain_params_list
        x0_list is set of initial state vectors for the chains for 
        different temperatures
        sigma0_list is jump proposal used until params.nu is obtained, after which it is adaptively estimated
        """
        dim = chain_params_list[0].dim
        N_burn = int(N/5)
        print('N_burn', N_burn)
        chain_list = self._init_chains(x0_list, sigma0_list,chain_params_list, N-N_burn)
        num_chains = len(chain_params_list)

        swap_mat = np.zeros((num_chains, N))
        swap_mat[:,0] = np.arange(num_chains)
       
        # get log posterior and likelihood of first sample for each chain
        curr_log_p_list = [] 
        curr_log_lh_list = []
        curr_log_prior_list = []
        x_curr_list = []
        num_accepted_list = [0]*num_chains
        for chain in chain_list:
            beta = chain.params.beta 
            xpt = chain.samples[:,0] # get first sample 
            log_lh = self.f_log_lh(xpt, **self.f_log_lh_kwargs)
            log_prior = self.f_log_prior(xpt, **self.f_log_prior_kwargs)
            log_p = beta*log_lh + log_prior # posterior
            curr_log_p_list.append(log_p)
            curr_log_lh_list.append(log_lh)
            curr_log_prior_list.append(log_prior)
            x_curr_list.append(xpt)

        for i in range(1, N): 
            # take a step on each chain
            for chain_ind in range(num_chains):
                # get chain
                chain = chain_list[chain_ind] 
                # get current sample and probs
                x_curr = x_curr_list[chain_ind]
                log_p_curr = curr_log_p_list[chain_ind]
                log_lh_curr = curr_log_lh_list[chain_ind]
                log_prior_curr = curr_log_prior_list[chain_ind]

                # propose new sample
                chain_params = chain.params
                sigma_curr =chain.sigma
                mu_curr = chain.mu
                sd = chain_params.sd
                eps = chain_params.eps
                prop_sigma = sd*sigma_curr + sd*eps*np.eye(dim)
                prop_kwargs = {'sigma': prop_sigma}
                x_prop, qxx = gaussian_proposal(x_curr, **prop_kwargs)
    
                # get log posterior and likelihood of proposed sample
                beta = chain.params.beta
                log_lh_prop = self.f_log_lh(x_prop, **self.f_log_lh_kwargs) 
                log_prior_prop = self.f_log_prior(x_prop, **self.f_log_prior_kwargs)
                log_p_prop = beta*log_lh_prop + log_prior_prop

                
                # accept or reject
                alpha = np.exp(log_p_prop - log_p_curr)
                alpha *= qxx # posterior asymmetry ratio
                u = np.random.rand()
                if u < alpha:
                    x_curr = x_prop
                    log_p_curr = log_p_prop
                    log_lh_curr = log_lh_prop
                    log_prior_curr = log_prior_prop
                    x_curr_list[chain_ind] = x_curr
                    if i > N_burn:
                        num_accepted_list[chain_ind] += 1


                # update current state for chain
                curr_log_p_list[chain_ind] = log_p_curr
                curr_log_lh_list[chain_ind] = log_lh_curr
                curr_log_prior_list[chain_ind] = log_prior_curr

                # add sample to chain
                if i > N_burn:
                    chain.samples[:, i-N_burn] = x_curr.copy()
                    chain.log_probs[i-N_burn] = log_p_curr
                    chain.acceptance_ratios[i-N_burn] = num_accepted_list[chain_ind] / (i - N_burn)  # acc. ratio


                # update chain covariance matrix (adaptive step)
                if i == chain_params.nu:
                    sigma = np.cov(chain.samples[:, :i])
                    mu = np.mean(chain.samples[:, :i], axis=1)
                    chain.sigma = sigma
                    chain.mu = mu
                if i > chain_params.nu:
                    if i > N_burn and chain_params.update_after_burn:
                        sigma = update_scm(sigma_curr, mu_curr, x_curr, i)
                        mu = update_mean(mu_curr, x_curr, i)
                        chain.sigma = sigma
                        chain.mu = mu
                    elif i < N_burn:
                        sigma = update_scm(sigma_curr, mu_curr, x_curr, i)
                        mu = update_mean(mu_curr, x_curr, i)
                        chain.sigma = sigma
                        chain.mu = mu


            # swaps (DEO)
            swap_mat[:,i] = swap_mat[:,i-1]
            if i > N_burn and (i%swap_interval==0):
                # ........
                X = i / swap_interval
                even =  X% 2
                for chain_ind in range(num_chains-1):
                    # get two chains up for swap proposal
                    hot_chain_ind = num_chains - 1 - chain_ind
                    hot_chain = chain_list[hot_chain_ind]
                    cold_chain_ind = hot_chain_ind - 1
                    cold_chain = chain_list[cold_chain_ind]
                    if ((even) and (hot_chain_ind % 2 == 0)) or ((not even) and (hot_chain_ind % 2 == 1)):
                        #print('even', even, 'hot_chain_ind', hot_chain_ind, 'cold_chain_ind', cold_chain_ind)

                        # get their likelihood and temps for calculating proposal ratio
                        hot_lh = curr_log_lh_list[hot_chain_ind]
                        cold_lh = curr_log_lh_list[cold_chain_ind]
                        beta_hot = hot_chain.params.beta
                        beta_cold = cold_chain.params.beta

                        # compute proposal ratio
                        log_omega = (beta_hot-beta_cold)*(cold_lh - hot_lh)
                        omega = np.exp(log_omega)
                        u = np.random.rand()
                        if u < omega: # swap

                            # swap log likelihoods
                            curr_log_lh_list[hot_chain_ind] = cold_lh
                            curr_log_lh_list[cold_chain_ind] = hot_lh
                            # fetch log prios
                            cold_log_prior = curr_log_prior_list[cold_chain_ind] 
                            hot_log_prior = curr_log_prior_list[hot_chain_ind] 
                            # swap priors
                            curr_log_prior_list[hot_chain_ind] = cold_log_prior
                            curr_log_prior_list[cold_chain_ind] = hot_log_prior
                            # swap log posteriors
                            curr_log_p_list[hot_chain_ind] = beta_hot*cold_lh + cold_log_prior
                            curr_log_p_list[cold_chain_ind] = beta_cold*hot_lh + hot_log_prior
                            # swap samples
                            t1 = x_curr_list[cold_chain_ind].copy()
                            t2 = x_curr_list[hot_chain_ind].copy() 
                            x_curr_list[cold_chain_ind] = t2
                            x_curr_list[hot_chain_ind] = t1

                            # note it in the swap mat
                            swap_mat[hot_chain_ind, i] = cold_chain_ind
                            swap_mat[cold_chain_ind, i] = hot_chain_ind
                    

        self.chain_list = chain_list
        self.swap_mat = swap_mat
        return chain_list

    def calculate_iact(self, chain_i, f, N_burn_in):
        samples = self.chain_list[chain_i].samples
        M_list, tau_list = calculate_iact(samples, f, N_burn_in)
        return M_list, tau_list

    def diagnostic_plot(self, density=False):
        """
        After running gen_samples, plot the acceptance ratio and the samples
        """
        chain_fig_list = []
        for i in range(len(self.chain_list)):
            chain = self.chain_list[i]
            samples = chain.samples
            log_probs = chain.log_probs
            acceptance_ratios = chain.acceptance_ratios
            dim = samples.shape[0]
            fig1, axes = plt.subplots(2,1)
            fig1.suptitle('Log posterior and acceptance ratio for chain with $\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))
            axes[1].plot(acceptance_ratios)
            axes[0].plot(log_probs)
            #axes[0,0].hist(self.samples, bins=50)

            if dim == 1:
                fig2, ax = plt.subplots(1, dim)
                ax.hist(samples[0,:], bins=50, density=density)
            else:
                num_rows, num_cols = get_subplot_dims(dim)
                fig2, axes = plt.subplots(num_rows, num_cols)
                for i in range(dim):
                    ax_row = int(i/num_cols)
                    ax_col = i % num_cols
                    print(axes.shape, ax_col, i, num_cols)
                    if len(axes.shape) == 1:
                        axes[ax_col].hist(samples[i,:], bins=150, density=density)
                    else:
                        axes[ax_row, ax_col].hist(samples[i,:], bins=150, density=density)

            fig2.suptitle('$\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))
            chain_fig_list.append((fig1, fig2))

        swap_fig = plt.figure()
        #plt.imshow(self.swap_mat, aspect='auto', cmap='gray_r', interpolation='none')
        for i in range(len(self.chain_list)):
            plt.plot(self.swap_mat[i,:])

        return chain_fig_list, swap_fig


