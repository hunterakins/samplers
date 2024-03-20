""" Description: 
    Parallel tempering fixed dimension
    Proposal is hard coded to be a Gaussian
Date:
    12/4/2023

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


class SampleCovarianceMatrix:
    def __init__(self, mu0, sigma0, N_samp):
        """
        Keep track of number of samples used in covariance
        matrix estimation
        """
        self.mu = mu0 
        self.sigma = sigma0
        self.N_samp = N_samp

    def _update(self, x):
        """
        Recursively update the SCM
        to inclue the measurement x
        """
        mu = update_mean(self.mu, x, self.N_samp)
        self.mu = mu
        sigma = update_scm(self.sigma, mu, x, self.N_samp)
        self.N_samp += 1
        self.sigma = sigma
        return mu, sigma, self.N_samp

class ChainParams:
    def __init__(self, dim, beta, N, nu, update_after_burn, cov_interval): 
        """
        beta is chain inverse temp
        N is number of samples
        """
        self.dim = dim
        self.beta = beta
        self.N = N 
        self.nu = nu
        self.update_after_burn = update_after_burn
        self.cov_interval = cov_interval
        return

class Chain:
    def __init__(self, params):
        """
        Initialize a chain with params
        """
        self.params = params
        self.samples = np.zeros((params.dim, params.N))
        self.log_probs = np.zeros(params.N)
        self.acceptance_log = np.zeros(params.N-1) # I make N-1 steps to get N samples...
        self.pert_proposals = 0
        self.accepted = 0 # accepted perturbation steps
        self.acceptance_ratios = np.zeros(params.N-1)
        self.curr_log_prior = None # prior
        self.curr_log_lh = None # likelihood
        self.curr_log_p = None # posterior
        return

def mh_walk(x0, N_tune, f_proposal, prop_cov, f_log_prior, f_log_lh, beta):
    """
    Run a MH walk with Gaussian proposal with prop_cov
    """
    dim = x0.size
    curr_log_prior = f_log_prior(x0)
    curr_log_lh = f_log_lh(x0)
    curr_log_p = curr_log_prior + beta*curr_log_lh
    x_samples = np.zeros((x0.size, N_tune+1))
    x_samples[:,0] = x0
    accepted= 0
    acceptance_ratios = np.zeros((N_tune))
    log_post = np.zeros((N_tune + 1))
    log_post[0] = curr_log_p
    for j in range(1, N_tune+1):
        x = x_samples[:, j-1].copy()
        u, gu = f_proposal(prop_cov)
        xprime = x.copy()
        xprime += u # update coefficients
        log_prior = f_log_prior(xprime)
        log_lh = f_log_lh(xprime)
        log_p = log_prior + beta*log_lh
        alpha = (log_p) - (curr_log_p + 0.0) # assumes symmetric proposal 
        alpha = np.exp(alpha)
        if np.random.rand() < alpha: #accept move
            x_samples[:,j] = xprime
            curr_log_prior = log_prior
            curr_log_lh = log_lh
            curr_log_p = log_p
            accepted += 1
        else: # reject
            x_samples[:,j] = x
        log_post[j] = curr_log_p
        acceptance_ratios[j-1] = (accepted)/j
    return x_samples, acceptance_ratios, log_post

class AdaptivePTSampler:
    """
    Parallel tempered sampler Use adaptive sampling (Haario) and user specified temperature ladder
    """
    def __init__(self, move_probs, dim, beta_arr, 
            f_log_prior, f_log_lh, f_proposal):
        """
        move_prob_list - list of list of move probabilities for each dimension
            move_prob_list[0] = [prob_pert] for 
        beta_arr - list of inverse temperatures in temperature ladder
        f_log_prior - log prior func
        f_log_lh - log likelihood func
        f_proposal - proposal function
        """
        self.moves = ['pert']
        self.dim = dim
        self.move_probs = move_probs
        self.beta_arr = beta_arr
        self.f_log_prior = f_log_prior
        self.f_log_lh = f_log_lh
        self.f_proposal = f_proposal

    def tune_proposal(self, N_tune, f_prior, N_burn = 1000, N_scm = 1000, variance_range = np.logspace(-6, 0, 7)):
        """
        For each chain and each dimension, run a sequence of perturb steps using a diagonal Gaussian proposal 
        with varying variances
        Calculate the acceptance ratio for each, and choose the optimal scaling.
        Once the optimal scaling is selected, run the chain 1000 steps, and estimate the sample covariance matrix. 
        Store that in prop_cov_list and return that
        """
        num_chains = self.beta_arr.size
        dim = self.dim
        prop_covs = []
        #plt.figure()
        best_vars = []
        x0_list = []
        for i in range(num_chains):
            beta = self.beta_arr[i]


            opt_acc = get_opt_acc(dim)
            xvals = f_prior() # generate random sample
            x = xvals
            sd = 2.4 ** 2 / dim # scaling of variance in Gelman Roberts Gilks 1996
            ratios = np.zeros((len(variance_range)))
            #plt.figure()
            x_maps = []
            for var_i, var in enumerate(variance_range): # now generate 100 samples for each variance
                prop_cov = np.eye(dim)*var
                x_samples, acc_ratios, log_post = mh_walk(x, N_tune, self.f_proposal, prop_cov*sd, self.f_log_prior, self.f_log_lh, beta)
                x_map = x_samples[:,np.argmax(log_post)]
                x_maps.append(x_map)
                final_acc_ratio = acc_ratios[-1]
                ratios[var_i] = final_acc_ratio
                #plt.plot(acc_ratios)
            best_i = np.argmin(np.abs(ratios - opt_acc))
            x_map = x_maps[best_i]
            print('dim', dim)
            print('best var', variance_range[best_i])
            print('best ratio', ratios[best_i])
            best_vars.append(variance_range[best_i])
            #plt.plot(np.log10(variance_range), ratios, label='dim: {}'.format(dim))

            """
            Now run a chain of length N_scm with burn in ... to get a number of samples to use for proposal covariance
            """
            prop_cov = np.eye(dim)*variance_range[best_i] # use best scaling for SCM estimation
            x_samples, acc_ratios, log_posts = mh_walk(x_map, N_scm+N_burn, self.f_proposal, prop_cov*sd, self.f_log_prior, self.f_log_lh, beta)
            x_map = x_samples[:,np.argmax(log_posts)]
            x_samples = x_samples[:, N_burn:] # discard burn
            x0_list.append(x_samples[:,-1])
            print('print prop cov tune acc ratio', acc_ratios[-1])
            sigma0 = (np.cov(x_samples)) # sample covariance matrix
            mu0 = np.mean(x_samples, axis=1)
            scm = SampleCovarianceMatrix(mu0, sigma0, x_samples.shape[1])
            prop_covs.append(scm)
        return x0_list, prop_covs
                    
    def initialize_chains(self, N, N_burn_in, nu, f_prior, update_after_burn, swap_interval, prop_covs, cov_interval, x0_list=[]):
        """
        Initialize the chains 
        Initialize the proposal covariance matrices
        This proposal covariance will be used for the jump proposal until params.nu has been
        reached, at which point the SCM will be updated using the chain samples
        N - int
            number of samples for chain to produce. Includes the iniital sample 
            produced by running burn in
        f_prior - function
            takes in the dimension of the state and returns a random sample
        prop_covs - ...
        eps - float
            initial variance on the proposal covariance matrices
        """
        self.chain_list = []
        self.N_burn_in = N_burn_in
        self.N = N
        self.swap_interval = swap_interval
        dim = self.dim
        for i in range(len(self.beta_arr)): # for each temperature
            beta = self.beta_arr[i]
            chain_p = ChainParams(dim, beta, N, nu, update_after_burn, cov_interval)
            chain = Chain(chain_p)
            if len(x0_list) == 0:
                xvals = f_prior()
                x = xvals
            else:
                x = x0_list[i].copy()
            chain.samples[:,0] = x
            chain.curr_log_prior = self.f_log_prior(x)
            chain.curr_log_lh = self.f_log_lh(x)
            chain.curr_log_p = chain.curr_log_prior + beta*chain.curr_log_lh
            self.chain_list.append(chain)
            chain.log_probs[0] = chain.curr_log_p
        self.prop_covs = prop_covs
        self.swap_mat = np.zeros((len(self.beta_arr), N), dtype=int)
        self.swap_mat[:,0] = np.arange(len(self.beta_arr))
        return

    def _perturb_move(self, i, j):
        """
        Just perturb chain and preserve dimension
        """
        chain = self.chain_list[i]
        chain.pert_proposals += 1
        x = chain.samples[:,j] # 
        beta = chain.params.beta
        curr_log_p = chain.curr_log_p # log posterior of the current chain state
        prop_cov = self.prop_covs[i].sigma.copy()
        sd = 2.4**2 / self.dim # default from haario 2001

        u, gu = self.f_proposal(prop_cov*sd)
        xprime = x.copy()
        xprime += u # update coefficients
        log_prior = self.f_log_prior(xprime)
        log_lh = self.f_log_lh(xprime)
        log_p = log_prior + beta*log_lh
        alpha = (log_p) - (curr_log_p + 0) # assumes symmetric proposal 
        alpha = np.exp(alpha)
        if np.random.rand() < alpha: #accept move
            chain.samples[:,j+1] = xprime
            chain.curr_log_prior = log_prior
            chain.curr_log_lh = log_lh
            chain.curr_log_p = log_p
            chain.accepted += 1
            chain.log_probs[j+1] = log_p
        else: # reject 
            chain.samples[:,j+1] = x
            chain.log_probs[j+1] = curr_log_p
        return chain

    def _select_move(self):
        move = np.random.choice(self.moves, p=self.move_probs)
        return move

    def _update_cov(self, i, j):
        """
        Update proposal covariance matrix of the ith temperature chain
        with the appropriate dimension
        """
        chain = self.chain_list[i]
        prop_cov = self.prop_covs[i]
        cov_interval = chain.params.cov_interval
        if j % cov_interval == 0:
            x = chain.samples[:, j+1]
            N_samp = prop_cov.N_samp
            if N_samp > chain.params.nu: # only update after we have obtained nu samples
                if (j > self.N_burn_in) and  (chain.params.update_after_burn):
                    prop_cov._update(x)
                elif j < self.N_burn_in:
                    prop_cov._update(x)
        return

    def _update_chain(self, i, j):
        """
        Propose move
        This populates sample j+1
        """
        chain = self.chain_list[i]
        move = self._select_move()
        if move == 'pert':
            self._perturb_move(i, j)
        if chain.pert_proposals > 0:
            chain.acceptance_ratios[j] = (chain.accepted)/(chain.pert_proposals)
        return

    def _swap_chains(self, j):
        """
        Propose swaps between chains
        """

        # swaps (DEO)
        self.swap_mat[:,j+1] = self.swap_mat[:,j].copy()
        swap_interval = self.swap_interval
        num_chains = self.beta_arr.size
        if (j%swap_interval==0):
            # ........
            X = j / swap_interval
            even =  X% 2
            for chain_ind in range(num_chains-1):
                # get two chains up for swap proposal
                hot_chain_ind = num_chains - 1 - chain_ind
                hot_chain = self.chain_list[hot_chain_ind]
                cold_chain_ind = hot_chain_ind - 1
                cold_chain = self.chain_list[cold_chain_ind]
                if ((even) and (hot_chain_ind % 2 == 0)) or ((not even) and (hot_chain_ind % 2 == 1)):

                    # get their likelihood and temps for calculating proposal ratio
                    hot_lh = hot_chain.curr_log_lh
                    cold_lh = cold_chain.curr_log_lh
                    beta_hot = hot_chain.params.beta
                    beta_cold = cold_chain.params.beta
                    if beta_hot > beta_cold:
                        print('beta hot, beta_cold', beta_hot, beta_cold)

                    # compute proposal ratio
                    log_omega = (beta_hot-beta_cold)*(cold_lh - hot_lh)
                    omega = np.exp(log_omega)
                    u = np.random.rand()
                    if u < omega: # swap accepted
                        x_hot = hot_chain.samples[:,j+1].copy()
                        x_cold = cold_chain.samples[:,j+1].copy()
                       
                        cold_prior = cold_chain.curr_log_prior
                        hot_prior = hot_chain.curr_log_prior
                       
                        hot_chain.samples[:,j+1] = x_cold
                        cold_chain.samples[:,j+1] = x_hot

                        hot_chain.curr_log_lh = cold_lh
                        cold_chain.curr_log_lh = hot_lh

                        hot_chain.curr_log_prior = cold_prior
                        cold_chain.curr_log_prior = hot_prior

                        hot_chain.curr_log_p = beta_hot * hot_chain.curr_log_lh + hot_chain.curr_log_prior
                        cold_chain.curr_log_p = beta_cold * cold_chain.curr_log_lh + cold_chain.curr_log_prior

                        # note it in the swap mat
                        t1 =  self.swap_mat[hot_chain_ind, j].copy()
                        t2 = self.swap_mat[cold_chain_ind, j].copy()
                        self.swap_mat[hot_chain_ind, j+1] = t2
                        self.swap_mat[cold_chain_ind, j+1] = t1


                        """
                        ### The chains maintain their history but get a new temperature
                        ### Their curr prob values need to be updated to reflect new temperatue
                        hot_chain.curr_log_p = beta_cold * hot_chain.curr_log_lh + hot_chain.curr_log_prior
                        cold_chain.curr_log_p = beta_hot * cold_chain.curr_log_lh + cold_chain.curr_log_prior


                        #swap temperatures
                        cold_chain.params.beta = beta_hot
                        hot_chain.params.beta = beta_cold

                        # note it in the swap mat
                        t1 =  self.swap_mat[hot_chain_ind, j].copy()
                        t2 = self.swap_mat[cold_chain_ind, j].copy()
                        self.swap_mat[hot_chain_ind, j+1] = t2
                        self.swap_mat[cold_chain_ind, j+1] = t1

                        # update log probs to reflect new temperature
                        hot_chain.log_probs[j+1] = hot_chain.curr_log_p
                        cold_chain.log_probs[j+1] = cold_chain.curr_log_p
                        """
        return

    def sample(self):
        """
        Make N steps on each chain
        """
        N = self.N
        num_temps = len(self.beta_arr)
        #self._burn_in()
        for j in range(N-1):
            for i in range(num_temps):
                self._update_chain(i,j) # propose plus accept/reject
                #self._update_cov(i,j) # updating proposal covariance matrix
            self._swap_chains(j) # proposing chain temperature swaps
        return

    def calculate_iact(self, chain_i, f, N_burn_in):
        samples = self.chain_list[chain_i].samples
        M_list, tau_list = calculate_iact(samples, f, N_burn_in)
        return M_list, tau_list

    def diagnostic_plot(self, density=False):
        """
        After running gen_samples, plot the acceptance ratio and the samples
        """
        chain_fig_list = []
        for i in range(len(self.chain_list)): # for each temperature
            chain = self.chain_list[i]
            samples = chain.samples
            log_probs = chain.log_probs
            acceptance_ratios = chain.acceptance_ratios
            print('i chain accpeted', chain.accepted)
            fig1, axes = plt.subplots(2,1)
            fig1.suptitle('Log posterior and acceptance ratio for chain with $\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))
            axes[1].plot(acceptance_ratios, label='pert acceptance')
            axes[1].legend()
            axes[0].plot(log_probs)
            #axes[0,0].hist(self.samples, bins=50)
        


        swap_fig = plt.figure()
        #plt.imshow(self.swap_mat, aspect='auto', cmap='gray_r', interpolation='none')
        for i in range(len(self.chain_list)):
            plt.plot(self.swap_mat[i,:])
        return chain_fig_list, swap_fig

    def get_chain_info(self, temp_i):
        samples = self.chain_list[temp_i].samples
        log_probs = self.chain_list[temp_i].log_probs
        acceptance_ratios = self.chain_list[temp_i].acceptance_ratios
        return samples, log_probs, acceptance_ratios

    def single_chain_diagnostic_plot(self, temp_i, density=False):
        """
        temp_i - int
            index of temperature in temperature ladder
        """
        plt.figure()
        beta = self.beta_arr[temp_i]
        t = 1/beta
        plt.suptitle('Chain index corresponding to chain with T={}'.format(t))
        plt.plot(self.swap_mat[temp_i,:])
        samples, log_probs, acceptance_ratios = self.get_chain_info(temp_i)
        log_p_ar_fig, axes = plt.subplots(2,1)
        log_p_ar_fig.suptitle('Log posterior and acceptance ratio for chain with $\\beta = {0}, T = {1} $'.format(round(beta, 2), round(1/beta, 2)))
        axes[1].plot(acceptance_ratios, label='pert acceptance')
        axes[1].legend()
        axes[0].plot(log_probs)
        #axes[0,0].hist(self.samples, bins=50)
        dim = self.dim 
        fig, ax = plt.subplots(1,self.dim)
        dim_samples = samples
        for k in range(dim):
            ax[k].hist(dim_samples[k,:], bins=150, density=density)

        fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(beta, 2), round(1/beta, 2)))

        fig, ax = plt.subplots(1,1)
        ax.plot(samples[0,:])
        fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(beta, 2), round(1/beta, 2)))


        #swap_fig = plt.figure()
        ##plt.imshow(self.swap_mat, aspect='auto', cmap='gray_r', interpolation='none')
        #for i in range(len(self.chain_list)):
        #    plt.plot(self.swap_mat[i,:])
        return  log_p_ar_fig, fig, ax

    def get_map_x(self, temp_i):
        """
        temp_i - int
            index of temperature in temperature ladder
        """
        samples, log_probs, acceptance_ratios = self.get_chain_info(temp_i)
        best_ind = np.argmax(log_probs)
        x_map = samples[:,best_ind]
        return x_map, log_probs[best_ind]
