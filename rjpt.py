""" Description: Reverse-jump MCMC for polynomial regression Proposal is hard coded to be a Gaussian
    Birth-death proposal randomly selected at each step
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
    def __init__(self, max_dim, beta, N, nu, update_after_burn): 
        """
        max dim is largest vector size
        beta is chain inverse temp
        N is number of samples
        """
        self.max_dim = max_dim
        self.beta = beta
        self.N = N 
        self.nu = nu
        self.update_after_burn = update_after_burn
        return

class Chain:
    def __init__(self, params):
        """
        Initialize a chain with params
        """
        self.params = params
        self.samples = np.zeros((params.max_dim+1, params.N+1))
        self.log_probs = np.zeros(params.N+1)
        self.acceptance_log = np.zeros(params.N)
        self.birth_acceptance_log = np.zeros(params.N)
        self.death_acceptance_log = np.zeros(params.N)
        self.birth_proposals = 0
        self.death_proposals = 0
        self.birth_accepted = 0
        self.death_accepted = 0
        self.accepted = 0 # accepted perturbation steps
        self.acceptance_ratios = np.zeros(params.N)
        self.birth_acceptance_ratios = np.zeros(params.N)
        self.death_acceptance_ratios = np.zeros(params.N)
        self.curr_log_prior = None # priot
        self.curr_log_lh = None # likelihood
        self.curr_log_p = None # posterior
        return

def mh_walk(x0, N_tune, f_proposal, prop_cov, f_log_prior, f_log_lh, beta):
    """
    Run a MH walk with Gaussian proposal with prop_cov
    """
    dim = int(x0[0])
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
        xprime[1:dim+1] += u # update coefficients
        log_prior = f_log_prior(xprime)
        log_lh = f_log_lh(xprime)
        log_p = log_prior + beta*log_lh
        alpha = (log_p) - (curr_log_p + 0) # assumes symmetric proposal 
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

class AdaptiveTransDPTSampler:
    """
    Trans-dimensional parallel tempered sampler Use adaptive sampling (Haario) and user specified temperature ladder
    """
    def __init__(self, move_probs, dim_list, beta_arr, f_log_prior, f_log_lh, f_proposal, f_log_gprime):
        """
        move_prob_list - list of list of move probabilities for each dimension
            move_prob_list[0] = [prob_pert, prob_birth, prob_death] for 
        beta_arr - list of inverse temperatures in temperature ladder
        f_log_prior - log prior func
        f_log_lh - log likelihood func
        f_proposal - proposal function
        f_log_gprime - proposal density function for the scalar birth proposal
            necessary for computing the GMH coefficient for death move
        """
        self.moves = ['birth', 'death']
        self.dim_list = dim_list
        self.num_dims = len(dim_list)
        self.move_probs = move_probs
        self.beta_arr = beta_arr
        self.f_log_prior = f_log_prior
        self.f_log_lh = f_log_lh
        self.f_proposal = f_proposal
        self.f_log_gprime = f_log_gprime
        self.birth_sigma_sq = np.array([1.0])
        self.no_birth_death=False

    def tune_proposal(self, N_tune, f_prior):
        """
        For each chain and each dimension, run a sequence of perturb steps using a diagonal Gaussian proposal 
        with varying variances
        Calculate the acceptance ratio for each, and choose the optimal scaling.
        Once the optimal scaling is selected, run the chain 1000 steps, and estimate the sample covariance matrix. 
        Store that in prop_cov_list and return that
        """
        num_chains = self.beta_arr.size
        num_dims = len(self.dim_list)
        max_dim = np.max(self.dim_list)
        prop_covs = []
        variance_range = np.logspace(-6, 0, 7)
        N_scm = 1000
        #plt.figure()
        for i in range(num_chains):
            dim_covs = []
            beta = self.beta_arr[i]
            best_vars = []
            for j in range(num_dims):
                dim = self.dim_list[j]


                opt_acc = get_opt_acc(dim)
                xvals = f_prior(dim) # generate random sample
                x = np.zeros(max_dim+1)
                x[0] = dim
                x[1:dim+1] = xvals
                sd = 2.4 ** 2 / dim # scaling of variance in Gelman Roberts Gilks 1996
                ratios = np.zeros((len(variance_range)))
                #plt.figure()

                for var_i, var in enumerate(variance_range): # now generate 100 samples for each variance
                    prop_cov = np.eye(dim)*var
                    x_samples, acc_ratios, _ = mh_walk(x, N_tune, self.f_proposal, prop_cov*sd, self.f_log_prior, self.f_log_lh, beta)
                    final_acc_ratio = acc_ratios[-1]
                    ratios[var_i] = final_acc_ratio
                    #plt.plot(acc_ratios)
                best_i = np.argmin(np.abs(ratios - opt_acc))
                print('dim', dim)
                print('best var', variance_range[best_i])
                print('best ratio', ratios[best_i])
                best_vars.append(variance_range[best_i])
                #plt.plot(np.log10(variance_range), ratios, label='dim: {}'.format(dim))

                """
                Now run a chain of length ... to get a number of samples to use for proposal covariance
                """
                prop_cov = np.eye(dim)*variance_range[best_i] # use best scaling for SCM estimation
                x_samples, acc_ratios, _ = mh_walk(x, N_scm, self.f_proposal, prop_cov*sd, self.f_log_prior, self.f_log_lh, beta)
                print('print prop cov tune acc ratio', acc_ratios[-1])
                sigma0 = (np.cov(x_samples[1:dim+1,:])) # sample covariance matrix
                mu0 = np.zeros(dim)
                scm = SampleCovarianceMatrix(mu0, sigma0, N_scm)
                dim_covs.append(scm)
            prop_covs.append(dim_covs)
            #plt.plot(range(num_dims), best_vars, label='beta: {}'.format(beta))
        #plt.show()
        return prop_covs
                    
    def initialize_chains(self, N, N_burn_in, nu, f_prior, update_after_burn, swap_interval, prop_covs):
        """
        Initialize the chains 
        Initialize the proposal covariance matrices
        This proposal covariance will be used for the jump proposal until params.nu has been
        reached, at which point the SCM will be updated using the chain samples
        N - int
            number of steps for chain to take (so final number of samples is N+1)
        f_prior - function
            takes in the dimension of the state and returns a random sample
        prop_covs - ...
        eps - float
            initial variance on the proposal covariance matrices
        """
        self.chain_list = []
        self.N_burn_in = N_burn_in
        self.N_steps = N
        self.swap_interval = swap_interval
        max_dim = max(self.dim_list)
        for i in range(len(self.beta_arr)): # for each temperature
            beta = self.beta_arr[i]
            chain_p = ChainParams(max_dim, beta, N, nu, update_after_burn)
            chain = Chain(chain_p)
            dim_ind = np.random.choice(self.num_dims)
            dim = self.dim_list[dim_ind]
            xvals = f_prior(dim)
            x = np.zeros(max_dim+1)
            x[0] = dim
            x[1:dim+1] = xvals
            chain.samples[:,0] = x
            chain.curr_log_prior = self.f_log_prior(x)
            chain.curr_log_lh = self.f_log_lh(x)
            chain.curr_log_p = chain.curr_log_prior + beta*chain.curr_log_lh
            self.chain_list.append(chain)
            chain.log_probs[0] = chain.curr_log_p

        """
        prop_covs = []
        for i in range(len(self.beta_arr)):
            dim_covs = []
            for j in range(len(self.dim_list)):
                dim = self.dim_list[j]
                sigma0 = prop_cov_arr[i]*np.eye(dim)
                mu0 = np.zeros(dim)
                N_samp = 1
                scm = SampleCovarianceMatrix(mu0, sigma0, N_samp)
                dim_covs.append(scm)
            prop_covs.append(dim_covs)
        """
        self.prop_covs = prop_covs
        self.swap_mat = np.zeros((len(self.beta_arr), N+1), dtype=int)
        self.swap_mat[:,0] = np.arange(len(self.beta_arr))
        return

    def _birth_move(self, i, j):
        """
        Birth move for chain i at iteration j
        Propose a state x' with dimension 1 greater than x
        Accept it with probability min(1, move_ratio)
        In particular, the (j+1)th sample is added to the chain
        move_ratio - ratio of proposing a birth at state x to proposing a death at state x'
        """
        chain = self.chain_list[i]
        chain.birth_proposals += 1
        x = chain.samples[:,j] # current state
        if self.no_birth_death:
            print('hi')
            chain.samples[:,j+1] = x.copy()
            return chain
        beta = chain.params.beta
        curr_log_p = chain.curr_log_p # log posterior of the current chain state (accounts for temp)
        dim = int(x[0])
        dim_ind = self.dim_list.index(dim)
        p_birth = self.move_probs[dim_ind][0]
        p_death = self.move_probs[dim_ind+1][1]
        u, log_gu = self.f_proposal(self.birth_sigma_sq)
        xprime = x.copy()
        xprime[0] += 1 # update dimension
        xprime[dim+1] = u # update new coefficient
        log_prior = self.f_log_prior(xprime)
        log_lh = self.f_log_lh(xprime)
        log_p = log_prior + beta*log_lh # use temperature of chain...
        alpha = (log_p + 0 + np.log(p_death)) - (curr_log_p + log_gu + np.log(p_birth)) 
        alpha = np.exp(alpha)
        if np.random.rand() < alpha: # accept step
            chain.samples[:,j+1] = xprime.copy()
            chain.curr_log_prior = log_prior
            chain.curr_log_lh = log_lh
            chain.curr_log_p = log_p
            chain.birth_acceptance_log[j] = 1
            chain.birth_accepted += 1
        else:
            chain.samples[:,j+1] = x.copy()
            chain.birth_acceptance_log[j] = 0
        return chain

    def _death_move(self, i, j):
        """
        Propose a death move for chain at iteration j
        Propose a state x' with dimension 1 less than x
        """
        chain = self.chain_list[i]
        chain.death_proposals += 1
        x = chain.samples[:,j]
        if self.no_birth_death:
            print('hi')
            chain.samples[:,j+1] = x.copy()
            return chain
        beta = chain.params.beta
        curr_log_p = chain.curr_log_p # log posterior of the current chain state
        dim = int(x[0])
        dim_ind = self.dim_list.index(dim)
        #move_p_ratio = self.move_probs[dim_ind][1]/self.move_probs[dim_ind-1][0] # death/birth
        p_death = self.move_probs[dim_ind][1]
        p_birth = self.move_probs[dim_ind-1][0]
        xprime = x.copy()
        xprime[0] -= 1 # update dimension
        u = xprime[dim] # coefficient to be removed
        xprime[dim] = 0 # update new coefficient
        log_prior = self.f_log_prior(xprime)
        log_lh = self.f_log_lh(xprime)
        sigma_sq = self.birth_sigma_sq
        log_guprime = self.f_log_gprime(u, **{'sigma_sq':sigma_sq})
        log_p = log_prior + beta*log_lh
        alpha = (log_p + log_guprime + np.log(p_birth)) - (curr_log_p + 0 + np.log(p_death))
        alpha = np.exp(alpha)
        
        if np.random.rand() < alpha: #accept move
            chain.samples[:,j+1] = xprime 
            chain.curr_log_prior = log_prior
            chain.curr_log_lh = log_lh
            chain.curr_log_p = log_p
            chain.death_acceptance_log[j] = 1
            chain.death_accepted += 1
        else: # reject and keep the original sample
            chain.samples[:,j+1] = x
            chain.death_acceptance_log[0] = 1
        return chain

    def _perturb_move(self, i, j):
        """
        Just perturb chain and preserve dimension
        """
        chain = self.chain_list[i]
        x = chain.samples[:,j+1] # since I have done either a birth or death move
        beta = chain.params.beta
        curr_log_p = chain.curr_log_p # log posterior of the current chain state
        dim = int(x[0])
        dim_ind = self.dim_list.index(dim)
        prop_cov = self.prop_covs[i][dim_ind].sigma.copy()
        sd = 2.4**2 / dim # default from haario 2001

        u, gu = self.f_proposal(prop_cov*sd)
        xprime = x.copy()
        xprime[1:dim+1] += u # update coefficients
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
        chain.acceptance_ratios[j] = (chain.accepted)/(j+1)
        return chain

    def _select_move(self, dim):
        dim_ind = self.dim_list.index(dim)
        move = np.random.choice(self.moves, p=self.move_probs[dim_ind])
        return move

    def _update_cov(self, i, j):
        """
        Update proposal covariance matrix of the ith temperature chain
        with the appropriate dimension
        """
        chain = self.chain_list[i]
        dim = int(chain.samples[0, j+1]) # dimension of the current state
        dim_ind = self.dim_list.index(dim)
        prop_cov = self.prop_covs[i][dim_ind]
        x = chain.samples[1:dim+1, j+1]
        N_samp = prop_cov.N_samp
        if N_samp > chain.params.nu: # only update after we have obtained nu samples
            if (j > self.N_burn_in) and  (chain_params.update_after_burn):
                prop_cov.update(x)
            elif j < self.N_burn_in:
                prop_cov.update(x)
        return

    def _update_chain(self, i, j):
        """
        Propose birth or death
        This populates sample j+1
        Then do a perturb move on sample j+1
        """
        chain = self.chain_list[i]
        dim = int(chain.samples[0,j])
        move = self._select_move(dim)
        if move == 'birth':
            self._birth_move(i, j)
        elif move == 'death':
            self._death_move(i, j)
        if chain.birth_proposals > 0:
            chain.birth_acceptance_ratios[j] = chain.birth_accepted/chain.birth_proposals
        if chain.death_proposals > 0:
            chain.death_acceptance_ratios[j] = chain.death_accepted/chain.death_proposals
        self._perturb_move(i, j)
        return

    def _swap_chains(self, j):
        """
        Propose swaps between chains
        """

        # swaps (DEO)
        self.swap_mat[:,j+1] = self.swap_mat[:,j].copy()
        swap_interval = self.swap_interval
        num_chains = self.beta_arr.size
        if j > self.N_burn_in and (j%swap_interval==0):
            # ........
            X = j / swap_interval
            even =  X% 2
            for chain_ind in range(num_chains-1):
                # get two chains up for swap proposal
                hot_chain_ind = num_chains - 1 - chain_ind
                hot_chain = self.chain_list[self.swap_mat[hot_chain_ind, j+1]]
                cold_chain_ind = hot_chain_ind - 1
                cold_chain = self.chain_list[self.swap_mat[cold_chain_ind, j+1]]
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
        return

    def sample(self):
        """
        Make N steps on each chain
        """
        N = self.N_steps
        num_temps = len(self.beta_arr)
        for j in range(N):
            for i in range(num_temps):
                self._update_chain(i,j) # propose plus accept/reject
                self._update_cov(i,j) # updating proposal covariance matrix
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
            birth_acceptance_ratios = chain.birth_acceptance_ratios
            death_acceptance_ratios = chain.death_acceptance_ratios
            #dim = samples.shape[0]
            fig1, axes = plt.subplots(2,1)
            fig1.suptitle('Log posterior and acceptance ratio for chain with $\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))
            axes[1].plot(acceptance_ratios, label='pert acceptance')
            axes[1].plot(death_acceptance_ratios, label='death acceptance')
            axes[1].plot(birth_acceptance_ratios, label='birth acceptance')
            axes[1].legend()
            axes[0].plot(log_probs)
            #axes[0,0].hist(self.samples, bins=50)
        
            # now get samples from same dims together...
            """
            fig_ax_list = [plt.subplots(1, x) for x in self.dim_list]
            for dim_ind, dim in enumerate(self.dim_list):
                dim_mask = samples[0,:] == dim
                fig, ax = fig_ax_list[dim_ind]
                dim_samples = samples[:, dim_mask][1:dim+1]
                for k in range(dim):
                    ax[k].hist(dim_samples[k,:], bins=150, density=density)

                fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))

            """
            fig, ax = plt.subplots(1,1)
            ax.hist(samples[0,:], bins=len(self.dim_list), range=(min(self.dim_list)-.5, max(self.dim_list) + .5), density=density)
            fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))

            fig, ax = plt.subplots(1,1)
            ax.plot(samples[0,:])
            fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))


        swap_fig = plt.figure()
        #plt.imshow(self.swap_mat, aspect='auto', cmap='gray_r', interpolation='none')
        for i in range(len(self.chain_list)):
            plt.plot(self.swap_mat[i,:])
        return chain_fig_list, swap_fig

    def get_chain_info(self, temp_i):
        inds = self.swap_mat[temp_i,:] 
        N = self.N_steps
        samples = np.zeros((max(self.dim_list) + 1, N+1))
        log_probs = np.zeros((N+1))
        acceptance_ratios = np.zeros(N)
        birth_acceptance_ratios = np.zeros(N)
        death_acceptance_ratios = np.zeros(N)

        for k in range(N+1):
            samples[:,k] = self.chain_list[inds[k]].samples[:,k]
            log_probs[k] = self.chain_list[inds[k]].log_probs[k]
            if k < N:
                acceptance_ratios[k] = self.chain_list[inds[k]].acceptance_ratios[k]
                birth_acceptance_ratios[k] = self.chain_list[inds[k]].birth_acceptance_ratios[k]
                death_acceptance_ratios[k] = self.chain_list[inds[k]].death_acceptance_ratios[k]
        return samples, log_probs, acceptance_ratios, birth_acceptance_ratios, death_acceptance_ratios

    def single_chain_diagnostic_plot(self, temp_i, density=False):
        """
        temp_i - int
            index of temperature in temperature ladder
        """
        plt.figure()
        plt.suptitle('Chain index corresponding to cold chain')
        beta = self.beta_arr[temp_i]
        t = 1/beta
        plt.plot(self.swap_mat[temp_i,:])
        samples, log_probs, acceptance_ratios, birth_acceptance_ratios, death_acceptance_ratios = self.get_chain_info(temp_i)
        log_p_ar_fig, axes = plt.subplots(2,1)
        log_p_ar_fig.suptitle('Log posterior and acceptance ratio for chain with $\\beta = {0}, T = {1} $'.format(round(beta, 2), round(1/beta, 2)))
        axes[1].plot(acceptance_ratios, label='pert acceptance')
        axes[1].plot(death_acceptance_ratios, label='death acceptance')
        axes[1].plot(birth_acceptance_ratios, label='birth acceptance')
        axes[1].legend()
        axes[0].plot(log_probs)
        #axes[0,0].hist(self.samples, bins=50)
    
        # now get samples from same dims together...
        fig_ax_list = [plt.subplots(1, x) for x in self.dim_list]
        for dim_ind, dim in enumerate(self.dim_list):
            dim_mask = (samples[0,:] == dim)
            fig, ax = fig_ax_list[dim_ind]
            dim_samples = samples[:, dim_mask][1:dim+1]
            for k in range(dim):
                ax[k].hist(dim_samples[k,:], bins=150, density=density)

            fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(beta, 2), round(1/beta, 2)))

        dim_fig, ax = plt.subplots(1,1)
        ax.hist(samples[0,:], bins=len(self.dim_list), color='k', range=(min(self.dim_list)-.5, max(self.dim_list) + .5), density=density)
        #dim_fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))

        fig, ax = plt.subplots(1,1)
        ax.plot(samples[0,:])
        fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(beta, 2), round(1/beta, 2)))


        #swap_fig = plt.figure()
        ##plt.imshow(self.swap_mat, aspect='auto', cmap='gray_r', interpolation='none')
        #for i in range(len(self.chain_list)):
        #    plt.plot(self.swap_mat[i,:])
        return  log_p_ar_fig, fig_ax_list, dim_fig
