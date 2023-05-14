"""
Description:
    Parallel tempering plus affine invariant

Date:
    4/28

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
from samplers.helpers import calculate_iact

def draw_g(a):
    Delta = np.sqrt(a) - 1/np.sqrt(a)
    U = Delta*(np.random.rand()) + 1/np.sqrt(a)
    z = np.square(U)
    return z

# Likelihood function given as a log-likelihood, proposal distribution
class ParallelTemperedAffineInvariantSampler: 
    def __init__(self, f_log_lh, f_log_prior, f_log_lh_kwargs={}, f_log_prior_kwargs={}):
        """ 
        f_log_lh is represents the unnormalized probability distribution (likelihood function)
        f_log_prior is the log prior distribution
        f_log_lh (x) / temp + f_log_prior is the posterior probability
        """
        self.f_log_lh = f_log_lh 
        self.f_log_lh_kwargs = f_log_lh_kwargs
        self.f_log_prior = f_log_prior 
        self.f_log_prior_kwargs = f_log_prior_kwargs
        self.weights = None # weights for proposal move selection
        self.samples = None #samples from chains
        self.log_probs = None # log posterior probabilities of samples
        self.acceptance_ratios = None #acceptance ratios of each walker  chain
        self.xcurr = None # buffer for current states for each walker chain
        self.log_post_curr = None # buffer for current log prob for each walker chain
        self.num_accepted = None # buffer to store number accepted in each walker chain
        self.a = None # stretch move param
        self.S = None # walk move param
        self.L = None # num walkers
        self.dim = None # dimension of parameter space
        self.M = None # number of tchains
        self.temps = None # temperatures for parallel tempering
        self.chain_index_array = None # keep track of temperature swaps
        self.prop_options = ['stretch', 'walk']

    def initialize_sampler(self, dim, a, S, L, M, temps, prop_weights):
        """
        Initialize the sampler parameters
        """
        self.dim = dim # dimension of space
        self.a = a  # stretch move param
        self.S = S  # num walkers in walk move
        self.L = L # num walkers
        self.M = M # number of temperature
        self.temps = temps # temperatures
        self.prop_weights = prop_weights
        self.prop_cycle = []
        for i in range(len(prop_weights)):
            for k in range(prop_weights[i]):
                self.prop_cycle.append(self.prop_options[i])

    def _make_stretch_move(self, temp_i, walker_i):
        """
        Make a stretch move
        """
        ind_vals = np.linspace(0, self.L-1, self.L, dtype=int)
        xk = self.xcurr[:, temp_i, walker_i]
        # select a random walker
        j = np.random.randint(0,self.L-1)
        xj = self.xcurr[:,temp_i, ind_vals!=j][:,j]
        # get stretch value and propose move
        z = draw_g(self.a)
        x_proposed = xj + z*(xk - xj)
        # get log probability of proposed move
        log_lh_proposed = self.f_log_lh(x_proposed, **self.f_log_lh_kwargs)
        log_prior_proposed = self.f_log_prior(x_proposed, **self.f_log_prior_kwargs)
        log_post_proposed = log_lh_proposed / self.temps[temp_i] + log_prior_proposed
        # get hastings ratio
        alpha = np.power(z, self.dim-1)*np.exp(log_post_proposed - self.log_post_curr[temp_i, walker_i])
        if np.random.rand() < alpha: # accept
            self.xcurr[:,temp_i, walker_i] = x_proposed
            self.log_lh_curr[temp_i, walker_i] = log_lh_proposed
            self.log_prior_curr[temp_i, walker_i] = log_prior_proposed
            self.log_post_curr[temp_i, walker_i] = log_post_proposed
            self.num_accepted[temp_i, walker_i] += 1
        return

    def _make_walk_move(self, temp_i, walker_i):
        """
        Make a walk move
        """
        # randomly select S walkers 
        ind_vals = np.linspace(0, self.L-1, self.L, dtype=int)
        x_set = self.xcurr[:,temp_i, ind_vals != walker_i]  # set of all other walkers
        ind_vals = np.linspace(0, self.L-2, self.L-1, dtype=int)
        ind_vals = np.random.permutation(ind_vals) # randomly permute indices
        x_set = x_set[:, ind_vals[:self.S]] # select S random walkers

        # get their covariance
        #xcov = np.cov(x_set) # get covariance of selected walkers
        #w = np.random.multivariate_normal(np.zeros(self.dim), xcov) # get random walk vector

        # propose a step
        xk = self.xcurr[:,temp_i, walker_i]
        x_proposed = xk.copy()
        xmean = np.mean(x_set, axis=1)
        for l in range(self.S): # equation 11 in Goodman and Weare
            xl = x_set[:, l]
            delta = xl - xmean
            x_proposed += np.random.randn()*delta # equation 11


        # get log probability of proposed move
        log_lh_proposed = self.f_log_lh(x_proposed, **self.f_log_lh_kwargs)
        log_prior_proposed = self.f_log_prior(x_proposed, **self.f_log_prior_kwargs)
        log_post_proposed = log_lh_proposed / self.temps[temp_i] + log_prior_proposed

        # get hastings ratio
        alpha = np.exp(log_post_proposed - self.log_post_curr[temp_i, walker_i])
        if np.random.rand() < alpha: # accept
            self.xcurr[:,temp_i, walker_i] = x_proposed
            self.log_post_curr[temp_i, walker_i] = log_post_proposed
            self.num_accepted[temp_i, walker_i] += 1
        return

    def _make_temp_swaps(self, i):
        """
        DEO swapping between temp chains
        """
        offset = i %2 # determine if even or odd swap
        num_swap_options = int(self.M // 2)
        for temp_i in range(self.M-offset-1,0, -2): # loop over ladder
            ihot = offset + temp_i
            icold = offset + temp_i - 1

            beta_hot = 1/self.temps[ihot]
            beta_cold = 1/self.temps[icold]

            cold_inds = np.random.permutation(self.L) # scramble ensemble
            hot_inds = np.random.permutation(self.L)

            hot_lh = self.log_lh_curr[ihot, hot_inds]
            cold_lh = self.log_lh_curr[icold, cold_inds]

            log_omega = (beta_hot - beta_cold)*(cold_lh-hot_lh)

            u = np.random.uniform(size=self.L)
            acc = u < np.exp(log_omega) # swap inds
            

            # interchange sample states, log_lh, log_prior, and log_post of accepted
            hot_acc_inds = hot_inds[acc] # walkers at hot temp to exchange
            cold_acc_inds = cold_inds[acc] # walkers at cold temp to exchange
            t1 = self.xcurr[:, ihot, hot_acc_inds].copy()
            t2 = self.xcurr[:, icold, cold_acc_inds].copy()
            self.xcurr[:, ihot, hot_acc_inds] = t2
            self.xcurr[:, icold, cold_acc_inds] = t1
            self.samples[:, ihot, hot_acc_inds, i] = t2
            self.samples[:, icold, cold_acc_inds, i] = t1

            t1 = self.log_lh_curr[ihot, hot_inds[acc]].copy()
            t2 = self.log_lh_curr[icold, cold_inds[acc]].copy()
            self.log_lh_curr[ihot, hot_inds[acc]] = t2
            self.log_lh_curr[icold, cold_inds[acc]] = t1

            t1 = self.log_prior_curr[ihot, hot_inds[acc]].copy()
            t2 = self.log_prior_curr[icold, cold_inds[acc]].copy()
            self.log_prior_curr[ihot, hot_inds[acc]] = t2
            self.log_prior_curr[icold, cold_inds[acc]] = t1

            
            #t1 = self.log_post_curr[ihot, hot_inds[acc]].copy()
            #t2 = self.log_post_curr[icold, cold_inds[acc]].copy()
            # use the swapped prior and lh to recompute the posterior
            t2 = self.log_lh_curr[ihot, hot_inds[acc]] / self.temps[ihot] + self.log_prior_curr[ihot, hot_inds[acc]]
            self.log_post_curr[ihot, hot_inds[acc]] = t2
            self.log_probs[ihot, hot_inds[acc], i] = t2
            t1 = self.log_lh_curr[ihot, hot_inds[acc]] / self.temps[ihot] + self.log_prior_curr[ihot, hot_inds[acc]]

            self.log_post_curr[icold, cold_inds[acc]] = t1
            self.log_probs[icold, cold_inds[acc], i] = t1


            # record swap in level array
            self.chain_index_array[ihot, hot_inds[acc], i] = self.temps[icold] # these were lowered
            self.chain_index_array[icold, cold_inds[acc], i] = self.temps[ihot] # these were raised
            self.chain_index_array[ihot, hot_inds[~acc], i] = self.temps[ihot] # these were not swapped
            self.chain_index_array[icold, cold_inds[~acc], i] = self.temps[icold] # these were not swapped
        return

    def select_move(self):
        """
        Weights are an unnormalized array of probabilities
        for stretch, walk, and replace
        """
        prop_cycle = self.prop_cycle
        if len(prop_cycle) == 1:
            return prop_cycle[0]
        N = len(prop_cycle)
        j = np.random.randint(0,N-1)
        return prop_cycle[j]

    def sample(self, x0, N):
        """
        Get N samples from the distribution p
        x0 is 1d
        N is an int
        L is number of walkers in the ensemble
        """
        self.samples = np.zeros((self.dim, self.M, self.L, N))
        self.samples[:,:,:,0] = x0.copy()
        self.xcurr = x0.copy() # array storing all current camples
        self.log_lh_curr = np.zeros((self.M, self.L)) # array storing all curr. lh
        self.log_prior_curr = np.zeros((self.M, self.L)) # array storing all curr. prior
        self.log_post_curr = np.zeros((self.M, self.L)) # array storing all curr. posterior

        self.num_accepted = np.zeros((self.M, self.L))
        self.acceptance_ratios = np.zeros((self.M, self.L, N))
        self.log_probs = np.zeros((self.M, self.L, N))

        self.chain_index_array = np.zeros((self.M, self.L, N), dtype=int) # array to store chain indices
        for i in range(self.M):
            self.chain_index_array[i,:,0] = self.temps[i]

        for temp_i in range(self.M):
            for walker_i in range(self.L):
                xcurr_ii = x0[:,temp_i, walker_i]
                llh_curr = self.f_log_lh(xcurr_ii, **self.f_log_lh_kwargs)
                lp_curr = self.f_log_prior(xcurr_ii, **self.f_log_prior_kwargs)
                self.log_lh_curr[temp_i, walker_i] = llh_curr
                self.log_prior_curr[temp_i, walker_i] = lp_curr
                self.log_post_curr[temp_i, walker_i] = llh_curr / self.temps[temp_i] + lp_curr


        ind_vals = np.linspace(0, self.L-1, self.L, dtype=int)
        for i in range(1,N):
            for temp_i in range(self.M):
                for walker_i in range(self.L):
                    move = self.select_move()
                    if move == 'stretch':
                        self._make_stretch_move(temp_i, walker_i)
                        #print('stretching')
                    else:
                        #print('walking')
                        self._make_walk_move(temp_i, walker_i)
                    # store accepted or rejected state in the chain
                    self.samples[:,temp_i, walker_i, i] = self.xcurr[:,temp_i, walker_i]
                    self.log_probs[temp_i, walker_i, i] = self.log_post_curr[temp_i, walker_i]
                    self.acceptance_ratios[temp_i, walker_i, i] = self.num_accepted[temp_i, walker_i] / i
            self._make_temp_swaps(i)

    def calculate_iact(self, f, N_burn_in):
        samples = self.samples
        tau_list, M_list = calculate_iact(samples, f, N_burn_in)
        return tau_list, M_list

    def diagnostic_plot(self, N_burn_in, density=False):
        """
        After running sample, plot the acceptance ratio and the samples
        """
        num_temps = self.M
        fig_list = []
        for k in range(num_temps):
            fig1, axes = plt.subplots(2,1)
            fig1.suptitle('Temperature: {}'.format(self.temps[k]))
            print(np.mean(self.log_probs[k,...], axis=0)[N_burn_in:])
            axes[1].plot(np.mean(self.acceptance_ratios[k,...], axis=0)[N_burn_in:])
            #axes[0,1].plot(self.samples)
            axes[0].plot(np.mean(self.log_probs[k,...], axis=0)[N_burn_in:])
            axes[0].set_ylabel('log posterior')
            axes[1].set_ylabel('acceptance ratio')
            #axes[0,0].hist(self.samples, bins=50)

            if self.dim == 1:
                fig2, ax = plt.subplots(1, self.dim)
                ax.hist(self.samples[:,k,...].flatten()[N_burn_in:], bins=50, density=density)
            else:
                if np.sqrt(self.dim) % 1 == 0: # square
                    num_cols = int(np.sqrt(self.dim))
                    fig2, axes = plt.subplots(num_cols, num_cols)
                else:
                    num_cols = int(np.sqrt(self.dim))+1
                    fig2, axes = plt.subplots(num_cols-1, num_cols)
                for i in range(self.dim):
                    ax_row = int(i/num_cols)
                    ax_col = i % num_cols
                    if len(axes.shape) == 1:
                        axes[ax_col].hist(self.samples[i,k,...].flatten()[N_burn_in:], bins=50, density=density)
                        axes[ax_col].set_title('x{}'.format(i))
                    else:
                        axes[ax_row, ax_col].hist(self.samples[i,k,...].flatten()[N_burn_in:], bins=50, density=density)
                        axes[ax_row, ax_col].set_title('x{}'.format(i))
            fig2.suptitle('Temperature: {}'.format(self.temps[k]))
            fig_list.append((fig1, fig2))
        return fig_list
