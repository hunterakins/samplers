"""
Description:
   Affine-invariant MCMC (Goodman and Weare)

Date:
    4/6/2023

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
class AffineInvariantSampler: 
    def __init__(self, f_log_p, f_log_p_kwargs={}):
        """ 
        f_p is represents the unnormalized probability distribution (likelihood function)
        f is the proposal density (function of chain state and kwargs, returns a sample as well as the ratio of proposal density

        """
        self.f_log_p = f_log_p 
        self.f_log_p_kwargs = f_log_p_kwargs
        self.weights = None # weights for proposal move selection
        self.samples = None #samples from chains
        self.log_probs = None # log probabilities of samples
        self.acceptance_ratios = None #acceptance ratios of each walker  chain
        self.xcurr = None # buffer for current states for each walker chain
        self.log_p_curr = None # buffer for current log prob for each walker chain
        self.num_accepted = None # buffer to store number accepted in each walker chain
        self.a = None # stretch move param
        self.S = None # walk move param
        self.L = None # num walkers
        self.dim = None # dimension of parameter space
        self.N = None # number of samples in a chain
        self.prop_options = ['stretch', 'walk']

    def initialize_sampler(self, dim, a, S, L, prop_weights):
        """
        Initialize the sampler parameters
        """
        self.dim = dim
        self.a = a 
        self.S = S 
        self.L = L
        self.prop_weights = prop_weights
        self.prop_cycle = []
        for i in range(len(prop_weights)):
            for k in range(prop_weights[i]):
                self.prop_cycle.append(self.prop_options[i])

    def _make_stretch_move(self, walker_i):
        """
        Make a stretch move
        """
        ind_vals = np.linspace(0, self.L-1, self.L, dtype=int)
        xk = self.xcurr[:, walker_i]
        # select a random walker
        j = np.random.randint(0,self.L-1)
        xj = self.xcurr[:,ind_vals!=j][:,j]
        # get stretch value and propose move
        z = draw_g(self.a)
        x_proposed = xj + z*(xk - xj)
        # get log probability of proposed move
        log_p_proposed = self.f_log_p(x_proposed, **self.f_log_p_kwargs)
        # get hastings ratio
        alpha = np.power(z, self.dim-1)*np.exp(log_p_proposed - self.log_p_curr[walker_i])
        if np.random.rand() < alpha: # accept
            self.xcurr[:,walker_i] = x_proposed
            self.log_p_curr[walker_i] = log_p_proposed
            self.num_accepted[walker_i] += 1
        return

    def _make_walk_move(self, walker_i):
        """
        Make a walk move
        """
        # randomly select S walkers 
        ind_vals = np.linspace(0, self.L-1, self.L, dtype=int)
        x_set = self.xcurr[:,ind_vals != walker_i]  # set of all other walkers
        ind_vals = np.linspace(0, self.L-2, self.L-1, dtype=int)
        ind_vals = np.random.permutation(ind_vals) # randomly permute indices
        x_set = x_set[:,ind_vals[:self.S]] # select S random walkers

        # get their covariance
        #xcov = np.cov(x_set) # get covariance of selected walkers
        #w = np.random.multivariate_normal(np.zeros(self.dim), xcov) # get random walk vector

        # propose a step
        xk = self.xcurr[:, walker_i]
        x_proposed = xk.copy()
        xmean = np.mean(x_set, axis=1)
        for l in range(self.S): # equation 11 in Goodman and Weare
            xl = x_set[:,l]
            delta = xl - xmean
            x_proposed += np.random.randn()*delta # equation 11
        log_p_proposed = self.f_log_p(x_proposed, **self.f_log_p_kwargs)

        # get hastings ratio
        alpha = np.exp(log_p_proposed - self.log_p_curr[walker_i])
        if np.random.rand() < alpha: # accept
            self.xcurr[:,walker_i] = x_proposed
            self.log_p_curr[walker_i] = log_p_proposed
            self.num_accepted[walker_i] += 1
        return

    def select_move(self):
        """
        Weights are an unnormalized array of probabilities
        for stretch and walk
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
        self.N = N
        self.samples = np.zeros((self.dim, self.L, N))
        self.samples[:,:,0] = x0.copy()
        acceptance_ratios = np.zeros(N)
        self.xcurr = x0.copy()
        self.log_p_curr = np.zeros(self.L)
        for i in range(self.L):
            xcurr_i = x0[:,i]
            self.log_p_curr[i] = self.f_log_p(xcurr_i, **self.f_log_p_kwargs)
        self.num_accepted = np.zeros(self.L)
        self.acceptance_ratios = np.zeros((self.L, N))
        self.log_probs = np.zeros((self.L, N))

        ind_vals = np.linspace(0, self.L-1, self.L, dtype=int)
        for i in range(1,N):
            for walker_i in range(self.L):
                move = self.select_move()
                if move == 'stretch':
                    self._make_stretch_move(walker_i)
                else:
                    self._make_walk_move(walker_i)
                # store accepted or rejected state in the chain
                self.samples[:,walker_i, i] = self.xcurr[:,walker_i]
                self.log_probs[walker_i, i] = self.log_p_curr[walker_i]
                self.acceptance_ratios[walker_i, i] = self.num_accepted[walker_i] / i

    def calculate_iact(self, f, N_burn_in):
        """
        Get autocorrelation function of E(f) using a block of size B
        and maximum lag of M
        f is assumed to be ?
        """
        samples = self.samples
        M_list, tau_list = calculate_iact(samples, f, N_burn_in)
        return M_list, tau_list

    def diagnostic_plot(self, density=False):
        """
        After running sample, plot the acceptance ratio and the samples
        """
        num_walkers = self.samples.shape[1]
        fig_list = []
        fig1, axes = plt.subplots(2,1)
        axes[1].plot(np.mean(self.acceptance_ratios, axis=0))
        #axes[0,1].plot(self.samples)
        axes[0].plot(np.mean(self.log_probs, axis=0))
        #axes[0,0].hist(self.samples, bins=50)

        if self.dim == 1:
            fig2, ax = plt.subplots(1, self.dim)
            ax.hist(self.samples[0,:,:].flatten(), bins=50, density=density)
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
                print(axes.shape, ax_col)
                if len(axes.shape) == 1:
                    axes[ax_col].hist(self.samples[i,:,:].flatten(), bins=50, density=density)
                else:
                    axes[ax_row, ax_col].hist(self.samples[i,:,:].flatten(), bins=50, density=density)
        fig_list.append((fig1, fig2))
        return fig_list

def test_g():
    vals = np.zeros(10000)
    a = 2
    for i in range(10000):
        vals[i] = draw_g(2)
    plt.hist(vals, bins=100, density=True)
    z_vals = np.linspace(1/a, a, 100)
    norm = 1/(np.sqrt(a) - 1/np.sqrt(a))
    plt.plot(z_vals, norm/(2*np.sqrt(z_vals)), 'r')
    plt.show()

if __name__ == '__main__':
    test_g()
    #adaptive_multimodal_gaussian_test()
