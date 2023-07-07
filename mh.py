"""
Description:
   Metropolis-Hastings sampler

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
from samplers.online_scm import update_mean, update_scm
from samplers.likelihoods import GaussianLikelihood, MultimodalGaussianLikelihood, RosenbrockTwoD
from samplers.proposals import gaussian_proposal

# Likelihood function given as a log-likelihood, proposal distribution
class MHSampler: 
    def __init__(self, f_log_p, f_prop, f_log_p_kwargs={}, f_prop_kwargs={}):
        """ 
        f_p is represents the unnormalized probability distribution (likelihood function)
        f is the proposal density (function of chain state and kwargs, returns a sample as well as the ratio of proposal density

        """
        self.f_log_p = f_log_p 
        self.f_log_p_kwargs = f_log_p_kwargs
        self.f_prop = f_prop 
        self.f_prop_kwargs = f_prop_kwargs

    def gen_samples(self, x0, N):
        """
        Get N samples from the distribution p
        x0 is 1d
        N is an int
        alpha is Hastings ratio
        """
        dim = x0.size
        samples = np.zeros((dim,N))
        log_probs = np.zeros((N))
        samples[0] = x0.copy() # initial sample
        acceptance_ratios = np.zeros(N)
        xcurr = x0.copy()
        log_p_curr = self.f_log_p(xcurr, **self.f_log_p_kwargs)
        num_accepted = 0
        for i in range(1,N):
            proposed_sample, qxx = self.f_prop(xcurr, **self.f_prop_kwargs) 
            log_p_proposed = self.f_log_p(proposed_sample, **self.f_log_p_kwargs)
            alpha = np.exp(log_p_proposed - log_p_curr)
            alpha *= qxx # posterior asymmetry ratio
            if alpha >= 1:
                xcurr = proposed_sample
                num_accepted += 1
                log_p_curr = log_p_proposed
            else:
                u = np.random.rand()
                if u < alpha:
                    xcurr = proposed_sample
                    num_accepted += 1
                    log_p_curr = log_p_proposed
            samples[:,i] = xcurr.copy()
            log_probs[i] = log_p_curr
            acceptance_ratios[i] = num_accepted/(i)
        self.samples = samples
        self.log_probs = log_probs
        self.acceptance_ratios = acceptance_ratios
        return samples, log_probs, acceptance_ratios

    def calculate_acf(self):
        """
        Get autocorrelation function of chain
        """
        return

    def diagnostic_plot(self, density=False):
        """
        After running gen_samples, plot the acceptance ratio and the samples
        """
        dim = self.samples.shape[0]
        fig1, axes = plt.subplots(2,1)
        axes[1].plot(self.acceptance_ratios)
        #axes[0,1].plot(self.samples)
        axes[0].plot(self.log_probs)
        #axes[0,0].hist(self.samples, bins=50)

        if dim == 1:
            fig2, ax = plt.subplots(1, dim)
            ax.hist(self.samples, bins=50, density=density)
        else:
            if np.sqrt(dim) % 1 == 0: # square
                num_cols = int(np.sqrt(dim))
                fig2, axes = plt.subplots(num_cols, num_cols)
            else:
                num_cols = int(np.sqrt(dim))+1
                fig2, axes = plt.subplots(num_cols-1, num_cols)
            for i in range(dim):
                ax_row = int(i/num_cols)
                ax_col = i % num_cols
                print(axes.shape, ax_col)
                if len(axes.shape) == 1:
                    axes[ax_col].hist(self.samples[i,:], bins=50, density=density)
                else:
                    axes[ax_row, ax_col].hist(self.samples[i,:], bins=50, density=density)
        return fig1, fig2

def gaussian_test():
    """
    Test MHSampler on an unnormalized Gaussian
    """
    # define a Gaussian distribution
    def p(x):
        mu = 0
        sigma = 1
        return np.exp(-(x-mu)**2 / (2*sigma**2))

    def log_p(x):
        mu = 0
        sigma = 1
        return -(x-mu)**2 / (2*sigma**2)

    num_params = 1
    def f_prop(x, **kwargs):
        sigma_prop = kwargs['sigma']
        x_new = x + sigma_prop*np.random.randn(num_params)
        qxx = 1 # symmetric so ratio of densitys is 1
        return x_new, qxx

    # create an MHSampler object
    sigma_prop = 2
    prop_kwargs = {'sigma' : sigma_prop}
    sampler = MHSampler(log_p, f_prop, f_prop_kwargs=prop_kwargs)

    # get 1000 samples from the distribution
    x0 = np.array([10])
    N = 10000
    samples, log_probs, acceptance_ratios = sampler.gen_samples(x0, N)

    # plot the samples
    sampler.diagnostic_plot()

class AdaptiveSampler(MHSampler):
    def __init__(self, f_log_p, f_log_p_kwargs={}):
        f_prop = gaussian_proposal
        super().__init__(f_log_p, f_prop, f_log_p_kwargs=f_log_p_kwargs)

    def _init_chain(self, x0, N):
        """
        Initialize variables for the chain that will have N samples
        and has initial guess x0
        """
        dim = x0.size
        samples = np.zeros((dim, N))
        log_probs = np.zeros((N))
        samples[:,0] = x0.copy() # initial sample
        acceptance_ratios = np.zeros(N)
        return dim, samples, log_probs, acceptance_ratios

    def gen_samples(self, x0, N, sigma0=None, eps = 1e-6, sd=None, nu=1000, N_burnin=1000):
        """
        Get N samples from the distribution p
        x0 is 1d
        N is an int
        alpha is Hastings ratio
        nu is number of samples to use before begininning online estimation of 
        proposal covariance
        sigma0 is prior covariance
        sd is scale factor 
        """
        # set proposal covariance
        if sigma0 is None:
            sigma = eps*np.eye(x0.size) # default diagonal cov matrix
        else:
            sigma = sigma0

        if sd is None:
            sd =(2.4)**2 / x0.size # default scale factor
        self.f_prop_kwargs = {'sigma' : sigma}

        # initialize chain and probs
        dim, samples, log_probs, acceptance_ratios = self._init_chain(x0, N)

        # initialize state variables
        xcurr = x0.copy()
        log_p_curr = self.f_log_p(xcurr, **self.f_log_p_kwargs)

        # do a burn in period
        for i in range(N_burnin):
            # propose sample and compute hastings ratio
            proposed_sample, qxx = self.f_prop(xcurr, **self.f_prop_kwargs) 
            log_p_proposed = self.f_log_p(proposed_sample, **self.f_log_p_kwargs)
            alpha = np.exp(log_p_proposed - log_p_curr)
            alpha *= qxx # posterior asymmetry ratio

            # accept or reject
            if alpha >= 1:
                xcurr = proposed_sample
            else:
                u = np.random.rand()
                if u < alpha:
                    xcurr = proposed_sample
            log_p_curr = self.f_log_p(xcurr, **self.f_log_p_kwargs)

        log_probs[0] = log_p_curr
        num_accepted = 0
        # run chain
        for i in range(1,N):
            if i == nu: # initialize covariance from samples
                sigma = np.cov(samples[:,:nu])
                mu  = np.mean(samples, axis=1)
                prop_sigma = sd*sigma + sd*eps*np.eye(dim)
                self.f_prop_kwargs = {'sigma' : prop_sigma}
            
            # propose sample and compute hastings ratio
            proposed_sample, qxx = self.f_prop(xcurr, **self.f_prop_kwargs) 
            log_p_proposed = self.f_log_p(proposed_sample, **self.f_log_p_kwargs)
            alpha = np.exp(log_p_proposed - log_p_curr)
            alpha *= qxx # posterior asymmetry ratio

            # accept or reject
            u = np.random.rand()
            if u < alpha:
                xcurr = proposed_sample
                num_accepted += 1
                log_p_curr = log_p_proposed

            # store updated sample in chain, along with log prob and acceptance ratio
            samples[:,i] = xcurr.copy()
            log_probs[i] = log_p_curr
            acceptance_ratios[i] = num_accepted/(i)
           
            # update proposal covariance if after burnin
            if i > nu: 
                sigma = update_scm(sigma, mu, xcurr, i)
                mu = update_mean(mu, xcurr, i)
                prop_sigma = sd*sigma + sd*eps*np.eye(dim)
                self.f_prop_kwargs = {'sigma' : prop_sigma}

        # store samples, log probs, and acceptance ratios
        self.samples = samples
        self.log_probs = log_probs
        self.acceptance_ratios = acceptance_ratios
        self.sigma_f = sigma
        return samples, log_probs, acceptance_ratios

#class PTAdaptiveSampler(MHSampler):

def adaptive_gaussian_test():
    """
    Test MHSampler on an uncorrelated Gaussian
    """
    # define a multivariate Gaussian distribtino
    num_params = 10
    # set variance of mean distribution
    mu_std = 10*np.random.randn(num_params)
    # set variance of variance distribution
    sigmas = 20*np.random.randn(num_params)

    mu, sigma = get_multivariate_gaussian_params(num_params, mu_std, sigmas)

    # define likelihood function assuming a uniform prior on the data
    sigma_inv = np.eye(num_params) / np.square(sigma)
    def log_p(x):
        return -.5*np.dot((x-mu),sigma_inv @ (x-mu) )


    # create an MHSampler object
    sampler = AdaptiveSampler(log_p)

    # sample the distro with adaptive sampler
    x0 = np.zeros(num_params)
    N = 100000
    sigma0 = 20*np.eye(num_params) # standard deviation for mu
    sd =(2.4)**2 / x0.size # default scale factor # from Haario
    print('sd', sd)
    sd /= 5
    samples, log_probs, acceptance_ratios = sampler.gen_samples(x0, N, sigma0=sigma0, eps=1e-5, nu=100, sd=sd)

    # plot the samples
    fig1, fig2 = sampler.diagnostic_plot(density=True)
    axes = fig2.axes
    for i in range(mu.size):
        mui = mu[i]
        sigmai = sigma[i]
        grid = np.linspace(mui - 3*sigmai, mui+3*sigmai, 100)
        vals = np.exp(-(grid-mui)**2/(2*sigmai**2)) / np.sqrt(2*np.pi*sigmai**2)
        axes[i].plot(grid, vals)
    plt.show()
    print('sigma f', sampler.sigma_f)

def adaptive_colored_gaussian_test():
    """
    Test MHSampler on an correlated gaussian noise
    """
    # define a multivariate Gaussian distribtino
    num_params = 10
    seed = 1
    
    mu_std = 10
    likelihood = GaussianLikelihood(num_params, mu_std, 20, colored=True, seed=seed)
    log_p = likelihood.f_log_p

    # create an MHSampler object
    sampler = AdaptiveSampler(log_p)

    # get 1000 samples from the distribution
    x0 = np.zeros(num_params)
    N = 10000
    sigma0 = 20*np.eye(num_params) # standard deviation for mu
    sd =(2.4)**2 / x0.size # default scale factor # from Haario
    sd /= 1
    print('sd', sd)
    N_burnin=3000
    samples, log_probs, acceptance_ratios = sampler.gen_samples(x0, N, sigma0=sigma0, eps=1e-5, nu=1000, sd=sd, N_burnin=N_burnin)

    # plot the samples
    fig1, fig2 = sampler.diagnostic_plot(density=True)
    axes = fig2.axes
    num_cols = int(np.sqrt(num_params))
    for i in range(num_params):
        ax_row = int(i/num_cols)
        ax_col = i % num_cols
        mui = likelihood.mu[i]
        sigmai = np.sqrt(likelihood.sigma[i,i])
        grid = np.linspace(mui - 3*sigmai, mui+3*sigmai, 100)
        vals = np.exp(-(grid-mui)**2/(2*sigmai**2)) / np.sqrt(2*np.pi*sigmai**2)
        axes[num_cols*ax_row+ ax_col].plot(grid, vals)

    plt.figure()
    plt.suptitle('sigma f')
    plt.imshow(sampler.sigma_f)
    plt.colorbar()
    #print('sigma f', sampler.sigma_f)
    plt.show()

def adaptive_multimodal_gaussian_test():
    """ Test MHSampler on an correlated gaussian noise
    """
    # define a mulitmodal Gaussian distribtino
    num_params = 2
    seed = None
    num_peaks = 2
    mu_std = 10 # standard deviation of mean distribution
    sigma_std = 3 # standard deviation of variance distribution
    likelihood = MultimodalGaussianLikelihood(num_params, mu_std, sigma_std, num_peaks, colored=True, seed=seed)
    log_p = likelihood.f_log_p

    # create an MHSampler object
    sampler = AdaptiveSampler(log_p)

    # get 1000 samples from the distribution
    x0 = np.zeros(num_params)
    N = 10000
    sigma0 = 20*np.eye(num_params) # standard deviation for mu
    sd =(2.4)**2 / x0.size # default scale factor # from Haario
    sd /= 1
    print('sd', sd)
    N_burnin=3000
    samples, log_probs, acceptance_ratios = sampler.gen_samples(x0, N, sigma0=sigma0, eps=1e-5, nu=1000, sd=sd, N_burnin=N_burnin)

    # plot the samples
    fig1, fig2 = sampler.diagnostic_plot(density=True)
    axes = fig2.axes
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
        axes[num_cols*ax_row+ ax_col].plot(grid, vals)

    plt.figure()
    plt.suptitle('sigma f')
    plt.imshow(sampler.sigma_f)
    plt.colorbar()
    #print('sigma f', sampler.sigma_f)
    plt.show()

if __name__ == '__main__':
    #adaptive_gaussian_test()
    adaptive_colored_gaussian_test()
    #adaptive_multimodal_gaussian_test()
    #gaussian_test()
