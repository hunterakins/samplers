"""
Description:
    Various likelihood functions for testing samplers

Date:
    4/22/2023


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
from numba import njit, jit


def get_multivariate_gaussian_params(num_params, mu_std, sigma_std, seed=1):
    """
    Draw random means and variances to define a multivariate Gaussian
    """
    if type(seed) is not type(None):
        np.random.seed(seed)
    # draw a sample from these distributions
    mu = np.random.randn(num_params) * mu_std

    # draw a sample from these distributions
    sigma = np.random.randn(num_params) * sigma_std
    sigma = abs(sigma)
    return mu, sigma

def get_random_colorization(num_params, seed=None): 
    """
    Get eigenvectors defining correlations between parameters
    """
    L = np.random.randn(num_params, num_params) # random eigenvectors
    P = L @ L.T # covariance matrix
    s, U = np.linalg.eigh(P)
    return U

class GaussianLikelihood:
    """
    A class for a multivariate Gaussian likelihood
    """
    def __init__(self, dim, mu_std, sigma_std, colored=False, seed=None):
        self.dim = dim
        self.mu_std = mu_std
        self.sigma_std = sigma_std
        self.colored = colored
        self.seed = seed
        self.mu, self.sigma = get_multivariate_gaussian_params(dim, mu_std, sigma_std, seed=seed)
        if colored == False:
            self.U = np.eye(dim)
        else:
            self.U = get_random_colorization(dim)
        self.sigma_inv = self.U@np.eye(dim)/np.square(self.sigma)@self.U.T
        self.sigma = self.U@np.eye(dim)*np.square(self.sigma)@self.U.T # covariance matrix
        self.f_log_p = self._get_log_p()

    def _get_log_p(self):
        def log_p(x):
            return -.5*np.dot((x-self.mu),self.sigma_inv @ (x-self.mu) )
        return log_p

def get_random_mu_sigma_list(num_peaks, dim, mu_std, sigma_std,sigma_mean, seed_list=None):
    """
    Generate a random set of means and standard deviations for a multimodal Gaussian
    """
    mu_list, sigma_list = [], []
    for i in range(num_peaks):
        if type(seed_list) is not type(None):
            seed = seed_list[i]
        mui, sigmai = get_multivariate_gaussian_params(dim, mu_std, sigma_std, seed=seed)
        mu_list.append(mui)
        sigma_list.append(sigmai+sigma_mean)
    return mu_list, sigma_list

class MultimodalGaussianLikelihood:
    """
    A class for a multimodal mixture of Gaussians
    """
    def __init__(self, dim, mu_list, sigma_list, colored=False, seed=None):
        self.dim = dim
        self.colored = colored
        self.seed = seed
        self.num_peaks = len(mu_list)
        self.mu_list = mu_list
        self.sigma_list = sigma_list
        if colored == False:
            self.U_list = [np.eye(dim) for i in range(self.num_peaks)]
        else:
            self.U_list = [get_random_colorization(dim) for i in range(self.num_peaks)]
        self.f_log_p = self._get_log_p()

    def _get_log_p(self):
        Sigma_inv_list = []
        for i in range(self.num_peaks):
            sigma = self.sigma_list[i] # std deviation
            if self.colored: # get a power law set of sigmas...
                # sigma_i = sigma_0 / i
                ii = np.linspace(1., self.dim, self.dim)
                sigma_arr = sigma / ii
            else:
                sigma_arr = sigma*np.ones(self.dim)
            # inverse of covariance
            D = np.diag(sigma_arr**-2)
            Sigma_inv = self.U_list[i]@D@self.U_list[i].T
            Sigma_inv_list.append(Sigma_inv)
        mu_list = self.mu_list
        num_peaks = self.num_peaks
        def log_p(x):
            like = 0
            for i in range(num_peaks):
                mu = mu_list[i]
                Sigma_inv=Sigma_inv_list[i]
                mode_contribution = np.exp(-.5*np.dot((x-mu),Sigma_inv @ (x-mu) )) 
                # denomanitor (det covariance is product of eigenvalues)
                mode_contribution /= np.sqrt(2*np.pi*np.prod(np.square(sigma)))
                like += mode_contribution
            if like == 0:
                return -np.inf
            return  np.log(like)
        return log_p

class RosenbrockTwoD:
    def __init__(self, a=1, b=100, c=20):
        self.a = a
        self.b = b
        self.c=c
        self.f_log_p = self._get_log_p()

    def _get_log_p(self):
        a = self.a
        b = self.b
        c = self.c
        @njit
        def log_p(x):
            return (-(a - x[0])**2 - b*(x[1]-x[0]**2)**2) / c
        return log_p

    def plot(self):
        xgrid = np.linspace(-4, 6, 100)
        ygrid = np.linspace(0, 30, 100)
        zgrid = np.zeros((100, 100))
        log_p = self._get_log_p()
        for i in range(ygrid.size):
            for j in range(xgrid.size):
                zgrid[i,j] = np.exp(log_p(np.array([xgrid[j], ygrid[i]])))
        plt.figure()
        plt.pcolormesh(xgrid, ygrid, zgrid)
        plt.colorbar()

def plot_twod_gaussian():
    num_peaks = 2
    mu_std = 10 # standard deviation of mean distribution
    sigma_std = 4 # standard deviation of variance distribution
    num_params = 2
    likelihood = MultimodalGaussianLikelihood(num_params, mu_std, sigma_std, num_peaks, colored=True)
    log_p = likelihood.f_log_p

    xgrid = np.linspace(-5*mu_std, 5*mu_std, 200)
    ygrid = np.linspace(-5*mu_std, 5*mu_std, 200)
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = np.zeros_like(X)
    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            Z[i,j] = np.exp(log_p(np.array([xgrid[i], ygrid[j]])))
    plt.figure()
    plt.pcolormesh(X, Y, Z)
    plt.colorbar()
    print('likelihood mean: ', likelihood.mu_list)
    plt.show()

def plot_oned_multimodal():
    num_peaks = 2
    mu_std = 10 # standard deviation of mean distribution
    sigma_std = 4 # standard deviation of variance distribution
    num_params = 1
    likelihood = MultimodalGaussianLikelihood(num_params, mu_std, sigma_std, num_peaks, colored=True)
    log_p = likelihood.f_log_p

    xgrid = np.linspace(-5*mu_std, 5*mu_std, 200)
    Z = np.zeros_like(xgrid)
    for i in range(len(xgrid)):
        Z[i] = np.exp(log_p(np.array([xgrid[i]])))
    plt.figure()
    plt.plot(xgrid, Z)
    print('likelihood mean: ', likelihood.mu_list)
    plt.show()


if __name__ == '__main__':  
    for i in range(100):
        plot_twod_gaussian()

    plot_oned_multimodal()
