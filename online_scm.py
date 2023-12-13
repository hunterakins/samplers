"""
Description:
    Online calculation of sample covariance matrix

Date:
    4/17/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from numba import njit

@njit
def update_mean(mun, xn1, n):
    """
    mu_n = \frac{1}{n} \sum_{i=1}^{n} x_i
    xn1 = x_{n+1}
    return \frac{1}{n+1} \sum_{i=1}^{n} x_i = \frac{n}{n+1} \mu_n + \frac{1}{n+1} x_{n+1}
    """
    mu_n1 = n/(n+1) * mun + 1/(n+1)*xn1
    return mu_n1

@njit
def update_scm(Cn, mun, xn1, n):
    """
    Return recursive update to the biased (normalized by N)
    sample covariance matrix
    when new observation xn1 comes online after n observations
    Cn is scm using first n observations, mun is sample mean
    of first n observations

    mu_n = \frac{1}{n} \sum_{i=1}^{n} x_i
    xn1 = x_{n+1}
    Cn1 = C_n + 1/(n+1) outer(x_n1, x_n1) - ...
    """
    output = n/(n+1)*Cn
    term1 = (1/(n+1) - (1/(n+1)**2))*np.outer(xn1, xn1)
    term2 = -(2*n)/(n+1)**2*np.outer(xn1, mun)
    term3 = (n) / (n+1)**2 * np.outer(mun, mun)
    output += term1 + term2 + term3
    output = .5*(output + output.T)
    return output

def test_recursion(N, M):
    """ M is dim, N is num samples 
    I use biased cov...
    Doesn't really matter since I'm tuning a covariance matrix
    scale anyways..."""
    x_samples = np.random.randn(M, N)
    X = np.cov(x_samples, bias=True)
    mu = np.mean(x_samples, axis=1)

    # calculate recursively
    mun = np.zeros(M)
    Cn = np.zeros((M,M))
    for i in range(N):
        xn1 = x_samples[:,i]
        n = i
        Cn = update_scm(Cn, mun, xn1, n)
        mun = update_mean(mun, xn1, n)

    mu_diff = mun - mu
    print('x', x_samples)
    print('mun', mun)
    print('mu', mu)
    cov_diff = Cn - X
    print('Cn', Cn)
    print('X', X)
    print('Cn diff', cov_diff)
    print("Cn ratio", Cn/ X)

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
  
if __name__ == '__main__':
    np.random.seed(1)
    test_recursion(1000, 3)
