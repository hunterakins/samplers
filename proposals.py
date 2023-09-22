"""
Description:
    Proposal functions for MCMC

Date:
    4/24/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from numba import jit, njit


def gaussian_proposal(x, **kwargs):
    """
    A gaussian proposal distribution with
    covariance matrix passed in kwargs
    Note that it is not normalized by the determinant
    """
    sigma_prop = kwargs['sigma']
    num_params = x.size
    x_new = np.random.multivariate_normal(x, sigma_prop)
    qxx = 1 # symmetric so ratio of densitys is 1
    return x_new, qxx

@njit
def jit_gaussian_proposal(x, sigma_eigs, sigma_eigvecs):
    num_params = x.size
    x_new = x.copy()
    for i in range(num_params):
        tmp = np.random.randn() * np.sqrt(sigma_eigs[i])
        x_new += tmp * sigma_eigvecs[:, i]
    qxx = 1 # symmetric so ratio of densitys is 1
    return x_new, qxx
