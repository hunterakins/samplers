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
from matplotlib import rc
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


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
