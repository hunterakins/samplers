"""
Description:
    Try and test the SDE approach for sampling a Gaussian

Date:
    5/14/2023

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



dim = 1
N = 1000


def grad_log_gaussian(q, mu, sigma):
    return - (q-mu)**2 / sigma**2

def advance_sde(qn, pn, F, bq, dt):
    # qn, pn is current position and momentum
    # fq is the function to compute force  at qn (grad log probability)
    # bq is the function to compute the hessian inverse?j
    # dt is the timestep
    pn12 = pn + dt/2*F(qn)
    qn12 = qn + dt/2 * Bqn12
    pn12 = alpha*pn12 + (alpha+1)*dt / 2 div_Bqn12_T + np.sqrt(1-alpha**2)*np.random.randn(dim)
    qn1 = qn12 +dt/2*Bqn12 @ pn12
    p1 = p12 + dt/2 * F(qn1)


