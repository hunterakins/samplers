"""
Description:
    Miscellaneous helper functions for the project.

Date:

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
from scipy import special

def get_subplot_dims(dim):
    """
    For dim-dimensional parameter
    get the subplot num_rows, num_cols
    """
    if dim == 1:
        return 1,1
    if np.sqrt(dim) % 1 == 0: # square
        num_rows = int(np.sqrt(dim))
        num_cols = num_rows
    else:
        num_rows = int(np.sqrt(dim))
        num_cols = dim / num_rows
        if num_cols % 1 != 0: # if it does not divide it...
            num_cols = int(num_cols) + 1
        num_cols = int(num_cols)
    return num_rows, num_cols

def calculate_mean_acorr(vals, B, M):
    """
    Use block size B to compute the integrated auto-corr time with max. 
    lag M
    Return an array sum_acorr which represents the autocorrelation function
    c(t) averaged over all blocks
    The values sum_acorr[:,i] are the estimated autocovariance for time delay 
    associated with the index i
    The first col is the variance of the time series for each dimension
    """
    N = vals.shape[1] # num samples of each dim
    dim = vals.shape[0] # dim..
    num_blocks = int((N-B) / (M))
    if num_blocks == 0:
        raise ValueError('Desired window for acf estimation is too small given the data (2M > N)')
    for i in range(num_blocks):
        block_vals = vals[:, i*M:(i+1)*M+B]
        mean_vals = block_vals.mean(axis=1)[:,np.newaxis]
        block_vals = block_vals - mean_vals
        fvals = np.fft.rfft(block_vals, axis=1)/np.sqrt(B)
        power = fvals * np.conj(fvals)
        acorr = np.fft.irfft(power, axis=1)
        acorr = acorr[:, :M] # throwout copy from symmetry
        if i == 0:
            sum_acorr = acorr.copy()
        else:
            sum_acorr += acorr.copy()
    sum_acorr /= num_blocks # get mean
    return sum_acorr

def test_acorr():
    vals = 2+5*np.random.randn(1, 1000)
    M = 20
    acorr = calculate_acf(vals)
    c0 = acorr[:,0]
    rho = acorr / acorr[:,0]
    tau = 1 + 2*np.sum(rho[:,1:M], axis=1)
    print('tau', tau)
    plt.figure()
    for i in range(vals.shape[0]):
        plt.plot(acorr[i,:])
        plt.plot(rho[i,:])
    plt.show()

def calculate_acf(vals):
    """
    Calculate a single acf for vals
    Result will only be valid up to a lag << vals.shape[1]
    """
    N = vals.shape[1] # num samples of each dim
    dim = vals.shape[0] # dim..
    mean_vals = np.mean(vals, axis=1)[:,np.newaxis]
    vals = vals - mean_vals
    fvals = np.fft.rfft(vals, axis=1)/np.sqrt(N)
    power = fvals * np.conj(fvals)
    acorr = np.fft.irfft(power, axis=1)
    return acorr

def calculate_iact(samples, f, N_burn_in, verbose=False, strict=False):
    """
    Calculate integrated autocorrelation time
    Use various values M << vals.shape
    """
    #print('N_burn_in', N_burn_in)
    vals = samples[...,N_burn_in:] # get rid of burn in
    vals = f(vals) # apply f to the samples
    acorr = calculate_acf(vals)
    acorr = acorr / acorr[:,0][:,None] # normalize

    N = acorr.shape[1]

    M = 10 # smallest block size considered
    max_num_steps = int(np.log2(N/M)) # since I double each step
    M_list, tau_list = [], []
    for i in range(max_num_steps):
        rho = acorr[:,1:M]
        tau = 1 + 2 * np.sum(rho, axis=1)

        if verbose == True:
            print("M : {0}, tau : {1}".format(M, tau))

        M_list.append(M)
        tau_list.append(tau)

        if M > 10*np.max(tau):
            break
        elif M > N / 10: # M << N not true...
            break
        else:
            M *= 2
    if M < 10*np.max(tau):
        if strict == True:
            raise Warning('M is not large enough to get a good estimate of tau')
        else:
            print('M is not large enough to get a good estimate of tau')
    if M > N/10:
        if strict == True:
            raise Warning('M is too significant of a fraction of N for a good estimate')
        else:
            print('M is not large enough to get a good estimate of tau')
    return M_list, tau_list

def get_opt_acc(dim):
    """BAsed on table I in Gelman Robert Gilks 1996 """
    if dim == 1:
        return 0.44
    elif dim == 2:
        return 0.352
    elif dim == 3:
        return 0.316
    elif dim == 4:
        return 0.279
    elif dim == 5:
        return 0.275
    else: # return asymptotic acceptance rate
        return 0.234

def unif_order_stat(yi, N, i, L):
    """
    Evaluate distribution of ith order statistic of N uniform random variables 
    on interval [0, L] at the point yi
    i = 1, 2, dots N
    """
    if i < 1 or i > N:
        raise ValueError("i must be between 1 and N (inclusive)")
    p = special.gamma(N) / (special.gamma(i-1) * special.gamma(N-i)) * np.power(yi/L, i-1) * np.power(1-yi/L, N-i) / L
    return p
if __name__ == '__main__':
    test_acorr()
    
