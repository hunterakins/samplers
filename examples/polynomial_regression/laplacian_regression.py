"""
Description:
    Use a Markov chain to sample the posterior distribution on polynomial coefficients 
    The polynomial coefficients determine the system state at a sequence of times, which are measured.
    The measurements are corrupted with Gaussian noise of known variance. 

Date:
    2023/9/20

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from samplers.mh import MHSampler
pics_folder = '/home/hunter/research/samplers/notes/pics/'


def get_f_log_prior(poly_dim, poly_mean, poly_sigma):
    """
    Description:
        Returns a function that evaluates the log prior distribution on a polynomial coefficient vector
    Args:
        poly_dim: int, dimension of polynomial
        poly_mean: float, mean of prior distribution on polynomial coefficients
        poly_sigma: float, standard deviation of prior distribution on polynomial coefficients
    Returns:
        f_log_prior: function, evaluates the log prior distribution on a polynomial coefficient vector
    """
    def f_log_prior(poly_coeff):
        return -0.5 * np.sum((poly_coeff - poly_mean)**2) / poly_sigma**2
    return f_log_prior

def get_f_log_likelihood(poly_dim, y_meas, tgrid, b):
    """
    Description:
        Returns a function that evaluates the log likelihood function on a polynomial coefficient vector
    Args:
        poly_dim: int, dimension of polynomial (degree of polynomial plus 1)
        tgrid: numpy array, times at which polynomial is evaluated
        b : scale parameter for associated Laplacian distribution
    """
    def f_log_likelihood(m):
        y = np.polyval(m, tgrid)
        return -0.5 * np.sum(np.abs(y - y_meas)) / b
    return f_log_likelihood

def get_f_proposal(prop_sigma):
    def f_proposal(m):
        sample = m + np.random.randn(m.size) * prop_sigma
        qxx = 1
        return sample, qxx
    return f_proposal

def get_H(tgrid, poly_dim):
    """
    Linear model for poly evaluation
    """
    H = np.zeros((len(tgrid), poly_dim))
    for i in range(poly_dim):
        H[:, i] = tgrid**(poly_dim - i - 1)
    return H


def get_m_true():
    m_true =np.load('m_true.npy')
    return m_true

def make_m_true():
    poly_dim = 5
    m_true = np.random.randn(poly_dim)
    np.save('m_true.npy', m_true)
    return m_true

def get_post_mean_cov(H, y_meas, noise_sigma, prior_sigma):
    """
    Description:
        Returns the posterior mean and covariance for the linear model vector
        x, which is related to the measurements y_meas by
        y_meas = H @ x + noise
        Noise is zero-mean gaussian with covariance matrix noise_sigma**2 * I
        The prior on x is zero-mean Gaussian with covariance matrix prior_sigma**2 * I
    Args:
        H: numpy array, linear model matrix
        y_meas: numpy array, measurements
        noise_sigma: float, standard deviation of noise
        prior_sigma: float, standard deviation of prior
    Returns:
        x_mean: numpy array, posterior mean of x
        x_cov: numpy array, posterior covariance of x
    """
    xdim = H.shape[1]
    prior_cov = prior_sigma**2 * np.eye(xdim)
    noise_cov = noise_sigma**2 * np.eye(len(y_meas))
    Kgain = prior_cov @ H.T @ np.linalg.inv(H @ prior_cov @ H.T + noise_cov)
    x_cov = prior_cov - Kgain @ H @ prior_cov
    x_mean = Kgain@y_meas
    return x_mean, x_cov

def get_H(tgrid, poly_dim):
    """
    Linear model for poly evaluation
    """
    H = np.zeros((len(tgrid), poly_dim))
    for i in range(poly_dim):
        H[:, i] = tgrid**(poly_dim - i - 1)
    return H

def run_simulation():
    poly_dim = 5
    N = 100
    tgrid = np.linspace(-1.0, 1.0, N)

    poly_mean = np.zeros(poly_dim)
    poly_sigma = 1.0
    noise_b = 0.1
    prop_sigma = 1.0

    #m_true = np.random.randn(poly_dim) * poly_sigma
    m_true = np.load('m_true.npy')
    y_true = np.polyval(m_true, tgrid)
    y_meas = y_true + np.random.laplace(loc=0.0, scale=noise_b, size= len(tgrid))

    noise = y_meas - y_true
    noise_sigma = np.std(noise)


    f_log_prior = get_f_log_prior(poly_dim, poly_mean, poly_sigma)
    f_log_likelihood = get_f_log_likelihood(poly_dim, y_meas, tgrid, noise_b)
    f_log_posterior = lambda m: f_log_prior(m) + f_log_likelihood(m)


    for sd in [1e-2, 2*1e-2, 5*1e-2,  1e-1]:
        N = int(1e3)
        f_proposal = get_f_proposal(sd*prop_sigma)
        mh_sampler = MHSampler(f_log_posterior, f_proposal)
        x0 = np.random.randn(poly_dim) * poly_sigma
        samples, log_probs, acceptance_ratios = mh_sampler.gen_samples(x0, N)
        plt.figure()
        plt.suptitle('sd = ' + str(sd))
        plt.plot(acceptance_ratios)



    sd = 5 * 1e-2 # based on the plots above...
    N= int(1e5)
    N_burn = 1e4
    f_proposal = get_f_proposal(sd*prop_sigma)
    mh_sampler = MHSampler(f_log_posterior, f_proposal)
    x0 = np.random.randn(poly_dim) * poly_sigma
    samples, log_probs, acceptance_ratios = mh_sampler.gen_samples(x0, N)

    samples = samples[:, int(N_burn):]
    log_probs = log_probs[int(N_burn):]
    acceptance_ratios = acceptance_ratios[int(N_burn):]

    vals = np.zeros((tgrid.size, samples.shape[1]))
    for i in range(samples.shape[1]):
        vals[:, i] = np.polyval(samples[:, i], tgrid)
    mean_val = np.mean(vals, axis=-1)
    std = np.std(vals, axis=-1)
    
    plt.figure()
    plt.plot(tgrid, y_true, 'r', label='\'True\'')
    plt.plot(tgrid, y_meas, 'o', label='Msmt.')
    plt.plot(tgrid, mean_val, 'k--', label='Mean')
    plt.fill_between(tgrid, mean_val - 2*std, mean_val + 2*std, alpha=0.2, label='2 std')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('y')
    plt.grid(True)
    #plt.savefig(pics_folder + 'laplac_poly_gaussian_regression_data.pdf')
    plt.savefig(pics_folder + 'laplac_poly_regression_estimation.pdf')

    m_est = np.mean(samples, axis=-1)
    cov = np.cov(samples)
    std = np.sqrt(np.diag(cov))
    xax = poly_dim - 1 - np.arange(poly_dim)

    H = get_H(tgrid, poly_dim)
    gaussian_m_est, gaussian_m_cov = get_post_mean_cov(H, y_meas, noise_sigma, poly_sigma)

    plt.figure()
    plt.errorbar(xax, m_est, color='g', marker='+', linewidth=0, elinewidth=1, yerr=2*std, label='estimated')
    plt.plot(xax, m_true, 'ko', label='true')
    plt.plot(xax, gaussian_m_est, 'r+', label='gaussian est.')
    plt.grid()
    plt.xlabel('Coeff. power')
    plt.legend()
    plt.savefig(pics_folder + 'laplac_poly_estimation.pdf')
    plt.figure()
    plt.suptitle('Accept. ratio')
    plt.plot(acceptance_ratios)
    plt.xlabel('Iteration')
    plt.savefig(pics_folder + 'laplac_poly_acceptance_ratio.pdf')

    plt.figure()
    plt.suptitle('Log probs')
    plt.plot(log_probs)
    plt.xlabel('Iteration')
    plt.savefig(pics_folder + 'laplac_poly_log_probs.pdf')


    y_est = np.polyval(m_est, tgrid)
    y_gauss_est = np.polyval(gaussian_m_est, tgrid)

    #plt.figure()
    #plt.plot(tgrid, y_true, 'k--', label='\'True\'')
    #plt.plot(tgrid, y_meas, 'b', label='Msmt.')
    #plt.plot(tgrid, y_est, 'g', label='Est.')
    #plt.plot(tgrid, y_gauss_est, 'r', label='Gaussian est.')
    #plt.legend()
    #plt.xlabel('Time')
    #plt.ylabel('y')
    #plt.grid()
    #plt.savefig(pics_folder + 'laplac_poly_regression_estimation.pdf')


    plt.show()


if __name__ == '__main__':
    #make_m_true()
    run_simulation()



