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

def get_f_log_likelihood(poly_dim, y_meas, tgrid, noise_sigma):
    """
    Description:
        Returns a function that evaluates the log likelihood function on a polynomial coefficient vector
    Args:
        poly_dim: int, dimension of polynomial (degree of polynomial plus 1)
        tgrid: numpy array, times at which polynomial is evaluated
        noise_sigma: float, standard deviation of Gaussian noise
    """
    def f_log_likelihood(m):
        y = np.polyval(m, tgrid)
        return -0.5 * np.sum((y - y_meas)**2) / noise_sigma**2
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


def get_evidence(H, y_meas, noise_sigma, prior_sigma):
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
    n = y_meas.size
    xdim = H.shape[1]
    x_mean, x_cov = get_post_mean_cov(H, y_meas, noise_sigma, prior_sigma)
    prior_cov = prior_sigma**2 * np.eye(xdim)
    noise_cov = noise_sigma**2 * np.eye(len(y_meas))
    noise_cov_inv = np.linalg.inv(noise_cov)

    term1 = y_meas[:, None].T @ noise_cov_inv @ y_meas[:, None]
    term2 = x_mean[:,None].T @ np.linalg.inv(x_cov) @ x_mean[:,None]  
    term3 = np.sqrt(np.linalg.det(x_cov))
    term4 = np.sqrt(np.linalg.det(prior_cov))
    term5 = np.sqrt(np.linalg.det(noise_cov))
    term6 = np.sqrt((2*np.pi)**n)
    evidence = np.exp(-.5*(term1 - term2)) * (term3/term4) /(term5*term6)
    return evidence

def run_simulation():
    poly_dim = 5
    N = 30
    tgrid = np.linspace(-1.0, 2.0, N)

    poly_mean = np.zeros(poly_dim)
    poly_sigma = 1.0
    snr_db = 20
    prop_sigma = 1.0

    #m_true = np.random.randn(poly_dim) * poly_sigma
    m_true = np.load('m_true.npy')
    y_true = np.polyval(m_true, tgrid)

    noise_var = (np.var(y_true)/(10**(snr_db/10)))
    noise_sigma = np.sqrt(noise_var)
    y = np.copy(y_true)
    y += noise_sigma*np.random.randn(tgrid.size)
    y_meas = y_true + np.random.randn(len(tgrid)) * noise_sigma


    f_log_prior = get_f_log_prior(poly_dim, poly_mean, poly_sigma)
    f_log_likelihood = get_f_log_likelihood(poly_dim, y_meas, tgrid, noise_sigma)
    f_log_posterior = lambda m: f_log_prior(m) + f_log_likelihood(m)


    for sd in [1e-2, 2*1e-2, 1e-1]:
        N = int(1e3)
        f_proposal = get_f_proposal(sd*prop_sigma)
        mh_sampler = MHSampler(f_log_posterior, f_proposal)
        x0 = np.random.randn(poly_dim) * poly_sigma
        samples, log_probs, acceptance_ratios = mh_sampler.gen_samples(x0, N)
        plt.figure()
        plt.suptitle('sd = ' + str(sd))
        plt.plot(acceptance_ratios)

    sd = 1 * 1e-1 # based on the plots above...
    N= int(1e5)
    N_burn = 1e4
    f_proposal = get_f_proposal(sd*prop_sigma)
    mh_sampler = MHSampler(f_log_posterior, f_proposal)
    x0 = np.random.randn(poly_dim) * poly_sigma
    samples, log_probs, acceptance_ratios = mh_sampler.gen_samples(x0, N)

    samples = samples[:, int(N_burn):]
    log_probs = log_probs[int(N_burn):]
    acceptance_ratios = acceptance_ratios[int(N_burn):]



    plt.figure()
    vals = np.zeros((tgrid.size, samples.shape[1]))
    for i in range(samples.shape[1]):
        vals[:, i] = np.polyval(samples[:, i], tgrid)
    mean_val = np.mean(vals, axis=-1)
    std = np.std(vals, axis=-1)

    plt.plot(tgrid, y_true, 'r', label='\'True\'')
    plt.plot(tgrid, y_meas, 'o', label='Msmt.')
    plt.plot(tgrid, mean_val, 'k--', label='Mean')
    plt.fill_between(tgrid, mean_val - 2*std, mean_val + 2*std, alpha=0.2, label='2 std')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig(pics_folder + 'poly_gaussian_regression_data.pdf')

    xmean, xcov = get_post_mean_cov(get_H(tgrid, poly_dim), y_meas, noise_sigma, poly_sigma)


    m_est = np.mean(samples, axis=-1)
    cov = np.cov(samples)
    std = np.sqrt(np.diag(cov))
    xax = poly_dim - 1 - np.arange(poly_dim)
    plt.figure()
    plt.errorbar(xax, m_est, color='b', marker='+', linewidth=0, elinewidth=1, yerr=2*std, label='estimated')
    plt.plot(xax, m_true, 'ko', label='true')
    plt.plot(xax, xmean, 'r*', label='posterior mean')
    plt.grid()
    plt.xlabel('Coeff. power')
    plt.ylabel('Coeff. value')
    plt.legend()
    plt.savefig(pics_folder + 'poly_estimation.pdf')
    plt.figure()
    plt.ylabel('Accept. ratio')
    plt.plot(acceptance_ratios, 'k')
    plt.grid()
    plt.xlabel('Iteration')
    plt.savefig(pics_folder + 'poly_acceptance_ratio.pdf')

    plt.figure()
    plt.ylabel('Log probs')
    plt.plot(log_probs, 'k')
    plt.grid()
    plt.xlabel('Iteration')
    plt.savefig(pics_folder + 'poly_log_probs.pdf')

    fig, axes = plt.subplots(1,2, sharey=True)
    cs0=axes[0].pcolormesh(xcov, rasterized=True)
    axes[0].set_title('Posterior covariance')
    fig.colorbar(cs0, ax=axes[0])
    cs1=axes[1].pcolormesh(np.cov(samples), rasterized=True)
    axes[1].set_title('Sample covariance')
    fig.colorbar(cs1, ax=axes[1])
    plt.savefig(pics_folder + 'poly_covariances.pdf')

    plt.show()

if __name__ == '__main__':
    run_simulation()



