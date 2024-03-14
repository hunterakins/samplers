"""
Description:
    Helpers for testing a piecewise polynomial sampler

Date:
    2024/03/04

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
import samplers.cp_rj as cpr
from scipy import special


def log_zm_gaussian(x, sigma):
    """
    Log of a zero mean gaussian with identical std sigma
    """
    N = x.size
    norm_term = -N/2 * np.log(2 * np.pi * sigma**2)
    return -0.5 * np.sum(x**2) / sigma**2 + norm_term

class PiecewisePolynomial:
    def __init__(self, t0, tf, num_cp, dim):
        self.t0 = t0
        self.tf = tf
        self.num_cp = num_cp
        self.dim = dim

    def randomly_initialize(self, kmax):
        self.cp = np.sort(np.random.uniform(self.t0, self.tf, self.num_cp))
        positions_list = [x for x in self.cp]
        value_list = []
        for i in range(self.num_cp+1):
            value_list.append(np.random.randn(self.dim))
        x = cpr.fill_x(kmax, self.dim, positions_list, value_list)
        x[0] = self.num_cp
        self.x = x
        return x

    def plot_x(self, tgrid, kmax):
        plot_x(tgrid, self.x, kmax, self.dim)

def plot_x(tgrid, x, kmax, state_dim):
    y = eval_pp(tgrid, x, kmax, state_dim)
    plt.plot(tgrid, y, 'k')

def eval_pp(tgrid, x, kmax, state_dim):
    num_cp = cpr.get_num_cp(x)
    num_segs = num_cp + 1
    positions = cpr.get_cp_positions(x)
    y = np.zeros(tgrid.size)
    for i in range(num_segs):
        if i == 0:
            if num_segs == 1: # 
                inds = (tgrid <= tgrid.max())
            else:
                inds = tgrid <= positions[i]
        elif i == num_segs-1:
            inds = tgrid > positions[i-1]
        else:
            inds = (tgrid > positions[i-1]) & (tgrid <= positions[i])
        t = tgrid[inds]
        poly_coeffs = cpr.get_ith_segment_state(x, i, kmax, state_dim)
        vals = np.polyval(poly_coeffs, t)
        y[inds] = vals
    return y

def get_log_lh(tgrid, y_true, sigma_n, kmax, state_dim):
    """
    True state is x
    Samples evaluated at tgrid
    Noise std sigma_n
    """
    N = tgrid.size
    norm_term = -N/2 * np.log(2 * np.pi * sigma_n**2)
    def log_lh(x):
        y = eval_pp(tgrid, x, kmax, state_dim)
        return -0.5 * np.sum((y - y_true)**2) / sigma_n**2 + norm_term
    return log_lh

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

def get_log_prior(sigma_prior, lambd, kmax, state_dim, interval, dirichlet=False):
    """
    Independent zero-mean Gaussian prior on the states with Std sigma_prior

    Node number has Poisson distribution prior with parameter lam
    p(k) = e^{-\lambda} \frac{\lambda^{k}}{k !}

    L is the size of the interval

  
    For node positions:

    Green approach
    For node number use the even order statistics of (2k+1) uniform random variables where k is number of nodes
    and the uniform distribution is U(0, L) where L is the total interval
    of days
    Distribution for ith order statistic Y_i of N uniform random variables X_1, X_2, dots, X_N
    is 
    p_Yi(y) = Gamma(n) / (Gamma(i-1) * Gamma(n-i)) * (y/L)^{i-1} * (1-y/L)^{n-i} / L
    where Gamma is the gamma function

    
    Dirichlet distribution doesn't care where the layers
    are:
    p(positions) = k ! / (L)^{k}
    where k is number of node positions
    """
    L = interval[1] - interval[0]
    
    def log_prior(x):
        """
        x is a vector containing info for the 
        model
        """
        num_nodes = cpr.get_num_cp(x)
        log_p_num = -lambd + num_nodes * np.log(lambd) - np.log(special.gamma(num_nodes))

        positions = cpr.get_cp_positions(x)
        if dirichlet:
            p_pos = special.gamma(num_nodes) / np.power(L, num_nodes)
            log_p_pos = np.log(special.gamma(num_nodes)) - num_nodes * np.log(L)
        
        else:
            log_p_pos = 0.0
            N = 2*num_nodes + 1
            for ind in range(num_nodes): # even order statistics
                i = 2*(ind+1) # 
                sj = positions[ind]
                pj = unif_order_stat(sj-interval[0], N, i, L)
                log_p_pos += np.log(pj)

        values = cpr.get_segment_values(x, kmax, state_dim)
        log_p_values = [log_zm_gaussian(x, sigma_prior) for x in values]

        log_p_vals = sum(log_p_values)
        log_p = log_p_num + log_p_pos + log_p_vals
        return log_p
    return log_prior

def get_f_prior(sigma_prior, lambd, kmax, dim, interval):
    """
    Draw from the prior
    Number of change points prior is
    poisson with parameter lambd conditioned
    on being less than or equal to kmax
    state dimension in each segment is dim
    """
    L = interval[1] - interval[0]
    s0 = interval[0]
    def f_prior():
        k = int(np.random.poisson(lam=lambd))
        while k > kmax: #redraw
            k = int(np.random.poisson(lam=lambd))

        positions = []
        for i in range(k):
            positions.append(s0 + np.random.rand() * L)
        positions.sort()

        values = []
        for i in range(k+1):
            val = sigma_prior*np.random.randn(dim)
            values.append(val)
        x = cpr.fill_x(kmax, dim, positions, values)
        return x
    return f_prior

def get_birth_sample(dim, birth_sigma):
    def f_birth_sample():
        return np.random.randn(dim) * birth_sigma
    return f_birth_sample

def get_birth_log_prob(birth_sigma):
    def f_birth_log_prob(x):
        return log_zm_gaussian(x, birth_sigma)
    return f_birth_log_prob

def get_pert_sample(dim, pert_sigma):
    def f_pert_sample():
        return np.random.randn(dim, ) * pert_sigma
    return f_pert_sample

def get_pert_log_prob(pert_sigma):
    def f_pert_log_prob(x):
        return log_zm_gaussian(x, pert_sigma)
    return f_pert_log_prob



if __name__ == '__main__':

    t0, tf, num_cp, dim = 0, 1, 2, 2

    pp = PiecewisePolynomial(t0, tf, num_cp, dim)
    kmax = 7
    x = pp.randomly_initialize(kmax)
    tgrid = np.linspace(pp.t0, pp.tf, 20)
    pp.plot_x(tgrid, kmax)
    #plot_x(tgrid, x, kmax, pp.dim)
    y_true = eval_pp(tgrid, x, kmax, dim)
    sigma_n = 0.1
    plt.figure()
    plt.plot(tgrid, y_true, 'b')
    y_true += np.random.randn(y_true.size) * sigma_n
    plt.plot(tgrid, y_true, 'k.')
    f_log_lh = get_log_lh(tgrid, y_true, sigma_n, kmax, dim)

    lambd_prior = 3
    f_prior_sample = get_f_prior(1, lambd_prior, kmax, dim, [t0, tf])

    log_lh_vals = np.zeros(10)
    # look at contour of log likelihood as I vary the position of the first change point
    positions = cpr.get_cp_positions(x)
    """
    pos_cands = np.linspace(t0, positions[1], 10)
    for i in range(pos_cands.size):
        cpr.replace_cp_pos(x, 0, pos_cands[i])
        log_lh_vals[i] = f_log_lh(x)
        y_tmp = eval_pp(tgrid, x, kmax, dim)
        plt.plot(tgrid, y_tmp, 'r.', alpha=0.3)

    prior_log_lh_vals = np.zeros(10)
    for i in range(pos_cands.size):
        xpr = f_prior_sample()
        prior_log_lh_vals[i] = f_log_lh(xpr)
        y_tmp = eval_pp(tgrid, xpr, kmax, dim)
        plt.plot(tgrid, y_tmp, 'b.', alpha=0.3)

    plt.figure()
    plt.plot(pos_cands, log_lh_vals)
    plt.plot(pos_cands, prior_log_lh_vals)

    """

    # do an example death move
    istar = np.random.randint(0, cpr.get_num_cp(x)-1)
    print('istar', istar)
    positions= cpr.get_cp_positions(x)
    print('x', x)
    print('positions', positions)
    full_positions = [t0] + list(positions) + [tf]
    print(full_positions)
    xleft = full_positions[istar]
    xright = full_positions[istar+2]
    print('xleft, xright', xleft, xright)
    xdel = positions[istar]
    print('xdel', xdel)
    positions = np.delete(positions, istar)
    print('positions after', positions)
    values = cpr.get_segment_values(x, kmax, dim)
    alpha1 = (xdel-xleft) / (xright - xleft)
    alpha2 = 1 - alpha1
    xj = values[istar].copy()
    xj1= values[istar+1].copy()
    xprime = alpha1 * xj + alpha2 * xj1
    new_values = values[:istar] + [xprime] + values[istar+2:]
    xprime =cpr.fill_x(kmax, dim, positions, new_values)
    print('xprime[0], x[0]', xprime[0], x[0])

    plt.figure()
    plt.plot(tgrid, eval_pp(tgrid, x, kmax, dim), 'b.')
    plt.plot(tgrid, eval_pp(tgrid, xprime, kmax, dim), 'r.')
    plt.show()

