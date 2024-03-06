"""
Description:
    One example of a changepoint polynomial regression

Date:
    2024/03/04

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from samplers.examples.piecewise_poly import pp_helpers as ph
from samplers import cp_poly_rj as cpr


dim = 4
birth_sigma = .1
pert_sigma = 0.1
f_birth_sample = ph.get_birth_sample(dim, birth_sigma)
f_birth_log_prob = ph.get_birth_log_prob(birth_sigma)
f_pert_sample = ph.get_pert_sample(dim, pert_sigma)
f_pert_log_prob = ph.get_pert_log_prob(pert_sigma)

sigma_prior = 1.0
lambd = 2
kmax = 10
interval = [0,1]
L = interval[1] - interval[0]

f_prior_sample = ph.get_f_prior(sigma_prior, lambd, kmax, dim, interval)
f_log_prior = ph.get_log_prior(sigma_prior, lambd, kmax, dim, interval, dirichlet=False)

""" Draw a random state and generate some fake data """
noise_std = 0.1
x_true = f_prior_sample()
print('x_true', x_true)
print('true num cp', x_true[0])
N = 20
tgrid = np.linspace(0, 1, N)
y_true = ph.eval_pp(tgrid, x_true, kmax, dim)
y_msmt = y_true + noise_std * np.random.randn(N)

f_log_lh = ph.get_log_lh(tgrid, y_msmt, noise_std, kmax, dim)
k_list = [x for x in range(kmax+1)]
move_probs = [[1/3, 1/3, 1/3, 0]] +  [[1/3,1/3,1/6,1/6]]*(kmax-1) + [[1/3, 1/3, 0, 1/3]]
print(move_probs)
print(len(move_probs))
#beta_arr = np.array([1.0, 10.0, 30.0])
beta_arr = np.array([1.0,1/10, 1/30, 1/100.0])
#beta_arr = np.array([1.0,5.0,20.0])

sampler = cpr.ChangePointSampler(move_probs, k_list, dim, beta_arr,
                                f_log_prior, f_log_lh, interval,
                                f_pert_sample, f_pert_log_prob,
                                f_birth_sample, f_birth_log_prob)

for sampler_num in range(2):
    N = 10000
    N_burn_in = 3000
    sampler.initialize_chains(N, N_burn_in, f_prior_sample, swap_interval=1)
    sampler.sample()
    sampler.diagnostic_plot()

    samples, log_probs, acceptance_ratios, birth_acceptance_ratios, death_acceptance_ratios = sampler.get_chain_info(0)


    plt.figure()
    plt.plot(tgrid, y_msmt, 'k.')
    plt.plot(tgrid, y_true, 'g')

    plt.figure()
    ypts = np.zeros((N, len(tgrid)))
    for i in range(N):
        x = samples[:,i]
        y = ph.eval_pp(tgrid, x, kmax, dim)
        ypts[i,:] = y
        if (i % 10 == 0):
            plt.plot(tgrid, y, 'b', alpha=0.1)

    mean_y = np.mean(ypts[:,:], axis=0)
    plt.plot(tgrid, y_msmt, 'k.')
    plt.plot(tgrid, mean_y, 'r--')
    plt.plot(tgrid, y_true, 'g')
    plt.show()



