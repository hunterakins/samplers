"""
Description:
    Test parallel-tempering affine invariant sampler on a multimodal Gaussian

Date:
    4/28/2023

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
from samplers.pt_emcee import ParallelTemperedAffineInvariantSampler
from samplers.likelihoods import GaussianLikelihood, MultimodalGaussianLikelihood, RosenbrockTwoD, get_random_mu_sigma_list


def f_log_prior(x):
    """
    Two-d x
    Uniform prior
    on x inside square centered at origin
    with side length 200
    """
    if max(abs(x)) > 100:
        return -np.inf
    else:
        return 0
    
def multimodal_test():
    """
    Use parallel tempered affine invariant sampler to sample from 
    multimodal distribution (mixture of Gaussians)
    Parameters to play with are:
    dim - dimension of space
    mu_list - list of means for each Gaussian
    sigma_list - list of standard deviations for each Gaussian
    colored - whether or not to use colored noise model (covariance is randomized)
        if colored=True, the covariance matrix has a power law distribution of eigenvalues
        with maximum eigenvalue equal to the variance in sigma_list


    """
    movie = False
    seed = None
    dim = 2
    mu_list = [np.ones(dim)*(10), -np.ones(dim)*10, np.array([1,-1])*10, np.array([-1, 1])*10]
    sigma_list = [2*np.ones(dim),2*np.ones(dim), 2*np.ones(dim),2*np.ones(dim)]
    likelihood = MultimodalGaussianLikelihood(dim, mu_list, sigma_list, seed=seed, colored=True)
    log_p = likelihood.f_log_p
    plt.figure()
    plt.suptitle('PDF to sample')
    x_test = np.linspace(-50, 50, 100)
    y_test = np.linspace(-50, 50, 100)
    X, Y = np.meshgrid(x_test, y_test)
    Z = np.zeros_like(X)
    for i in range(len(x_test)):
        for j in range(len(y_test)):
            Z[i,j] = np.exp(log_p([x_test[i], y_test[j]]))
    plt.contourf(Y, X, Z.T, 100)#, vmin=Z.max() - 40)
    plt.colorbar()
    plt.savefig('pics/multimodal_pdf.png')

    # create an Sampler object
    sampler = ParallelTemperedAffineInvariantSampler(log_p, f_log_prior)

    # set run parameters
    N = 2000
    N_burn_in = 100
    a = 2 # stretch parameter
    L = 100 # number of walkers
    S = 6 # number of walkers used for covariance estimation

    # set the chain parameters
    M = 2
    Tmax = 100
    temp_ladder = np.exp(np.linspace(0, np.log(Tmax), M))
    print(temp_ladder) # must be cold to hot
    prop_weights = [10,0]
    sampler.initialize_sampler(dim, a, S, L, M, temp_ladder, prop_weights)

    # draw from the prior
    var_prior = 20**2
    sigma0 = var_prior*np.eye(2)
    x0 = np.zeros((dim, M, L))
    for temp_i in range(M):
        for walker_i in range(L):
            x0[:,temp_i, walker_i] = np.random.multivariate_normal(np.zeros(dim), sigma0)

    sampler.sample(x0, N)

    samples = sampler.samples.copy()
    # add samples to the ...

    if movie == True:
        fig, axes = plt.subplots(2,1,sharex=True, sharey=True)
        cs = axes[0].contourf(X, Y, Z, 100)#, vmin=Z.max() - 40)
        axes[0].set_xlim([-40, 40])
        axes[0].set_ylim([-40, 40])
        plt.colorbar(cs, ax=axes[0])

        cs = axes[1].contourf(X, Y, np.power(Z,1/100), 100)#, vmin=Z.max() - 40)
        axes[1].set_xlim([-40, 40])
        axes[1].set_ylim([-40, 40])
        plt.colorbar(cs, ax=axes[1])
        fig.tight_layout()
        for i in range(N):
            line1 = (axes[0].plot(samples[0,0,:,i], samples[1,0,:,i], 'k.'))[0]
            line2 = (axes[1].plot(samples[0,1,:,i], samples[1,1,:,i], 'k.'))[0]
            plt.savefig('pics/{0}.png'.format(str(i).zfill(3)))
            line1.remove()
            line2.remove()
        import os
        os.system('ffmpeg -r 3 -f image2 -s 1920x1080 -i pics/%03d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p pics/pt_emcee_multimodal_{0}_{1}.mp4'.format(prop_weights[0], prop_weights[1]))
        os.system('rm pics/*png')

    #samples = samples.transpose((0, 2, 1))

    # now plot the results
    cold_samples = samples[:,0,...]
    hot_samples = samples[:,-1,...]

    # reshape so that walkers are flattened out
    hot_samples = hot_samples.reshape(hot_samples.shape[0], hot_samples.shape[1]*hot_samples.shape[2])
    cold_samples = cold_samples.reshape(cold_samples.shape[0], cold_samples.shape[1]*cold_samples.shape[2])

    # plot the various histograms
    sampler.diagnostic_plot(N_burn_in)
    hot_samples = hot_samples[:,N_burn_in:]
    cold_samples = cold_samples[:,N_burn_in:]
    plt.figure()
    plt.plot(cold_samples[0,:])
    plt.suptitle('Cold chain samples')

    plt.figure()
    plt.suptitle('Hot chain histogram')
    plt.hist2d(hot_samples[0,:], hot_samples[1,:], bins=[100,100])
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.colorbar()

    plt.figure()
    plt.suptitle('Cold chain histogram')
    plt.hist2d(cold_samples[0,:], cold_samples[1,:], bins=[100,100])
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.colorbar()

    def f(x):
        print(x.shape)
        x = x[:,0,...] # pick the cold chain
        print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        print(x.shape)
        #avg = np.mean(x, axis=1)
        #print(avg.shape)
        return x
    M_list, tau_list = sampler.calculate_iact(f, N_burn_in)
    plt.figure()
    plt.suptitle('IACT est.')
    plt.ylabel('IACT (tau)')
    plt.xlabel('M (maximum  lag in ACF for IACT estimation)')
    plt.plot(M_list, tau_list)
    #plt.plot(mean_acorr[i,:])


    # also plot some of the swap maps...
    plt.figure()
    plt.suptitle('Some example swap tracks of two different walkers\nNot sure I implemented this correctly')
    plt.plot(sampler.chain_index_array[0,0,::2])
    plt.plot(sampler.chain_index_array[-1,0,::2])
    plt.ylabel('Chain temp.')
    plt.xlabel('Iteration')
    plt.show()

multimodal_test()
