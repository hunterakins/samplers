""" Description: 
    Reverse-jump MCMC for change-point problem from Green
    Proposal is hard coded
Date:
    12/4/2023

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
from samplers.proposals import gaussian_proposal
from samplers.online_scm import update_mean, update_scm, SampleCovarianceMatrix
from samplers.helpers import *

def get_dim(k):
    """
    k is number of changepoints
    a state is given by the position of the change points
    plus the values on the k+1 segments
    """
    return 2*k + 1

def fill_x(max_dim, positions_list, values_list):
    """
    Fill a vector with information for problem with N nodes
    and values in values_list for the N+1 segments  

    """
    N_nodes = len(positions_list)
    N_segs = N_nodes + 1
    x = np.zeros(max_dim+1)
    x[0] = N_nodes
    if N_nodes > 0:
        x[1:N_nodes+1] = positions_list
        x[N_nodes+1: N_nodes + 1 + N_segs] = values_list
    else:
        print(values_list)
        x[1] = values_list[0]
    return x

def get_positions(x):
    """
    Get positions of change points from x
    """
    N_nodes = int(x[0])
    if N_nodes == 0:
        return []
    positions = list(x[1:N_nodes+1])
    return positions

def get_values(x):
    """
    Get values of segments from x
    """
    N_nodes = int(x[0])
    N_segs = N_nodes + 1
    if N_nodes == 0:
        return [x[1]]
    values = list(x[N_nodes+1: N_nodes + 1 + N_segs])
    return values

def update_value(x, seg_ind, value):
    """
    Update value of segment seg_ind
    """
    N_nodes = int(x[0])
    N_segs = N_nodes + 1
    x[N_nodes+1 + seg_ind] = value
    return x

def update_position(x, seg_ind, position):
    """
    Update position of change point seg_ind
    """
    N_nodes = int(x[0])
    x[1 + seg_ind] = position
    return x

class ChainParams:
    def __init__(self, max_dim, beta, N): 
        """
        max dim is largest vector size
        beta is chain inverse temp
        N is number of samples
        """
        self.max_dim = max_dim
        self.beta = beta
        self.N = N 
        return

class Chain:
    def __init__(self, params):
        """
        Initialize a chain with params
        """
        self.params = params
        self.samples = np.zeros((params.max_dim+1, params.N+1))
        self.log_probs = np.zeros(params.N+1)
        self.acceptance_log = np.zeros(params.N)
        self.birth_acceptance_log = np.zeros(params.N)
        self.death_acceptance_log = np.zeros(params.N)
        self.birth_proposals = 0
        self.death_proposals = 0
        self.birth_accepted = 0
        self.death_accepted = 0
        self.accepted = 0 # accepted perturbation steps
        self.acceptance_ratios = np.zeros(params.N)
        self.birth_acceptance_ratios = np.zeros(params.N)
        self.death_acceptance_ratios = np.zeros(params.N)
        self.curr_log_prior = None # priot
        self.curr_log_lh = None # likelihood
        self.curr_log_p = None # posterior
        return

class ChangePointSampler:
    """
    Change point sampler to reproduce the result in Green
    """
    def __init__(self, move_probs, k_list, beta_arr, 
            f_log_prior, f_log_lh, interval):
                       #f_log_gprime, h_diffeo_birth, J_birth, h_diffeo_death, J_death, interval):
        """
        move_prob_list - list of list of move probabilities for each dimension
            move_prob_list[0] = [prob_pert, prob_birth, prob_death] for 
        beta_arr - list of inverse temperatures in temperature ladder
        f_log_prior - log prior func
        f_log_lh - log likelihood func
        interval - list of two floats
            domain of the change point function
        """
        self.moves = ['birth', 'death']
        self.k_list = k_list
        self.num_k = len(k_list)
        self.move_probs = move_probs
        self.beta_arr = beta_arr
        self.f_log_prior = f_log_prior
        self.f_log_lh = f_log_lh
        self.interval = interval
        self.max_dim = get_dim(max(k_list))
        #self.f_log_gprime = f_log_gprime
        #self.birth_sigma_sq = np.array([.25])
        #self.no_birth_death=False
        #self.h_diffeo_birth = h_diffeo_birth
        #self.J_birth = J_birth
        #self.h_diffeo_death = h_diffeo_death
        #self.J_death = J_death
        self.death_log_p_ratio = []
        self.birth_log_p_ratio = []

    def initialize_chains(self, N, N_burn_in, f_prior, swap_interval):
        """
        Initialize the chains 
        no swapping
        N - int
            number of steps for chain to take (so final number of samples is N+1)
        f_prior - function
            takes in the dimension of the state and returns a random sample
        """
        self.chain_list = []
        self.N_burn_in = N_burn_in
        self.N_steps = N
        self.swap_interval = swap_interval
        max_dim = get_dim(max(self.k_list))
        for i in range(len(self.beta_arr)): # for each temperature
            beta = self.beta_arr[i]
            chain_p = ChainParams(max_dim, beta, N)
            chain = Chain(chain_p)
            x = f_prior()
            chain.samples[:,0] = x
            chain.curr_log_prior = self.f_log_prior(x)
            chain.curr_log_lh = self.f_log_lh(x)
            chain.curr_log_p = chain.curr_log_prior + beta*chain.curr_log_lh
            self.chain_list.append(chain)
            chain.log_probs[0] = chain.curr_log_p

        self.swap_mat = np.zeros((len(self.beta_arr), N+1), dtype=int)
        self.swap_mat[:,0] = np.arange(len(self.beta_arr))
        return

    def _birth_move(self, i, j):
        """
        Birth move for chain i at iteration j
        Insert a new node point
        """
        chain = self.chain_list[i]
        chain.birth_proposals += 1
        x = chain.samples[:,j+1] # current state
        beta = chain.params.beta
        curr_log_p = chain.curr_log_p # log posterior of the current chain state (accounts for temp)
        k = int(x[0]) # number of nodes
        k_ind = self.k_list.index(k)
        p_birth = self.move_probs[k_ind][0]
        p_death = self.move_probs[k_ind+1][1]

        #xprime = x.copy()
        positions = get_positions(x)
        full_positions = [self.interval[0]] + positions + [self.interval[1]]
        values = get_values(x)

        # draw a new node position randomly
        new_pos = np.random.rand()*(self.interval[1] - self.interval[0]) + self.interval[0]

        # find which segment to insert it into 
        count = 1 # full_positions[count-1], full_positions[count] will bracket the new position
        while new_pos > full_positions[count]:
            count += 1
        sj = full_positions[count-1]
        sj1 = full_positions[count]
        sstar = new_pos
        new_positions = full_positions[:count] + [new_pos] + full_positions[count:]
        #print('positions', positions)
        #print('new positions', new_positions)
        pos_prime = new_positions[1:-1] # throw out interval points for proposed 
        curr_val = values[count-1]
        # need to suggest two values for the two new segments
        L = self.interval[1] - self.interval[0] # length of interval
        u = np.random.rand()
        log_gu = -np.log(L) # log probability of drawing sstar
        log_guprime = -np.log(k+1) # log probability of drawing that node to delete in reverse move

        ratio = (1-u)/u
        Delta = sj1 - sj
        Delta1 = sstar - sj
        Delta2 = sj1 - sstar
        hj_prime = curr_val * np.power(ratio, -Delta2 / Delta) 
        hj1_prime = curr_val * np.power(ratio, Delta1 / Delta)
        new_values = values[:count-1] + [hj_prime, hj1_prime] + values[count:]
        xprime = fill_x(self.max_dim, pos_prime, new_values)

        # Jacobian
        J_birth = np.square(hj_prime + hj1_prime)/ curr_val # equation on page 721 Green 1995

        log_prior = self.f_log_prior(xprime)
        log_lh = self.f_log_lh(xprime)
        log_p = log_prior + beta*log_lh # use temperature of chain...
        alpha = (log_p - curr_log_p) +  (log_guprime - log_gu) +  (np.log(p_death) - np.log(p_birth)) + np.log(J_birth)
        if i == 1:
            self.birth_log_p_ratio.append(alpha)
        alpha = np.exp(alpha)
        if np.random.rand() < alpha: # accept step
            chain.samples[:,j+1] = xprime.copy()
            chain.curr_log_prior = log_prior
            chain.curr_log_lh = log_lh
            chain.curr_log_p = log_p
            chain.birth_acceptance_log[j] = 1
            chain.birth_accepted += 1
        else:
            chain.birth_acceptance_log[j] = 0
        return chain

    def _death_move(self, i, j):
        """
        Death move 
        Remove a node point and merge the two values
        neighboring values
        """
        chain = self.chain_list[i]
        chain.death_proposals += 1
        x = chain.samples[:,j+1] # current state
        beta = chain.params.beta
        curr_log_p = chain.curr_log_p # log posterior of the current chain state (accounts for temp)
        k = int(x[0]) # number of nodes
        k_ind = self.k_list.index(k)
        p_birth = self.move_probs[k_ind-1][0]
        p_death = self.move_probs[k_ind][1]


        #xprime = x.copy()
        positions = get_positions(x)
        istar = np.random.randint(0, k) # index of the node to remove
        #print('positions', positions)
        #print('istar', istar)
        sstar = positions[istar]
        full_positions = [self.interval[0]] + positions + [self.interval[1]]
        Delta = full_positions[istar+2] - full_positions[istar] 
        Delta1 = sstar - full_positions[istar]
        Delta2 = full_positions[istar+2] - sstar

        values = get_values(x)
        hj = values[istar]
        hj1 = values[istar+1]
        hjprime = np.power(hj, Delta1/Delta) * np.power(hj1, Delta2/Delta)

        new_positions = full_positions[:istar] + full_positions[istar+1:]
        new_positions = new_positions[1:-1]

        L = self.interval[1] - self.interval[0] # length of interval
        log_guprime = -np.log(L) # log of the uniform density used to draw sstar in the reverse move
        log_gu = -np.log(k) # probability of selecting that node to delete

        new_values = values[:istar] + [hjprime] + values[istar+2:]

        xprime = fill_x(self.max_dim, new_positions, new_values)
        #print('proposing death', x[0], xprime[0])


        # Jacobian
        J_death = hjprime / (np.square(hj + hj1)) # equation on page 721 Green 1995

        log_prior = self.f_log_prior(xprime)
        log_lh = self.f_log_lh(xprime)
        log_p = log_prior + beta*log_lh # use temperature of chain...
        alpha = (log_p - curr_log_p) +  (log_guprime - log_gu) +  (np.log(p_birth) - np.log(p_death)) + np.log(J_death)
        if i == 1:
            self.death_log_p_ratio.append(alpha)
        alpha = np.exp(alpha)
        if np.random.rand() < alpha: # accept step
            chain.samples[:,j+1] = xprime.copy()
            chain.curr_log_prior = log_prior
            chain.curr_log_lh = log_lh
            chain.curr_log_p = log_p
            chain.death_acceptance_log[j] = 1
            chain.death_accepted += 1
        else:
            chain.death_acceptance_log[j] = 0
        return chain

    def _position_perturb_move(self, i, j):
        """
        Perturb the value in one of the segments
        """
        chain = self.chain_list[i]
        x = chain.samples[:,j+1] # 
        beta = chain.params.beta
        curr_log_p = chain.curr_log_p # log posterior of the current chain state
        num_nodes = int(x[0])

        if num_nodes == 0:
            return

        xprime = x.copy()
        """
        Randomly select a node
        """
        if num_nodes == 1:
            node_ind = 0
        else:
            node_ind = np.random.randint(0,num_nodes)
        positions = get_positions(xprime)
        sj = get_positions(xprime)[node_ind]
        """
        Get the bracketing positions
        sjm1, sjp1
        """
        if num_nodes == 1:
            sjm1 = self.interval[0]
            sjp1 = self.interval[1]
        else:
            if node_ind == 0 :
                sjm1 = self.interval[0]
                sjp1 = positions[node_ind + 1]
            elif node_ind == num_nodes-1:
                sjm1 = positions[node_ind-1]
                sjp1 = self.interval[1]
            else:
                sjm1 = positions[node_ind - 1]
                sjp1 = positions[node_ind + 1]

        """
        Propose new position uniformly in the interval
        This is a symmetric proposal distribution
        """
        u = np.random.rand() * (sjp1 - sjm1)
        sjprime = sjm1 + u
        xprime = update_position(xprime, node_ind, sjprime)

        log_prior = self.f_log_prior(xprime)
        log_lh = self.f_log_lh(xprime)

        log_p = log_prior + beta*log_lh
        alpha = (log_p) - (curr_log_p + 0) # assumes symmetric proposal 
        alpha = np.exp(alpha)
        if np.random.rand() < alpha: #accept move
            chain.samples[:,j+1] = xprime
            chain.curr_log_prior = log_prior
            chain.curr_log_lh = log_lh
            chain.curr_log_p = log_p
            chain.accepted += 1
            chain.log_probs[j+1] = log_p
        else: # reject
            chain.samples[:,j+1] = x
            chain.log_probs[j+1] = curr_log_p
        chain.acceptance_ratios[j] = (chain.accepted)/(j+1)
        return chain

    def _value_perturb_move(self, i, j):
        """
        Perturb the value in one of the segments
        """
        chain = self.chain_list[i]
        x = chain.samples[:,j+1] # since I have done either a birth or death move
        beta = chain.params.beta
        curr_log_p = chain.curr_log_p # log posterior of the current chain state
        num_nodes = int(x[0])

        xprime = x.copy()
        # choose segment to perturb
        seg_ind = np.random.randint(0,num_nodes+1)
        h = get_values(xprime)[seg_ind]
        u = np.random.rand() - 1/2 # uniform between -1/2 and 1/2
        hprime = np.exp(u) * h
        xprime = update_value(xprime, seg_ind, hprime)

        log_prop_ratio = hprime - h # NEED TO DOUBLE CHECK THIS

        log_prior = self.f_log_prior(xprime)
        log_lh = self.f_log_lh(xprime)

        log_p = log_prior + beta*log_lh
        alpha = (log_p) - (curr_log_p + 0) # assumes symmetric proposal 
        alpha = np.exp(alpha)
        if np.random.rand() < alpha: #accept move
            chain.samples[:,j+1] = xprime
            chain.curr_log_prior = log_prior
            chain.curr_log_lh = log_lh
            chain.curr_log_p = log_p
            chain.accepted += 1
            chain.log_probs[j+1] = log_p
        else: # reject
            chain.samples[:,j+1] = x
            chain.log_probs[j+1] = curr_log_p
        chain.acceptance_ratios[j] = (chain.accepted)/(j+1)
        return chain

    def _select_move(self, dim): 
        dim_ind = self.k_list.index(dim)
        #print('k list', self.k_list)
        #print('dim ind', dim_ind)
        #print('moves', self.moves)
        move = np.random.choice(self.moves, p=self.move_probs[dim_ind])
        return move

    def _update_chain(self, i, j):
        """
        Propose birth or death
        This populates sample j+1
        Then do a perturb move on sample j+1
        """
        chain = self.chain_list[i]
        """
        Copy chain values into next sample
        """
        x = chain.samples[:,j] # current state
        chain.samples[:,j+1] = x.copy()

        dim = int(chain.samples[0,j])
        move = self._select_move(dim)
        if move == 'birth':
            self._birth_move(i, j)
        elif move == 'death':
            self._death_move(i, j)
        if chain.birth_proposals > 0:
            chain.birth_acceptance_ratios[j] = chain.birth_accepted/chain.birth_proposals
        if chain.death_proposals > 0:
            chain.death_acceptance_ratios[j] = chain.death_accepted/chain.death_proposals
        self._position_perturb_move(i, j)
        self._value_perturb_move(i, j)
        return

    def _swap_chains(self, j):
        """
        Propose swaps between chains
        """

        # swaps (DEO)
        self.swap_mat[:,j+1] = self.swap_mat[:,j].copy()
        swap_interval = self.swap_interval
        num_chains = self.beta_arr.size
        if j > self.N_burn_in and (j%swap_interval==0):
            # ........
            X = j / swap_interval
            even =  X% 2
            for chain_ind in range(num_chains-1):
                # get two chains up for swap proposal
                hot_chain_ind = num_chains - 1 - chain_ind
                hot_chain = self.chain_list[self.swap_mat[hot_chain_ind, j+1]]
                cold_chain_ind = hot_chain_ind - 1
                cold_chain = self.chain_list[self.swap_mat[cold_chain_ind, j+1]]
                if ((even) and (hot_chain_ind % 2 == 0)) or ((not even) and (hot_chain_ind % 2 == 1)):

                    # get their likelihood and temps for calculating proposal ratio
                    hot_lh = hot_chain.curr_log_lh
                    cold_lh = cold_chain.curr_log_lh
                    beta_hot = hot_chain.params.beta
                    beta_cold = cold_chain.params.beta
                    if beta_hot > beta_cold:
                        print('beta hot, beta_cold', beta_hot, beta_cold)

                    # compute proposal ratio
                    log_omega = (beta_hot-beta_cold)*(cold_lh - hot_lh)
                    omega = np.exp(log_omega)
                    u = np.random.rand()
                    if u < omega: # swap accepted
                        ### The chains maintain their history but get a new temperature
                        ### Their curr prob values need to be updated to reflect new temperatue
                        hot_chain.curr_log_p = beta_cold * hot_chain.curr_log_lh + hot_chain.curr_log_prior
                        cold_chain.curr_log_p = beta_hot * cold_chain.curr_log_lh + cold_chain.curr_log_prior


                        #swap temperatures
                        cold_chain.params.beta = beta_hot
                        hot_chain.params.beta = beta_cold

                        # note it in the swap mat
                        t1 =  self.swap_mat[hot_chain_ind, j].copy()
                        t2 = self.swap_mat[cold_chain_ind, j].copy()
                        self.swap_mat[hot_chain_ind, j+1] = t2
                        self.swap_mat[cold_chain_ind, j+1] = t1
        return

    def sample(self):
        """
        Make N steps on each chain
        """
        N = self.N_steps
        num_temps = len(self.beta_arr)
        for j in range(N):
            for i in range(num_temps):
                self._update_chain(i,j) # propose plus accept/reject
            self._swap_chains(j) # proposing chain temperature swaps
        return

    def calculate_iact(self, chain_i, f, N_burn_in):
        samples = self.chain_list[chain_i].samples
        M_list, tau_list = calculate_iact(samples, f, N_burn_in)
        return M_list, tau_list

    def diagnostic_plot(self, density=False):
        """
        After running gen_samples, plot the acceptance ratio and the samples
        """
        chain_fig_list = []
        for i in range(len(self.chain_list)): # for each temperature
            chain = self.chain_list[i]
            samples = chain.samples
            log_probs = chain.log_probs
            acceptance_ratios = chain.acceptance_ratios
            print('i chain accpeted', chain.accepted)
            birth_acceptance_ratios = chain.birth_acceptance_ratios
            death_acceptance_ratios = chain.death_acceptance_ratios
            #dim = samples.shape[0]
            fig1, axes = plt.subplots(2,1)
            fig1.suptitle('Log posterior and acceptance ratio for chain with $\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))
            axes[1].plot(acceptance_ratios, label='pert acceptance')
            axes[1].plot(death_acceptance_ratios, label='death acceptance')
            axes[1].plot(birth_acceptance_ratios, label='birth acceptance')
            axes[1].legend()
            axes[0].plot(log_probs)
            #axes[0,0].hist(self.samples, bins=50)
        
            # now get samples from same dims together...
            """
            fig_ax_list = [plt.subplots(1, x) for x in self.k_list]
            for dim_ind, dim in enumerate(self.k_list):
                dim_mask = samples[0,:] == dim
                fig, ax = fig_ax_list[dim_ind]
                dim_samples = samples[:, dim_mask][1:dim+1]
                for k in range(dim):
                    ax[k].hist(dim_samples[k,:], bins=150, density=density)

                fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))

            """
            fig, ax = plt.subplots(1,1)
            ax.hist(samples[0,:], bins=len(self.k_list), range=(min(self.k_list)-.5, max(self.k_list) + .5), density=density)
            fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))

            fig, ax = plt.subplots(1,1)
            ax.plot(samples[0,:])
            fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))


        swap_fig = plt.figure()
        #plt.imshow(self.swap_mat, aspect='auto', cmap='gray_r', interpolation='none')
        for i in range(len(self.chain_list)):
            plt.plot(self.swap_mat[i,:])
        return chain_fig_list, swap_fig

    def get_chain_info(self, temp_i):
        inds = self.swap_mat[temp_i,:] 
        N = self.N_steps
        samples = np.zeros((max(self.k_list) + 1, N+1))
        log_probs = np.zeros((N+1))
        acceptance_ratios = np.zeros(N)
        birth_acceptance_ratios = np.zeros(N)
        death_acceptance_ratios = np.zeros(N)

        for k in range(N+1):
            samples[:,k] = self.chain_list[inds[k]].samples[:,k]
            log_probs[k] = self.chain_list[inds[k]].log_probs[k]
            if k < N:
                acceptance_ratios[k] = self.chain_list[inds[k]].acceptance_ratios[k]
                birth_acceptance_ratios[k] = self.chain_list[inds[k]].birth_acceptance_ratios[k]
                death_acceptance_ratios[k] = self.chain_list[inds[k]].death_acceptance_ratios[k]
        return samples, log_probs, acceptance_ratios, birth_acceptance_ratios, death_acceptance_ratios

    def single_chain_diagnostic_plot(self, temp_i, density=False):
        """
        temp_i - int
            index of temperature in temperature ladder
        """
        plt.figure()
        plt.suptitle('Chain index corresponding to cold chain')
        beta = self.beta_arr[temp_i]
        t = 1/beta
        plt.plot(self.swap_mat[temp_i,:])
        samples, log_probs, acceptance_ratios, birth_acceptance_ratios, death_acceptance_ratios = self.get_chain_info(temp_i)
        log_p_ar_fig, axes = plt.subplots(2,1)
        log_p_ar_fig.suptitle('Log posterior and acceptance ratio for chain with $\\beta = {0}, T = {1} $'.format(round(beta, 2), round(1/beta, 2)))
        axes[1].plot(acceptance_ratios, label='pert acceptance')
        axes[1].plot(death_acceptance_ratios, label='death acceptance')
        axes[1].plot(birth_acceptance_ratios, label='birth acceptance')
        axes[1].legend()
        axes[0].plot(log_probs)
        #axes[0,0].hist(self.samples, bins=50)
    
        # now get samples from same dims together...
        fig_ax_list = [plt.subplots(1, x) for x in self.k_list]
        for dim_ind, dim in enumerate(self.k_list):
            dim_mask = (samples[0,:] == dim)
            fig, ax = fig_ax_list[dim_ind]
            dim_samples = samples[:, dim_mask][1:dim+1]
            for k in range(dim):
                ax[k].hist(dim_samples[k,:], bins=150, density=density)

            fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(beta, 2), round(1/beta, 2)))

        dim_fig, ax = plt.subplots(1,1)
        ax.hist(samples[0,:], bins=len(self.k_list), color='k', range=(min(self.k_list)-.5, max(self.k_list) + .5), density=density)
        #dim_fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(chain.params.beta, 2), round(1/chain.params.beta, 2)))

        fig, ax = plt.subplots(1,1)
        ax.plot(samples[0,:])
        fig.suptitle('$\\beta = {0}, T = {1} $'.format(round(beta, 2), round(1/beta, 2)))


        #swap_fig = plt.figure()
        ##plt.imshow(self.swap_mat, aspect='auto', cmap='gray_r', interpolation='none')
        #for i in range(len(self.chain_list)):
        #    plt.plot(self.swap_mat[i,:])
        return  log_p_ar_fig, fig_ax_list, dim_fig
