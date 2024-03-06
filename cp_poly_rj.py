""" Description: 
    Reverse-jump for a piecewise polynomial problem where 
    polynomial has a fixed dimension that is shared among all segments
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


"""
State of the chain is an array
First value is number of changepoints
Next kmax values are allocated for the changepoint locations
(kmax is maximum number of allowed change points)
There are kmax+1 allowed possible segments, each of which contains the state
model vector, which consists of a fixed number of coefficients
"""

def get_max_dim(kmax, state_dim):
    """
    Vector needs to store 
    a) the number of changepoints 
    b) the position of the changepoints (potentially kmax of them)
    c) the state values within each segment potentially (kmax+1) segments
    """
    max_dim = (1 + kmax) + (kmax + 1)*state_dim
    return max_dim

def fill_x(kmax, state_dim, positions, values_list):
    """
    Create a numpy array that represents the state contained using 
    the change point positions in the list positions
    values_list is list of arrays, each array is a model state vector
    """
    x0 = np.zeros(get_max_dim(kmax, state_dim))
    num_cp = len(positions)
    update_cp_positions(x0, positions)
    for i in range(len(values_list)):
        replace_state_val(x0, i, values_list[i], kmax, state_dim)
    return x0

def get_num_cp(x):
    """
    Get number of changepoints from the state
    """
    return int(x[0])

def get_cp_positions(x):
    num_cp = get_num_cp(x)
    positions = x[1:num_cp+1]
    return positions

def get_segment_values(x, kmax, state_dim):
    num_segs = int(x[0]) + 1
    values_list = []
    for i in range(num_segs):
        xstate = get_ith_segment_state(x,i, kmax, state_dim)
        values_list.append(xstate)
    return values_list
        
def update_cp_positions(x, positions):
    num_cp = len(positions)
    x[0] = num_cp
    x[1:num_cp+1] = positions
    return x

def get_ith_state_inds(x, i, kmax, state_dim):
    num_cp = get_num_cp(x)
    if i > num_cp:
        raise ValueError('Attempting to extract non-existent state (i > num segments)')
    N_state = state_dim 
    i0 = (kmax + 1) + (N_state)*i
    i1 = i0 + N_state
    return i0, i1

def get_ith_segment_state(x, i, kmax, state_dim):
    """
    x - state
    i - index (starts at 0)
    kmax - max number of allowed changepoints
    state_dim  - maximum number of allowed state vector dimension
    """
    i0, i1 = get_ith_state_inds(x, i, kmax, state_dim)
    xstate = x[i0:i1]
    return xstate
    print(get_max_dim(kmax, state_dim))

def replace_cp_pos(x, i, pos):
    """
    i is index of change point (0, 1, 2)
    Update the position of the ith changepoint to be pos
    """
    num_cp = get_num_cp(x)
    if i >= num_cp:
        raise ValueError
    x[i+1] = pos
    return x

def replace_state_val(x, i, values, kmax,  state_dim):
    """
    Replace state of the eith segment with values
    """
    i0, i1 = get_ith_state_inds(x, i, kmax, state_dim)
    x[i0:i1] = values
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
        self.samples = np.zeros((params.max_dim, params.N+1))
        self.log_probs = np.zeros(params.N+1)
        self.acceptance_log = np.zeros(params.N)
        self.birth_acceptance_log = np.zeros(params.N)
        self.death_acceptance_log = np.zeros(params.N)
        self.pos_pert_acceptance_log = np.zeros(params.N)
        self.val_pert_acceptance_log = np.zeros(params.N)
        self.birth_proposals = 0
        self.death_proposals = 0
        self.pos_pert_proposals = 0
        self.val_pert_proposals = 0
        self.birth_accepted = 0
        self.death_accepted = 0
        self.pos_pert_accepted = 0
        self.val_pert_accepted = 0
        self.acceptance_ratios = np.zeros(params.N)
        self.birth_acceptance_ratios = np.zeros(params.N)
        self.death_acceptance_ratios = np.zeros(params.N)
        self.val_pert_acceptance_ratios = np.zeros(params.N)
        self.pos_pert_acceptance_ratios = np.zeros(params.N)
        self.curr_log_prior = None # priot
        self.curr_log_lh = None # likelihood
        self.curr_log_p = None # posterior
        return

class ChangePointSampler:
    """
    Change point sampler to reproduce the result in Green
    """
    def __init__(self, move_probs, k_list, state_dim, beta_arr, 
            f_log_prior, f_log_lh, interval, 
            f_val_pert_sample, f_val_pert_log_prob, 
            f_birth_pert_sample, f_birth_pert_log_prob):
        """
        move_prob_list - list of list of move probabilities for each dimension
            move_prob_list[0] = [prob_pos_pert, prob_val_pert, prob_birth, prob_death] for 
        k_list - list of allowed number of changepoints
        beta_arr - list of inverse temperatures in temperature ladder
        f_log_prior - log prior func
        f_log_lh - log likelihood func
        interval - list of two floats
            domain of the change point function
        f_val_pert_sample - function
            sample from perturbation distribution
        f_val_pert_log_prob - function
            log probability of the perturbation
        f_birth_pert_sample - function
            sample from perturbation distribution
        f_birth_pert_log_prob - function
            log probability of the perturbation
        """
        self.moves = ['pos_pert', 'val_pert', 'birth', 'death']
        self.k_list = k_list
        self.num_k = len(k_list)
        self.kmax = max(self.k_list)
        self.state_dim = state_dim
        self.move_probs = move_probs
        self.beta_arr = beta_arr
        self.f_log_prior = f_log_prior
        self.f_log_lh = f_log_lh
        self.interval = interval
        self.max_dim = get_max_dim(self.kmax, self.state_dim)
        self.death_log_p_ratio = []
        self.birth_log_p_ratio = []
        self.f_val_pert_sample = f_val_pert_sample
        self.f_val_pert_log_prob = f_val_pert_log_prob
        self.f_birth_pert_sample = f_birth_pert_sample
        self.f_birth_pert_log_prob = f_birth_pert_log_prob

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
        max_dim = self.max_dim
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
        Add a new node and initialize the values of the new segments
        to match the existing segment
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
        positions = get_cp_positions(x)
        full_positions = [self.interval[0]] + list(positions) + [self.interval[1]]
    
        # get values in a list for the current segments
        values = get_segment_values(x, self.kmax, self.state_dim)

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
        pos_prime = new_positions[1:-1] # throw out interval points for proposed vector

        curr_val = get_ith_segment_state(x, count-1, self.kmax, self.state_dim)
        L = self.interval[1] - self.interval[0] # length of interval
        log_gu = -np.log(L) # log probability of drawing sstar
        log_guprime = -np.log(k+1) # log probability of drawing that node to delete in reverse move

        # need to suggest two values for the two new segments
        dx = sj1 - sj
        dx1 = sstar - sj
        dx2 = sj1 - sstar
        alpha1 = dx1 / dx
        alpha2 = dx2 / dx
        u = self.f_birth_pert_sample()
        log_gu_pert = self.f_birth_pert_log_prob(u)
        xj_prime = curr_val  - alpha2*u
        xj1_prime = curr_val + alpha1*u
        new_values = values[:count-1] + [xj_prime, xj1_prime] + values[count:]
        xprime = fill_x(self.kmax, self.state_dim, pos_prime, new_values)
        #xprime = fill_x(self.max_dim, pos_prime, new_values)

        # Jacobian h(x, u) = (x-u, x+u) = [[I, -I], [I, I]]([[x], [i]]) = [[x1'],[x2']]
        J_birth = np.power(1, self.state_dim) #  because of the way I picked alpha1, alpha2
        
        log_prior = self.f_log_prior(xprime)
        log_lh = self.f_log_lh(xprime)
        log_p = log_prior + beta*log_lh # use temperature of chain...
        alpha = (log_p - curr_log_p) +  (log_guprime - log_gu - log_gu_pert) +  (np.log(p_death) - np.log(p_birth)) + np.log(J_birth)
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
        positions = get_cp_positions(x)
        istar = np.random.randint(0, k) # index of the node to remove
        #print('positions', positions)
        #print('istar', istar)
        sstar = positions[istar] # node to delete

        full_positions = [self.interval[0]] + list(positions) + [self.interval[1]]
        xleft = full_positions[istar]
        xright = full_positions[istar+2]
        dx1 = sstar - xleft
        dx2 = xright - sstar
        dx = xright - xleft

        new_positions = full_positions[:istar] + full_positions[istar+1:]
        new_positions = new_positions[1:-1]

        # get values in a list for the current segments
        values = get_segment_values(x, self.kmax, self.state_dim)
        xj = values[istar].copy()
        xj1 = values[istar+1].copy()
        alpha1 = dx1 / dx
        alpha2 = dx2 / dx
        xjprime = alpha1*xj + alpha2*xj1
        uprime = xj1 - xjprime
        log_gu_prime_pert = self.f_birth_pert_log_prob(uprime)


        L = self.interval[1] - self.interval[0] # length of interval
        log_guprime = -np.log(L) + log_gu_prime_pert # log of the uniform density used to draw sstar in the reverse move PLUS prob to draw the u that gets me to xjprime
        log_gu = -np.log(k) # probability of selecting the node to delete

        new_values = values[:istar] + [xjprime] + values[istar+2:]

        xprime = fill_x(self.kmax, self.state_dim, new_positions, new_values)
        #print('proposing death', x[0], xprime[0])

        # Jacobian
        J_death = np.power(1.0, -self.state_dim) # inverse of the birth because of choice of alpha

        log_prior = self.f_log_prior(xprime)
        log_lh = self.f_log_lh(xprime)
        log_p = log_prior + beta*log_lh # use temperature of chain...
        alpha = (log_p - curr_log_p) +  (log_guprime  - log_gu) +  (np.log(p_birth) - np.log(p_death)) + np.log(J_death)
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
        chain.pos_pert_proposals += 1
        x = chain.samples[:,j]  
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
        positions = get_cp_positions(xprime)
        sj = get_cp_positions(xprime)[node_ind]
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
        xprime = replace_cp_pos(xprime, node_ind, sjprime)

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
            chain.log_probs[j+1] = log_p
            chain.pos_pert_acceptance_log[j] = 1
            chain.pos_pert_accepted += 1
        else: # reject
            chain.pos_pert_acceptance_log[j] = 0
            chain.samples[:,j+1] = x
            chain.log_probs[j+1] = curr_log_p
        return chain

    def _value_perturb_move(self, i, j):
        """
        Perturb the value in one of the segments
        """
        chain = self.chain_list[i]
        chain.val_pert_proposals += 1
        x = chain.samples[:,j] 
        beta = chain.params.beta
        curr_log_p = chain.curr_log_p # log posterior of the current chain state
        num_nodes = int(x[0])

        xprime = x.copy()
        # choose segment to perturb
        seg_ind = np.random.randint(0,num_nodes+1)
        statei = get_ith_segment_state(xprime, seg_ind, self.kmax, self.state_dim)
        u = self.f_val_pert_sample() 
        #print('xprime before replace', xprime)
        statei += u
        #print('u', u)
        #print('seg ind', seg_ind)
        #print('statei', statei)
        #print('xprime after', xprime)
        #replace_state_val(xprime, seg_ind, statei, self.kmax,  self.state_dim)

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
            chain.val_pert_accepted += 1
            chain.val_pert_acceptance_log[j] = 1
            chain.log_probs[j+1] = log_p
        else: # reject
            chain.samples[:,j+1] = x
            chain.log_probs[j+1] = curr_log_p
            chain.val_pert_acceptance_log[j] = 0
        return chain

    def _select_move(self, dim): 
        dim_ind = self.k_list.index(dim)
        #print('k list', self.k_list)
        #print('dim ind', dim_ind)
        #print('moves', self.moves)
        move = np.random.choice(self.moves, p=self.move_probs[dim_ind])
        return move

    def _update_chain(self, i, j, burnin=False):
        """
        Propose birth or death or perturb
        """
        chain = self.chain_list[i]
        """
        Copy chain values into next sample
        """
        x = chain.samples[:,j] # current state
        chain.samples[:,j+1] = x.copy()

        dim = int(chain.samples[0,j])
        #print('dim', dim)
        move = self._select_move(dim)
        #print('j', j)
        #print('move', move)
        if move == 'birth':
            self._birth_move(i, j)
        elif move == 'death':
            self._death_move(i, j)
        elif move == 'pos_pert':
            self._position_perturb_move(i, j)
        else:
            self._value_perturb_move(i, j)

        if not burnin: # only keep track of acceptance ratio after burn in 
            if chain.birth_proposals > 0:
                chain.birth_acceptance_ratios[j] = chain.birth_accepted/chain.birth_proposals
            if chain.death_proposals > 0:
                chain.death_acceptance_ratios[j] = chain.death_accepted/chain.death_proposals
            if chain.pos_pert_proposals > 0:
                chain.pos_pert_acceptance_ratios[j] = chain.pos_pert_accepted/chain.pos_pert_proposals
            if chain.val_pert_proposals > 0:
                chain.val_pert_acceptance_ratios[j] = chain.val_pert_accepted/chain.val_pert_proposals
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

    def _burn_in(self):
        num_temps = len(self.beta_arr)
        for j in range(self.N_burn_in):
            for i in range(num_temps):
                self._update_chain(i,0, burnin=True) # propose plus accept/reject
                chain = self.chain_list[i]
                chain.samples[:,0] = chain.samples[:,1] # copy the sample back
        return

    def sample(self):
        """
        Make N steps on each chain
        """
        N = self.N_steps
        num_temps = len(self.beta_arr)
        self._burn_in()
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
            samples = chain.samples[:,:]
            log_probs = chain.log_probs[:]
            acceptance_ratios = chain.val_pert_acceptance_ratios
            print('i chain accpeted', chain.val_pert_accepted)
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
        samples = np.zeros((self.max_dim, N+1))
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
