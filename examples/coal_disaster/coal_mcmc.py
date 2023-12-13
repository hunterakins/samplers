"""
Description:
    Sample test for the coal disaster problem

Date:
    12/7/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from coal_data import get_accident_days
from samplers.coal_rj import fill_x, get_positions, get_values, update_value, update_position,  ChangePointSampler, get_dim
from scipy import special


def get_log_lh():
    y = get_accident_days()
    s0 = y[0]
    slast = y[-1]
    def log_lh(x):
        """ 
        x is a vector containing info for the 
        model
        x[0] is the number of node points
        x[1:num_nodes+1] are the position of the nodes
        x[num_nodes+1:num_nodes + 1 + num_nodes + 1] are the values
        (the process rate) for the segments defined by those nodes
        The intervals the nodes live within are given by s0 and slast
        """
        i = 0
        curr_seg_ind = 0
        log_lh_val = 0.0
        positions = get_positions(x)
        positions = [s0] + positions + [slast] # append the start and end
        values = get_values(x)
        for i in range(y.size):
            while y[i] > positions[curr_seg_ind + 1]: # find segment that y[i] is in
                curr_seg_ind += 1
            log_lh_val += np.log(values[curr_seg_ind]) # add the value of that segment

        """ 
        Now add in the normalization term
        \int_0^L values \dd L = \sum_{i=0}^{num_nodes} values[i] * (positions[i+1] - positions[i])
        """
        log_lh_val -= np.sum(np.array(values) * np.diff(np.array(positions)))
        return log_lh_val
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

def get_log_prior(alpha, beta, lambd, kmax, dirichlet=False):
    """
    Independent gamma distribution on the values ('heights') of the segments
    with parameters alpha and beta
    p(h) = \beta^\alpha h^{\alpha - 1} e^{-\beta h} / \Gamma(\alpha)
    logp = alpha * log(beta) - (alpha-1) * h - beta * h - log(Gamma(alpha))

    Node number has Poisson distribution prior with parameter lam
    p(k) = e^{-\lambda} \frac{\lambda^{k}}{k !}

   
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
    y = get_accident_days()
    L = y[-1] - y[0]
    t1 = alpha*np.log(beta)
    t2 = special.gamma(alpha)
    
    def log_prior(x):
        """
        x is a vector containing info for the 
        model
        x[0] is the number of node points
        x[1:num_nodes+1] are the position of the nodes
        x[num_nodes+1:num_nodes + 1 + num_nodes + 1] are the values
        (the process rate) for the segments defined by those nodes
        The intervals the nodes live within are given by s0 and slast
        """
        num_nodes = int(x[0])
        #p_num = np.exp(-lambd) * (np.power(lambd, num_nodes)) / special.gamma(num_nodes) # use gamma to compute factorial
        log_p_num = -lambd + num_nodes * np.log(lambd) - np.log(special.gamma(num_nodes))

        positions = get_positions(x)
        if dirichlet:
            p_pos = special.gamma(num_nodes) / np.power(L, num_nodes)
            log_p_pos = np.log(special.gamma(num_nodes)) - num_nodes * np.log(L)
        
        else:
            log_p_pos = 0.0
            N = 2*num_nodes + 1
            for ind in range(num_nodes): # even order statistics
                i = 2*(ind+1) # 
                sj = positions[ind]
                pj = unif_order_stat((sj - y[0]), N, i, L)
                log_p_pos += np.log(pj)

        values = get_values(x)
        log_p_values = [np.log(t1) + (alpha-1)*np.log(x)-beta*x -np.log(t2) for x in values]

        log_p_vals = sum(log_p_values)
        log_p = log_p_num + log_p_pos + log_p_vals
        return log_p
    return log_prior

def get_f_prior(alpha, beta, lambd, kmax, interval):
    """
    Draw from the prior
    Number of change points prior is
    poisson with parameter lambd conditioned
    on being less than or equal to kmax
    """
    shape = alpha
    scale = 1/beta
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
            val = np.random.gamma(shape, scale)
            values.append(val)
        max_dim = get_dim(kmax)
        x = fill_x(max_dim, positions, values)
        return x
    return f_prior


def make_green_figure():
    # gamma distribution parameters
    alpha = 1
    beta = 200
    # poisson for num changepoints
    lambd = 3
    y = get_accident_days()
    interval = [y[0], y[-1]]
    k_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    num_k = len(k_list)
    kmax = max(k_list)
    f_prior = get_f_prior(alpha, beta, lambd, kmax, interval)

    x0 = f_prior()
    k = int(x0[0])
    values = get_values(x0)
    positions = list(get_positions(x0))
    positions = [interval[0]] + positions + [interval[1]]
    print('Initial model number of change points k', k)

    swap_interval = 1
    T_max = 100
    num_T = 1
    t_arr = np.linspace(1, T_max, num_T)
    beta_arr = 1/t_arr
    print('temperatures ', t_arr)

    move_probs = [[1, 0]] + [[1/2, 1/2]]*(num_k-2) + [[0,1]]
    f_log_lh = get_log_lh()
    f_log_prior = get_log_prior(alpha, beta, lambd, kmax)
    cps = ChangePointSampler(move_probs, k_list, beta_arr, f_log_prior, f_log_lh, interval)

    N = 40000
    N_burn_in = 2000
    cps.initialize_chains(N, N_burn_in, f_prior, swap_interval)

    cps.sample()
    cps.diagnostic_plot()

    best_fit = np.argmax(cps.chain_list[0].log_probs)
    x = cps.chain_list[0].samples[:,best_fit]
    positions = get_positions(x)

    values = get_values(x)
    positions = [interval[0]] + positions + [interval[1]]
    print('optimum node positions', positions)
    plt.figure()
    k = int(x[0])
    for i in range(k+1):
        plt.plot([positions[i], positions[i+1]], [values[i], values[i]], 'k')
    plt.title('Best fit')

    

    plt.show()



make_green_figure()






