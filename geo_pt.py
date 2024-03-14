""" Description:

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from samplers.pt import AdaptivePTSampler
from optimiz import gi_helpers as gh

class FixedDGeoSampler(AdaptivePTSampler):
    """
    Parallel tempered sampler Use adaptive sampling (Haario) and user specified temperature ladder
    """
    def __init__(self, move_probs, dim, beta_arr, 
            f_log_prior, f_log_lh, f_proposal, 
            k, pos_range, c_range, rho_range, attn_range):
        super().__init__(move_probs, dim, beta_arr, 
            f_log_prior, f_log_lh, f_proposal)
        self.k = k # num layers
        self.pos_range = pos_range
        self.c_range = c_range
        self.rho_range = rho_range
        self.attn_range = attn_range


    def plot_dist(self, temp_i,N_bins=20, Nz=100):
        """
        temp_i - int
            index of temperature in temperature ladder
        """
        zgrid = np.linspace(self.pos_range[0], self.pos_range[1], Nz)
        samples, log_probs, acceptance_ratios = self.get_chain_info(temp_i)
        u_samples= np.zeros(samples.shape)
        c_samples = np.zeros((zgrid.size, self.N))
        rho_samples = np.zeros((zgrid.size, self.N))
        attn_samples = np.zeros((zgrid.size, self.N))

        """ Unscale """
        for i in range(samples.shape[1]):
            ux = gh.unscale_x(samples[:,i][:,None], self.k, self.pos_range, self.c_range, self.rho_range, self.attn_range)
            num_segments = self.k+1
            interface_pts = ux[:self.k]
            ci = ux[self.k::3]
            rhoi = ux[self.k+1::3]
            attni = ux[self.k+2::3]
            full_pos = [self.pos_range[0]] + list(interface_pts) + [self.pos_range[1]]

            for seg_i in range(num_segments):
                inds = (zgrid <= full_pos[seg_i+1]) & (zgrid >= full_pos[seg_i])
                c_samples[inds, i] = ci[seg_i]
                rho_samples[inds, i] = rhoi[seg_i]
                attn_samples[inds, i] = attni[seg_i]

        val_arrs = [c_samples, rho_samples, attn_samples]
        labels = ['c', 'rho', 'attn']
        ranges = [self.c_range, self.rho_range, self.attn_range]
        for i in range(len(val_arrs)):
            bins = np.linspace(ranges[i][0], ranges[i][1], N_bins)
            histo = np.zeros((zgrid.size, N_bins-1))
            for zi in range(zgrid.size):
                hist, _ = np.histogram(val_arrs[i][zi,:], bins=bins)
                histo[zi,:] = hist
            plt.figure()
            plt.suptitle(labels[i])
            bin_vals = 0.5*(bins[1:] + bins[:-1])
            plt.pcolormesh(bin_vals, zgrid, histo)
            plt.ylabel('Depth (m)')
            plt.xlabel(labels[i])
            plt.gca().invert_yaxis()
            plt.colorbar()
        return
        
