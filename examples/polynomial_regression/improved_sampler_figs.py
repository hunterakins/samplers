"""
Description:
    Make some figures and tests for the improved sampler idea

Date:
    12/6/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt

def get_H(t, m):
    """
    Given time samples t and model dimension m
    make the model matrix H
    """
    H = np.zeros((t.size, m))
    for i in range(m):
        H[:, i] = t**i
    H = H[:, ::-1] # leading coeff first
    return H

def get_A(Gn1, Gn):
    """
    Matrix for death move...
    Gn1 is the model matrix for dimension n-1
    subsampled at n-1 points (so it is square
    and hopefully invertible)
    Gn is the model matrix for dimension n
    subsampled at n-1 points
    """
    n = Gn.shape[1]
    mat1 = np.linalg.inv(Gn1)@Gn
    A = np.zeros((n, n))
    A[0,0] = 1 # leading coeff becomes uprime
    A[1:, :] = mat1
    A_inv = np.linalg.inv(A)
    return A, A_inv


def get_sub_inds(N, m):
    """
    Get m indices from N
    start at end points and move in
    """
    stride = int((N-1)/(m-1))
    left_over = N -1 - stride*(m-1)
    inds = [0]
    for i in range(m-1):
        ind = stride  + inds[-1]
        if left_over > 0:
            ind += 1
            left_over -= 1
        inds.append(ind)
    return inds

def test_death():
    t = np.linspace(0, 1, 20)
    m_list = [2,3,4,5,6,7]
    inds = np.arange(t.size)
    xfig, xax = plt.subplots(1,1)
    fig, ax = plt.subplots(1,1)
    for m in m_list[:-1]:
        Hn1 = get_H(t, m) # dim I will die to
        sub_inds = get_sub_inds(t.size, m)
        H = get_H(t, m+1)
        Gn1 = Hn1[sub_inds, :]
        Gn = H[sub_inds, :]
        print(Hn1.shape, H.shape)
        print(Gn1.shape, Gn.shape)
        A, A_inv = get_A(Gn1, Gn)

        x = np.random.randn(m+1)
        y = H@x
        ax.plot(t, y, 'ko')
        ax.plot(t[sub_inds], y[sub_inds], 'k*')

        xn1 = A@x
        print('det A', np.linalg.det(A))
        u = xn1[0]
        y = Hn1@xn1[1:]  
        ax.plot(t, y, 'bo')
        ax.plot(t[sub_inds], y[sub_inds], 'b*')

        xax.plot(x, 'k')
        xax.plot(xn1[:-1], 'b')

def test_birth():
    t = np.linspace(0, 1, 20)
    m_list = [2,3,4,5,6,7]
    inds = np.arange(t.size)
    xfig, xax = plt.subplots(1,1)
    fig, ax = plt.subplots(1,1)
    for m in m_list[:-1]:
        H = get_H(t, m)
        sub_inds = get_sub_inds(t.size, m)
        H1 = get_H(t, m+1)
        Gn1 = H[sub_inds, :]
        Gn = H1[sub_inds, :]
        A, A_inv = get_A(Gn1, Gn)

        x1 = np.random.randn(m)
        u = np.random.rand()
        y = H@x1
        ax.plot(t, y, 'ko')
        ax.plot(t[sub_inds], y[sub_inds], 'k*')

        h_vec = np.zeros(m+1)
        h_vec[1:] = x1
        h_vec[0] = u
        x2 = A_inv@h_vec
        print('det A', np.linalg.det(A))
        y = H1@x2
        ax.plot(t, y, 'bo')
        ax.plot(t[sub_inds], y[sub_inds], 'b*')

        xax.plot(x1, 'k')
        xax.plot(x2[:-1], 'b')

test_death()
plt.show()
test_birth()
plt.show()




