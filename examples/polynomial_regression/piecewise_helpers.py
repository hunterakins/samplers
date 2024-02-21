"""
Description:
    Helpers for piecewise function

Date:
    12/7/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt

class PiecewisePolynomial:
    def __init__(self, node_pts, coeff_list):
        """
        Node pts is a list of n+1 values that define the n intervals
        coeff_list is a list of coefficients for each interval

        The polynomials are defined in reference to the previous node point
        They also carry over the value of the previous node point
        (see eval for details)
        """
        self.node_pts = node_pts
        self.coeff_list = coeff_list

    def eval(self, x):
        """
        Evaluate piecewise polynomial at x
        """
        if x < self.node_pts[0]:
            raise ValueError("x is less than the first node point")
        if x > self.node_pts[-1]:
            raise ValueError("x is greater than the last node point") 
        i = 0
        val = 0.0
        while x > self.node_pts[i+1]:
            arg = self.node_pts[i+1] - self.node_pts[i]
            val += np.polyval(self.coeff_list[i], arg)
            i += 1
        # now x is in the interval
        if x == self.node_pts[i]:
            return val + self.coeff_list[i][-1] # return the constant term
        else:
            arg = x - self.node_pts[i]
            return val + np.polyval(self.coeff_list[i], arg)

def simple_lin_example():
    node_pts = [0, 1, 2, 3, 4]
    coeff_list = [[1,1], [2,0], [.5, 0], [5, 0]]
    pw = PiecewisePolynomial(node_pts, coeff_list)
    xvals = np.linspace(0, 4, 100)
    yvals = [pw.eval(x) for x in xvals]
    plt.figure()
    plt.plot(xvals, yvals)

def second_order_example():
    node_pts = [0, 1, 2, 3, 4]
    coeff_list = [[1,1, 0], [2,-1, 0], [20, .5, 0], [-20, 1, 0]]
    pw = PiecewisePolynomial(node_pts, coeff_list)
    xvals = np.linspace(0, 4, 1000)
    yvals = [pw.eval(x) for x in xvals]
    plt.figure()
    plt.plot(xvals, yvals)


simple_lin_example()
second_order_example()
plt.show()

