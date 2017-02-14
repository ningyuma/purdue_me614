import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard # pdb package allows you to interrupt the python script with the keyboard() command
import spatial_discretization as sd  # like include .h file in C++, calls another file
import matplotlib.pyplot as plt
import prob1_plot as p1
import scipy.sparse.linalg as splinalg

def prob2_plot(order_scheme, iiter):
    N = 10
    RMS_error = np.zeros(iiter)
    deltaX = np.zeros(iiter)
    for ii in range(iiter):
        x_mesh = np.linspace(-.3, .3, N)
        f = x_mesh ** 4 + 3 * x_mesh ** 3 + 2 * x_mesh ** 2 + x_mesh + 1  # define a function by myself
        f_der1 = 4 * x_mesh ** 3 + 9 * x_mesh ** 2 + 4 * x_mesh + 1
        f_der3 = 24 * x_mesh + 18
        derivative_order = 3
        pts = 4
        D = sd.Generate_Spatial_Operators(x_mesh, order_scheme, derivative_order)
        D = D.tolil()
        row_1 = np.zeros(len(x_mesh))
        row_1[0] = 1
        D[0, :] = row_1
        f_der3[0] = f[0]
        row_2 = np.zeros(len(x_mesh))
        row_2_weights = sd.Generate_Weights(x_mesh[0:pts], x_mesh[0], 1)
        row_2[0:pts] = row_2_weights
        D[-2, :] = row_2
        f_der3[-2] = f_der1[0]
        row_end = np.zeros(len(x_mesh))
        row_end_weights = sd.Generate_Weights(x_mesh[-pts:], x_mesh[-1], 1)
        row_end[-pts:] = row_end_weights
        D[-1, :] = row_end
        f_der3[-1] = f_der1[-1]
        D = D.tocsr()
        u_hat = splinalg.spsolve(D, f_der3)
        error = (u_hat - f)
        RMS_error[ii] = np.sqrt(np.mean(error ** 2))
        deltaX[ii] = (.6 / (N - 1))
        print RMS_error
        N = N * 2
    deltaX_1 = [ii ** 1 for ii in deltaX]
    deltaX_2 = [ii ** 2 for ii in deltaX]
    deltaX_3 = [ii ** 3 for ii in deltaX]
    deltaX_4 = [ii ** 4 for ii in deltaX]
    deltaX_5 = [ii ** 5 for ii in deltaX]
    deltaX_6 = [ii ** 6 for ii in deltaX]
    plt.loglog(deltaX ** -1, RMS_error, linewidth=3, label=order_scheme)
    plt.loglog(np.reciprocal(deltaX), deltaX_1, '-.', linewidth=2, label='n=1')
    plt.loglog(np.reciprocal(deltaX), deltaX_2, '-.', linewidth=2, label='n=2')
    plt.loglog(np.reciprocal(deltaX), deltaX_3, '-.', linewidth=2, label='n=3')
    plt.loglog(np.reciprocal(deltaX), deltaX_4, '-.', linewidth=2, label='n=4')
    plt.loglog(np.reciprocal(deltaX), deltaX_5, '-.', linewidth=2, label='n=5')
    plt.loglog(np.reciprocal(deltaX), deltaX_6, '-.', linewidth=2, label='n=6')
    plt.legend(loc='lower left')
    plt.grid()
    plt.title(r'RMS of Error $\epsilon$ vs. $\Delta x^{-1}$')
    plt.xlabel(r'$\Delta x^{-1}$')
    plt.ylabel(r'RMS')
