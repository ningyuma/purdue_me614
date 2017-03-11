import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
import spatial_discretization
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
import pylab as plt
from matplotlib import rc as matplotlibrc


def plot_uf(Nx, u_dict, u_anal, order):
    Lx = 1.0
    xx = np.linspace(0., Lx, Nx + 1)
    x_mesh = 0.5 * (xx[:-1] + xx[1:])

    N_finest = 100
    xx_finest = np.linspace(0., Lx, N_finest + 1)
    x_mesh_finest = 0.5 * (xx_finest[:-1] + xx_finest[1:])

    figwidth = 10
    figheight = 6
    lineWidth = 2
    textFontSize = 28
    gcafontSize = 30

    ymin = -12.
    ymax = 12.

    fig = plt.figure(order, figsize=(figwidth, figheight))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    plt.axes(ax)
    ax.plot(x_mesh, u_dict['c2nd_ee'], linewidth=lineWidth, label="Second Order Central & Explicit Euler")
    ax.plot(x_mesh, u_dict['c2nd_cn'], linewidth=lineWidth, label="Second Order Central & Crank-Nicolson")
    ax.plot(x_mesh, u_dict['u1st_ee'], linewidth=lineWidth, label="First Order Upwind & Explicit Euler")
    ax.plot(x_mesh, u_dict['u1st_cn'], linewidth=lineWidth, label="First Order Upwind & Crank-Nicolson")
    ax.plot(x_mesh, u_dict['u2nd_ee'], linewidth=lineWidth, label="Second Order Upwind & Explicit Euler")
    ax.plot(x_mesh, u_dict['u2nd_cn'], linewidth=lineWidth, label="Second Order Upwind & Crank-Nicolson")
    ax.plot(x_mesh_finest, u_anal, '--', linewidth=lineWidth, label="Analytical Solution")
    plt.legend(loc=2, fontsize='x-small')
    ax.grid('on', which='both')
    plt.setp(ax.get_xticklabels(), fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(), fontsize=gcafontSize)
    ax.set_xlabel(r"$x$", fontsize=textFontSize)
    ax.set_ylabel(r"$u(x,t)$", fontsize=textFontSize, rotation=90)
    ax.set_ylim([ymin, ymax])
