from user_input import *
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
import time_space_discr as tsd
from plot_uf import plot_uf
from get_rms import get_rms


def plot_contour(time_advancement, advection_scheme, order ):
    na = 10
    nc = 10
    Ca = np.linspace(0.1, 3., na)
    Cc = np.linspace(0.1, 2., nc)
    spec_radius = np.zeros((na, nc))
    xv, yv = np.meshgrid(Ca, Cc, sparse=False, indexing='ij')
    for i in range(na):
        for j in range(nc):
            spec_radius[i, j] = tsd.get_r(xv[i, j], yv[i, j], time_advancement, advection_scheme)

    levels = np.linspace(0., 2., 100)
    ff = plt.figure(order, figsize=(10, 10))
    ax = ff.add_subplot(111)
    cc = ax.contourf(xv, yv, spec_radius, levels=levels)

    ax.set_xlabel(r'$C_{\alpha}$')
    ax.set_ylabel(r'$C_{c}$')
    # ax.set_title(r'Contour plot of Spectral Radius')

    ff.subplots_adjust(right=0.8)
    caxb = ff.add_axes([0.81, 0.1, 0.05, 0.8])

    cb = ff.colorbar(cc, cax=caxb, orientation='vertical')
    cb.ax.set_ylabel('Spectral Radius')
