import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
import spatial_discretization_nonpd
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
import pylab as plt
from matplotlib import rc as matplotlibrc


def get_delta(u, a, x_mesh, Lx):

    delta = -1
    u_delta = 0.01 * a
    for ii, x in enumerate(x_mesh):
        if x_mesh[ii] > Lx / 2.:
            if u[ii] < u_delta:
                delta = x_mesh[ii]
                break
    return delta
