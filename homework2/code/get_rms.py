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


def get_rms(Nx, u_num):
    u_anal = tsd.get_u_anal(Nx)
    rms = np.sqrt(np.mean((u_anal - u_num) ** 2))
    return rms
