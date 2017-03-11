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

Lx = 1.
c_x = 5.  # (linear) convection speed
alpha = 5.  # diffusion coefficients
gamma1 = 1.0
gamma2 = 1.0
c1 = 5.0
c2 = 5.0
m = 2.0
w1 = 2.0 * np.pi / Lx
w2 = 2.0 * m * np.pi / Lx
Tf = 1 / (w2 ** 2 * alpha)  # one complete cycle