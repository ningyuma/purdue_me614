from mpi4py import MPI
import numpy as np
import math
import os
import sys
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# array of deltaX
deltaX = np.array([2.0, 0.2, 0.02, 0.002, 0.0002, 2e-05, 2e-06, 2e-7, 2e-8])
# load the data pickled in the data generation part
error_truc_p1 = pickle.load(open("error_p1"))
error_truc_p2 = pickle.load(open("error_p2"))
error_truc_p4 = pickle.load(open("error_p4"))
error_truc_p8 = pickle.load(open("error_p8"))
error_truc_p16 = pickle.load(open("error_p16"))
# plot the figure
plt.loglog(1 / deltaX, error_truc_p1, linewidth=2, label='1 process')
plt.loglog(1 / deltaX, error_truc_p2, linewidth=2, label='2 processes')
plt.loglog(1 / deltaX, error_truc_p4, linewidth=2, label='4 processes')
plt.loglog(1 / deltaX, error_truc_p8, linewidth=2, label='8 processes')
plt.loglog(1 / deltaX, error_truc_p16, linewidth=2, label='16 processes')
plt.loglog(1 / deltaX, deltaX ** 1, '-.', linewidth=1, label='$\Delta x^{1}$')
plt.loglog(1 / deltaX, deltaX ** 2, '-.', linewidth=1, label='$\Delta x^{2}$')
plt.loglog(1 / deltaX, deltaX ** 3, '-.', linewidth=1, label='$\Delta x^{3}$')
plt.loglog(1 / deltaX, deltaX ** 4, '-.', linewidth=1, label='$\Delta x^{4}$')
plt.loglog(1 / deltaX, deltaX ** 5, '-.', linewidth=1, label='$\Delta x^{5}$')
plt.loglog(1 / deltaX, deltaX ** 6, '-.', linewidth=1, label='$\Delta x^{6}$')
plt.ylim([10 ** -12, 10 ** 3])
plt.legend(loc='upper right')
plt.grid()
plt.title(r'Truncation Error $\epsilon$ vs. $\Delta x^{-1}$')
plt.xlabel(r'$\Delta x^{-1}$')
plt.ylabel(r'$\epsilon$')
plt.show()
plt.savefig("../../../report/figures/part_a.png")
