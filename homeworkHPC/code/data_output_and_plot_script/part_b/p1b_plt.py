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

# Strong Scaling
# array initialization for plot
num_proc = np.array([1, 2, 4, 8, 16])
time_strong = np.zeros(len(num_proc), dtype=np.float128)
strong_eff = np.zeros(len(num_proc), dtype=np.float128)
# data recorded when runing part_a
time_strong[0] = 35.056201
time_strong[1] = 18.600436
time_strong[2] = 9.4895389
time_strong[3] = 5.3930619
time_strong[4] = 3.350652

for ii in range(5):  # eff = t1 / ( N * tN )
    strong_eff[ii] = time_strong[0] / (num_proc[ii] * time_strong[ii]) * 100
# plot the figure
plt.figure(1)
plt.plot(num_proc, strong_eff, linewidth=2, color='k', label='Strong Scaling Efficiencies')
plt.axhline(y=50, linestyle='--', linewidth=1, color='b', label='50% Efficiency')
plt.axhline(y=75, linestyle='--', linewidth=1, color='g', label='75% Efficiency')
plt.axhline(y=100, linestyle='--', linewidth=1, color='r', label='100% Efficiency')
plt.legend(loc='upper right')
plt.grid()
plt.title(r'Strong Scaling Efficiencies [%] vs. Number of Processes')
plt.xlabel(r'$N$')
plt.ylabel(r'Efficiency [%]')
plt.ylim([45, 105])
plt.show()
plt.savefig("../../../report/figures/part_b_strong.png")

# Weak Scaling
time_weak = np.zeros(len(num_proc), dtype=np.float128)
weak_eff = np.zeros(len(num_proc), dtype=np.float128)
# load the data pickled in the data generation part
time_weak[0] = pickle.load(open("time_p1"))
time_weak[1] = pickle.load(open("time_p2"))
time_weak[2] = pickle.load(open("time_p4"))
time_weak[3] = pickle.load(open("time_p8"))
time_weak[4] = pickle.load(open("time_p16"))

for ii in range(5):  # eff = t1 / tN
    weak_eff[ii] = time_weak[0] / time_weak[ii] * 100
# plot the figure
plt.figure(2)
plt.plot(num_proc, weak_eff, linewidth=2, color='k', label='Weak Scaling Efficiencies')
plt.axhline(y=50, linestyle='--', linewidth=1, color='b', label='50% Efficiency')
plt.axhline(y=75, linestyle='--', linewidth=1, color='g', label='75% Efficiency')
plt.axhline(y=100, linestyle='--', linewidth=1, color='r', label='100% Efficiency')
plt.legend(loc='lower right')
plt.grid()
plt.title(r'Weak Scaling Efficiencies [%] vs. Number of Processes')
plt.xlabel(r'$N$')
plt.ylabel(r'Efficiency [%]')
plt.ylim([45, 105])
plt.show()
plt.savefig("../../../report/figures/part_b_weak.png")
