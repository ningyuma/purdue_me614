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

# initialize num_proc array for plot
num_proc = np.linspace(1, 16, 16)
error_truc = np.zeros(len(num_proc), dtype=np.float128)
# load the data pickled in the data generation part
error_truc[0] = pickle.load(open("error_c_p1"))
error_truc[1] = pickle.load(open("error_c_p2"))
error_truc[2] = pickle.load(open("error_c_p2"))
error_truc[3] = pickle.load(open("error_c_p4"))
error_truc[4] = pickle.load(open("error_c_p5"))
error_truc[5] = pickle.load(open("error_c_p6"))
error_truc[6] = pickle.load(open("error_c_p7"))
error_truc[7] = pickle.load(open("error_c_p8"))
error_truc[8] = pickle.load(open("error_c_p9"))
error_truc[9] = pickle.load(open("error_c_p10"))
error_truc[10] = pickle.load(open("error_c_p11"))
error_truc[11] = pickle.load(open("error_c_p12"))
error_truc[12] = pickle.load(open("error_c_p13"))
error_truc[13] = pickle.load(open("error_c_p14"))
error_truc[14] = pickle.load(open("error_c_p15"))
error_truc[15] = pickle.load(open("error_c_p16"))
# get the mean of error
error_mean = np.mean(error_truc)
# plot the figure
plt.plot(num_proc, error_truc, 'ro', label='Truncation Error')
plt.axhline(y=error_mean, linestyle='--', linewidth=1, label='Mean Error')
plt.legend(loc='upper left')
plt.grid()
plt.title(r'Truncation Error $\epsilon$ vs. Number of Processes')
plt.xlabel(r'$N$')
plt.ylabel(r'$\epsilon$')
plt.show()
plt.savefig("../../../report/figures/part_c.png")
