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

########################################################
# This version is a version with comments
# you can read the comments without run the code
# part_b and part_c are basically the same as part_a
# so I just comment out the difference part.
# for plot script, please look at p1a_plt.py, p1b_plt.py, p1c_plt.py
# under the data_output_and_plot_script directory
########################################################

# MPI environment set up, getting rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# define a function that outputs f(x) given x
def f(x):
    return np.exp(x) * np.tanh(x)


# define a global integration function using simpson's rule
def integration(f, lower_bound, upper_bound, N, deltaX):
    x_stencil = np.linspace(lower_bound, upper_bound, N, dtype=np.float128)  # generate grid
    f_x1 = f(x_stencil[:-1])  # calculate 3 parts respectively, using vectorized method
    f_x2 = f(x_stencil[1:])
    middle = 4 * f((x_stencil[1:] + x_stencil[:-1]) / 2)
    int_sum = (np.sum(f_x1) + np.sum(f_x2) + np.sum(middle)) * deltaX / 6  # sum each part and sum them up
    return int_sum

# Switch control
Part_a = False
Part_b = False
Part_c = False

if Part_a:
    # calculate analytical integral
    int_analytical = -1 / math.exp(10) + math.exp(10) + 2 * math.atan(1 / math.exp(10)) - 2 * math.atan(math.exp(10))
    lower = -10.  # lower and upper bound
    upper = 10.
    iteration = 9  # maximum amounts of grid points
    int_numerical = np.zeros(iteration, dtype=np.float128)  # global array initialization
    local_int_numerical = np.zeros(1, dtype=np.float128)
    global_int_numerical = np.zeros(1, dtype=np.float128)
    error_truc = np.zeros(iteration, dtype=np.float128)
    deltaX = np.zeros(iteration, dtype=np.float128)
    time_use = np.zeros(iteration, dtype=np.float128)
    for ii in range(iteration):
        comm.Barrier()  # timing start
        time_start = time.time()
        N = 10 ** (ii + 1) + 1  # total number of points
        deltaX[ii] = (upper - lower) / (N - 1)  # calculate delta_x
        num_per_proc = N // size  # calculate number of points each process using integer division
        reminder = N % size  # get extra number points
        if rank == size-1:  # give the extra number points to the last process
            local_lower = lower + rank * num_per_proc * deltaX[ii]  # calculate local lower and upper bound
            local_upper = lower + ((rank + 1) * num_per_proc + (reminder - 1)) * deltaX[ii]
            local_int_numerical = integration(f, local_lower, local_upper, (num_per_proc + reminder), deltaX[ii])
        else:
            local_lower = lower + rank * num_per_proc * deltaX[ii]
            local_upper = lower + (rank + 1) * num_per_proc * deltaX[ii]
            local_int_numerical = integration(f, local_lower, local_upper, num_per_proc + 1, deltaX[ii])
        # use comm.reduce to sum up the integral calculated from each process to root 0
        global_int_numerical = comm.reduce(local_int_numerical, op=MPI.SUM, root=0)
        if rank == 0:
            int_numerical[ii] = global_int_numerical
            error_truc[ii] = abs(int_numerical[ii] - int_analytical)  # calculate truncation error in root 0
        comm.Barrier()  # timing end
        time_end = time.time()
        time_use[ii] = time_end - time_start
    if rank == 0:
        print "p = 1"
        print "time use is", time_use
        print "deltaX is", deltaX
        print "error_truc is", error_truc
        print "int_numerical is", int_numerical
        print "int_analytical is", int_analytical
        writer = open("error_p1", "wb")  # pickle the truncation error and write it in a file
        pickle.dump(error_truc, writer)
        writer.close()

if Part_b:
    strong_scaling = False
    weak_scaling = False
    if strong_scaling:
        pass
        # data can be get from part_a
        # data is recorded in the plot script, check p1b_plt.py
        # time_strong[0] = 35.056201
        # time_strong[1] = 18.600436
        # time_strong[2] = 9.4895389
        # time_strong[3] = 5.3930619
        # time_strong[4] = 3.350652
    if weak_scaling:
        comm.Barrier()
        time_start = time.time()
        int_analytical = -1 / math.exp(10) + math.exp(10) + 2 * math.atan(1 / math.exp(10)) - 2 * math.atan(
            math.exp(10))
        lower = -10.
        upper = 10.
        int_numerical = np.zeros(1, dtype=np.float128)
        local_int_numerical = np.zeros(1, dtype=np.float128)
        global_int_numerical = np.zeros(1, dtype=np.float128)
        error_truc = np.zeros(1, dtype=np.float128)
        deltaX = np.zeros(1, dtype=np.float128)
        num_per_proc = 10 ** 7  # each process has the same number of points
        N = size * num_per_proc
        deltaX = (upper - lower) / (N - 1)
        local_lower = lower + rank * num_per_proc * deltaX
        local_upper = lower + (rank + 1) * num_per_proc * deltaX
        local_int_numerical = integration(f, local_lower, local_upper, num_per_proc + 1, deltaX)
        global_int_numerical = comm.reduce(local_int_numerical, op=MPI.SUM, root=0)
        comm.Barrier()
        time_end = time.time()
        time_use = time_end - time_start
        if rank == 0:
            int_numerical = global_int_numerical
            error_truc = abs(int_numerical - int_analytical)
            print "p = xx"
            print "time use is", time_use
            print "deltaX is", deltaX
            print "error_truc is", error_truc
            print "int_numerical is", int_numerical
            print "int_analytical is", int_analytical
            writer = open("time_p1", "wb")
            pickle.dump(time_use, writer)  # pickle the time recorded in a file
            writer.close()

if Part_c:
    comm.Barrier()
    time_start = time.time()
    int_analytical = -1 / math.exp(10) + math.exp(10) + 2 * math.atan(1 / math.exp(10)) - 2 * math.atan(
        math.exp(10))
    lower = -10.
    upper = 10.
    int_numerical = np.zeros(1, dtype=np.float128)
    local_int_numerical = np.zeros(1, dtype=np.float128)
    global_int_numerical = np.zeros(1, dtype=np.float128)
    error_truc = np.zeros(1, dtype=np.float128)
    deltaX = np.zeros(1, dtype=np.float128)
    N = 10 ** 5 + 1
    deltaX = (upper - lower) / (N - 1)
    num_per_proc = N / size
    reminder = N % size
    if rank == size - 1:
        local_lower = lower + rank * num_per_proc * deltaX
        local_upper = lower + ((rank + 1) * num_per_proc + (reminder - 1)) * deltaX
        local_int_numerical = integration(f, local_lower, local_upper, (num_per_proc + reminder), deltaX)
    else:
        local_lower = lower + rank * num_per_proc * deltaX
        local_upper = lower + (rank + 1) * num_per_proc * deltaX
        local_int_numerical = integration(f, local_lower, local_upper, num_per_proc + 1, deltaX)
    global_int_numerical = comm.reduce(local_int_numerical, op=MPI.SUM, root=0)
    comm.Barrier()
    time_end = time.time()
    time_use = time_end - time_start
    if rank == 0:
        int_numerical = global_int_numerical
        error_truc = abs(int_numerical - int_analytical)  # calculate truncation error for each case
        print "p = xx"
        print "time use is", time_use
        print "deltaX is", deltaX
        print "error_truc is", error_truc
        print "int_numerical is", int_numerical
        print "int_analytical is", int_analytical
        writer = open("error_c_p1", "wb")
        pickle.dump(error_truc, writer)  # pickle the truncation error and write it in a file
        writer.close()
