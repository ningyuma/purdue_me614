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

# outfile = TemporaryFile()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#data = numpy.arange(10, dtype=numpy.float64)
#comm.Send([data, MPI.INT], dest=1, tag=77)


def f(x):
    return np.exp(x) * np.tanh(x)


def integration(f, lower_bound, upper_bound, N, deltaX):
    x_stencil = np.linspace(lower_bound, upper_bound, N, dtype=np.float128)
    f_x1 = f(x_stencil[:-1])
    f_x2 = f(x_stencil[1:])
    middle = 4 * f((x_stencil[1:] + x_stencil[:-1]) / 2)
    int_sum = (np.sum(f_x1) + np.sum(f_x2) + np.sum(middle)) * deltaX / 6
    return int_sum

Part_a = False
Part_b = False
Part_c = True

if Part_a:
    int_analytical = -1 / math.exp(10) + math.exp(10) + 2 * math.atan(1 / math.exp(10)) - 2 * math.atan(math.exp(10))
    lower = -10.
    upper = 10.
    iteration = 9
    int_numerical = np.zeros(iteration, dtype=np.float128)
    local_int_numerical = np.zeros(1, dtype=np.float128)
    global_int_numerical = np.zeros(1, dtype=np.float128)
    error_truc = np.zeros(iteration, dtype=np.float128)
    deltaX = np.zeros(iteration, dtype=np.float128)
    time_use = np.zeros(iteration, dtype=np.float128)
    for ii in range(iteration):
        comm.Barrier()
        time_start = time.time()
        N = 10 ** (ii + 1) + 1
        deltaX[ii] = (upper - lower) / (N - 1)
        num_per_proc = N // size
        reminder = N % size
        if rank == size-1:
            local_lower = lower + rank * num_per_proc * deltaX[ii]
            local_upper = lower + ((rank + 1) * num_per_proc + (reminder - 1)) * deltaX[ii]
            local_int_numerical = integration(f, local_lower, local_upper, (num_per_proc + reminder), deltaX[ii])
        else:
            local_lower = lower + rank * num_per_proc * deltaX[ii]
            local_upper = lower + (rank + 1) * num_per_proc * deltaX[ii]
            local_int_numerical = integration(f, local_lower, local_upper, num_per_proc + 1, deltaX[ii])
        global_int_numerical = comm.reduce(local_int_numerical, op=MPI.SUM, root=0)
        if rank == 0:
            int_numerical[ii] = global_int_numerical
            error_truc[ii] = abs(int_numerical[ii] - int_analytical)
        comm.Barrier()
        time_end = time.time()
        time_use[ii] = time_end - time_start
    if rank == 0:
        print "p = 1"
        print "time use is", time_use
        print "deltaX is", deltaX
        print "error_truc is", error_truc
        print "int_numerical is", int_numerical
        print "int_analytical is", int_analytical
        writer = open("error_p1", "wb")
        pickle.dump(error_truc, writer)
        writer.close()

if Part_b:
    strong_scaling = False
    weak_scaling = True
    if strong_scaling:
        pass
        # N = 10 ** 8 + 1
        # data recorded in the plot script
        # strong_scaling_eff = t1 / (N * tN) * 100
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
        num_per_proc = 10 ** 7
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
            pickle.dump(time_use, writer)
            writer.close()
            #weak_scaling_eff = (t1 / tN) * 100

if Part_c:
    #Nc = 10 ** 5 + 1
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
        error_truc = abs(int_numerical - int_analytical)
        print "p = 9"
        print "time use is", time_use
        print "deltaX is", deltaX
        print "error_truc is", error_truc
        print "int_numerical is", int_numerical
        print "int_analytical is", int_analytical
        writer = open("error_c_p9", "wb")
        pickle.dump(error_truc, writer)
        writer.close()
