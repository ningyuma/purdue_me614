import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard # pdb package allows you to interrupt the python script with the keyboard() command
import spatial_discretization as sd  # like include .h file in C++, calls another file
import matplotlib.pyplot as plt
import prob1_plot as p1
import prob2_plot as p2
import scipy.sparse.linalg as splinalg

Plot_Problem1 = True
Plot_Problem2 = True
Plot_Problem3 = True

############################################################
# Problem 1
############################################################

if Plot_Problem1:
    plt.figure(1)
    p1.plotC_centered(6., '(C-centered) l=r=3')
    plt.savefig("../report/figures/ccl3r3.png")
    plt.figure(2)
    p1.plotC_centered(4., '(C-centered) l=r=2')
    plt.savefig("../report/figures/ccl2r2.png")
    plt.figure(3)
    p1.plotC_centered(2., '(C-centered) l=r=1')
    plt.savefig("../report/figures/ccl1r1.png")
    plt.figure(4)
    p1.plotC_biased(3., '(C-biased) l=0 r=3')
    plt.savefig("../report/figures/cbl0r3.png")
    plt.figure(5)
    p1.plotC_biased(2., '(C-biased) l=0 r=2')
    plt.savefig("../report/figures/cbl0r2.png")
    plt.figure(6)
    p1.plotC_biased(1., '(C-biased) l=0 r=1')
    plt.savefig("../report/figures/cbl0r1.png")
    plt.figure(7)
    p1.plotS_centered(6., '(S-centered) l=r=3')
    plt.savefig("../report/figures/scl3r3.png")
    plt.figure(8)
    p1.plotS_centered(4., '(S-centered) l=r=2')
    plt.savefig("../report/figures/scl2r2.png")
    plt.figure(9)
    p1.plotS_centered(2., '(S-centered) l=r=1, same as Figure 12')
    plt.savefig("../report/figures/scl1r1.png")
    plt.figure(10)
    p1.plotS_biased(4., '(S-biased) l=1 r=3')
    plt.savefig("../report/figures/sbl1r3.png")
    plt.figure(11)
    p1.plotS_biased(3., '(S-biased) l=1 r=2')
    plt.savefig("../report/figures/sbl1r2.png")
    plt.figure(12)
    p1.plotS_biased(2., '(S-biased) l=1 r=1, same as Figure 9')
    plt.savefig("../report/figures/sbl1r1.png")

############################################################
# Problem 2a
############################################################

if Plot_Problem2:
    N = 10
    x_mesh = np.linspace(-.3, .3, N)
    order_scheme = "3rd-order"
    derivative_order = 1
    D = sd.Generate_Spatial_Operators(x_mesh, order_scheme, derivative_order)
    plt.figure(13)
    plt.spy(D)
    plt.savefig("../report/figures/p2_3p1d.png")
    order_scheme = "5th-order"
    derivative_order = 1
    D = sd.Generate_Spatial_Operators(x_mesh, order_scheme, derivative_order)
    plt.figure(14)
    plt.spy(D)
    plt.savefig("../report/figures/p2_5p1d.png")
    order_scheme = "3rd-order"
    derivative_order = 3
    D = sd.Generate_Spatial_Operators(x_mesh, order_scheme, derivative_order)
    plt.figure(15)
    plt.spy(D)
    plt.savefig("../report/figures/p2_3p3d.png")
    order_scheme = "5th-order"
    derivative_order = 3
    D = sd.Generate_Spatial_Operators(x_mesh, order_scheme, derivative_order)
    plt.figure(16)
    plt.spy(D)
    plt.savefig("../report/figures/p2_5p3d.png")

############################################################
# Problem 2b
############################################################

    plt.figure(17)
    iiter = 13
    order_scheme = "3rd-order"
    p2.prob2_plot(order_scheme, iiter)
    plt.savefig("../report/figures/p2b_3p.png")
    plt.figure(18)
    iiter = 7  # increase to 8, get warning: RankWarning: Polyfit may be poorly conditioned, warnings.warn(msg, RankWarning)
    order_scheme = "5th-order"
    p2.prob2_plot(order_scheme, iiter)
    plt.savefig("../report/figures/p2b_5p.png")
    # print RMS in every iteration as a process bar for the code
############################################################
# Problem 3
############################################################
# Have not got here yet.....T T