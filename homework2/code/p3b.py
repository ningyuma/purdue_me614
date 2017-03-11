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
from get_delta import get_delta


def p3_b_ply_get_delta(Nx, Lx, c_x, alpha, beta, w, a, num_curve, get_del=False):

    #########################################
    ############### User Input ##############

    machine_epsilon = np.finfo(float).eps
    # Lx = 1.
    CFL = 0.5  # Courant-Friedrichs-Lewy number (i.e. dimensionless time step)
    # c_x_ref = 2.0
    # c_x = 10.  # (linear) convection speed, affect the Tf in my case, c_x down, dt the same, more lines
    # alpha = 2.  # diffusion coefficients, \ahpha up, dt down, more lines
    # w = 100000.
    # beta = 10.
    # a = 1.

    # n_prd = 10.
    prd = 2. * np.pi / w
    # n_per_prd = 10
    # Tf = n_prd * prd # one complete cycle
    # dt = Tf / (n_prd * n_per_prd)
    Tf = 500.
    dt = 0.1
    # num_curve =
    plot_every = Tf / (dt * num_curve)
    num_prd = int(Tf / prd)
    # plot_every_steady = np.floor(num_prd / 20)

    ## Diffusion Scheme
    diffusion_scheme = "2nd-order-central"  # always-second-order central

    ## Advection_Scheme
    advection_scheme = "2nd-order-upwind"

    u_initial = np.zeros(Nx)

    # dx = Lx / (Nx - 2)
    # dx2 = dx * dx
    # x_mesh = np.linspace(-dx / 2., Lx + dx / 2., Nx)
    sigma = np.linspace(0, Nx - 1, Nx) / (Nx - 1)
    x_mesh = Lx * (sigma ** 2)
    # x_mesh = np.zeros(Nx)
    # x_mesh[1:-1] = x_mesh_tmp
    x_mesh[0] = 0. - x_mesh[1]
    dx_last = (Lx - x_mesh[-2])
    x_mesh[-1] = Lx + dx_last
    dx = np.diff(x_mesh)
    dx2 = dx * dx
    # actual mesh points are off the boundaries x=0, x=Lx
    # non-periodic boundary conditions created with ghost points
    # x_mesh = 0.5 * (xx[:-1] + xx[1:])
    # dx = np.diff(xx)[0]

    #########################################
    ######## Preprocessing Stage ############

    # for linear advection/diffusion time step is a function
    # of c,alpha,dx only; we use reference limits, ballpark
    # estimates for Explicit Euler
    # dt_max_advective = dx / (c_x + machine_epsilon)  # think of machine_epsilon as zero
    # dt_max_diffusive = dx2 / (alpha + machine_epsilon)
    # dt_max = np.min([dt_max_advective, dt_max_diffusive])
    # print dt_max_advective
    # dt = CFL * dt_max
    # print "dt = ", dt

    # Creating identity matrix
    Ieye = scysparse.identity(Nx)

    # Creating first derivative
    Dx = spatial_discretization_nonpd.Generate_Spatial_Operators(x_mesh, advection_scheme, derivation_order=1)
    # plt.figure(200)
    # plt.spy(Dx)
    # plt.show()
    # Creating second derivative
    D2x2 = spatial_discretization_nonpd.Generate_Spatial_Operators(x_mesh, diffusion_scheme, derivation_order=2)
    # plt.figure(100)
    # plt.spy(D2x2)
    # plt.show()
    # Creating A,B matrices such that:
    #     A*u^{n+1} = B*u^{n} + q
    # if time_advancement == "Explicit-Euler":
    #     A = Ieye
    #     B = Ieye - dt * c_x * Dx + dt * alpha * D2x2
    # if time_advancement == "Crank-Nicolson":
    beta_eye = scysparse.identity(Nx) * beta
    adv_diff_Op = -dt * c_x * Dx + dt * alpha * D2x2 - beta_eye
    A = Ieye - 0.5 * adv_diff_Op
    B = Ieye + 0.5 * adv_diff_Op

    A = A.todense()
    B = B.todense()

    A[0, :] = 0.
    A[0, :2] = [0.5, 0.5]
    # A_rend = np.zeros(Nx, dtype=np.float)
    # A_rend[-2:] = [-1 / dx, 1 / dx]
    A[-1, :] = 0.
    A[-1, -2:] = [-1 / dx_last, 1 / dx_last]

    B[0, :] = 0.
    B[-1, :] = 0.

    # forcing csr ordering..
    A, B = scysparse.csr_matrix(A), scysparse.csr_matrix(B)

    #########################################
    ########## Time integration #############

    u = u_initial  # initializing solution
    it = 0
    time = 0.0
    thr = 0
    # cfl = dt * alpha / (dx ** 2)
    # cfl = dt * c_x / dx
    delta = np.zeros(num_curve)
    while time < Tf:
        it += 1
        time += dt
        bc_at_0 = a * np.cos(w * time)
        bc_at_l = 0
        q = np.zeros(Nx, dtype=np.float)
        q[0] = bc_at_0
        q[-1] = bc_at_l
        # Update solution
        # solving : A*u^{n+1} = B*u^{n} + q
        # where q is zero for periodic and zero source terms
        u = spysparselinalg.spsolve(A, B.dot(u) + q)
        # A_fac = spysparselinalg.factorized(A)
        # u = A_fac(B.dot(u))
    # A, B = scysparse.csr_matrix(A), scysparse.csr_matrix(B)
    #     if it % plot_every == 0:  # plot every plot_every time steps  np.mod(x1,x2)=x1-floor(x1/x2)*x2
    #         if thr == num_curve -1:
        if it == Tf / dt:
                delta = get_delta(u, a, x_mesh, Lx)
            # thr += 1
            # plt.legend(loc=1, fontsize='x-small')
            # plt.grid('on', which='both')
            # plt.xlabel(r"$x$")
            # plt.ylabel(r"$u(x,t)$", rotation=90)
            # plt.xlim([0, 1.4 * Lx])
    # if get_del:
    #     delta_diff = np.diff(delta)
    #     tol = 1e-6
    #     delta_ss = -1
    #     for ii in xrange(num_curve - 2):
    #         if np.abs(delta_diff[ii]) < tol:
    #             delta_ss = delta[ii + 1]
    return delta

            # if time - time
        # for ii in xrange(num_prd) + 1:
        #     if time == prd * ii :
        #         plt.figure(302)
        #         plt.plot(x_mesh[1:-1], u[1:-1])
        #         plt.grid('on', which='both')
        #         plt.xlabel(r"$x$")
        #         plt.ylabel(r"$u(x,t)$", rotation=90)


    #     if it == Tf / dt:
    #         plt.plot(x_mesh[1:-1], u[1:-1])





