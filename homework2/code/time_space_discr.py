from user_input import *
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


def get_u(Nx, advection_scheme, time_advancement):
    # Lx = 1.
    # c_x = 5.  # (linear) convection speed
    # alpha = 5.  # diffusion coefficients
    # gamma1 = 1.0
    # gamma2 = 1.0
    # c1 = 5.0
    # c2 = 5.0
    # m = 2.0
    # w1 = 2.0 * np.pi / Lx
    # w2 = 2.0 * m * np.pi / Lx
    # Tf = 1 / (w2 ** 2 * alpha)  # one complete cycle
    machine_epsilon = np.finfo(float).eps

    CFL = 0.5  # Courant-Friedrichs-Lewy number (i.e. dimensionless time step)

    ## Diffusion Scheme
    diffusion_scheme = "2nd-order-central"  # always-second-order central

    def u_initial(X):
        return c1 * np.sin(w1 * X - gamma1) - c2 * np.cos(w2 * X - gamma2)

    xx = np.linspace(0., Lx, Nx + 1)
    # actual mesh points are off the boundaries x=0, x=Lx
    # non-periodic boundary conditions created with ghost points
    x_mesh = 0.5 * (xx[:-1] + xx[1:])
    dx = np.diff(xx)[0]
    dx2 = dx * dx

    #########################################
    ######## Preprocessing Stage ############

    # for linear advection/diffusion time step is a function
    # of c,alpha,dx only; we use reference limits, ballpark
    # estimates for Explicit Euler
    dt_max_advective = dx / (abs(c_x) + machine_epsilon)
    dt_max_diffusive = dx2 / (alpha + machine_epsilon)
    dt_max = np.min([dt_max_advective, dt_max_diffusive])
    dt = CFL * dt_max


    # Creating identity matrix
    Ieye = scysparse.identity(Nx)

    # Creating first derivative
    Dx = spatial_discretization.Generate_Spatial_Operators(x_mesh, advection_scheme, derivation_order=1)
    # Creating second derivative
    D2x2 = spatial_discretization.Generate_Spatial_Operators(x_mesh, diffusion_scheme, derivation_order=2)

    # Creating A,B matrices such that:
    #     A*u^{n+1} = B*u^{n} + q
    if time_advancement == "Explicit-Euler":
        A = Ieye
        B = Ieye - dt * c_x * Dx + dt * alpha * D2x2
    if time_advancement == "Crank-Nicolson":
        adv_diff_Op = -dt * c_x * Dx + dt * alpha * D2x2
        A = Ieye - 0.5 * adv_diff_Op
        B = Ieye + 0.5 * adv_diff_Op

    # forcing csr ordering..
    A, B = scysparse.csr_matrix(A), scysparse.csr_matrix(B)
    #########################################
    ########## Time integration #############

    u = u_initial(x_mesh)  # initializing solution

    time = 0.0
    # cfl = dt * alpha / (dx ** 2)
    # cfl = dt * c_x / dx
    while time < Tf:
        time += dt

        # Update solution
        # solving : A*u^{n+1} = B*u^{n} + q
        # where q is zero for periodic and zero source terms
        u = spysparselinalg.spsolve(A, B.dot(u))
        # A_fac = spysparselinalg.factorized(A)
        # u = A_fac(B.dot(u))
    # A, B = scysparse.csr_matrix(A), scysparse.csr_matrix(B)
    return u, A, B


def get_u_anal(Nx):
    # Lx = 1.
    # c_x = 5.  # (linear) convection speed
    # #c_x = 100
    # alpha = 5.  # diffusion coefficients
    # gamma1 = 1.0
    # gamma2 = 1.0
    # c1 = 5.0
    # c2 = 5.0
    # m = 2.0
    # w1 = 2.0 * np.pi / Lx
    # w2 = 2.0 * m * np.pi / Lx
    # Tf = 1 / (w2 ** 2 * alpha)  # one complete cycle
    xx = np.linspace(0., Lx, Nx + 1)
    x_mesh = 0.5 * (xx[:-1] + xx[1:])

    def u_analytical(X, t):
        return c1 * np.exp(-w1 ** 2 * alpha * t) * np.sin(w1 * (X - c_x * t) - gamma1) - \
               c2 * np.exp(-w2 ** 2 * alpha * t) * np.cos(w2 * (X - c_x * t) - gamma2)

    u_anal = u_analytical(x_mesh, Tf)

    return u_anal


def get_u_fixed_t(Nx, advection_scheme, time_advancement, N_max):
    machine_epsilon = np.finfo(float).eps
    # CFL = 0.4  # Courant-Friedrichs-Lewy number (i.e. dimensionless time step)

    ## Diffusion Scheme
    diffusion_scheme = "2nd-order-central"  # always-second-order central

    def u_initial(X):
        return c1 * np.sin(w1 * X - gamma1) - c2 * np.cos(w2 * X - gamma2)

    xx = np.linspace(0., Lx, Nx + 1)
    # actual mesh points are off the boundaries x=0, x=Lx
    # non-periodic boundary conditions created with ghost points
    x_mesh = 0.5 * (xx[:-1] + xx[1:])
    dx = np.diff(xx)[0]
    dx2 = dx * dx

    #########################################
    ######## Preprocessing Stage ############

    # for linear advection/diffusion time step is a function
    # of c,alpha,dx only; we use reference limits, ballpark
    # estimates for Explicit Euler
    # dt_max_advective = dx / (c_x + machine_epsilon)
    # dt_max_diffusive = dx2 / (alpha + machine_epsilon)
    # dt_max = np.min([dt_max_advective, dt_max_diffusive])
    # dt = 0.1 / (c_x + machine_epsilon) * 0.9
    # dt = (Lx / N_max) * 0.99 / c_x
    dt = Tf / 100.

    # Creating identity matrix
    Ieye = scysparse.identity(Nx)

    # Creating first derivative
    Dx = spatial_discretization.Generate_Spatial_Operators(x_mesh, advection_scheme, derivation_order=1)
    # Creating second derivative
    D2x2 = spatial_discretization.Generate_Spatial_Operators(x_mesh, diffusion_scheme, derivation_order=2)

    # Creating A,B matrices such that:
    #     A*u^{n+1} = B*u^{n} + q
    if time_advancement == "Explicit-Euler":
        A = Ieye
        B = Ieye - dt * c_x * Dx + dt * alpha * D2x2
    if time_advancement == "Crank-Nicolson":
        adv_diff_Op = -dt * c_x * Dx + dt * alpha * D2x2
        A = Ieye - 0.5 * adv_diff_Op
        B = Ieye + 0.5 * adv_diff_Op

    # forcing csr ordering..
    A, B = scysparse.csr_matrix(A), scysparse.csr_matrix(B)
    #########################################
    ########## Time integration #############

    u = u_initial(x_mesh)  # initializing solution

    time = 0.0
    # cfl = dt * alpha / (dx ** 2)
    # cfl = dt * c_x / dx
    while time < Tf:
        time += dt

        # Update solution
        # solving : A*u^{n+1} = B*u^{n} + q
        # where q is zero for periodic and zero source terms
        u = spysparselinalg.spsolve(A, B.dot(u))
        # A_fac = spysparselinalg.factorized(A)
        # u = A_fac(B.dot(u))
    # A, B = scysparse.csr_matrix(A), scysparse.csr_matrix(B)
    return u, A, B


def get_u_fixed_x(dt, advection_scheme, time_advancement):
    machine_epsilon = np.finfo(float).eps
    # CFL = 0.4  # Courant-Friedrichs-Lewy number (i.e. dimensionless time step)

    ## Diffusion Scheme
    diffusion_scheme = "2nd-order-central"  # always-second-order central

    def u_initial(X):
        return c1 * np.sin(w1 * X - gamma1) - c2 * np.cos(w2 * X - gamma2)

    Nx = 100
    xx = np.linspace(0., Lx, Nx + 1)
    # actual mesh points are off the boundaries x=0, x=Lx
    # non-periodic boundary conditions created with ghost points
    x_mesh = 0.5 * (xx[:-1] + xx[1:])
    #dx = np.diff(xx)[0]
    # dx2 = dx * dx

    ##########################################
    ######## Preprocessing Stage ############

    # for linear advection/diffusion time step is a function
    # of c,alpha,dx only; we use reference limits, ballpark
    # estimates for Explicit Euler
    # dt_max_advective = dx / (c_x + machine_epsilon)
    # dt_max_diffusive = dx2 / (alpha + machine_epsilon)
    # dt_max = np.min([dt_max_advective, dt_max_diffusive])
    # dt = 0.1 / (c_x + machine_epsilon) * 0.9

    # Creating identity matrix
    Ieye = scysparse.identity(Nx)

    # Creating first derivative
    Dx = spatial_discretization.Generate_Spatial_Operators(x_mesh, advection_scheme, derivation_order=1)
    # Creating second derivative
    D2x2 = spatial_discretization.Generate_Spatial_Operators(x_mesh, diffusion_scheme, derivation_order=2)

    # Creating A,B matrices such that:
    #     A*u^{n+1} = B*u^{n} + q
    if time_advancement == "Explicit-Euler":
        A = Ieye
        B = Ieye - dt * c_x * Dx + dt * alpha * D2x2
    if time_advancement == "Crank-Nicolson":
        adv_diff_Op = -dt * c_x * Dx + dt * alpha * D2x2
        A = Ieye - 0.5 * adv_diff_Op
        B = Ieye + 0.5 * adv_diff_Op

    # forcing csr ordering..
    A, B = scysparse.csr_matrix(A), scysparse.csr_matrix(B)
    #########################################
    ########## Time integration #############

    u = u_initial(x_mesh)  # initializing solution

    time = 0.0
    # cfl = dt * alpha / (dx ** 2)
    # cfl = dt * c_x / dx
    while time < Tf:
        time += dt

        # Update solution
        # solving : A*u^{n+1} = B*u^{n} + q
        # where q is zero for periodic and zero source terms
        u = spysparselinalg.spsolve(A, B.dot(u))
        # A_fac = spysparselinalg.factorized(A)
        # u = A_fac(B.dot(u))
    # A, B = scysparse.csr_matrix(A), scysparse.csr_matrix(B)
    return u, Nx


def get_r(Cc, Ca, time_advancement, advection_scheme):
    c_x = 100.
    alpha = 1.
    dx = (Cc / c_x) / (Ca / alpha)
    dt = Cc * dx / c_x
    Nx = int(Lx / dx) + 1
    xx = np.linspace(0., Lx, Nx + 1)
    x_mesh = 0.5 * (xx[:-1] + xx[1:])

    Ieye = scysparse.identity(Nx)

    ## Diffusion Scheme
    diffusion_scheme = "2nd-order-central"  # always-second-order central
    # Creating first derivative
    Dx = spatial_discretization.Generate_Spatial_Operators(x_mesh, advection_scheme, derivation_order=1)
    # Creating second derivative
    D2x2 = spatial_discretization.Generate_Spatial_Operators(x_mesh, diffusion_scheme, derivation_order=2)

    if time_advancement == "Explicit-Euler":
        A = Ieye
        B = Ieye - dt * c_x * Dx + dt * alpha * D2x2
    if time_advancement == "Crank-Nicolson":
        adv_diff_Op = -dt * c_x * Dx + dt * alpha * D2x2
        A = Ieye - 0.5 * adv_diff_Op
        B = Ieye + 0.5 * adv_diff_Op

    A, B = scysparse.csr_matrix(A), scysparse.csr_matrix(B)
    T = (scylinalg.inv(A.todense())).dot(B.todense())  # T = A^{-1}*B
    lambdas, _ = scylinalg.eig(T)
    r = np.max(abs(lambdas))

    return r
