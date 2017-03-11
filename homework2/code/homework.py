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
import time_space_discr as tsd
import time_space_discr_nonpd as tsd_np
from plot_uf import plot_uf
from get_rms import get_rms
import plot_contour as plt_cont
import p3b as p3b
from p3_get_u import p3_get_u


Problem_1 = False
Problem_2b = False
Problem_2c = True
Problem_2d = False
Problem_2e = False
Problem_3a = False
Problem_3b = False

if Problem_1:
    # Overflow happens when your variable gets too big for the variable type. You can use numpy seterr to have it
    # raise an exception for overflow, then use a try/except statement. I think there are other ways of doing it too.
    pi_f16 = np.float16(np.pi)
    ii = 1
    pi_star_f16 = np.float16(pi_f16 ** (ii + 1))
    pi_hat_f16 = np.float16(pi_star_f16 * np.float16(1 / pi_f16) ** ii)
    error_f16 = []
    error_f16 = np.float64(abs(pi_f16 - np.float64(pi_hat_f16)))

    while np.isfinite(pi_star_f16 * pi_f16):

        ii += 1
        pi_star_f16 = np.float16(pi_f16 ** (ii + 1))
        pi_hat_f16 = np.float16(pi_star_f16 * np.float16(1 / pi_f16) ** ii)
        error_f16_new = np.float64(abs(pi_f16 - np.float64(pi_hat_f16)))
        error_f16 = np.append(error_f16, error_f16_new)
    plt.figure(18)
    plt.plot(range(ii), error_f16)
    plt.xlabel(r"$n_{max}$")
    plt.ylabel(r"$\epsilon$")
    plt.savefig("../report/figures/p1_f16.png")

    ii = 1
    pi_f32 = np.float32(np.pi)
    pi_star_f32 = np.float32(pi_f32 ** (ii + 1))
    pi_hat_f32 = np.float32(pi_star_f32 * np.float32(1 / pi_f32) ** ii)
    error_f32 = []
    error_f32 = np.float64(abs(pi_f32 - np.float64(pi_hat_f32)))

    while np.isfinite(pi_star_f32 * pi_f32):

        ii += 1
        pi_star_f32 = np.float32(pi_f32 ** (ii + 1))
        pi_hat_f32 = np.float32(pi_star_f32 * np.float32(1 / pi_f32) ** ii)
        error_f32_new = np.float64(abs(pi_f32 - np.float64(pi_hat_f32)))
        error_f32 = np.append(error_f32, error_f32_new)
    plt.figure(19)
    plt.plot(range(ii), error_f32)
    plt.xlabel(r"$n_{max}$")
    plt.ylabel(r"$\epsilon$")
    plt.savefig("../report/figures/p1_f32.png")

    ii = 1
    pi_f64 = np.float64(np.pi)
    pi_star_f64 = np.float64(pi_f64 ** (ii + 1))
    pi_hat_f64 = np.float64(pi_star_f64 * np.float64(1 / pi_f64) ** ii)
    error_f64 = []
    error_f64 = np.float64(abs(pi_f64 - np.float64(pi_hat_f64)))

    while np.isfinite(pi_star_f64 * pi_f64):

        ii += 1
        pi_star_f64 = np.float64(pi_f64 ** (ii + 1))
        pi_hat_f64 = np.float64(pi_star_f64 * np.float64(1 / pi_f64) ** ii)
        error_f64_new = np.float64(abs(pi_f64 - np.float64(pi_hat_f64)))
        error_f64 = np.append(error_f64, error_f64_new)
    plt.figure(20)
    plt.plot(range(ii), error_f64)
    plt.xlabel(r"$n_{max}$")
    plt.ylabel(r"$\epsilon$")
    plt.savefig("../report/figures/p1_f64.png")



if Problem_2b:
    ############################
    ######## Part b ############

    best_scheme = []
    least_scheme = []
    u_anal = tsd.get_u_anal(100)
    u_dict_10 = {}
    u_dict_25 = {}
    u_dict_50 = {}
    u_dict_100 = {}

    u_dict_10['c2nd_ee'] = tsd.get_u(10, "2nd-order-central", "Explicit-Euler")[0]
    u_dict_10['c2nd_cn'] = tsd.get_u(10, "2nd-order-central", "Crank-Nicolson")[0]
    u_dict_10['u1st_ee'] = tsd.get_u(10, "1st-order-upwind", "Explicit-Euler")[0]
    u_dict_10['u1st_cn'] = tsd.get_u(10, "1st-order-upwind", "Crank-Nicolson")[0]
    u_dict_10['u2nd_ee'] = tsd.get_u(10, "2nd-order-upwind", "Explicit-Euler")[0]
    u_dict_10['u2nd_cn'] = tsd.get_u(10, "2nd-order-upwind", "Crank-Nicolson")[0]

    plot_uf(10, u_dict_10, u_anal, 1)

    rms_10 = {k: get_rms(10, v) for k, v in u_dict_10.items()}
    best_scheme.append(min(rms_10, key=lambda x: rms_10[x]))
    least_scheme.append(max(rms_10, key=lambda x: rms_10[x]))

    plt.savefig("../report/figures/p2b_n10.png")

    u_dict_25['c2nd_ee'] = tsd.get_u(25, "2nd-order-central", "Explicit-Euler")[0]
    u_dict_25['c2nd_cn'] = tsd.get_u(25, "2nd-order-central", "Crank-Nicolson")[0]
    u_dict_25['u1st_ee'] = tsd.get_u(25, "1st-order-upwind", "Explicit-Euler")[0]
    u_dict_25['u1st_cn'] = tsd.get_u(25, "1st-order-upwind", "Crank-Nicolson")[0]
    u_dict_25['u2nd_ee'] = tsd.get_u(25, "2nd-order-upwind", "Explicit-Euler")[0]
    u_dict_25['u2nd_cn'] = tsd.get_u(25, "2nd-order-upwind", "Crank-Nicolson")[0]
    plot_uf(25, u_dict_25, u_anal, 2)

    rms_25 = {k: get_rms(25, v) for k, v in u_dict_25.items()}
    best_scheme.append(min(rms_25, key=lambda x: rms_25[x]))
    least_scheme.append(max(rms_25, key=lambda x: rms_25[x]))

    plt.savefig("../report/figures/p2b_n25.png")

    u_dict_50['c2nd_ee'] = tsd.get_u(50, "2nd-order-central", "Explicit-Euler")[0]
    u_dict_50['c2nd_cn'] = tsd.get_u(50, "2nd-order-central", "Crank-Nicolson")[0]
    u_dict_50['u1st_ee'] = tsd.get_u(50, "1st-order-upwind", "Explicit-Euler")[0]
    u_dict_50['u1st_cn'] = tsd.get_u(50, "1st-order-upwind", "Crank-Nicolson")[0]
    u_dict_50['u2nd_ee'] = tsd.get_u(50, "2nd-order-upwind", "Explicit-Euler")[0]
    u_dict_50['u2nd_cn'] = tsd.get_u(50, "2nd-order-upwind", "Crank-Nicolson")[0]
    plot_uf(50, u_dict_50, u_anal, 3)

    rms_50 = {k: get_rms(50, v) for k, v in u_dict_50.items()}
    best_scheme.append(min(rms_50, key=lambda x: rms_50[x]))
    least_scheme.append(max(rms_50, key=lambda x: rms_50[x]))

    plt.savefig("../report/figures/p2b_n50.png")

    u_dict_100['c2nd_ee'] = tsd.get_u(100, "2nd-order-central", "Explicit-Euler")[0]
    u_dict_100['c2nd_cn'] = tsd.get_u(100, "2nd-order-central", "Crank-Nicolson")[0]
    u_dict_100['u1st_ee'] = tsd.get_u(100, "1st-order-upwind", "Explicit-Euler")[0]
    u_dict_100['u1st_cn'] = tsd.get_u(100, "1st-order-upwind", "Crank-Nicolson")[0]
    u_dict_100['u2nd_ee'] = tsd.get_u(100, "2nd-order-upwind", "Explicit-Euler")[0]
    u_dict_100['u2nd_cn'] = tsd.get_u(100, "2nd-order-upwind", "Crank-Nicolson")[0]
    plot_uf(100, u_dict_100, u_anal, 4)

    rms_100 = {k: get_rms(100, v) for k, v in u_dict_100.items()}
    best_scheme.append(min(rms_100, key=lambda x: rms_100[x]))
    least_scheme.append(max(rms_100, key=lambda x: rms_100[x]))

    plt.savefig("../report/figures/p2b_n100.png")

    print "best scheme is", best_scheme
    print "least scheme is", least_scheme
    # best scheme is ['c2nd_cn', 'c2nd_cn', 'c2nd_cn', 'c2nd_cn']
    # least scheme is ['u1st_ee', 'u1st_ee', 'u1st_ee', 'u1st_ee']

if Problem_2c:
    ############################
    ######## Part c ############

    # fix dt
    Nx = np.array([10, 50, 100, 500, 1000, 5000, 10000])
    rms_c2nd_cn = np.zeros(len(Nx))
    rms_u1st_cn = np.zeros(len(Nx))
    rms_u2nd_cn = np.zeros(len(Nx))

    for iN, N in enumerate(Nx):
        u_temp_2c = tsd.get_u_fixed_t(N, "2nd-order-central", "Crank-Nicolson", Nx[len(Nx) - 1])[0]
        rms_c2nd_cn[iN] = get_rms(N, u_temp_2c)
        u_temp_1u = tsd.get_u_fixed_t(N, "1st-order-upwind", "Crank-Nicolson", Nx[len(Nx) - 1])[0]
        rms_u1st_cn[iN] = get_rms(N, u_temp_1u)
        u_temp_2u = tsd.get_u_fixed_t(N, "2nd-order-upwind", "Crank-Nicolson", Nx[len(Nx) - 1])[0]
        rms_u2nd_cn[iN] = get_rms(N, u_temp_2u)
    plt.figure(5)
    plt.loglog(Nx, rms_u2nd_cn, linewidth=2, label="2nd-order-upwind")
    plt.loglog(Nx, rms_c2nd_cn, linewidth=2, label="2nd-order-central")
    plt.loglog(Nx, rms_u1st_cn, linewidth=2, label="1st-order-upwind")
    plt.loglog(Nx, (1. / Nx) ** 2, '--', linewidth=2, label="2nd-order")
    plt.loglog(Nx, (1. / Nx) ** 1, '--', linewidth=2, label="1st-order")
    plt.loglog(Nx, (1. / Nx) ** 3, '--', linewidth=2, label="3rd-order")
    plt.legend(loc=1, fontsize='x-small')
    plt.grid()
    plt.xlabel(r'$N_{x}$')
    plt.ylabel('Root-Mean-Square Error')
    plt.savefig("../report/figures/p2c_fixed_dt.png")
    # fix dx
    Nt = []
    Nt = [Tf / ii for ii in np.array([30, 40, 50, 90, 100, 200, 500, 1000, 10000, 100000])]
    rms_c2nd_cn = np.zeros(len(Nt))
    rms_u1st_cn = np.zeros(len(Nt))
    rms_u2nd_cn = np.zeros(len(Nt))

    for it, dt in enumerate(Nt):
        u_temp_2c = tsd.get_u_fixed_x(dt, "2nd-order-central", "Crank-Nicolson")
        rms_c2nd_cn[it] = get_rms(u_temp_2c[1], u_temp_2c[0])
        u_temp_1u = tsd.get_u_fixed_x(dt, "1st-order-upwind", "Crank-Nicolson")
        rms_u1st_cn[it] = get_rms(u_temp_1u[1], u_temp_1u[0])
        u_temp_2u = tsd.get_u_fixed_x(dt, "2nd-order-upwind", "Crank-Nicolson")
        rms_u2nd_cn[it] = get_rms(u_temp_2u[1], u_temp_2u[0])

    Nt = np.asarray(Nt)
    plt.figure(6)
    plt.loglog(1. / Nt, rms_c2nd_cn, linewidth=1, label="2nd-order-central")
    plt.loglog(1. / Nt, rms_u1st_cn, linewidth=1, label="1st-order-upwind")
    plt.loglog(1. / Nt, rms_u2nd_cn, linewidth=1, label="2nd-order-upwind")
    plt.loglog(1. / Nt, Nt ** 1, '--', linewidth=1, label="1st-order")
    plt.loglog(1. / Nt, Nt ** 2, '--', linewidth=1, label="2nd-order")
    plt.loglog(1. / Nt, Nt ** 3, '--', linewidth=1, label="3rd-order")
    plt.legend(loc=3, fontsize='x-small')
    plt.grid()
    plt.xlabel(r'$\Delta t^{-1}$')
    plt.ylabel('Root-Mean-Square Error')
    plt.savefig("../report/figures/p2c_fixed_dx.png")

if Problem_2d:
    ############################
    ######## Part d ############
    A_dict = {}
    A_dict['c2nd_ee'] = tsd.get_u(10, "2nd-order-central", "Explicit-Euler")[1]
    A_dict['c2nd_cn'] = tsd.get_u(10, "2nd-order-central", "Crank-Nicolson")[1]
    A_dict['u1st_ee'] = tsd.get_u(10, "1st-order-upwind", "Explicit-Euler")[1]
    A_dict['u1st_cn'] = tsd.get_u(10, "1st-order-upwind", "Crank-Nicolson")[1]
    A_dict['u2nd_ee'] = tsd.get_u(10, "2nd-order-upwind", "Explicit-Euler")[1]
    A_dict['u2nd_cn'] = tsd.get_u(10, "2nd-order-upwind", "Crank-Nicolson")[1]
    B_dict = {}
    B_dict['c2nd_ee'] = tsd.get_u(10, "2nd-order-central", "Explicit-Euler")[2]
    B_dict['c2nd_cn'] = tsd.get_u(10, "2nd-order-central", "Crank-Nicolson")[2]
    B_dict['u1st_ee'] = tsd.get_u(10, "1st-order-upwind", "Explicit-Euler")[2]
    B_dict['u1st_cn'] = tsd.get_u(10, "1st-order-upwind", "Crank-Nicolson")[2]
    B_dict['u2nd_ee'] = tsd.get_u(10, "2nd-order-upwind", "Explicit-Euler")[2]
    B_dict['u2nd_cn'] = tsd.get_u(10, "2nd-order-upwind", "Crank-Nicolson")[2]
    plt.figure(7)
    plt.subplot(1, 2, 1)
    plt.spy(A_dict['c2nd_ee'])
    plt.xlabel('A matrix')
    plt.subplot(1, 2, 2)
    plt.spy(B_dict['c2nd_ee'])
    plt.xlabel('B matrix')
    plt.savefig("../report/figures/p2d_c2nd_ee.png")
    plt.figure(8)
    plt.subplot(1, 2, 1)
    plt.spy(A_dict['c2nd_cn'])
    plt.xlabel('A matrix')
    plt.subplot(1, 2, 2)
    plt.spy(B_dict['c2nd_cn'])
    plt.xlabel('B matrix')
    plt.savefig("../report/figures/p2d_c2nd_cn.png")
    plt.figure(9)
    plt.subplot(1, 2, 1)
    plt.spy(A_dict['u1st_ee'])
    plt.xlabel('A matrix')
    plt.subplot(1, 2, 2)
    plt.spy(B_dict['u1st_ee'])
    plt.xlabel('B matrix')
    plt.savefig("../report/figures/p2d_u1st_ee.png")
    plt.figure(10)
    plt.subplot(1, 2, 1)
    plt.spy(A_dict['u1st_cn'])
    plt.xlabel('A matrix')
    plt.subplot(1, 2, 2)
    plt.spy(B_dict['u1st_cn'])
    plt.xlabel('B matrix')
    plt.savefig("../report/figures/p2d_1st_cn.png")
    plt.figure(11)
    plt.subplot(1, 2, 1)
    plt.spy(A_dict['u2nd_ee'])
    plt.xlabel('A matrix')
    plt.subplot(1, 2, 2)
    plt.spy(B_dict['u2nd_ee'])
    plt.xlabel('B matrix')
    plt.savefig("../report/figures/p2d_u2nd_ee.png")
    plt.figure(12)
    plt.subplot(1, 2, 1)
    plt.spy(A_dict['u2nd_cn'])
    plt.xlabel('A matrix')
    plt.subplot(1, 2, 2)
    plt.spy(B_dict['u2nd_cn'])
    plt.xlabel('B matrix')
    plt.savefig("../report/figures/p2d_u2nd_cn.png")

if Problem_2e:
    ############################
    ######## Part e ############

    plt_cont.plot_contour("Explicit-Euler", "2nd-order-central", 13)
    plt.savefig("../report/figures/p2e_contours_c2nd.png")
    plt_cont.plot_contour("Explicit-Euler", "1st-order-upwind", 14)
    plt.savefig("../report/figures/p2e_contours_u1st.png")
    plt_cont.plot_contour("Explicit-Euler", "2nd-order-upwind", 15)
    plt.savefig("../report/figures/p2e_contours_u2nd.png")

if Problem_3a:
    # steady state
    delta_ss = tsd_np.p3_a_ply_get_delta(Nx=50, Lx=1., c_x=10., alpha=2., beta=10., w=100000., a=1., num_curve=20,
                                         order=301, get_del=True)
    plt.savefig("../report/figures/p3a_ss.png")
    print "delta is:", delta_ss
    # original
    u_org = p3_get_u(Nx=50, Lx=1., c_x=10., alpha=2., beta=10., w=100000., a=1.)[0]
    x_mesh = p3_get_u(Nx=50, Lx=1., c_x=10., alpha=2., beta=10., w=100000., a=1.)[1]
    # plt.savefig("../report/figures/p3a_change_original.png")
    # change c
    u_change_c_1 = p3_get_u(Nx=50, Lx=1., c_x=20., alpha=2., beta=10., w=100000., a=1.)[0]
    u_change_c_2 = p3_get_u(Nx=50, Lx=1., c_x=30., alpha=2., beta=10., w=100000., a=1.)[0]
    plt.figure(302)
    plt.plot(x_mesh[1:-1], u_org[1:-1], linewidth=2, label="c=10")
    plt.plot(x_mesh[1:-1], u_change_c_1[1:-1], linewidth=2, label="c=20")
    plt.plot(x_mesh[1:-1], u_change_c_2[1:-1], linewidth=2, label="c=30")
    plt.legend(loc=1, fontsize=12)
    plt.grid('on', which='both')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$", rotation=90)
    plt.savefig("../report/figures/p3a_change_c.png")
    # change alpha
    u_change_alpha_1 = p3_get_u(Nx=50, Lx=1., c_x=10., alpha=5., beta=10., w=100000., a=1.)[0]
    u_change_alpha_2 = p3_get_u(Nx=50, Lx=1., c_x=10., alpha=10., beta=10., w=100000., a=1.)[0]
    plt.figure(303)
    plt.plot(x_mesh[1:-1], u_org[1:-1], linewidth=2, label=r"$\alpha=2$")
    plt.plot(x_mesh[1:-1], u_change_alpha_1[1:-1], linewidth=2, label=r"$\alpha=5$")
    plt.plot(x_mesh[1:-1], u_change_alpha_2[1:-1], linewidth=2, label=r"$\alpha=10$")
    plt.legend(loc=1, fontsize=12)
    plt.grid('on', which='both')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$", rotation=90)
    plt.savefig("../report/figures/p3a_change_alpha.png")
    # change beta
    u_change_beta_1 = p3_get_u(Nx=50, Lx=1., c_x=10., alpha=2., beta=100., w=100000., a=1.)[0]
    u_change_beta_2 = p3_get_u(Nx=50, Lx=1., c_x=10., alpha=2., beta=1000., w=100000., a=1.)[0]
    plt.figure(304)
    plt.plot(x_mesh[1:-1], u_org[1:-1], linewidth=2, label=r"$\beta=10$")
    plt.plot(x_mesh[1:-1], u_change_beta_1[1:-1], linewidth=2, label=r"$\beta=100$")
    plt.plot(x_mesh[1:-1], u_change_beta_2[1:-1], linewidth=2, label=r"$\beta=1000$")
    plt.legend(loc=1, fontsize=12)
    plt.grid('on', which='both')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$", rotation=90)
    plt.savefig("../report/figures/p3a_change_beta.png")
    # change w
    u_change_omega_1 = p3_get_u(Nx=50, Lx=1., c_x=10., alpha=2., beta=10., w=10000., a=1.)[0]
    u_change_omega_2 = p3_get_u(Nx=50, Lx=1., c_x=10., alpha=2., beta=10., w=1000., a=1.)[0]
    plt.figure(305)
    plt.plot(x_mesh[1:-1], u_org[1:-1], linewidth=2, label=r"$\omega=100000$")
    plt.plot(x_mesh[1:-1], u_change_omega_1[1:-1], linewidth=2, label=r"$\omega=10000$")
    plt.plot(x_mesh[1:-1], u_change_omega_2[1:-1], linewidth=2, label=r"$\omega=1000$")
    plt.legend(loc=1, fontsize=12)
    plt.grid('on', which='both')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$", rotation=90)
    plt.savefig("../report/figures/p3a_change_w.png")
    # change a
    u_change_a_1 = p3_get_u(Nx=50, Lx=1., c_x=10., alpha=2., beta=10., w=100000., a=2.)[0]
    u_change_a_2 = p3_get_u(Nx=50, Lx=1., c_x=10., alpha=2., beta=10., w=100000., a=3.)[0]
    plt.figure(306)
    plt.plot(x_mesh[1:-1], u_org[1:-1], linewidth=2, label=r"$a=1$")
    plt.plot(x_mesh[1:-1], u_change_a_1[1:-1], linewidth=2, label=r"$a=2$")
    plt.plot(x_mesh[1:-1], u_change_a_2[1:-1], linewidth=2, label=r"$a=3$")
    plt.legend(loc=1, fontsize=12)
    plt.grid('on', which='both')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$", rotation=90)
    plt.savefig("../report/figures/p3a_change_a.png")
if Problem_3b:
    # delta/L = f(c/(L*w), alpha/(L^2 * w), beta/w, a/Lw)
    amp = np.array([1., 2., 4., 8., 16.])
    Lx_org = 1.
    c_x_org = 10.
    w_org = 100000.
    alpha_org = 2.
    beta_org = 10.
    a_org = 1.
    c_x_amp = 4
    w_amp = 2
    alpha_amp = 8
    beta_amp = 2
    a_amp = 4
    # Lx = np.array([1., 2., 4., 8., 16.])
    # c_x = np.array([1., 4., 16., 64., 256.]) * 10
    # w = np.array([1., 2., 4., 8., 16.]) * 100000
    # alpha = np.array([1., 8., 64., 512., 4096.]) * 2
    # beta = np.array([1., 2., 4., 8., 16.]) * 10
    # a = np.array([1., 4., 16., 64., 256.])
    delta = np.zeros(len(amp))
    for ii in xrange(len(amp)):
        amp_it = amp[ii]
        delta[ii] = p3b.p3_b_ply_get_delta(50, Lx_org * amp_it, c_x_org * amp_it * c_x_amp, alpha_org * amp_it * alpha_amp,
                beta_org * amp_it * beta_amp, w_org * amp_it * w_amp, a_org * amp_it * a_amp, num_curve=10, get_del=True)
    print delta
    plt.figure(400)
    plt.plot(amp, delta, linewidth=2)
    plt.grid('on', which='both')
    plt.xlabel(r"$L_{x}$")
    plt.ylabel(r"$\delta$", rotation=90)
    plt.savefig("../report/figures/p3b.png")






