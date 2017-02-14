import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard # pdb package allows you to interrupt the python script with the keyboard() command
import spatial_discretization as sd # like include .h file in C++, calls another file
import matplotlib.pyplot as plt

def plotC_centered(N,reconstruction):
    deltaX = []
    discretization_error = []
    deltaX.append(1)
    ii = 0
    while deltaX[ii] > 1e-13:
        L = (N * deltaX[ii]) / 2
        x_stencil = np.linspace(-L, L, N+1)
        x_eval = 0.0
        f = np.tanh(x_stencil) * np.sin(5. * x_stencil + 1.5)
        dfdx_analytical = (1 / np.cosh(x_eval) ** 2) * np.sin(5 * x_eval + 1.5) + 5 * np.tanh(x_eval) * np.cos(
        5 * x_eval + 1.5)
        derivation_order = 1
        w_der = sd.Generate_Weights(x_stencil, x_eval, derivation_order)
        dfdx_hat = w_der.dot(f)
        discretization_error.append(np.abs(dfdx_hat - dfdx_analytical))
        deltaX.append(deltaX[ii] * 0.7)
        ii = ii + 1
    deltaX.pop()
    deltaX_1 = [ii ** 1 for ii in deltaX]
    deltaX_2 = [ii ** 2 for ii in deltaX]
    deltaX_3 = [ii ** 3 for ii in deltaX]
    deltaX_4 = [ii ** 4 for ii in deltaX]
    deltaX_5 = [ii ** 5 for ii in deltaX]
    deltaX_6 = [ii ** 6 for ii in deltaX]
    plt.loglog(np.reciprocal(deltaX), discretization_error, linewidth=3, label=reconstruction)
    plt.loglog(np.reciprocal(deltaX), deltaX_1, '-.', linewidth=2, label='n=1')
    plt.loglog(np.reciprocal(deltaX), deltaX_2, '-.', linewidth=2, label='n=2')
    plt.loglog(np.reciprocal(deltaX), deltaX_3, '-.', linewidth=2, label='n=3')
    plt.loglog(np.reciprocal(deltaX), deltaX_4, '-.', linewidth=2, label='n=4')
    plt.loglog(np.reciprocal(deltaX), deltaX_5, '-.', linewidth=2, label='n=5')
    plt.loglog(np.reciprocal(deltaX), deltaX_6, '-.', linewidth=2, label='n=6')
    plt.legend(loc='lower left')
    plt.grid()
    plt.title(r'Truncation Error $\epsilon$ vs. $\Delta x^{-1}$')
    plt.xlabel(r'$\Delta x^{-1}$')
    plt.ylabel(r'$\epsilon$')


def plotC_biased(N,reconstruction):
    deltaX = []
    discretization_error = []
    deltaX.append(1)
    ii = 0
    while deltaX[ii] > 1e-13:
        L = N * deltaX[ii]
        x_stencil = np.linspace(0, L, N+1)
        x_eval = 0.0
        f = np.tanh(x_stencil) * np.sin(5. * x_stencil + 1.5)
        dfdx_analytical = (1 / np.cosh(x_eval) ** 2) * np.sin(5 * x_eval + 1.5) + 5 * np.tanh(x_eval) * np.cos(
        5 * x_eval + 1.5)
        derivation_order = 1
        w_der = sd.Generate_Weights(x_stencil, x_eval, derivation_order)
        dfdx_hat = w_der.dot(f)
        discretization_error.append(np.abs(dfdx_hat - dfdx_analytical))
        deltaX.append(deltaX[ii] * 0.8)
        ii = ii + 1
    deltaX.pop()
    deltaX_1 = [ii ** 1 for ii in deltaX]
    deltaX_2 = [ii ** 2 for ii in deltaX]
    deltaX_3 = [ii ** 3 for ii in deltaX]
    deltaX_4 = [ii ** 4 for ii in deltaX]
    deltaX_5 = [ii ** 5 for ii in deltaX]
    deltaX_6 = [ii ** 6 for ii in deltaX]
    plt.loglog(np.reciprocal(deltaX), discretization_error, linewidth=3, label=reconstruction)
    plt.loglog(np.reciprocal(deltaX), deltaX_1, '-.', linewidth=2, label='n=1')
    plt.loglog(np.reciprocal(deltaX), deltaX_2, '-.', linewidth=2, label='n=2')
    plt.loglog(np.reciprocal(deltaX), deltaX_3, '-.', linewidth=2, label='n=3')
    plt.loglog(np.reciprocal(deltaX), deltaX_4, '-.', linewidth=2, label='n=4')
    plt.loglog(np.reciprocal(deltaX), deltaX_5, '-.', linewidth=2, label='n=5')
    plt.loglog(np.reciprocal(deltaX), deltaX_6, '-.', linewidth=2, label='n=6')
    plt.legend(loc='lower left')
    plt.grid()
    plt.title(r'Truncation Error $\epsilon$ vs. $\Delta x^{-1}$')
    plt.xlabel(r'$\Delta x^{-1}$')
    plt.ylabel(r'$\epsilon$')


def plotS_centered(N,reconstruction):
    deltaX = []
    discretization_error = []
    deltaX.append(1)
    ii = 0
    while deltaX[ii] > 1e-13:
        L = ((N - 1) * deltaX[ii]) / 2
        x_stencil = np.linspace(-L, L, N)
        x_eval = 0.0
        f = np.tanh(x_stencil) * np.sin(5. * x_stencil + 1.5)
        dfdx_analytical = (1 / np.cosh(x_eval) ** 2) * np.sin(5 * x_eval + 1.5) + 5 * np.tanh(x_eval) * np.cos(
        5 * x_eval + 1.5)
        derivation_order = 1
        w_der = sd.Generate_Weights(x_stencil, x_eval, derivation_order)
        dfdx_hat = w_der.dot(f)
        discretization_error.append(np.abs(dfdx_hat - dfdx_analytical))
        deltaX.append(deltaX[ii] * 0.8)
        ii = ii + 1
    deltaX.pop()
    deltaX_1 = [ii ** 1 for ii in deltaX]
    deltaX_2 = [ii ** 2 for ii in deltaX]
    deltaX_3 = [ii ** 3 for ii in deltaX]
    deltaX_4 = [ii ** 4 for ii in deltaX]
    deltaX_5 = [ii ** 5 for ii in deltaX]
    deltaX_6 = [ii ** 6 for ii in deltaX]
    plt.loglog(np.reciprocal(deltaX), discretization_error, linewidth=3, label=reconstruction)
    plt.loglog(np.reciprocal(deltaX), deltaX_1, '-.', linewidth=2, label='n=1')
    plt.loglog(np.reciprocal(deltaX), deltaX_2, '-.', linewidth=2, label='n=2')
    plt.loglog(np.reciprocal(deltaX), deltaX_3, '-.', linewidth=2, label='n=3')
    plt.loglog(np.reciprocal(deltaX), deltaX_4, '-.', linewidth=2, label='n=4')
    plt.loglog(np.reciprocal(deltaX), deltaX_5, '-.', linewidth=2, label='n=5')
    plt.loglog(np.reciprocal(deltaX), deltaX_6, '-.', linewidth=2, label='n=6')
    plt.legend(loc='lower left')
    plt.grid()
    plt.title(r'Truncation Error $\epsilon$ vs. $\Delta x^{-1}$')
    plt.xlabel(r'$\Delta x^{-1}$')
    plt.ylabel(r'$\epsilon$')


def plotS_biased(N,reconstruction):
    deltaX = []
    discretization_error = []
    deltaX.append(1)
    ii = 0
    while deltaX[ii] > 1e-13:
        L = (N - 1) * deltaX[ii]
        x_stencil = np.linspace(-deltaX[ii] / 2, L - deltaX[ii] / 2, N)
        x_eval = 0.0
        f = np.tanh(x_stencil) * np.sin(5. * x_stencil + 1.5)
        dfdx_analytical = (1 / np.cosh(x_eval) ** 2) * np.sin(5 * x_eval + 1.5) + 5 * np.tanh(x_eval) * np.cos(
        5 * x_eval + 1.5)
        derivation_order = 1
        w_der = sd.Generate_Weights(x_stencil, x_eval, derivation_order)
        dfdx_hat = w_der.dot(f)
        discretization_error.append(np.abs(dfdx_hat - dfdx_analytical))
        deltaX.append(deltaX[ii] * 0.8)
        ii = ii + 1
    deltaX.pop()
    deltaX_1 = [ii ** 1 for ii in deltaX]
    deltaX_2 = [ii ** 2 for ii in deltaX]
    deltaX_3 = [ii ** 3 for ii in deltaX]
    deltaX_4 = [ii ** 4 for ii in deltaX]
    deltaX_5 = [ii ** 5 for ii in deltaX]
    deltaX_6 = [ii ** 6 for ii in deltaX]
    plt.loglog(np.reciprocal(deltaX), discretization_error, linewidth=3, label=reconstruction)
    plt.loglog(np.reciprocal(deltaX), deltaX_1, '-.', linewidth=2, label='n=1')
    plt.loglog(np.reciprocal(deltaX), deltaX_2, '-.', linewidth=2, label='n=2')
    plt.loglog(np.reciprocal(deltaX), deltaX_3, '-.', linewidth=2, label='n=3')
    plt.loglog(np.reciprocal(deltaX), deltaX_4, '-.', linewidth=2, label='n=4')
    plt.loglog(np.reciprocal(deltaX), deltaX_5, '-.', linewidth=2, label='n=5')
    plt.loglog(np.reciprocal(deltaX), deltaX_6, '-.', linewidth=2, label='n=6')
    plt.legend(loc='lower left')
    plt.grid()
    plt.title(r'Truncation Error $\epsilon$ vs. $\Delta x^{-1}$')
    plt.xlabel(r'$\Delta x^{-1}$')
    plt.ylabel(r'$\epsilon$')
