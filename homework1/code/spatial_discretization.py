import numpy as np
import scipy.sparse as scysparse
import sys
from pdb import set_trace as keyboard

def Generate_Weights(x_stencil, x_eval, der_order):
    if x_stencil.ndim > 1:
        sys.exit("stencil array is not a 1D numpy array")

    derivation_order = int(der_order)  # making sure derivation order is integer
    polynomial_order = len(x_stencil) - 1

    weights = np.zeros(x_stencil.shape)
    N = x_stencil.size

    for ix, x in enumerate(x_stencil):
        base_func = np.zeros(N, )
        base_func[ix] = 1.0
        poly_coefs = np.polyfit(x_stencil, base_func, polynomial_order)
        weights[ix] = np.polyval(np.polyder(poly_coefs, der_order), x_eval)

    return weights

############################################################
############################################################

def Generate_Spatial_Operators(x_mesh,order_scheme,der_order):

    N = x_mesh.size

    D = scysparse.lil_matrix((N, N), dtype=np.float64)

    if order_scheme == "3rd-order":

        for i, x_eval in enumerate(x_mesh):
        
            if i == 0:
                x_stencil = x_mesh[:4] # this includes points 0,1,2
                D[i, :4] = Generate_Weights(x_stencil, x_eval, der_order)
            elif i == N-1:
                x_stencil = x_mesh[-4:]
                D[i, -4:] = Generate_Weights(x_stencil, x_eval, der_order)
            elif i == N-2:
                x_stencil = x_mesh[-4:]
                D[i, -4:] = Generate_Weights(x_stencil, x_eval, der_order)
            else:
                x_stencil = x_mesh[i-1:i+3]
                D[i, i-1:i+3] = Generate_Weights(x_stencil, x_eval, der_order)

    if order_scheme == "5th-order":

        for i, x_eval in enumerate(x_mesh):

            if i == 0 or i == 1:
                x_stencil = x_mesh[:6]  # this includes points 0,1,2,3,4,5
                D[i, :6] = Generate_Weights(x_stencil, x_eval, der_order)

            elif i == N - 1 or i == N-2:
                x_stencil = x_mesh[-6:]
                D[i, -6:] = Generate_Weights(x_stencil, x_eval, der_order)

            elif i == N-3:
                x_stencil = x_mesh[-6:]
                D[i, -6:] = Generate_Weights(x_stencil, x_eval, der_order)

            else:
                x_stencil = x_mesh[i - 2:i + 4]
                D[i, i - 2:i + 4] = Generate_Weights(x_stencil, x_eval, der_order)

    return D.tocsr()