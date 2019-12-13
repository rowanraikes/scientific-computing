import numpy as np
import pylab as pl
from math import pi
from matplotlib import pyplot as plt

from pde_methods import solve_heat_eq

# Exact solution of heat equation
# with dirichlet conditions u(0) = u(L) = 0
def u_exact(x,t,L):
   
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def test(solutions, x, T, L):

	error_tol = 0.01

	if L_2_norm(solutions,x,0.1) > error_tol:
		print("Test failed")
		exit()

	print("Test passed")

def L_2_norm(num1, num2, dx):

    # solutions arrays must be of odd length
    g_sq = np.power( abs(np.subtract(num1, num2)), 2)

    # simpsons ruke for integration
    I = (dx/3)*(g_sq[0] + np.sum(g_sq[1:g_sq.size-1:2]) + np.sum(g_sq[2:g_sq.size-1:2]) + g_sq[-1])

    return np.sqrt(I)

# set numerical parameters
mx = 10     # number of gridpoints in space
mt = 1000 # number of gridpoints in time

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.1        # total time to solve for

u_j = solve_heat_eq(mx, mt, L, T, kappa, dirichlet, forward_euler, u_I)


# Test the numerical solution
test(u_j, x, T, L)
