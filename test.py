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

	for i in range(solutions.size):
		if abs(solutions[i] - u_exact(x[i], T, L)) > error_tol:
			print("Test failed at point of index: ", i)
			exit()

	print("Test passed")

# set numerical parameters
mx = 10     # number of gridpoints in space
mt = 1000 # number of gridpoints in time

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.1        # total time to solve for

u_j = solve_heat_eq(mx, mt, L, T, kappa)

# plot the final result and exact solution
fig = plt.figure()
x = np.linspace(0, L, mx+1)

plt.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
plt.plot(xx,u_exact(xx,T,L),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.show()

# Test the numerical solution
test(u_j, x, T, L)
