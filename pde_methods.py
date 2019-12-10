# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import pylab as pl
from math import pi
from matplotlib import pyplot as plt

def u_I(x,L):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def solve_heat_eq_matrix(mx, mt, L, T, kappa):

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2) 

    u_j = np.zeros(x.size)

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i], L)

    # create A_fe matrix
    A_fe = np.zeros((x.size, x.size))

    # create boundary condition array
    bound_array = np.zeros(x.size)

    for i in range(x.size):
        A_fe[i,i] = (1 - 2*lmbda)

        if i < x.size - 1:
            A_fe[i+1, i] = lmbda
            A_fe[i, i+1] = lmbda

    # loop over every time point
    for j in range(mt):

        # set up boundary conditions for jth time point
        bound_array[0] = p(j)
        bound_array[x.size-1] = q(j)

        u_jp1 = dirichlet_bound_cond(u_j, lmbda, j)

        #u_jp1 = np.add(np.dot(A_fe, np.transpose(u_j)), lmbda*bound_array)

        # Boundary conditions (outer mesh points)
        #u_jp1[0] = p(j); u_jp1[mx] = q(j)
        
        # Update u_j
        u_j[:] = u_jp1[:]

    return(u_j)

def dirichlet_bound_cond(u_j, lmbda, j):

    # create forward euler matrix
    A_fe = np.zeros((u_j.size, u_j.size))

    for i in range(u_j.size):
        A_fe[i,i] = (1 - 2*lmbda)

        if i < u_j.size - 1:
            A_fe[i+1, i] = lmbda
            A_fe[i, i+1] = lmbda

    # create boundary condition array
    bound_array = np.zeros(u_j.size)
    bound_array[0] = p(j)
    bound_array[u_j.size-1] = q(j)


    # compute u(j+1) array
    u_jp1 = np.add(np.dot(A_fe, np.transpose(u_j)), lmbda*bound_array)

    return u_jp1

# Dirichlet conditions
def p(j): # end at x=0
    return 0

def q(j): # end at x = L
    return 0.1

# set numerical parameters
mx = 10     # number of gridpoints in space
mt = 1000    # number of gridpoints in time

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5     # total time to solve for

u_j = solve_heat_eq_matrix(mx, mt, L, T, kappa)

print(u_j)

# plot the final result and exact solution
x = np.linspace(0, L, mx+1)

plt.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
plt.plot(xx,u_exact(xx,T),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.show()

