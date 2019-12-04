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

    # create matrix
    A_fe = np.zeros((x.size, x.size))

    for i in range(x.size):
        A_fe[i,i] = (1 - 2*lmbda)

        if i < x.size - 1:
            A_fe[i+1, i] = lmbda
            A_fe[i, i+1] = lmbda

    for n in range(mt):

        u_jp1 = np.dot(A_fe, np.transpose(u_j) )

        # Boundary conditions (outer mesh points)
        u_jp1[0] = 0; u_jp1[mx] = 0
        
        # Update u_j
        u_j[:] = u_jp1[:]


    return(u_j)



def solve_heat_eq(mx, mt, L, T, kappa):

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2) 

    pj = 0   # boundary conditions
    qj = 0   #

    descritisation = forward_euler_descritisation

    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)   

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i], L)


    for n in range(1, mt+1):

        # Calculate inner mesh points
        for i in range(1, mx):



            u_jp1[i] = descritisation(u_j, u_jp1, lmbda, i)

        #print(u_jp1)

        # Boundary conditions (outer mesh points)
        u_jp1[0] = pj; u_jp1[mx] = qj
        
        # Update u_j
        u_j[:] = u_jp1[:]

    return u_j

def forward_euler_dirchlet(u_j, u_jp1, lmbda, i, mx, pj, qj):


    # first boundary timestep
    if i == 1:
        u_jp1 = u_j[i] + lmbda*(2*u_j[i] + u_j[i+1] + pj)

    if i == mx:
        u_jp1 = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + qj)

    # normal timestep 
    else:
        u_jp1 = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])

    return u_jp1


def forward_euler_descritisation(u_j, u_jp1, lmbda, i):

    # normal timestep 
    return u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])

def backward_euler_descritisation(u_j, u_jp1, lmbda, i):

    # Backward Euler timestep   
    return u_j[i] + (lmbda / 2)*(u_j[i+1] - 2*u_j[i] + u_j[i-1] + u_jp1[i+1] - 2*u_jp1[i] + u_jp1[i-1])

def crank_nicholson_descritisation(u_j, u_jp1, lmbda, i):

    # Crank Nicholson timestep   
    return u_j[i] + (lmbda / 2)*(u_jp1[i+1] - 2*u_jp1[i] + u_jp1[i-1] + u_j[i+1] - 2*u_j[i] + u_j[i-1])


# set numerical parameters
mx = 10     # number of gridpoints in space
mt = 10    # number of gridpoints in time

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.01     # total time to solve for

u_j = solve_heat_eq(mx, mt, L, T, kappa)

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

solve_heat_eq_matrix(mx, mt, L, T, kappa)