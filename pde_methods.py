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

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5        # total time to solve for
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def solve_heat_eq(u_j, u_jp1, lmbda, descritation, mx, mt):

    for n in range(1, mt+1):

        # Descritation at each time step
        for i in range(1, mx):

            u_jp1[i] = descritation(u_j, u_jp1, lmbda, i)

        #print(u_jp1)

        # Boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0
        
        # Update u_j
        u_j[:] = u_jp1[:]

    return u_j

def forward_euler_descritation(u_j, u_jp1, lmbda, i):

    # Forward Euler timestep   
    return u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])

def backward_euler_descritation(u_j, u_jp1, lmbda, i):

    # Backward Euler timestep   
    return u_j[i] + (lmbda / 2)*(u_j[i+1] - 2*u_j[i] + u_j[i-1] + u_jp1[i+1] - 2*u_jp1[i] + u_jp1[i-1])

def crank_nicholson_descritation(u_j, u_jp1, lmbda, i):

    # Crank Nicholson timestep   
    return u_j[i] + (lmbda / 2)*(u_jp1[i+1] - 2*u_jp1[i] + u_jp1[i-1] + u_j[i+1] - 2*u_j[i] + u_j[i-1])


# set numerical parameters
mx = 10     # number of gridpoints in space
mt = 1000 # number of gridpoints in time

# set up the numerical environment variables
x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
print("deltax=",deltax)
print("deltat=",deltat)
print("lambda=",lmbda)

# set up the solution variables
u_j = np.zeros(x.size)        # u at current time step
u_jp1 = np.zeros(x.size)      # u at next time step

# Set initial condition
for i in range(0, mx+1):
    u_j[i] = u_I(x[i])


u_j = solve_heat_eq(u_j, u_jp1, lmbda, crank_nicholson_descritation, mx, mt)

# plot the final result and exact solution
plt.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
plt.plot(xx,u_exact(xx,T),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.show()