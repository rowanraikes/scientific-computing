
import numpy as np
import pylab as pl
from math import pi
from matplotlib import pyplot as plt



def u_I(x,L):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y


def solve_heat_eq(mx, mt, L, T, kappa):

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2) 

    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)   

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i], L)

    for j in range(1, mt+1):

        # Calculate inner mesh points
        for i in range(1, mx):

            u_jp1[i] = forward_euler_dirchlet(u_j, u_jp1, lmbda, i, j, mx)
        
        # Update u_j
        u_j[:] = u_jp1[:]

    return u_j

def forward_euler_dirchlet(u_j, u_jp1, lmbda, i, j, mx):


    # first boundary timestep
    if i == 1:
        u_jp1 = u_j[i] + lmbda*(2*u_j[i] + u_j[i+1] + p(j) )

    if i == mx:
        u_jp1 = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + q(j) )

    # normal timestep 
    else:
        u_jp1 = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1]) + RHS_fun(i,j)

    return u_jp1

# Dirichlet conditions
def p(j): # end at x = 0
    return 0

def q(j): # end at x = L
    return 0


def RHS_fun(i,j):

    if i == 3:
        s_ij = 0.001
    else:
        s_ij = 0
    
    return s_ij


# set numerical parameters
mx = 30     # number of gridpoints in space
mt = 1000    # number of gridpoints in time

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5   # total time to solve for

u_j1 = solve_heat_eq(mx, mt, L, 0.1, kappa)
u_j2 = solve_heat_eq(mx, mt, L, 0.2, kappa)
u_j3 = solve_heat_eq(mx, mt, L, 0.3, kappa)
u_j4 = solve_heat_eq(mx, mt, L, 0.4, kappa)
u_j5 = solve_heat_eq(mx, mt, L, 0.5, kappa)


# plot the final result and exact solution
x = np.linspace(0, L, mx+1)

plt.plot(x,u_j1,'r',label='t=0.1')
plt.plot(x,u_j2,'b',label='t=0.2')
plt.plot(x,u_j3,'g',label='t=0.3')
plt.plot(x,u_j4,'c',label='t=0.4')
plt.plot(x,u_j5,'m',label='t=0.5')
xx = np.linspace(0,L,250)
#plt.plot(xx,u_exact(xx,T),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.show()