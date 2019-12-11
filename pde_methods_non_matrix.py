
import numpy as np
import pylab as pl
from math import pi
from matplotlib import pyplot as plt

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def u_I(x,L):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y


def solve_heat_eq(mx, mt, L, T, kappa):

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    dx = x[1] - x[0]            # gridspacing in x
    dt = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*dt/(dx**2) 

    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)   

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i], L)

    for j in range(1, mt+1):

        u_jp1 = forward_euler(u_j, lmbda, i, j, mx, dx, dt)
        print(u_jp1)
 
        # Update u_j
        u_j[:] = u_jp1[:]

    return u_j

def backward_euler(u_j, lmbda, i, j, mx, dx, dt):

    u_jp1 = np.zeros(u_j.size)


    # Dirichlet boundary conds ###############

    # inner mesh points
    for i in range(1, mx):

        # grid points adjacent to boundary
        if i == 1:
            u_jp1[i] = u_j[i] + lmbda*( u_jp1[i+1] - 2*u_jp1[i] + p(j+1) )  

        if i == mx-1:
            
            u_jp1[i] = u_j[i] + lmbda*(u_jp1[i-1] - 2*u_jp1[i] + q(j+1) )

        # normal timestep 
        else:
            u_jp1[i] = u_j[i] + lmbda*( u_jp1[i+1] - 2*u_jp1[i] + u_jp1[i-1]) + dt*RHS_fun(i,j)

    u_jp1[0] = p(j)
    u_jp1[u_j.size-1] = q(j)

    return u_jp1



def forward_euler(u_j, lmbda, i, j, mx, dx, dt):

    u_jp1 = np.zeros(u_j.size)

    # Dirichlet boundary conds ###############

    # inner mesh points
    for i in range(1, mx):

        # grid points adjacent to boundary
        if i == 1:
            u_jp1[i] = u_j[i] + lmbda*(2*u_j[i] + u_j[i+1] + p(j) )

        if i == mx-1:
            
            u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + q(j) )

        # normal timestep 
        else:
            u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1]) + dt*RHS_fun(i,j)

    # end points
    u_jp1[0] = p(j)    
    u_jp1[u_j.size-1] = q(j)


    # Neumann #################

    # for i in range(0, mx):

    #     if i == 0:
    #         u_jp1[i] = u_j[0] + lmbda*(2*u_j[1] - 2*u_j[0]) - 2*lmbda*dx*P(j)

    #     if i == mx:
    #         u_jp1[i] = u_j[mx] + lmbda*(2*u_j[mx-1] - 2*u_j[0]) + 2*lmbda*dx*Q(j) 

    #     # normal timestep 
    #     else:
    #         u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1]) + RHS_fun(i,j)


    return u_jp1

# Dirichlet conditions
def p(j): # end at x = 0
    return 0.1

def q(j): # end at x = L
    return 0.1

# Neumann conditions
def P(j): # end at x = 0
    return 0

def Q(j): # end at x = L
    return 0


def RHS_fun(i,j):

    if i == 3:
        s_ij = 0
    else:
        s_ij = 0
    
    return s_ij


# set numerical parameters
mx = 10     # number of gridpoints in space
mt = 10   # number of gridpoints in time

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5   # total time to solve for

u_j = solve_heat_eq(mx, mt, L, 0.5, kappa)

# plot the final result and exact solution
x = np.linspace(0, L, mx+1)

plt.plot(x,u_j,'ro',label='t=0.5')
xx = np.linspace(0,L,250)
plt.plot(xx,u_exact(xx,T),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.show()