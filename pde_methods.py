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

from scipy.sparse import diags

def u_I(x,L):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def solve_heat_eq_matrix(mx, mt, L, T, kappa, bound_cond):

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

        u_jp1 = bound_cond(u_j, lmbda, deltat, j)
        
        # Update u_j
        u_j[:] = u_jp1[:]

    return(u_j)

def forward_euler_neumann(u_j, lmbda, dx, j):

    # create forward euler matrix 
    A = np.zeros((u_j.size, u_j.size))

    for i in range(u_j.size):
        A[i,i] = (1 - 2*lmbda)

        if i < u_j.size - 1:
            A[i+1, i] = lmbda
            A[i, i+1] = lmbda

    A[0,1] = 2*lmbda
    A[u_j.size-1, u_j.size-2] = 2*lmbda

    # create bound array
    bound_array = np.zeros(u_j.size)
    bound_array[0] = -P(j)
    bound_array[u_j.size-1] = Q(j)

    # compute u(j+1) array
    u_jp1 = np.add(np.dot(A,u_j), 2*lmbda*dx*bound_array)

    return u_jp1

def forward_euler_dirichlet(u_j, lmbda, dt, j):

    u_jp1 = np.zeros(u_j.size)

    # create forward euler matrix
    # A_fe = np.zeros((u_j.size-2, u_j.size-2))

    # for i in range(0,u_j.size-2):

    #     A_fe[i,i] = (1 - 2*lmbda)

    #     if i < u_j.size - 3:
    #         A_fe[i+1, i] = lmbda
    #         A_fe[i, i+1] = lmbda

    # Use sparse matrix
    A_diagonals = (lmbda, 1-2*lmbda, lmbda)
    A_fe = diags(A_diagonals, [-1,0,1], shape = (mx-1,mx-1))

    # create boundary condition array
    bound_array = np.zeros(u_j.size-2)
    bound_array[0] = p(j)
    bound_array[u_j.size-3] = q(j)

    # evaluate RHS function
    s_j = RHS_fun(j, u_j)

    # compute u(j+1) array
    u_jp1[1:-1] = np.add(A_fe.dot(np.transpose(u_j[1:-1])) , lmbda*bound_array )

    # add RHS function
    u_jp1 = np.add(u_jp1, dt*s_j)

    # update boundary nodes
    u_jp1[0] = p(j)
    u_jp1[u_j.size-1] = q(j)

    return u_jp1

def backward_euler_dirichlet(u_j, lmbda, dt, j):

    u_jp1 = np.zeros(u_j.size)

    # create forward euler matrix
    A_be = np.zeros((u_j.size-2, u_j.size-2))

    for i in range(0,u_j.size-2):

        A_be[i,i] = (1 + 2*lmbda)

        if i < u_j.size - 3:
            A_be[i+1, i] = -lmbda
            A_be[i, i+1] = -lmbda

    # create boundary condition array
    bound_array = np.zeros(u_j.size-2)
    bound_array[0] = p(j)
    bound_array[u_j.size-3] = q(j)

    # evaluate RHS function
    s_j = RHS_fun(j, u_j)

    # compute u(j+1) array
    u_jp1[1:-1] = np.add(np.dot( np.linalg.inv(A_be), np.transpose(u_j[1:-1]) ) , lmbda*bound_array )

    # add RHS function
    u_jp1 = np.add(u_jp1, dt*s_j)

    # update boundary nodes
    u_jp1[0] = p(j)
    u_jp1[u_j.size-1] = q(j)

    return u_jp1

# def thomas(A, X):

#     # solves Y = AX where A is tridiaganol 
#     Y = np.zeros(X.size) # solutions array
#     ddash = np.zeros(X.size)
#     cdash = np.zeros(X.size)

#     Y[X.size-1] = ddash[X.size-1]

#     cdash[0] = A[0,1] / A[0,0]

#     for i in range(1,X.size-1):

#         cdash[i] = A[i,i+1] / A[i,i] - A[i,i-1]*cdash[i-1]

#     print(cdash)


#     for i in range(X.size):
#         Y[-i] = ddash[-i] - cdash[-i]*Y[-i+1]


def RHS_fun(j, u_j):

    s_j = np.zeros(u_j.size)

    for i in range(u_j.size):

        if i == 1:
            s_j[i] = 0
        else:
            s_j[i] = 0
    
    return s_j

# Dirichlet conditions
def p(j): # end at x = 0
    return 0

def q(j): # end at x = L
    return 0

# Neumann conditions
def P(j): # end at x = 0
    return 2

def Q(j): # end at x = L
    return -2

def RMSE(exact, num):

    error = np.sqrt( np.sum( abs(np.subtract(exact,num))))

    return error

# set numerical parameters
mx = 10   # number of gridpoints in space
mt = 100    # number of gridpoints in time

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5   # total time to solve for

#u_j = solve_heat_eq_matrix(mx, mt, L, T, kappa, dirichlet_bound_cond_fe )


errors = []
mxs = []

# for n in range(1,3): # exponent

#     for i in range(1,10): # number of points for each exponent

#         mx = np.power(10,n)*i

#         mt = int(mx*T / np.power(L,2))


#         xx = np.linspace(0,L,mx+1)

#         u_j = solve_heat_eq_matrix(mx, mt, L, T, kappa, forward_euler_dirichlet)

#         error = RMSE(u_j, u_exact(xx,T) )

#         errors.append(error)
#         mxs.append(mx)

# #slope, intercept = np.polyfit(np.log(mxs), np.log(errors), 1)

# #print("Gradient = ",slope)

# plt.loglog(mxs, errors)
# plt.xlabel("Number of grid points in space")
# plt.ylabel("RMSE")

u_j = solve_heat_eq_matrix(mx, mt, L, T, kappa, forward_euler_dirichlet )

#plot the final result and exact solution
x = np.linspace(0, L, mx+1)

plt.plot(x,u_j,'ro',label='t=')

xx = np.linspace(0,L,250)
plt.plot(xx,u_exact(xx,T),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.show()
