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
from scipy.sparse.linalg import inv


def u_I(x,L):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def solve_heat_eq_matrix(mx, mt, L, T, kappa, bound_cond, scheme):

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2) 

    u_j = np.zeros(x.size)

    # Set up matrices 
    if scheme == "forward_euler":
    # create forward euler matrix
    # Use sparse matrix
        A_diagonals = (lmbda, 1-2*lmbda, lmbda)
        A = diags(A_diagonals, [-1,0,1], shape = (mx-1,mx-1))

    # create backward euler matrix
    if scheme == "backward_euler":
        A_diagonals = (-lmbda, 1+2*lmbda, -lmbda)
        A = diags(A_diagonals, [-1,0,1], shape = (mx-1,mx-1))
        A = inv(A) # performing inversion here saves computation


    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i], L)

    # loop over every time point
    for j in range(mt):

        u_jp1 = dirichlet(u_j, A, lmbda, deltat, j)
        
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

def dirichlet(u_j, A, lmbda, dt, j):

    u_jp1 = np.zeros(u_j.size)

    # create boundary condition array
    bound_array = np.zeros(u_j.size-2)
    bound_array[0] = p(j)
    bound_array[u_j.size-3] = q(j)

    # evaluate RHS function
    s_j = RHS_fun(j, u_j)

    # compute u(j+1) array
    u_jp1[1:-1] = np.add(A.dot(np.transpose(u_j[1:-1])) , lmbda*bound_array )

    # add RHS function
    u_jp1 = np.add(u_jp1, dt*s_j)

    # update boundary nodes
    u_jp1[0] = p(j)
    u_jp1[u_j.size-1] = q(j)

    return u_jp1

def backward_euler_dirichlet(u_j, lmbda, dt, j):

    u_jp1 = np.zeros(u_j.size)

    A_diagonals = (-lmbda, 1+2*lmbda, -lmbda)
    A_fe = diags(A_diagonals, [-1,0,1], shape = (mx-1,mx-1))

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

    error = np.sqrt( np.sum( np.power( np.subtract(exact,num), 2) ) / num.size )

    return error

# set numerical parameters
mx = 10   # number of gridpoints in space
mt = 1000  # number of gridpoints in time

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5   # total time to solve for

#u_j = solve_heat_eq_matrix(mx, mt, L, T, kappa, dirichlet_bound_cond_fe )


errors = []
mxs = []

#mt = 1000



for n in range(1,9): # number of points for each exponent
   
    mx = np.power(2,n)
   
    print(mx)
    mt = np.power(mx, 2)

    xx = np.linspace(0,L,mx+1)

    u_j = solve_heat_eq_matrix(mx, mt, L, T, kappa, dirichlet, 'forward_euler')

    error = RMSE(u_j, u_exact(xx,T) )

    errors.append(error)
    mxs.append(mx)

slope, intercept = np.polyfit(np.log(mxs), np.log(errors), 1)

print("Gradient = ",slope)

plt.loglog(mxs, errors)
plt.xlabel("Number of grid points in space")
plt.ylabel("RMSE")
plt.title("Error Plot for Forward Euler")

# u_j = solve_heat_eq_matrix(mx, mt, L, T, kappa, dirichlet, 'backward_euler' )

# #plot the final result and exact solution
# x = np.linspace(0, L, mx+1)

# plt.plot(x,u_j,'ro',label='t=')

# xx = np.linspace(0,L,250)
# plt.plot(xx,u_exact(xx,T),'b-',label='exact')
# plt.xlabel('x')
# plt.ylabel('u(x,0.5)')
# plt.legend(loc='upper right')
# # plt.show()
plt.show()
