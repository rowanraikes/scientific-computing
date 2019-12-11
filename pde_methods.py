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

def solve_heat_eq(mx, mt, L, T, kappa, bound_cond, scheme, initial_distribution):

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2) 

    u_j = np.zeros(x.size)

    # Set up matrices for dirichlet boundary conditions
    if scheme == "forward_euler":
    # create forward euler matrix

        A_diagonals = (lmbda, 1-2*lmbda, lmbda)
        A = diags(A_diagonals, [-1,0,1], shape = (mx+1,mx+1), format='csc')

    # create backward euler matrix
    if scheme == "backward_euler":
        A_diagonals = (-lmbda, 1+2*lmbda, -lmbda)
        A = diags(A_diagonals, [-1,0,1], shape = (mx-1,mx-1), format='csc')
        A = inv(A) # performing inversion here saves computation

    # create Crank Nicholson matrix
    if scheme == "crank_nicholson":
        A_diagonals = (-lmbda/2, 1+lmbda, -lmbda/2)
        A = diags(A_diagonals, [-1,0,1], shape = (mx-1,mx-1), format='csc')

        B_diagonals = (lmbda/2, 1-lmbda, lmbda/2)
        B = diags(B_diagonals, [-1,0,1], shape = (mx-1,mx-1),  format='csc')

        A = inv(A).dot(B)


    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = initial_distribution(x[i], L)

    # loop over every time point
    for j in range(mt):

        # evaluate RHS function
        s_j = RHS_fun(j, u_j)

        u_jp1 = bound_cond(u_j, A, lmbda, deltax, deltat, j, s_j, scheme)
        
        # Update u_j
        u_j[:] = u_jp1[:]

    return(u_j)

def neumann(u_j, A, lmbda, dx, dt, j, s_j, scheme):
  
    # edit FE matrix for neumann conditions
    if scheme == "forward_euler":
        A[0,1] = 2*lmbda
        A[u_j.size-1, u_j.size-2] = 2*lmbda

    # create bound array
    bound_array = np.zeros(u_j.size)
    bound_array[0] = -P(j)
    bound_array[u_j.size-1] = Q(j)

    # compute u(j+1) array
    u_jp1 = np.add(A.dot(u_j), 2*lmbda*dx*bound_array)

    # add RHS function
    u_jp1 = np.add(u_jp1, dt*s_j)

    return u_jp1

def dirichlet(u_j, A, lmbda, dx, dt, j, s_j, scheme):

    u_jp1 = np.zeros(u_j.size)

    A = A[2:,2:] # reduce size of A for calculating inner grid points

    # create boundary condition array
    bound_array = np.zeros(u_j.size-2)
    bound_array[0] = p(j)
    bound_array[u_j.size-3] = q(j)

    # compute u(j+1) array
    u_jp1[1:-1] = np.add(A.dot(np.transpose(u_j[1:-1])) , lmbda*bound_array )

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
def p(j): # u at end at x = 0
    return 0

def q(j): # u at end at x = L
    return 0

# Neumann conditions
def P(j): # du/dx at end at x = 0
    return 0

def Q(j): # du/dx at end at x = L
    return 0

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



# for n in range(1,4): # number of points for each exponent

#     for m in range(1,2):
   
#         mx = m*np.power(10,n)
   
#         print(mx)
#         mt = mx

#         xx = np.linspace(0,L,mx+1)

#         u_j = solve_heat_eq(mx, mt, L, T, kappa, dirichlet, 'crank_nicholson')

#         error = RMSE(u_j, u_exact(xx,T) )

#         errors.append(error)
#         mxs.append(mx)

# slope, intercept = np.polyfit(np.log(mxs), np.log(errors), 1)

# print("Gradient = ",slope)

# plt.loglog(mxs, errors)
# plt.xlabel("Number of grid points in space")
# plt.ylabel("RMSE")
# plt.title("Error Plot for Crank-Nicholson")

u_j = solve_heat_eq(mx, mt, L, T, kappa, neumann, 'forward_euler', u_I )

#plot the final result and exact solution
x = np.linspace(0, L, mx+1)

plt.plot(x,u_j,'ro',label='t=')

xx = np.linspace(0,L,250)
#plt.plot(xx,u_exact(xx,T),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
# # plt.show()
plt.show()
