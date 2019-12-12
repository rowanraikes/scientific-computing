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
    y = np.sin(pi*x/L) + 0.1
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)

    return y

def forward_euler(mx, lmbda, bound_cond):

    A_diagonals = (lmbda, 1-2*lmbda, lmbda)
    A = diags(A_diagonals, [-1,0,1], shape = (mx+1,mx+1), format='csc')

    # edit matrix for neumann conditions 
    if bound_cond == neumann: 
        A[0,1] = 2*lmbda
        A[mx, mx-1] = 2*lmbda

    if bound_cond == dirichlet:

        A = A[2:,2:] 

    return A

def backward_euler(mx, lmbda, bound_cond):

    #A_diagonals = (-lmbda, 2*lmbda-1, -lmbda)
    #A = diags(A_diagonals, [-1,0,1], shape = (mx+1,mx+1), format='csc')

    # Edit matrix for neumann condition
    if bound_cond == neumann:
        A[1,0] = 2*lmbda
        A[mx-1,mx] = 2*lmbda

    if bound_cond == dirichlet:

        A_diagonals = (-lmbda, 2*lmbda+1, -lmbda)
        A = diags(A_diagonals, [-1,0,1], shape = (mx-1,mx-1), format='csc')

        #A[0,1] = 0
        #A[mx-2, mx-3] = 0

    A = inv(A)
    return A

def crank_nicholson(mx, lmbda, bound_cond):

    if bound_cond == dirichlet:

        A_diagonals = (-lmbda/2, 1+lmbda, -lmbda/2)
        A = diags(A_diagonals, [-1,0,1], shape = (mx-1,mx-1), format='csc')

        #A[0,1] = 0
        #A[mx-2,mx-3] = 0

        

    A = inv(A)

    return A

def solve_heat_eq(mx, mt, L, T, kappa, bound_cond, scheme, initial_distribution):

    """
    Solves a 1D diffusion problem, boundary conditions 
    can be selected by user.

    Parameters
    ----------
    mx : int
        The number of grid spacings in space.
    mt : int
        The number of grid spacings in time.
    L : float
        The size of the domain in space.
    T : float
        The size of the domain in time.
    kappa : float
        The diffusion constant
    bound_cond : function
        Function that returns array for next point in time, different
        ones may be selected for different boundary conditions.
    scheme : function
        Function that returns the necessary evolution matrix based 
        upon the choice of descretisation and boundary condition.
    initial_distribution : function
        Returns the initial temperature didtribution. 

    Returns
    ---------
    A 1D numpy.array of the solutions at the desired point in time.

    """

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2) 

    u_j = np.zeros(x.size) 

    # calculate evolution matrix 
    A = scheme(mx, lmbda, bound_cond)

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = initial_distribution(x[i], L)

    # loop over every time point
    for j in range(mt):

        # evaluate RHS function
        s_j = RHS_fun(j, u_j)

        u_jp1 = bound_cond(u_j, A, lmbda, deltax, deltat, j, s_j, scheme, bound_cond)
        
        # Update u_j
        u_j[:] = u_jp1[:]

    return(u_j)

def neumann(u_j, A, lmbda, dx, dt, j, s_j, scheme, bound_cond):

    """
    Function that solves for the next time point, with Neumann
    boundary conditions.

    Parameters
    ----------
    u_j : numpy.array
        Array of solutions in space for current point in time.
    A : scipy.sparse.diags array
        The evolution matrix.
    lmbda : float
        The mesh fourier number.
    dx : float
        Size of grid spacings in space.
    dt : float
        Size of grid spacings in time.
    j : int
        The index of the point in time being solved for.
    s_j : numpy.array
        Array containing values of right hand side function.
    bound_cond : function
        Function that returns array for next point in time, different
        ones may be selected for different boundary conditions.
    scheme : function
        Function that returns the necessary evolution matrix based 
        upon the choice of descretisation and boundary condition.

    Returns
    ----------
    A 1D numpy.array of the solutions for the next point in time.

    """

    # create rhs vector
    if scheme == forward_euler:
        rhs_vect = np.zeros(u_j.size)
        rhs_vect[0] = -P(j)
        rhs_vect[u_j.size-1] = Q(j)

    if scheme == backward_euler:
        rhs_vect = np.zeros(u_j.size)
        rhs_vect[0] = P(j+1)
        rhs_vect[u_j.size-1] = -Q(j+1)


    # compute u(j+1) array
    u_jp1 = np.add(A.dot(u_j), 2*lmbda*dx*rhs_vect)

    # add RHS function
    u_jp1 = np.add(u_jp1, dt*s_j)

    return u_jp1

def dirichlet(u_j, A, lmbda, dx, dt, j, s_j, scheme, bound_cond):

    """
    Function that solves for the next time point, with Dirichlet
    boundary conditions.

    Parameters
    -----------
    (The same as for the neumann function)

    Returns
    -----------
    A 1D numpy.array of the solutions for the next point in time.

    """

    u_jp1 = np.zeros(u_j.size)

    # create boundary condition array
    #bound_array = np.zeros(u_j.size-2)
    #bound_array[0] = p(j)
    #bound_array[u_j.size-3] = q(j)

    # compute u(j+1) array
    if scheme == backward_euler:
    
        u_jp1[1:-1] = A.dot( np.add(np.transpose(u_j[1:-1]),  lmbda*bound_array(j, u_j)) )

    if scheme == forward_euler:

        u_jp1[1:-1] = np.add( A.dot(np.transpose(u_j[1:-1])) ,  lmbda*bound_array(j, u_j) )

    if scheme == crank_nicholson:

        B_diagonals = (lmbda/2, 1-lmbda, lmbda/2)
        B = diags(B_diagonals, [-1,0,1], shape = (mx-1,mx-1),  format='csc')

        u_jp1[1:-1] = A.dot( np.add( np.add( 0.5*lmbda*bound_array(j, u_j), 0.5*lmbda*bound_array(j, u_j) ), B.dot(np.transpose(u_j[1:-1]))))

    # add RHS function
    u_jp1 = np.add(u_jp1, dt*s_j)

    # update boundary nodes
    u_jp1[0] = p(j)
    u_jp1[u_j.size-1] = q(j)

    return u_jp1


def RHS_fun(j, u_j):

    """
    Function that calculates the distribution of a heat source inside 
    the domain.

    Parameters
    ----------
    j : int  
        The index of the point in time being solved for.
    u_j : numpy.array
        Array of solutions in space for current point in time.

    Returns
    ---------
    numpy.array containing the distribution of the heat source at the 
    required point in time.

    """

    s_j = np.zeros(u_j.size) # array containing distribution of heat source

    for i in range(u_j.size):

        if i == 1:
            s_j[i] = 0
        else:
            s_j[i] = 0
    
    return s_j # return s_j = 0 for all i if no heat source required.

# Dirichlet conditions
def p(j): # u at end at x = 0
    return 0.2

def q(j): # u at end at x = L
    return 0.1

def bound_array(j, u_j):

    bound_array = np.zeros(u_j.size-2)
    bound_array[0] = p(j)
    bound_array[u_j.size-3] = q(j)

    return bound_array

# Neumann conditions
def P(j): # du/dx at end at x = 0
    return 0

def Q(j): # du/dx at end at x = L
    return 0

def RMSE(exact, num):

    """
    Function calculates the RMSE of a numerical solution.

    Parameters
    ----------
    exact : numpy.array
        Array containing exact solutions
    num : numpy.array
        Array containing numerical solutions

    Returns
    ----------
    A float whose value is the root mean squared error of the numerical solution.
     
    """

    error = np.sqrt( np.sum( np.power( np.subtract(exact,num), 2) ) / num.size )

    return error

def L_2_norm(exact, num, dx):

    # solutions arrays must be of odd length
    g_sq = np.power( abs(np.subtract(exact, num)), 2)

    # simpsons ruke for integration
    I = (dx/3)*(g_sq[0] + np.sum(g_sq[1:g_sq.size-1:2]) + np.sum(g_sq[2:g_sq.size-1:2]) + g_sq[-1])

    return np.sqrt(I)

# set numerical parameters
mx = 10  # number of gridpoints in space
mt = 1000  # number of gridpoints in time

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5   # total time to solve for

#u_j = solve_heat_eq_matrix(mx, mt, L, T, kappa, dirichlet_bound_cond_fe )


errors = []
mxs = []

#mt = 1000



# for n in range(1,8): # number of points for each exponent

#     mx = np.power(2,n)
   
#     mt = np.power(mx, 2)

#     #mt = int( T/L*np.power(mx, 2) )

#     xx = np.linspace(0,L,mx+1)

#     u_j = solve_heat_eq(mx, mt, L, T, kappa, dirichlet, forward_euler, u_I)
#     u_j2 = solve_heat_eq(mx*2, mt*2, L, T, kappa, dirichlet, forward_euler, u_I)

#     error = L_2_norm(u_j, u_j2, L/mx )
#     #error = RMSE(u_j, u_exact(xx,T))

#     errors.append(error)
#     mxs.append(mx)

# #slope, intercept = np.polyfit(np.log(mxs), np.log(errors), 1)

# #print("Gradient = ",slope)

# plt.loglog(mxs, errors)
# plt.xlabel("Number of grid points in space")
# plt.ylabel("RMSE")
# plt.title("Error Plot for Forward Euler")

u_j = solve_heat_eq(mx, mt, L, T, kappa, dirichlet, crank_nicholson, u_I )

print(u_j)

#xx = np.linspace(0,L,mx+1)

#print(L_2_norm(u_j, u_exact(xx,T), L/mx ))

#print(u_j)

#print(u_j)

#plot the final result and exact solution
x = np.linspace(0, L, mx+1)

plt.plot(x,u_j,'ro',label='t=')

#plt.plot(xx,u_exact(xx,T),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
# # # # plt.show()
plt.show()
