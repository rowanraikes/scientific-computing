This repository contains a numerical solver for the 1D heat/diffusion equation, a 
partial differential equation. The code uses a number of libraries available in 
the Anaconda package. I recommend executing it using using the Anaconda prompt
so as to avoid any import errors. (>>python pde_methods.py)

The software allows for the selection of neumann or dirichlet boundary conditions, 
these effect how heat diffuses from either end of the 1D bar being modelled.

The code also supports three different descretisation techniques; forward euler,
backward euler or crank nicholson.


Executing the code in its current state will simulate heat diffusion in a 1D bar 
with neumann boundary conditions. This means that heat cannot pass through the 
boundaries of the domain being simulated. Snapshots of the heat distribution at
a number of points in time will be plotted.