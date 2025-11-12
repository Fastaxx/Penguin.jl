# Johansen-Colella Problems

This directory contains implementations of the Johansen-Colella benchmark problems
for diffusion equations with interfaces, as described in:
- Johansen, H. and Colella, P. (1998). A Cartesian grid embedded boundary method for
  Poisson's equation on irregular domains. Journal of Computational Physics, 147(1),
  60-85. https://doi.org/10.1006/jcph.1998.5965
  
## Problem 1: Poisson Equation with Constant Coefficient
Implemented in `Problem1_PoissonConstant.jl`, this problem involves solving the Poisson
equation with a constant diffusion coefficient on a star-shaped domain. The exact solution
is known, allowing for error analysis and convergence studies.

## Problem 2: Poisson Equation with Variable Coefficient
Implemented in `Problem2_PoissonVariable.jl`, this problem extends the first by introducing
a variable diffusion coefficient. The domain remains star-shaped, and the exact solution
is also known for this case. NOTE: THIS SCRIPT DOES NOT PRODUCE ACCURATE SOLUTIONS DUE TO A BUG IN THE
VARIABLE-COEFFICIENT DIFFUSION OPERATOR IMPLEMENTATION.

## Problem 3: Laplace Equation outside the star-shaped domain
Implemented in `Problem3_FlowerLaplace.jl`, this problem focuses on solving the Laplace equation
outside the star-shaped domain, with Dirichlet boundary conditions applied on the interface.
Diagnostic outputs are the evaluations of potential overshoots and undershoots in the domain.

## Problem 4: Poisson Equation in 3D with Spherical Interface
Implemented in `Problem4_SchwartzColella_Poisson3D.jl`, this problem involves solving the Poisson equation
in a three-dimensional domain with a spherical interface. The exact solution is known, and
error norms are computed for convergence analysis.

## Problem 5: Heat Equation in 3D with Spherical Interface
Implemented in `Problem5_SchwartzColella_Heat3D.jl`, this problem addresses the heat equation
in a three-dimensional domain with a spherical interface. The solution evolves over time,
and error norms are calculated at the final time for convergence studies.
