# Scalar Problems

## List of Problems
- Steady Diffusion (Poisson Equation) with Dirichlet BCs : `Scalar_1D_Diffusion_Poisson_Dirichlet.jl`, `Scalar_2D_Diffusion_Poisson_Dirichlet.jl`, `Scalar_3D_Diffusion_Poisson_Dirichlet.jl`


## Problem Descriptions
**Steady Diffusion (Poisson Equation) with Dirichlet BCs**:
Given $\mathbf{x} \in \Omega$, find the scalar field $\phi(\mathbf{x})$ satisfying the Poisson equation:
$$ -\nabla^2 \phi(\mathbf{x}) = f(\mathbf{x}), \quad \mathbf{x} \in \Omega $$
with Dirichlet boundary conditions:
$$ \phi(\mathbf{x}) = g(\mathbf{x}), \quad \mathbf{x} \in \partial \Omega $$
where $f(\mathbf{x})$ is a source term, and $g(\mathbf{x})$ specifies the value of $\phi$ on the boundary $\partial \Omega$. The problem is implemented in 1D, 2D, and 3D domains.


