# Scalar Problems

## List of Problems
- Steady Diffusion (Poisson Equation) with Dirichlet BCs : `Scalar_1D_Diffusion_Poisson_Dirichlet.jl`, `Scalar_2D_Diffusion_Poisson_Dirichlet.jl`, `Scalar_3D_Diffusion_Poisson_Dirichlet.jl`
- Unsteady Diffusion (Heat Equation) with Dirichlet BCs : `Scalar_1D_Diffusion_Heat_Dirichlet.jl`, `Scalar_2D_Diffusion_Heat_Dirichlet.jl`, `Scalar_3D_Diffusion_Heat_Dirichlet.jl`
- Unsteady Diffusion (Heat Equation) with Robin BCs : `Scalar_1D_Diffusion_Heat_Robin.jl`, `Scalar_2D_Diffusion_Heat_Robin.jl`, `Scalar_3D_Diffusion_Heat_Robin.jl`

## Problem Descriptions
**Steady Diffusion (Poisson Equation) with Dirichlet BCs**:
Given $\mathbf{x} \in \Omega$, find the scalar field $\phi(\mathbf{x})$ satisfying the Poisson equation:
$$ -\nabla^2 \phi(\mathbf{x}) = f(\mathbf{x}), \quad \mathbf{x} \in \Omega $$
with Dirichlet boundary conditions:
$$ \phi(\mathbf{x}) = g(\mathbf{x}), \quad \mathbf{x} \in \partial \Omega $$
where $f(\mathbf{x})$ is a source term, and $g(\mathbf{x})$ specifies the value of $\phi$ on the boundary $\partial \Omega$. The problem is implemented in 1D, 2D, and 3D domains.

**Unsteady Diffusion (Heat Equation) with Dirichlet BCs**:
Given $\mathbf{x} \in \Omega$ and time $t \in (0, T]$, find the scalar field $u(\mathbf{x}, t)$ satisfying the heat equation:
$$ \frac{\partial u(\mathbf{x}, t)}{\partial t} - \kappa \nabla^2 u(\mathbf{x}, t) = f(\mathbf{x}, t), \quad \mathbf{x} \in \Omega, \; t \in (0, T] $$
with initial condition:
$$ u(\mathbf{x}, 0) = u_0(\mathbf{x}), \quad \mathbf{x} \in \Omega $$
and Dirichlet boundary conditions:
$$ u(\mathbf{x}, t) = h(\mathbf{x}, t), \quad \mathbf{x} \in \partial \Omega, \; t \in (0, T] $$
where $\kappa$ is the thermal diffusivity, $u_0(\mathbf{x})$ is the initial temperature distribution, and $h(\mathbf{x}, t)$ specifies the temperature on the boundary $\partial \Omega$. The problem is implemented in 1D, 2D, and 3D domains.

**Unsteady Diffusion (Heat Equation) with Robin BCs**:
Given $\mathbf{x} \in \Omega$ and time $t \in (0, T]$, find the scalar field $u(\mathbf{x}, t)$ satisfying the heat equation:
$$ \frac{\partial u(\mathbf{x}, t)}{\partial t} - \kappa \nabla^2 u(\mathbf{x}, t) = f(\mathbf{x}, t), \quad \mathbf{x} \in \Omega, \; t \in (0, T] $$
with initial condition:
$$ u(\mathbf{x}, 0) = u_0(\mathbf{x}), \quad \mathbf{x} \in \Omega $$
and Robin boundary conditions:
$$ \alpha u(\mathbf{x}, t) + \beta \frac{\partial u(\mathbf{x}, t)}{\partial n} = r(\mathbf{x}, t), \quad \mathbf{x} \in \partial \Omega, \; t \in (0, T] $$
where $\kappa$ is the thermal diffusivity, $u_0(\mathbf{x})$ is the initial temperature distribution, $\alpha$ and $\beta$ are coefficients defining the Robin boundary condition, and $r(\mathbf{x}, t)$ specifies the boundary condition on $\partial \Omega$. The problem is implemented in 1D, 2D, and 3D domains.

## References
- J. Crank, "The Mathematics of Diffusion," Oxford University Press, 1975
- H. S. Carslaw and J. C. Jaeger, "Conduction of Heat in Solids," Oxford University Press, 1959.
