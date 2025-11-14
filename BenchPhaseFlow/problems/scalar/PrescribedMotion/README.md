# Prescribed Motion Problems

This directory contains benchmark problems for scalar diffusion with prescribed
motion of the immersed body. 
The problems here are used to verify the accuracy and convergence of the
PrescribedMotion module in BenchPhaseFlow, especially the Space-Time Cut-Cell method.

## Mathematical Formulation
Given $\Omega(t)$ as the time-dependent domain occupied by the fluid and
$\Gamma(t)$ as the moving boundary of the immersed body, the scalar diffusion
problem is formulated as:
$$
\frac{\partial u}{\partial t} = D \nabla^2 u \quad \text{in } \Omega(t), \quad t > 0
$$
with boundary conditions:
$$
u = u_b \quad \text{on } \Gamma(t), \quad t >
0$$
and initial condition:
$$
u(x, 0) = u_0(x) \quad \text{in } \Omega(0)
$$
where $D$ is the diffusion coefficient, $u_b$ is the prescribed boundary value on
the moving boundary, and $u_0(x)$ is the initial scalar field.
    

## Available Problems
- `Heat_2D_Moving.jl`: A 2D heat diffusion problem with a circular body that oscillates in size over time. The problem tests the ability of the method to handle moving boundaries and verifies convergence rates.

- `SchwartzColella/FixedDisk.jl`: A 2D heat diffusion problem with a fixed circular body. This problem serves as a baseline to verify the accuracy of the method without the added complexity of motion.

- `SchwartzColella/ExpandingDisk.jl`: A 2D heat diffusion problem with a circular body that expands over time. This problem tests the method's capability to handle freshly created cut-cells and verifies convergence rates.

- `SchwartzColella/ShrinkingDisk.jl`: A 2D heat diffusion problem with a circular body that shrinks over time. This problem tests the method's capability to handle disappearing cut-cells and verifies convergence rates.

- `Heat_1D_Moving_ConstantBC.jl`: A 1D heat diffusion problem with a moving boundary where the boundary condition is constant. This problem verifies that the numerical solution preserves the constant state while the cut location moves.

- `JohansenColella/FixedDirichlet.jl`: A 2D heat diffusion problem with a fixed elliptical multiple bodies and Dirichlet boundary conditions on both the outer box and the ellipse boundaries. This problem verifies convergence rates for fixed boundaries.

- `JohansenColella/MovingDirichlet.jl`: A 2D heat diffusion problem with a moving elliptical body and Dirichlet boundary conditions on both the outer box and the ellipse boundary. This problem verifies convergence rates for moving boundaries.