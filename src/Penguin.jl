module Penguin

using SparseArrays
using StaticArrays
using CartesianGeometry
using ImplicitIntegration
using LinearAlgebra
using IterativeSolvers
using CairoMakie
using WriteVTK
# Write your package code here.

include("mesh.jl")
export Mesh, get_border_cells, nC

include("capacity.jl")
export Capacity

include("operators.jl")
export AbstractOperators, ẟ_m, δ_p, Σ_m, Σ_p, I
export ∇, ∇₋
export DiffusionOps, ConvectionOps

include("boundary.jl")
export AbstractBoundary, Dirichlet, Neumann, Robin, Periodic
export AbstractInterfaceBC, ScalarJump, FluxJump, BorderConditions, InterfaceConditions

include("phase.jl")
export Phase

include("solver.jl")
export TimeType, PhaseType, EquationType
export Solver, solve_system!
export build_I_bc, build_I_D, build_source, build_g_g
export BC_border_mono!

include("solver/diffusion.jl")
export DiffusionSteadyMono, solve_DiffusionSteadyMono!

include("vizualize.jl")
export plot_solution

include("convergence.jl")
export check_convergence

include("vtk.jl")
export write_vtk
end
