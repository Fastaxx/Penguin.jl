module Penguin

using SparseArrays
using StaticArrays
using CartesianGeometry
using ImplicitIntegration
using LinearAlgebra
using IterativeSolvers
using CairoMakie
using WriteVTK
using LsqFit
using SpecialFunctions
using Roots
using Interpolations

# Write your package code here.

include("mesh.jl")
export Mesh, get_border_cells, nC
export SpaceTimeMesh

include("capacity.jl")
export Capacity

include("operators.jl")
export AbstractOperators, ẟ_m, δ_p, Σ_m, Σ_p, I
export ∇, ∇₋
export DiffusionOps, ConvectionOps

include("boundary.jl")
export AbstractBoundary, Dirichlet, Neumann, Robin, Periodic
export AbstractInterfaceBC, ScalarJump, FluxJump, BorderConditions, InterfaceConditions
export GibbsThomson

include("phase.jl")
export Phase

include("utils.jl")
export initialize_temperature_uniform!, initialize_temperature_square!, initialize_temperature_circle!, initialize_temperature_function!
export initialize_rotating_velocity_field, initialize_radial_velocity_field, initialize_poiseuille_velocity_field
export volume_redefinition!

include("solver.jl")
export TimeType, PhaseType, EquationType
export Solver, solve_system!
export build_I_bc, build_I_D, build_source, build_g_g
export BC_border_mono!, BC_border_diph!
export cfl_restriction

include("solver/diffusion.jl")
export DiffusionSteadyMono, solve_DiffusionSteadyMono!
export DiffusionSteadyDiph, solve_DiffusionSteadyDiph!
export DiffusionUnsteadyMono, solve_DiffusionUnsteadyMono!
export DiffusionUnsteadyDiph, solve_DiffusionUnsteadyDiph!

include("solver/advectiondiffusion.jl")
export AdvectionDiffusionSteadyMono, solve_AdvectionDiffusionSteadyMono!
export AdvectionDiffusionSteadyDiph, solve_AdvectionDiffusionSteadyDiph!
export AdvectionDiffusionUnsteadyMono, solve_AdvectionDiffusionUnsteadyMono!
export AdvectionDiffusionUnsteadyDiph, solve_AdvectionDiffusionUnsteadyDiph!

include("solver/darcy.jl")
export DarcyFlow, solve_DarcyFlow!, solve_darcy_velocity
export DarcyFlowUnsteady, solve_DarcyFlowUnsteady!

include("prescribedmotionsolver/diffusion.jl")
export MovingDiffusionUnsteadyMono, solve_MovingDiffusionUnsteadyMono!
export MovingDiffusionUnsteadyDiph, solve_MovingDiffusionUnsteadyDiph!

include("liquidmotionsolver/diffusion.jl")
export MovingLiquidDiffusionUnsteadyMono, solve_MovingLiquidDiffusionUnsteadyMono!
export MovingLiquidDiffusionUnsteadyDiph, solve_MovingLiquidDiffusionUnsteadyDiph!

include("vizualize.jl")
export plot_solution, animate_solution

include("convergence.jl")
export check_convergence, check_convergence_diph

include("vtk.jl")
export write_vtk
end
