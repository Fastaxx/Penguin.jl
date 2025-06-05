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
using FFTW
using DSP
using Colors
using Statistics
using LibGEOS
using GeoInterface

# Write your package code here.

include("mesh.jl")
export Mesh, get_border_cells, nC
export SpaceTimeMesh

include("front_tracking.jl")
export FrontTracker, create_circle!, create_rectangle!, create_ellipse!
export set_markers!, get_markers, add_marker!, get_fluid_polygon, is_point_inside, get_intersection, get_markers, sdf, compute_marker_normals, compute_volume_jacobian
export compute_capacities, fluid_cell_properties, compute_surface_capacities, compute_second_type_capacities
export compute_intercept_jacobian, compute_segment_cell_intersections, create_segment_line, compute_segment_parameters, update_front_with_intercept_displacements!
export compute_spacetime_capacities

include("front_tracking1D.jl")
export FrontTracker1D, compute_capacities_1d, sdf, is_point_inside
export compute_spacetime_capacities_1d

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

include("interpolation.jl")
export lin_interpol, quad_interpol, cubic_interpol

include("solver.jl")
export TimeType, PhaseType, EquationType
export Solver, solve_system!
export build_I_bc, build_I_D, build_source, build_g_g
export BC_border_mono!, BC_border_diph!
export cfl_restriction
export remove_zero_rows_cols!

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

include("liquidmotionsolver/diffusion2d.jl")
export MovingLiquidDiffusionUnsteadyMono2D, solve_MovingLiquidDiffusionUnsteadyMono2D!

include("liquidmotionsolver/stefan.jl")
export StefanMono2D, solve_StefanMono2D!,  solve_StefanAdvection!
export compute_volume_jacobian

include("concentrationsolver/species.jl")
export DiffusionUnsteadyConcentration, solve_DiffusionUnsteadyConcentration!

include("binarysolver/binary.jl")
export DiffusionUnsteadyBinary, solve_DiffusionUnsteadyBinary!

include("vizualize.jl")
export plot_solution, animate_solution
export plot_isotherms

include("vizualize_mov.jl")
export analyze_convergence_rates_newton, plot_timestep_history, plot_interface_evolution, plot_newton_residuals, analyze_interface_spectrum

include("convergence.jl")
export check_convergence, check_convergence_diph

include("vtk.jl")
export write_vtk
end
