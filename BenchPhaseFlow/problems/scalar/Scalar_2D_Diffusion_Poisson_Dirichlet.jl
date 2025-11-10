using Penguin
using LsqFit
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using CSV
using Test

"""
    Scalar 2D Diffusion Poisson Problem with Dirichlet Boundary Conditions
This script performs a mesh-convergence study for the 2D Poisson equation
using the Penguin.jl library. It computes errors and estimated orders of convergence
for different mesh sizes and writes the results to a CSV file.
Analytical solution is : u(x,y) = 1 - (x-center_x)^2 - (y-center_y)^2
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

"""
    run_mesh_convergence_ft(nx_list, ny_list, radius, center, u_analytical; kwargs...)

Compute convergence data for the front tracking method.
"""
function run_mesh_convergence_ft(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    norm,
    relative::Bool=false
)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    inside_cells = Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    for (nx, ny) in zip(nx_list, ny_list)
        println("Front Tracking mesh $(nx)×$(ny)")
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))

        front = FrontTracker()
        resolution = max(500, min(nx, ny) * 10)
        create_circle!(front, center[1], center[2], radius, resolution)

        capacity = Capacity(front, mesh)
        operator = DiffusionOps(capacity)

        bc_boundary = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(
            :left   => bc_boundary,
            :right  => bc_boundary,
            :top    => bc_boundary,
            :bottom => bc_boundary
        ))
        phase = Phase(capacity, operator, (x,y,_)->4.0, (x,y,_)->1.0)
        solver = DiffusionSteadyMono(phase, bc_b, Dirichlet(0.0))

        solve_DiffusionSteadyMono!(solver; method=Base.:\)

        _, _, all_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm, relative)

        push!(h_vals, min(lx / nx, ly / ny))
        push!(err_vals,       all_err)
        push!(err_full_vals,  full_err)
        push!(err_cut_vals,   cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity))
        Δx = lx / nx
        Δy = ly / ny
        coverage_x = ceil(Int, 2 * radius / Δx)
        coverage_y = ceil(Int, 2 * radius / Δy)
        push!(inside_cells_by_dim, [coverage_x, coverage_y])
    end

    return (
        h_vals = h_vals,
        err_vals = err_vals,
        err_full_vals = err_full_vals,
        err_cut_vals = err_cut_vals,
        err_empty_vals = err_empty_vals,
        inside_cells = inside_cells,
        inside_cells_by_dim = inside_cells_by_dim,
        orders = compute_orders(h_vals, err_vals, err_full_vals, err_cut_vals),
        norm = norm
    )
end

"""
    run_mesh_convergence(nx_list, ny_list, radius, center, u_analytical; kwargs...)

Compute convergence data for the implicit integration method.
"""
function run_mesh_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    norm,
    relative::Bool=false
)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    inside_cells = Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    for (nx, ny) in zip(nx_list, ny_list)
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))
        circle = (x,y) -> (sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

        capacity = Capacity(circle, mesh; method="ImplicitIntegration")
        operator = DiffusionOps(capacity)

        bc_boundary = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(
            :left   => bc_boundary,
            :right  => bc_boundary,
            :top    => bc_boundary,
            :bottom => bc_boundary
        ))
        phase = Phase(capacity, operator, (x,y,_)->4.0, (x,y,_)->1.0)
        solver = DiffusionSteadyMono(phase, bc_b, Dirichlet(0.0))

        solve_DiffusionSteadyMono!(solver; method=Base.:\)

        _, _, all_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm, relative)

        push!(h_vals, min(lx / nx, ly / ny))
        push!(err_vals,       all_err)
        push!(err_full_vals,  full_err)
        push!(err_cut_vals,   cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity))
        Δx = lx / nx
        Δy = ly / ny
        coverage_x = ceil(Int, 2 * radius / Δx)
        coverage_y = ceil(Int, 2 * radius / Δy)
        push!(inside_cells_by_dim, [coverage_x, coverage_y])
    end

    return (
        h_vals = h_vals,
        err_vals = err_vals,
        err_full_vals = err_full_vals,
        err_cut_vals = err_cut_vals,
        err_empty_vals = err_empty_vals,
        inside_cells = inside_cells,
        inside_cells_by_dim = inside_cells_by_dim,
        orders = compute_orders(h_vals, err_vals, err_full_vals, err_cut_vals),
        norm = norm
    )
end

function compare_methods(; csv_path=nothing)
    nx_list = [4, 8, 16, 32, 64, 128]
    ny_list = nx_list
    radius, center = 1.0, (2.0, 2.0)
    u_analytical(x,y) = 1.0 - (x-center[1])^2 - (y-center[2])^2

    println("Running Front Tracking convergence...")
    ft_data = run_mesh_convergence_ft(nx_list, ny_list, radius, center, u_analytical, norm=2, relative=false)

    println("Running implicit integration convergence...")
    vofi_data = run_mesh_convergence(nx_list, ny_list, radius, center, u_analytical, norm=2, relative=false)

    ft_df = make_convergence_dataframe("FrontTracking", ft_data)
    vofi_df = make_convergence_dataframe("ImplicitIntegration", vofi_data)

    combined = vcat(ft_df, vofi_df)

    results_dir = csv_path === nothing ?
        joinpath(BENCH_ROOT, "results", "scalar") :
        dirname(csv_path)

    mkpath(results_dir)
    csv_out = csv_path === nothing ?
        joinpath(results_dir, "Scalar_2D_Diffusion_Poisson_Dirichlet_Convergence.csv") :
        csv_path

    CSV.write(csv_out, combined)

    return (
        csv_path = csv_out,
        front_tracking = (data = ft_data, table = ft_df),
        implicit_integration = (data = vofi_data, table = vofi_df)
    )
end

function main(; csv_path=nothing)
    results = compare_methods(; csv_path=csv_path)
    return results
end

    results = main()

    @testset "Poisson convergence benchmarks" begin
        ft_orders = results.front_tracking.data.orders
        vofi_orders = results.implicit_integration.data.orders

        @test ft_orders.all > 1.0
        @test vofi_orders.all > 1.0
        @test length(results.front_tracking.data.h_vals) == length(results.implicit_integration.data.h_vals)
        @test length(results.front_tracking.data.h_vals) == length(results.front_tracking.data.err_vals)
        @test minimum(results.front_tracking.data.err_vals) <
              maximum(results.front_tracking.data.err_vals)
        @test isfile(results.csv_path)
    end
