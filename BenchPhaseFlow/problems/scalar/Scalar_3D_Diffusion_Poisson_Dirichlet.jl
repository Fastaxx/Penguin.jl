using Penguin
using LsqFit
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using CSV
using Test

"""
    Scalar 3D Diffusion Poisson Problem with Dirichlet Boundary Conditions
This script performs a mesh-convergence study for the 3D Poisson equation
using the Penguin.jl library. It computes errors and estimated orders of convergence
for different mesh sizes and writes the results to a CSV file.
Analytical solution is : u(x,y,z) = 1/6 * (radius^2 - ((x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2))
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

function run_mesh_convergence_3d(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    nz_list::Vector{Int},
    center::Tuple{Float64,Float64,Float64},
    radius::Float64,
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    lz::Float64=4.0,
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

    for (nx, ny, nz) in zip(nx_list, ny_list, nz_list)
        println("Mesh $(nx)×$(ny)×$(nz)")
        mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (0.0, 0.0, 0.0))

        body = (x, y, z, _=0) -> sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2) - radius
        capacity = Capacity(body, mesh)
        operator = DiffusionOps(capacity)

        bc_boundary = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(
            :left   => bc_boundary,
            :right  => bc_boundary,
            :top    => bc_boundary,
            :bottom => bc_boundary,
            :front  => bc_boundary,
            :back   => bc_boundary
        ))
        phase = Phase(capacity, operator, (x,y,z)->1.0, (x,y,z)->1.0)
        solver = DiffusionSteadyMono(phase, bc_b, Dirichlet(0.0))

        solve_DiffusionSteadyMono!(solver; method=Base.:\)

        _, _, all_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm, relative)

        push!(h_vals, min(lx / nx, ly / ny, lz / nz))
        push!(err_vals,       all_err)
        push!(err_full_vals,  full_err)
        push!(err_cut_vals,   cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity))
        Δx = lx / nx
        Δy = ly / ny
        Δz = lz / nz
        coverage_x = ceil(Int, 2 * radius / Δx)
        coverage_y = ceil(Int, 2 * radius / Δy)
        coverage_z = ceil(Int, 2 * radius / Δz)
        push!(inside_cells_by_dim, [coverage_x, coverage_y, coverage_z])
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

function write_convergence_csv(method_name, data; csv_path=nothing)
    df = make_convergence_dataframe(method_name, data)

    results_dir = csv_path === nothing ?
        joinpath(BENCH_ROOT, "results", "scalar") :
        dirname(csv_path)

    mkpath(results_dir)
    csv_out = csv_path === nothing ?
        joinpath(results_dir, "Scalar_3D_Diffusion_Poisson_Dirichlet_Convergence.csv") :
        csv_path

    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing, ny_list=nothing, nz_list=nothing)
    center = (2.0, 2.0, 2.0)
    radius = 0.5
    nx_vals = isnothing(nx_list) ? [8, 16, 32, 64] : nx_list
    ny_vals = isnothing(ny_list) ? [8, 16, 32, 64] : ny_list
    nz_vals = isnothing(nz_list) ? [8, 16, 32, 64] : nz_list

    u_analytical(x,y,z) = 1.0/6 * (radius^2 - ((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2))

    data = run_mesh_convergence_3d(
        nx_vals, ny_vals, nz_vals, center, radius, u_analytical;
        lx=4.0, ly=4.0, lz=4.0, norm=2, relative=false
    )

    csv_info = write_convergence_csv("Scalar_3D_Diffusion_Poisson_Dirichlet_Convergence", data; csv_path=csv_path)

    return (data=data, csv_path=csv_info.csv_path, table=csv_info.table)
end

results = main()

@testset "Poisson 3D convergence" begin
    orders = results.data.orders
    @test orders.all > 1.0
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
