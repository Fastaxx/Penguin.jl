using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

"""
Scalar 1D Diffusion Poisson Problem with Dirichlet Boundary Conditions
This script performs a mesh-convergence study for the 1D Poisson equation
using the Penguin.jl library. It computes errors and estimated orders of convergence
for different mesh sizes and writes the results to a CSV file.
Analytical solution is : u(x) = - (x-center)^3/6 - (center*(x-center)^2)/2 + radius^2/6*(x-center) + center*radius^2/2
"""


const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

function run_mesh_convergence_1d(
    nx_list::Vector{Int},
    center::Float64,
    radius::Float64,
    u_analytical::Function;
    lx::Float64 = 1.0,
    bc_value::Float64 = 0.0,
    norm = 2,
    relative::Bool = false
)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    inside_cells = Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    for nx in nx_list
        println("Mesh size: $nx")
        mesh = Penguin.Mesh((nx,), (lx,), (0.0,))
        body = (x, _=0) -> abs(x - center) - radius

        capacity = Capacity(body, mesh;method="ImplicitIntegration")
        operator = DiffusionOps(capacity)

        bc = Dirichlet(bc_value)
        bc_b = BorderConditions(Dict(:top => bc, :bottom => bc))
        phase = Phase(capacity, operator, (x,y,z)-> x, (x,y,z)->1.0)
        solver = DiffusionSteadyMono(phase, bc_b, bc)

        solve_DiffusionSteadyMono!(solver; method=Base.:\)

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm, relative)

        push!(h_vals, lx / nx)
        push!(err_vals,       global_err)
        push!(err_full_vals,  full_err)
        push!(err_cut_vals,   cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity))
        Δx = lx / nx
        coverage_x = ceil(Int, 2 * radius / Δx)
        push!(inside_cells_by_dim, [coverage_x])
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
        joinpath(results_dir, "$(method_name).csv") :
        csv_path

    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing)
    center = 0.5
    radius = 0.11
    nx_vals = isnothing(nx_list) ? [2, 4, 8, 16, 32, 64, 128, 256] : nx_list

    u_analytical(x) = - (x-center)^3/6 - (center*(x-center)^2)/2 +
                      radius^2/6*(x-center) + center*radius^2/2

    data = run_mesh_convergence_1d(
        nx_vals, center, radius, u_analytical;
        lx=1.0, bc_value=0.0, norm=2, relative=false
    )

    csv_info = write_convergence_csv("Scalar_1D_Diffusion_Poisson_Dirichlet_Convergence", data; csv_path=csv_path)

    return (data=data, csv_path=csv_info.csv_path, table=csv_info.table)
end

results = main()

@testset "Poisson 1D convergence" begin
    orders = results.data.orders
    @test orders.all > 1.0
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
