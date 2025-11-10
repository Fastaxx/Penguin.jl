using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

"""
Scalar 1D Diffusion Heat Equation Convergence Test
This script performs a mesh-convergence study for the 1D heat equation
using the Penguin.jl library. It computes errors and estimated orders of convergence
for different mesh sizes and writes the results to a CSV file.
Analytical solution : u(x,t) = 4/π * Σ (1/(2n+1)) * sin((2n+1) * π * (x - (center - radius)) / (2 * radius)) * exp(-κ * ((2n+1) * π / (2 * radius))^2 * t)
where the sum is over n = 0 to ∞.
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

function heating_slab_solution(center::Float64, radius::Float64, t::Float64;
    κ::Float64 = 1.0,
    Nterms::Int = 800,
    tol::Float64 = 1e-12,
    w0::Float64 = 1.0
)
    x_left = center - radius
    x_right = center + radius
    L = 2 * radius

    return function (x::Float64)
        if (x < x_left) || (x > x_right)
            return 1.0
        end
        xi = x - x_left
        theta = 0.0
        for m in 0:(Nterms-1)
            n = 2m + 1
            lambda = n * π / L
            term =  (sin(lambda * xi)/n) * exp(-κ * lambda^2 * t)
            theta += term
            if abs(term) < tol
                break
            end
        end
        theta *= 4 / π
        return theta
    end
end

function run_mesh_convergence_heat_1d(
    nx_list::Vector{Int},
    radius::Float64,
    center::Float64,
    u_analytical::Function;
    lx::Float64 = 1.0,
    norm::Int = 2,
    Tend::Float64 = 0.1
)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    inside_cells = Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    for nx in nx_list
        mesh = Penguin.Mesh((nx,), (lx,), (0.0,))
        interval = (x, _=0) -> sqrt((x - center)^2) - radius
        capacity = Capacity(interval, mesh)
        operator = DiffusionOps(capacity)

        bc_boundary = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(
            :left  => Dirichlet(0.0),
            :right => Dirichlet(0.0)
        ))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        ndofs = (nx + 1)
        u0o = ones(ndofs)
        u0y = zeros(ndofs)
        u0 = vcat(u0o, u0y)

        Δt = 0.5 * (lx / nx)^2
        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "CN")
        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, "CN"; method=Base.:\)

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm, false)

        push!(h_vals, lx / nx)
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
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
    results_dir = isnothing(csv_path) ? joinpath(BENCH_ROOT, "results", "scalar") : dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "Scalar_1D_Diffusion_Heat_Dirichlet_Convergence.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing)
    nx_vals = isnothing(nx_list) ? [2, 4, 8, 16, 32] : nx_list
    radius = 0.25
    center = 0.5
    Tend = 0.1
    u_analytical = heating_slab_solution(center, radius, Tend; κ=1.0, Nterms=400)

    data = run_mesh_convergence_heat_1d(
        nx_vals, radius, center, u_analytical;
        lx = 1.0, norm = 2, Tend = Tend
    )

    csv_info = write_convergence_csv("Scalar_1D_Diffusion_Heat_Dirichlet", data; csv_path=csv_path)
    return (data = data, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()


@testset "Heat 1D convergence" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test orders.all > 1.0
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
