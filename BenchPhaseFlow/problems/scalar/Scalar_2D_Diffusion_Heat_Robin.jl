using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using Roots
using CSV
using Test

"""
Benchmark: Scalar 2D Heat Equation with Robin Interface (legacy Heat.jl port)
This script reproduces the benchmark/Heat.jl workflow but in the BenchPhaseFlow
structure: it performs a mesh-convergence study for the 2D transient diffusion
problem around a heated circle with Robin interface conditions, logs errors, and
writes a single CSV summary (no plots or timestamped folders).
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

function robin_radial_heat_solution(center::Tuple{Float64,Float64}, R::Float64;
    t::Float64 = 0.1,
    k::Float64 = 1.0,
    a::Float64 = 1.0,
    wr::Float64 = 1.0,
    w0::Float64 = 0.0,
    Nzeros::Int = 200
)
    function j0_zeros_robin(N, k, R; guess_shift = 0.25)
        eq(alpha) = alpha * besselj1(alpha) - k * R * besselj0(alpha)
        zs = zeros(Float64, N)
        for m in 1:N
            x_left  = max((m - guess_shift - 0.5) * π, 1e-6)
            x_right = (m - guess_shift + 0.5) * π
            zs[m] = find_zero(eq, (x_left, x_right))
        end
        return zs
    end

    alphas = j0_zeros_robin(Nzeros, k, R)

    return function (x::Float64, y::Float64)
        r = sqrt((x - center[1])^2 + (y - center[2])^2)
        if r >= R
            return w0
        end
        s = 0.0
        for α in alphas
            An = 2.0 * k * R / ((k^2 * R^2 + α^2) * besselj0(α))
            s += An * exp(-a * α^2 * t / R^2) * besselj0(α * (r / R))
        end
        return (1.0 - s) * (wr - w0) + w0
    end
end

function run_heat_benchmark_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64 = 4.0,
    ly::Float64 = 4.0,
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

    for (nx, ny) in zip(nx_list, ny_list)
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))
        circle = (x,y,_=0) -> (sqrt((x-center[1])^2 + (y-center[2])^2) - radius)
        capacity = Capacity(circle, mesh)
        operator = DiffusionOps(capacity)

        w0 = 0.0
        wr = 1.0
        bc_boundary = Robin(1.0, 1.0, wr)
        bc0 = Dirichlet(w0)
        bc_b = BorderConditions(Dict(
            :left   => bc0,
            :right  => bc0,
            :top    => bc0,
            :bottom => bc0
        ))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        u0ₒ = fill(w0, (nx+1)*(ny+1))
        u0ᵧ = zeros((nx+1)*(ny+1))
        u0 = vcat(u0ₒ, u0ᵧ)

        Δt = 0.25 * (lx / nx)^2
        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "BE")
        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, "CN"; method=Base.:\)

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm)

        push!(h_vals, min(lx / nx, ly / ny))
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
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

function write_convergence_csv(method_name, data; csv_path=nothing)
    df = make_convergence_dataframe(method_name, data)
    results_dir = isnothing(csv_path) ? joinpath(BENCH_ROOT, "results", "scalar") : dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "$(method_name)_Convergence.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing)
    nx_vals = isnothing(nx_list) ? [4, 8, 16, 32, 64, 128] : nx_list
    ny_vals = nx_vals
    radius = 1.0
    center = (2.01, 2.01)
    u_analytical = robin_radial_heat_solution(center, radius)

    data = run_heat_benchmark_convergence(
        nx_vals, ny_vals, radius, center, u_analytical;
        lx = 4.0, ly = 4.0, norm = 2, Tend = 0.1
    )

    csv_info = write_convergence_csv("Scalar_2D_Diffusion_Heat_Robin", data; csv_path=csv_path)
    return (data = data, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()

@testset "Heat benchmark convergence" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test orders.all > 1.0
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
