using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using DataFrames
using Test

"""
Johansen–Colella Problem 3: Laplace equation inside a flower-shaped interface
(`r = 0.25 + 0.05 cos 6θ`) embedded in `[0,1]²`. Dirichlet data are ϕ = 1 on the
immersed boundary and ϕ = 0 on the outer box. This script mirrors the
`benchmark/flower.jl` setup but focuses solely on overshoot diagnostics, logging
statistics versus mesh size to a CSV file.
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const FLOWER_CENTER = (0.5, 0.5)

flower_level_set(x, y, _=0) = begin
    dx, dy = x - FLOWER_CENTER[1], y - FLOWER_CENTER[2]
    r = sqrt(dx^2 + dy^2)
    θ = atan(dy, dx)
    return -(r - (0.25 + 0.05 * cos(6θ)))
end

function solve_flower(nx::Int, ny::Int;
    lx::Float64 = 1.0,
    ly::Float64 = 1.0
)
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))
    capacity = Capacity(flower_level_set, mesh; method="ImplicitIntegration")
    operator = DiffusionOps(capacity)

    bc_outer = Dirichlet(0.0)
    bc_b = BorderConditions(Dict(
        :left   => bc_outer,
        :right  => bc_outer,
        :top    => bc_outer,
        :bottom => bc_outer
    ))
    phase = Phase(capacity, operator, (x,y,z)->0.0, (x,y,z)->1.0)
    solver = DiffusionSteadyMono(phase, bc_b, Dirichlet(1.0))
    solve_DiffusionSteadyMono!(solver; method=Base.:\)
    return solver, capacity
end

function overshoot_metrics(solver, capacity, nx::Int, ny::Int)
    ndofs = (nx + 1) * (ny + 1)
    values = solver.x[1:ndofs]

    cut_idx = findall(capacity.cell_types .== -1)
    inside_idx = findall(capacity.cell_types .== 1)

    overshoot_cells = cut_idx === [] ? 0 : count(values[cut_idx] .> 1.0)
    overshoot_pct = cut_idx === [] ? 0.0 : 100 * overshoot_cells / length(cut_idx)

    return (
        nx = nx,
        ny = ny,
        max_value = maximum(values),
        min_value = minimum(values),
        cut_cells = length(cut_idx),
        inside_cells = length(inside_idx),
        overshoot_cells = overshoot_cells,
        overshoot_percent = overshoot_pct
    )
end

function run_flower_study(nx_list::Vector{Int})
    stats = Vector{NamedTuple}(undef, length(nx_list))
    for (i, nx) in enumerate(nx_list)
        solver, capacity = solve_flower(nx, nx)
        stats[i] = overshoot_metrics(solver, capacity, nx, nx)
    end
    return stats
end

function write_stats_csv(method_name, stats; csv_path=nothing)
    df = DataFrame(stats)
    results_dir = isnothing(csv_path) ? joinpath(BENCH_ROOT, "results", "scalar") : dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "$(method_name)_Diagnostics.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; nx_list=nothing, csv_path=nothing)
    nx_vals = isnothing(nx_list) ? [4, 8, 16, 32, 64, 128, 256] : nx_list
    stats = run_flower_study(nx_vals)
    csv_info = write_stats_csv("JohansenColella_P3", stats; csv_path=csv_path)
    return (stats = stats, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()

@testset "Johansen-Colella Problem 3" begin
    @test length(results.stats) > 0
    @test isfile(results.csv_path)
    max_vals = [stat.max_value for stat in results.stats]
    @test maximum(max_vals) <= 1.1
end
