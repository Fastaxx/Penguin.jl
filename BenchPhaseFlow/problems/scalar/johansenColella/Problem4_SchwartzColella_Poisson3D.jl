using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

"""
Johansen–Colella Problem 4 (Schwartz Colella 3D Poisson):
Solve ΔΦ = -14 f inside a sphere of radius 0.392 embedded in a unit cube,
with the analytical solution Φ = f, where
    f(x,y,z) = sin(x) sin(2y) sin(3z).
Dirichlet boundary conditions enforce Φ = f on the immersed sphere surface and
Φ = 0 on the outer box. A mesh-convergence sweep records the errors.
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

radius() = 0.392
center_default() = (0.5, 0.5, 0.5)
domain_size() = (1.0, 1.0, 1.0)

function sphere_level_set(x,y,z)
    return sqrt((x-center_default()[1])^2 + (y-center_default()[2])^2 + (z-center_default()[3])^2) - radius()
end

f_exact(x,y,z) = sin(x) * sin(2y) * sin(3z)
function forcing(x,y,z) 
    return 14.0 * f_exact(x,y,z)
end

function run_poisson3d_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    nz_list::Vector{Int},
    center::Tuple{Float64,Float64,Float64};
    norm::Int = 2
)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    inside_cells = Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    lx, ly, lz = domain_size()
    body = sphere_level_set

    for (nx, ny, nz) in zip(nx_list, ny_list, nz_list)
        mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (0.0, 0.0, 0.0))
        capacity = Capacity(body, mesh; method="VOFI")
        operator = DiffusionOps(capacity)

        bc_outer = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(
            :left   => bc_outer,
            :right  => bc_outer,
            :top    => bc_outer,
            :bottom => bc_outer,
            :front  => bc_outer,
            :back   => bc_outer
        ))
        phase = Phase(capacity, operator, (x,y,z)->forcing(x,y,z), (x,y,z)->1.0)
        bc_interface = Dirichlet((x,y,z)->f_exact(x,y,z))

        solver = DiffusionSteadyMono(phase, bc_b, bc_interface)
        solve_DiffusionSteadyMono!(solver; method=Base.:\)

        u_func = (x,y,z) -> f_exact(x,y,z)
        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(u_func, solver, capacity, norm)

        push!(h_vals, min(lx / nx, ly / ny, lz / nz))
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity))
        Δx = lx / nx
        Δy = ly / ny
        Δz = lz / nz
        coverage_x = ceil(Int, 2 * radius() / Δx)
        coverage_y = ceil(Int, 2 * radius() / Δy)
        coverage_z = ceil(Int, 2 * radius() / Δz)
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
    results_dir = isnothing(csv_path) ? joinpath(BENCH_ROOT, "results", "scalar") : dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "$(method_name)_Convergence.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing, ny_list=nothing, nz_list=nothing)
    center = center_default()
    nx_vals = isnothing(nx_list) ? [8, 16, 32, 64] : nx_list
    ny_vals = isnothing(ny_list) ? nx_vals : ny_list
    nz_vals = isnothing(nz_list) ? nx_vals : nz_list

    data = run_poisson3d_convergence(nx_vals, ny_vals, nz_vals, center; norm=2)
    csv_info = write_convergence_csv("JohansenColella_P4", data; csv_path=csv_path)

    return (data=data, csv_path=csv_info.csv_path, table=csv_info.table)
end

results = main()

@testset "Johansen-Colella Problem 4" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
    println(orders)
end
