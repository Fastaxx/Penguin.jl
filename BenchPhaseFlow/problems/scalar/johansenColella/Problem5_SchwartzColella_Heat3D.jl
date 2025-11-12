using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

"""
Johansen–Colella Problem 5: 3D heat equation inside a sphere embedded in a unit
box with the analytic solution
    Φ(x,y,z,t) = 4 / (5π (t+1)) * exp(-(x² + y² + z²)/(5 (t+1))).
The source term is
    f(x,y,z,t) = 4 * (x² + y² + z² + 5(t+1)) / (125 π (t+1)³)
                 * exp(-(x² + y² + z²)/(5 (t+1))),
and Dirichlet boundary conditions enforce Φ on the immersed sphere. A transient
simulation to `Tend` collects the final-time error across several meshes.
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

radius() = 0.392
center_default() = (0.5, 0.5, 0.5)
domain_size() = (1.0, 1.0, 1.0)

function sphere_level_set(center::Tuple{Float64,Float64,Float64})
    (x,y,z,_=0) -> sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - radius()
end

function phi_exact(x,y,z,t)
    r2 = x^2 + y^2 + z^2
    denom = 5 * π * (t + 1)
    return 4.0 / denom * exp(-r2 / (5 * (t + 1)))
end

function source_term(x,y,z,t)
    r2 = x^2 + y^2 + z^2
    denom = 125 * π * (t + 1)^3
    return 4.0 * (r2 + 5 * (t + 1)) / denom * exp(-r2 / (5 * (t + 1)))
end

function run_heat3d_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    nz_list::Vector{Int},
    center::Tuple{Float64,Float64,Float64};
    Tend::Float64 = 0.1,
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
    body = sphere_level_set(center)

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
        bc_interface = Dirichlet((x,y,z)->phi_exact(x,y,z,Tend))

        phase = Phase(capacity, operator, (x,y,z,t)->source_term(x,y,z,t), (x,y,z)->1.0)

        ndofs = (nx + 1)*(ny + 1)*(nz + 1)
        u0 = vcat(fill(phi_exact(0.0,0.0,0.0,0.0), ndofs), zeros(ndofs))

        Δx = min(lx / nx, ly / ny, lz / nz)
        Δt = 0.25 * Δx^2
        solver = DiffusionUnsteadyMono(phase, bc_b, bc_interface, Δt, u0, "BE")
        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_interface, "CN"; method=Base.:\)

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence((x,y,z)->phi_exact(x,y,z,Tend), solver, capacity, norm)

        push!(h_vals, Δx)
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity))
        coverage_x = ceil(Int, 2 * radius() / (lx / nx))
        coverage_y = ceil(Int, 2 * radius() / (ly / ny))
        coverage_z = ceil(Int, 2 * radius() / (lz / nz))
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
    nx_vals = isnothing(nx_list) ? [8, 12, 18] : nx_list
    ny_vals = isnothing(ny_list) ? nx_vals : ny_list
    nz_vals = isnothing(nz_list) ? nx_vals : nz_list
    Tend = 0.1

    data = run_heat3d_convergence(nx_vals, ny_vals, nz_vals, center; Tend=Tend)

    csv_info = write_convergence_csv("JohansenColella_P5", data; csv_path=csv_path)
    return (data=data, csv_path=csv_info.csv_path, table=csv_info.table)
end

results = main()

@testset "Johansen-Colella Problem 5" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
    println(orders)
end
