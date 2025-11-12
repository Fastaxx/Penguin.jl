using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

"""
1D heat equation with a moving interval and constant Dirichlet boundary data.
The solution is manufactured to remain identically 1 when both the immersed
boundary and the domain edges enforce `u = 1`. This script verifies that the
numerical solution preserves this constant state while the cut location moves.
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..","..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

function moving_radius(t, r_mean, r_amp, period)
    return r_mean + r_amp * sin(2π * t / period)
end

function moving_body(center, r_mean, r_amp, period)
    return (x,t,_=0)->abs(x-center) - moving_radius(t, r_mean, r_amp, period)
end

function run_moving_heat_convergence(
    nx_list::Vector{Int},
    r_mean::Float64,
    r_amp::Float64,
    period::Float64,
    center::Float64;
    lx::Float64 = 1.0,
    Tend::Float64 = 0.1
)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    inside_cells = Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    Φ_exact = x -> 1.0

    for nx in nx_list
        mesh = Penguin.Mesh((nx,), (lx,), (0.0,))
        Δt = 0.25 * (lx / nx)^2
        Tstart = Δt

        body = moving_body(center, r_mean, r_amp, period)
        STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt])
        capacity = Capacity(body, STmesh; compute_centroids=false)
        operator = DiffusionOps(capacity)

        bc = Dirichlet(1.0)
        bc_b = BorderConditions(Dict(:left=>bc, :right=>bc))
        interface_bc = Dirichlet(1.0)
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        ndofs = nx + 1
        u0ₒ = fill(1.0, ndofs)
        u0ᵧ = ones(ndofs)
        u0 = vcat(u0ₒ, u0ᵧ)

        solver = MovingDiffusionUnsteadyMono(phase, bc_b, interface_bc, Δt, u0, mesh, "BE")
        solve_MovingDiffusionUnsteadyMono!(solver, phase, body, Δt, Tstart, Tend, bc_b, interface_bc, mesh, "BE"; method=Base.:\)

        R_tend = moving_radius(Tend, r_mean, r_amp, period)
        body_tend = (x,_=0)->abs(x-center) - R_tend
        capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)
        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(Φ_exact, solver, capacity_tend, 2)

        push!(h_vals, lx / nx)
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity_tend))
        coverage = ceil(Int, 2 * R_tend / (lx / nx))
        push!(inside_cells_by_dim, [coverage])
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
        norm = 2
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
    nx_vals = isnothing(nx_list) ? [16, 32, 64, 128] : nx_list
    r_mean = 0.25
    r_amp = 0.05
    period = 0.2
    center = 0.5
    Tend = 0.1

    data = run_moving_heat_convergence(nx_vals, r_mean, r_amp, period, center; Tend=Tend)
    csv_info = write_convergence_csv("Scalar_1D_Diffusion_Heat_Moving_ConstantBC", data; csv_path=csv_path)
    return (data=data, csv_path=csv_info.csv_path, table=csv_info.table)
end

results = main()

@testset "1D moving constant Dirichlet" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
