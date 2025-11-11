using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using Roots
using CSV
using Test

"""
Temporal convergence benchmark for the scalar 2D heat equation with Robin
interface conditions. The geometry, physics, and analytical reference are the
same as `Scalar_2D_Diffusion_Heat_Robin.jl`, but the mesh is fixed while the
time step varies to compare Backward Euler (BE) and Crank–Nicolson (CN).
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

function run_temporal_convergence(
    dt_list::Vector{Float64},
    scheme::String,
    nx::Int,
    ny::Int,
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64 = 4.0,
    ly::Float64 = 4.0,
    norm::Int = 2,
    Tend::Float64 = 0.1
)
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))
    circle = (x,y,_=0) -> sqrt((x - center[1])^2 + (y - center[2])^2) - radius
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

    ndofs = (nx + 1) * (ny + 1)
    u_initial = vcat(fill(w0, ndofs), zeros(ndofs))

    dt_vals = Float64[]
    err_vals = Float64[]
    err_full = Float64[]
    err_cut = Float64[]
    err_empty = Float64[]

    for Δt in dt_list
        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, copy(u_initial), scheme)
        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, scheme; method=Base.:\)
        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm)

        push!(dt_vals, Δt)
        push!(err_vals, global_err)
        push!(err_full, full_err)
        push!(err_cut, cut_err)
        push!(err_empty, empty_err)
    end

    return (
        h_vals = dt_vals,
        err_vals = err_vals,
        err_full_vals = err_full,
        err_cut_vals = err_cut,
        err_empty_vals = err_empty,
        inside_cells = [count_inside_cells(capacity) for _ in dt_vals],
        inside_cells_by_dim = [ [ceil(Int, 2 * radius / (lx / nx)), ceil(Int, 2 * radius / (ly / ny))] for _ in dt_vals ],
        orders = compute_orders(dt_vals, err_vals, err_full, err_cut),
        norm = norm
    )
end

function write_convergence_csv(method_name, data; csv_path=nothing)
    df = make_convergence_dataframe(method_name, data)
    results_dir = isnothing(csv_path) ? joinpath(BENCH_ROOT, "results", "scalar") : dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "$(method_name)_Temporal.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path_BE=nothing, csv_path_CN=nothing, dt_list=nothing, nx=256, ny=256)
    radius = 1.0
    center = (2.0, 2.0)
    Tend = 0.1
    dt_vals = isnothing(dt_list) ? [0.02, 0.01, 0.005, 0.0025] : dt_list
    u_analytical = robin_radial_heat_solution(center, radius; t=Tend)

    data_BE = run_temporal_convergence(dt_vals, "BE", nx, ny, radius, center, u_analytical; Tend=Tend)
    data_CN = run_temporal_convergence(dt_vals, "CN", nx, ny, radius, center, u_analytical; Tend=Tend)

    csv_BE = write_convergence_csv("Scalar_2D_Diffusion_Heat_Robin_BE", data_BE; csv_path=csv_path_BE)
    csv_CN = write_convergence_csv("Scalar_2D_Diffusion_Heat_Robin_CN", data_CN; csv_path=csv_path_CN)

    return (
        data_BE = data_BE,
        data_CN = data_CN,
        csv_BE = csv_BE.csv_path,
        csv_CN = csv_CN.csv_path
    )
end

results = main()

@testset "Heat 2D Robin temporal convergence" begin
    @test results.data_BE.orders.all < 0 ? false : true
    @test results.data_CN.orders.all < 0 ? false : true
    @test length(results.data_BE.h_vals) == length(results.data_BE.err_vals)
    @test length(results.data_CN.h_vals) == length(results.data_CN.err_vals)
    @test isfile(results.csv_BE)
    @test isfile(results.csv_CN)
    println("BE orders: ", results.data_BE.orders)
    println("CN orders: ", results.data_CN.orders)
end
