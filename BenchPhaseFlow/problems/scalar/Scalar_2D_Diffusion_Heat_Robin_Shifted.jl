using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using Roots
using CSV
using DataFrames
using Test

"""
Scalar 2D transient diffusion with Robin interface conditions. The circular
interface of radius `R` is translated within the square domain and the global
error at final time is averaged over the translation space
`(x₀, y₀) ∈ [-L/2, L/2]²`, truncated so the circle remains inside the box.
The integral `∫∫ e(x₀, y₀) dx₀ dy₀` is approximated by the uniform mean of the
sampled errors.
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

function run_heat_convergence(
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

    for (nx, ny) in zip(nx_list, ny_list)
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))
        circle = (x,y,_=0) -> sqrt((x-center[1])^2 + (y-center[2])^2) - radius
        capacity = Capacity(circle, mesh; method="VOFI")
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
    end

    return (
        h_vals = h_vals,
        err_vals = err_vals,
        err_full_vals = err_full_vals,
        err_cut_vals = err_cut_vals,
        err_empty_vals = err_empty_vals
    )
end

function average_shifted_errors(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    lx::Float64,
    ly::Float64;
    k::Float64 = 1.0,
    Tend::Float64 = 0.1,
    shift_points::Int = 5
)
    base_center = (2.0, 2.0)
    δx = min(0.2, min(base_center[1] - radius, lx - base_center[1] - radius))
    δy = min(0.2, min(base_center[2] - radius, ly - base_center[2] - radius))
    shift_vals_x = collect(range(-δx, δx, length=shift_points))
    shift_vals_y = collect(range(-δy, δy, length=shift_points))

    total_err = zeros(length(nx_list))
    total_full = zeros(length(nx_list))
    total_cut = zeros(length(nx_list))
    total_empty = zeros(length(nx_list))
    sample_count = 0
    reference_h = nothing

    for sx in shift_vals_x
        for sy in shift_vals_y
            center = (base_center[1] + sx, base_center[2] + sy)
            u_analytical = robin_radial_heat_solution(center, radius; t=Tend, k=k)
            data = run_heat_convergence(
                nx_list, ny_list, radius, center, u_analytical;
                lx = lx, ly = ly, Tend = Tend
            )
            total_err .+= data.err_vals
            total_full .+= data.err_full_vals
            total_cut .+= data.err_cut_vals
            total_empty .+= data.err_empty_vals
            reference_h = data.h_vals
            sample_count += 1
        end
    end

    mean_err = total_err ./ sample_count
    mean_full = total_full ./ sample_count
    mean_cut = total_cut ./ sample_count
    mean_empty = total_empty ./ sample_count

    return (
        h_vals = reference_h,
        mean_err = mean_err,
        mean_full_err = mean_full,
        mean_cut_err = mean_cut,
        mean_empty_err = mean_empty,
        samples = sample_count,
        shift_vals_x = shift_vals_x,
        shift_vals_y = shift_vals_y
    )
end

function write_mean_error_csv(method_name, data; csv_path=nothing)
    df = DataFrame(
        h = data.h_vals,
        mean_all_err = data.mean_err,
        mean_full_err = data.mean_full_err,
        mean_cut_err = data.mean_cut_err,
        mean_empty_err = data.mean_empty_err,
        samples = fill(data.samples, length(data.h_vals))
    )
    results_dir = isnothing(csv_path) ? joinpath(BENCH_ROOT, "results", "scalar") : dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "$(method_name)_MeanError.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing, shift_points=5)
    nx_vals = isnothing(nx_list) ? [4, 8, 16, 32, 64] : nx_list
    ny_vals = nx_vals
    radius = 1.0
    lx = 4.0
    ly = 4.0
    Tend = 0.1

    data = average_shifted_errors(
        nx_vals, ny_vals, radius, lx, ly;
        k = 1.0, Tend = Tend, shift_points = shift_points
    )

    csv_info = write_mean_error_csv("Scalar_2D_Diffusion_Heat_Robin_Shifted", data; csv_path=csv_path)
    return (data = data, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()

@testset "Heat 2D Robin shifted mean error" begin
    @test results.data.samples > 0
    @test length(results.data.h_vals) == length(results.data.mean_err)
    @test isfile(results.csv_path)
    println("Orders of convergence (mean error):")
    for (h, err) in zip(results.data.h_vals, results.data.mean_err)
        order = log(err / results.data.mean_err[end]) / log(h / results.data.h_vals[end])
        println("h = $h, mean error = $err, order = $order")
    end
    println("Orders of convergence (mean full error):")
    for (h, err) in zip(results.data.h_vals, results.data.mean_full_err)
        order = log(err / results.data.mean_full_err[end]) / log(h / results.data.h_vals[end])
        println("h = $h, mean full error = $err, order = $order")
    end
    println("Orders of convergence (mean cut error):")
    for (h, err) in zip(results.data.h_vals, results.data.mean_cut_err)
        order = log(err / results.data.mean_cut_err[end]) / log(h / results.data.h_vals[end])
        println("h = $h, mean cut error = $err, order = $order")
    end
end
