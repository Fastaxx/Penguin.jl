using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using Roots
using SpecialFunctions
using CSV
using Test

"""
Scalar 1D Diffusion Heat Equation with Robin Boundary Conditions
This benchmark mirrors `examples/1D/Diffusion/Heat_robin.jl`: a 1D transient
diffusion problem on `x ∈ [0, 10]` with interface position `center = 0.25`,
diffusivity `a = 5`, and homogeneous Robin condition `uₓ + u = 0` at the cut.
The exact solution used for convergence is the complementary-error-function
profile
```
u(x,t) = erf(η) + exp(k(x-center) + a k² t) ⋅ erfc(η + k √(a t)),
η = (x - center)/(2 √(a t)), k = 1,
```
which satisfies the same boundary data (Dirichlet `u=1` on the left, `u=0`
on the right) and source-free diffusion equation `u_t = a u_{xx}`.
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

function robin_erf_solution(center::Float64;
    a::Float64 = 5.0,
    k::Float64 = 1.0,
    t::Float64 = 1.0
)
    return function (x::Float64, _args...)
        if t == 0.0
            return 1.0
        end
        ξ = x - center
        η = ξ / (2 * sqrt(a * t))
        return erf(η) + exp(k * ξ + a * k^2 * t) * erfc(η + k * sqrt(a * t))
    end
end

function run_robin_heat_1d(
    nx_list::Vector{Int},
    center::Float64,
    u_analytical::Function;
    lx::Float64 = 10.0,
    norm= 2,
    Tend::Float64 = 1.0,
    robin_k::Float64 = 1.0,
    diffusivity::Float64 = 5.0
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
        body = (x, _=0) -> -(x - center)
        capacity = Capacity(body, mesh; method="ImplicitIntegration")
        operator = DiffusionOps(capacity)

        bc_boundary = Robin(robin_k, 1.0, 0.0)
        bc_b = BorderConditions(Dict(
            :left  => Dirichlet(1.0),
            :right => Dirichlet(0.0)
        ))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->diffusivity)

        ndofs = nx + 1
        u0ₒ = ones(ndofs)
        u0ᵧ = ones(ndofs)
        u0 = vcat(u0ₒ, u0ᵧ)

        Δt = 0.5 * (lx / nx)^2
        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "BE")
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
        coverage_x = ceil(Int, 2 * 1.0 / Δx)
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
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "$(method_name)_Convergence.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing)
    nx_vals = isnothing(nx_list) ? [4, 8, 16, 32, 64, 128] : nx_list
    center = 0.51
    Tend = 1.0
    robin_k = 1.0
    diffusivity = 5.0
    u_analytical = robin_erf_solution(center; a=diffusivity, k=robin_k, t=Tend)

    data = run_robin_heat_1d(
        nx_vals, center, u_analytical;
        lx = 10.0, norm = Inf, Tend = Tend, robin_k = robin_k, diffusivity = diffusivity
    )

    csv_info = write_convergence_csv("Scalar_1D_Diffusion_Heat_Robin_Linf", data; csv_path=csv_path)
    return (data = data, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()

@testset "Heat 1D Robin convergence" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test orders.all > 1.0
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
    println(orders)
end
