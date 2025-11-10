using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using Roots
using CSV
using Test

"""
Scalar 1D Diffusion Heat Equation with Robin Boundary Conditions
Constant initial temperature `V` evolves under `u_t = κ u_{xx}` inside a slab,
with Robin interface conditions `∂ₙu + k u = 0`. The analytical solution follows
the cosine eigen-expansion
    u(x,t)/V = Σ_{n=0}^∞ (2L cos(αₙ ξ/L) sec(αₙ)) / (L(L+1)+αₙ²) * exp(-αₙ² t),
where `ξ = x - (center - radius)` maps the domain to `[0,L]`, `L = 2*radius`,
and the eigenvalues satisfy `α tan α = L`.

IN PROGRESS : ANALYTICAL SOLUTION NOT VERIFIED YET
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

function robin_alpha_roots(N::Int, L::Float64; eps::Float64 = 1e-6)
    eq(α) = α * tan(α) - L
    deq(α) = tan(α) + α / cos(α)^2
    roots = zeros(Float64, N)
    for n in 0:N-1
        guess = n * π + π / 4
        try
            roots[n+1] = find_zero((eq, deq), guess, Roots.Newton(); atol=1e-12, rtol=1e-12, maxiters=100)
        catch
            left = n * π + eps
            right = n * π + π / 2 - eps
            roots[n+1] = find_zero(eq, (left, right), Roots.Bisection(); atol=1e-12, maxiters=200)
        end
    end
    return roots
end

function robin_slab_solution(center::Float64, radius::Float64;
    V::Float64 = 1.0,
    t::Float64 = 0.1,
    κ::Float64 = 1.0,
    Nroots::Int = 400,
    tol::Float64 = 1e-12
)
    L = 2 * radius
    αs = robin_alpha_roots(Nroots, L)
    return function (x::Float64)
        if (x < center - radius) || (x > center + radius)
            return V
        end
        ξ = x - center
        s = 0.0
        for α in αs
            cα = cos(α)
            if abs(cα) < 1e-14
                continue
            end
            term = (2 * L * cos(α * ξ / L) / cα) / (L * (L + 1.0) + α^2) * exp(-κ * α^2 * t)
            s += term
            if abs(term) < tol
                break
            end
        end
        return V * s
    end
end

function run_robin_heat_1d(
    nx_list::Vector{Int},
    radius::Float64,
    center::Float64,
    u_analytical::Function;
    lx::Float64 = 1.0,
    norm::Int = 2,
    Tend::Float64 = 0.1,
    k::Float64 = 1.0,
    V::Float64 = 1.0
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
        body = (x, _=0) -> abs(x - center) - radius
        capacity = Capacity(body, mesh; method="ImplicitIntegration")
        operator = DiffusionOps(capacity)

        bc_boundary = Robin(k, 1.0, 0.0)
        bc_b = BorderConditions(Dict(
            :left  => Dirichlet(V),
            :right => Dirichlet(V)
        ))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        ndofs = nx + 1
        u0ₒ = fill(V, ndofs)
        u0ᵧ = zeros(ndofs)
        u0 = vcat(u0ₒ, u0ᵧ)

        Δt = 0.5 * (lx / nx)^2
        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "BE")
        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, "CN"; method=Base.:\)

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm)

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
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "$(method_name)_Convergence.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing)
    nx_vals = isnothing(nx_list) ? [8, 16, 32, 64, 128] : nx_list
    radius = 0.5
    center = 0.5
    Tend = 0.1
    k = 1.0
    V = 1.0
    u_analytical = robin_slab_solution(center, radius; V=V, t=Tend, κ=1.0, Nroots=400)

    data = run_robin_heat_1d(
        nx_vals, radius, center, u_analytical;
        lx = 1.0, norm = 2, Tend = Tend, k = k, V = V
    )

    csv_info = write_convergence_csv("Scalar_1D_Diffusion_Heat_Robin", data; csv_path=csv_path)
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
end
