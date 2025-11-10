using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

"""
Scalar 3D Diffusion Heat Equation Convergence Test
This script performs a mesh-convergence study for the 3D heat equation
using the Penguin.jl library. It mirrors the 2D benchmark but for a sphere,
reporting errors and estimated convergence orders across meshes and exporting
results to a CSV file (no plotting or ad-hoc outputs).
The analytical solution used is the classical series for a cooling sphere:
u(r,t) = Tb + (T0 - Tb) * (2R)/(πr) * Σ_{n=1}^∞ (-1)^{n+1} (1/n) sin(nπr/R) exp(-κ (nπ/R)^2 t),
with the r → 0 limit handled analytically.
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

function spherical_heat_solution(center::Tuple{Float64,Float64,Float64}, radius::Float64;
    κ::Float64 = 1.0,
    t::Float64 = 0.1,
    T0::Float64 = 0.0,
    Tb::Float64 = 1.0,
    max_terms::Int = 200,
    tol::Float64 = 1e-12
)
    return function (x::Float64, y::Float64, z::Float64)
        r = sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2)
        if r >= radius
            return Tb
        end

        if r < 1e-12
            s = 0.0
            for n in 1:max_terms
                λ = n * π / radius
                term = (-1)^(n + 1) * exp(-κ * λ^2 * t)
                s += term
                if abs(term) < tol
                    break
                end
            end
            return Tb + (T0 - Tb) * 2 * s
        end

        s = 0.0
        for n in 1:max_terms
            λ = n * π / radius
            term = ((-1)^(n + 1) / n) * sin(λ * r) * exp(-κ * λ^2 * t)
            s += term
            if abs(term) < tol
                break
            end
        end
        return Tb + (T0 - Tb) * (2 * radius / (π * r)) * s
    end
end

function run_mesh_convergence_heat_3d(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    nz_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64,Float64},
    u_analytical::Function;
    lx::Float64 = 4.0,
    ly::Float64 = 4.0,
    lz::Float64 = 4.0,
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

    for (nx, ny, nz) in zip(nx_list, ny_list, nz_list)
        mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (0.0, 0.0, 0.0))
        sphere = (x, y, z, _=0) -> sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2) - radius
        capacity = Capacity(sphere, mesh)
        operator = DiffusionOps(capacity)

        bc_boundary = Dirichlet(1.0)
        bc_b = BorderConditions(Dict(
            :left   => bc_boundary,
            :right  => bc_boundary,
            :top    => bc_boundary,
            :bottom => bc_boundary,
            :front  => bc_boundary,
            :back   => bc_boundary
        ))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        ndofs = (nx + 1) * (ny + 1) * (nz + 1)
        u0ₒ = fill(0.0, ndofs)
        u0ᵧ = zeros(ndofs)
        u0 = vcat(u0ₒ, u0ᵧ)

        Δx = lx / nx
        Δy = ly / ny
        Δz = lz / nz
        Δt = 0.25 * min(Δx^2, Δy^2, Δz^2)

        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "BE")
        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, "BE"; method=Base.:\)

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm)

        push!(h_vals, min(Δx, Δy, Δz))
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity))
        coverage_x = ceil(Int, 2 * radius / Δx)
        coverage_y = ceil(Int, 2 * radius / Δy)
        coverage_z = ceil(Int, 2 * radius / Δz)
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
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "Scalar_3D_Diffusion_Heat_Dirichlet_Convergence.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing, ny_list=nothing, nz_list=nothing)
    nx_vals = isnothing(nx_list) ? [8, 12, 16, 20] : nx_list
    ny_vals = isnothing(ny_list) ? nx_vals : ny_list
    nz_vals = isnothing(nz_list) ? nx_vals : nz_list
    radius = 1.0
    center = (2.0, 2.0, 2.0)
    Tend = 0.1
    u_analytical = spherical_heat_solution(center, radius; κ=1.0, t=Tend, T0=0.0, Tb=1.0, max_terms=200)

    data = run_mesh_convergence_heat_3d(
        nx_vals, ny_vals, nz_vals, radius, center, u_analytical;
        lx = 4.0, ly = 4.0, lz = 4.0, norm = 2, Tend = Tend
    )

    csv_info = write_convergence_csv("Scalar_3D_Diffusion_Heat_Dirichlet", data; csv_path=csv_path)
    return (data = data, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()

@testset "Heat 3D convergence" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test orders.all > 1.0
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
