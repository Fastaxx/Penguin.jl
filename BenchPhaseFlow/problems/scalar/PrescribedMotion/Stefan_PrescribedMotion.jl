using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using Roots
using CSV
using Test

"""
One-phase Stefan problem with prescribed interface motion. The analytical
solution
    T(x,t) = T₀ - T₀/erf(λ) * erf(x / (2√(k t)))
is imposed on the moving boundary as Dirichlet data while the interface
position `s(t) = 2 λ √(k t)` is prescribed in the level-set function. This
script runs a mesh-convergence study using `MovingDiffusionUnsteadyMono`.

NOTE : THIS SCRIPT DON'T RUN CORRECTY YET
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..","..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

function find_lambda(Stefan_number)
    f(λ) = λ * exp(λ^2) * erf(λ) - Stefan_number / sqrt(π)
    return find_zero(f, 0.1)
end

analytical_T(x, t, T₀, k, λ) =
    t <= 0 ? T₀ : T₀ - T₀ / erf(λ) * erf(x / (2 * sqrt(k * max(t, eps(Float64)))))

interface_position(t, k, λ) = 2 * λ * sqrt(k * t)

function run_prescribed_stefan_convergence(
    nx_list::Vector{Int},
    T₀::Float64,
    k::Float64,
    Stefan_number::Float64;
    lx::Float64 = 2.0,
    x0::Float64 = 0.0,
    Tend::Float64 = 0.1
)
    λ = find_lambda(Stefan_number)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    inside_cells = Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    for nx in nx_list
        mesh = Penguin.Mesh((nx,), (lx,), (x0,))
        Δt = 0.25 * (lx / nx)^2
        Tstart = Δt

        body = (x,t,_=0)->x - interface_position(t, k, λ)
        st_mesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt])
        capacity = Capacity(body, st_mesh)
        operator = DiffusionOps(capacity)

        bc_left = Dirichlet(0.0)
        bc_right = Dirichlet(1.0)
        bc_b = BorderConditions(Dict(:left=>bc_left, :right=>bc_right))
        interface_bc = Dirichlet(1.0)

        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->k)

        ndofs = nx + 1
        u0ₒ = [analytical_T(mesh.nodes[1][i], Tstart, T₀, k, λ) for i in 1:nx+1]
        u0ᵧ = zeros(ndofs)
        u0 = vcat(u0ₒ, u0ᵧ)

        solver = MovingDiffusionUnsteadyMono(phase, bc_b, interface_bc, Δt, u0, mesh, "BE")
        solve_MovingDiffusionUnsteadyMono!(solver, phase, body, Δt, Tstart, Tend, bc_b, interface_bc, mesh, "BE"; method=Base.:\)

        body_tend = (x,_=0)->x - interface_position(Tend, k, λ)
        capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(x->analytical_T(x, Tend, T₀, k, λ), solver, capacity_tend, 2)

        push!(h_vals, lx / nx)
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity_tend))
        coverage = ceil(Int, 2 * interface_position(Tend, k, λ) / (lx / nx))
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
    T₀ = 1.0
    k = 1.0
    Stefan_number = 1.0
    Tend = 0.1

    data = run_prescribed_stefan_convergence(nx_vals, T₀, k, Stefan_number; Tend=Tend)
    csv_info = write_convergence_csv("Scalar_1D_Stefan_PrescribedMotion", data; csv_path=csv_path)
    return (data=data, csv_path=csv_info.csv_path, table=csv_info.table)
end

results = main()

@testset "Prescribed Stefan convergence" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
