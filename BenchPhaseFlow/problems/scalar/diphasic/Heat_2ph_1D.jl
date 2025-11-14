using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using SpecialFunctions
using Test

"""
Diphasic 1D heat diffusion benchmark reproduced from `benchmark/Heat_2ph_1D.jl`.
Performs a mesh-convergence study for the unsteady diphasic heat equation with
a planar interface and writes convergence data (no plotting).
"""
const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

struct Heat2PhParams
    lx::Float64
    x0::Float64
    xint::Float64
    Tend::Float64
    He::Float64
    D1::Float64
    D2::Float64
end

Heat2PhParams(; lx=8.0, x0=0.0, xint=4.0, Tend=0.1, He=100.0, D1=1.0, D2=1.0) =
    Heat2PhParams(lx, x0, xint, Tend, He, D1, D2)

function heat_phase1_solution(params::Heat2PhParams)
    pref = -params.He / (1 + params.He * sqrt(params.D1 / params.D2))
    denom = 2 * sqrt(params.D1 * params.Tend)
    return x -> begin
        ξ = x - params.xint
        pref * (erfc(ξ / denom) - 2)
    end
end

function heat_phase2_solution(params::Heat2PhParams)
    pref = -params.He / (1 + params.He * sqrt(params.D1 / params.D2))
    denom = 2 * sqrt(params.D2 * params.Tend)
    return x -> begin
        ξ = x - params.xint
        pref * erfc(ξ / denom) + 1
    end
end

function run_heat_2ph_1d(
    nx_list::Vector{Int};
    params::Heat2PhParams=Heat2PhParams(),
    norm::Real=2,
    relative::Bool=false
)
    h_vals = Float64[]
    dt_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    phase1_all_errs = Float64[]
    phase1_full_errs = Float64[]
    phase1_cut_errs = Float64[]
    phase1_empty_errs = Float64[]
    phase2_all_errs = Float64[]
    phase2_full_errs = Float64[]
    phase2_cut_errs = Float64[]
    phase2_empty_errs = Float64[]
    inside_cells = Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()
    inside_cells_phase1 = Int[]
    inside_cells_phase2 = Int[]

    u1_exact = heat_phase1_solution(params)
    u2_exact = heat_phase2_solution(params)

    for nx in nx_list
        mesh = Penguin.Mesh((nx,), (params.lx,), (params.x0,))
        body = (x, _=0) -> x - params.xint
        body_c = (x, _=0) -> params.xint - x
        capacity1 = Capacity(body, mesh)
        capacity2 = Capacity(body_c, mesh)
        operator1 = DiffusionOps(capacity1)
        operator2 = DiffusionOps(capacity2)

        bc_b = BorderConditions(Dict(
            :bottom => Dirichlet(0.0),
            :top => Dirichlet(1.0)
        ))

        ic = InterfaceConditions(
            ScalarJump(1.0, params.He, 0.0),
            FluxJump(1.0, 1.0, 0.0)
        )

        f_zero = (x, y, z, t) -> 0.0
        D1_func = (x, y, z) -> params.D1
        D2_func = (x, y, z) -> params.D2

        phase1 = Phase(capacity1, operator1, f_zero, D1_func)
        phase2 = Phase(capacity2, operator2, f_zero, D2_func)

        ndofs = nx + 1
        u0 = vcat(zeros(ndofs), zeros(ndofs), ones(ndofs), ones(ndofs))
        Δt = 0.5 * (params.lx / nx)^2
        push!(dt_vals, Δt)

        solver = DiffusionUnsteadyDiph(phase1, phase2, bc_b, ic, Δt, u0, "CN")
        solve_DiffusionUnsteadyDiph!(solver, phase1, phase2, Δt, params.Tend, bc_b, ic, "CN"; method=Base.:\)
        push!(solver.states, solver.x)

        _, _, global_errs, full_errs, cut_errs, empty_errs =
            check_convergence_diph(u1_exact, u2_exact, solver, capacity1, capacity2, norm, relative)

        push!(h_vals, params.lx / nx)
        push!(err_vals, global_errs[3])
        push!(err_full_vals, full_errs[3])
        push!(err_cut_vals, cut_errs[3])
        push!(err_empty_vals, empty_errs[3])
        push!(phase1_all_errs, global_errs[1])
        push!(phase2_all_errs, global_errs[2])
        push!(phase1_full_errs, full_errs[1])
        push!(phase2_full_errs, full_errs[2])
        push!(phase1_cut_errs, cut_errs[1])
        push!(phase2_cut_errs, cut_errs[2])
        push!(phase1_empty_errs, empty_errs[1])
        push!(phase2_empty_errs, empty_errs[2])

        inside1 = count_inside_cells(capacity1)
        inside2 = count_inside_cells(capacity2)
        push!(inside_cells, inside1 + inside2)
        push!(inside_cells_by_dim, [inside1, inside2])
        push!(inside_cells_phase1, inside1)
        push!(inside_cells_phase2, inside2)
    end

    return (
        h_vals = h_vals,
        dt_vals = dt_vals,
        err_vals = err_vals,
        err_full_vals = err_full_vals,
        err_cut_vals = err_cut_vals,
        err_empty_vals = err_empty_vals,
        phase1_all_errs = phase1_all_errs,
        phase1_full_errs = phase1_full_errs,
        phase1_cut_errs = phase1_cut_errs,
        phase1_empty_errs = phase1_empty_errs,
        phase2_all_errs = phase2_all_errs,
        phase2_full_errs = phase2_full_errs,
        phase2_cut_errs = phase2_cut_errs,
        phase2_empty_errs = phase2_empty_errs,
        inside_cells = inside_cells,
        inside_cells_by_dim = inside_cells_by_dim,
        inside_cells_phase1 = inside_cells_phase1,
        inside_cells_phase2 = inside_cells_phase2,
        orders = compute_orders(h_vals, err_vals, err_full_vals, err_cut_vals),
        norm = norm
    )
end

function write_convergence_csv(method_name, data; csv_path=nothing)
    df = make_diphasic_convergence_dataframe(method_name, data)
    results_dir = isnothing(csv_path) ?
        joinpath(BENCH_ROOT, "results", "scalar", "diphasic") :
        dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ?
        joinpath(results_dir, "$(method_name)_Convergence.csv") :
        csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing, params::Heat2PhParams=Heat2PhParams())
    nx_vals = isnothing(nx_list) ? [4, 8, 16, 32, 64, 128, 256] : nx_list
    data = run_heat_2ph_1d(nx_vals; params=params)
    csv_info = write_convergence_csv("Heat_2ph_1D_He$(params.He)", data; csv_path=csv_path)
    return (data = data, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()

@testset "Diphasic Heat 1D convergence" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
