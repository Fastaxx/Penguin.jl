using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

"""
Liu-Fedkiw diphasic diffusion benchmark (Case 1).

Problem: u_xx = 0 on [0, 1] with u(0) = 0, u(1) = 2, interface at x = 0.5, jump
conditions [u] = 1 and [u_x] = 0. Exact solution is u = x on the left of the
interface and u = x + 1 on the right.
"""
const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

const X_INTERFACE = 0.5

u_left_case1(x) = x
u_right_case1(x) = x + 1.0

function build_solver_case1(mesh)
    body = (x, _=0) -> x - X_INTERFACE
    body_c = (x, _=0) -> X_INTERFACE - x
    capacity1 = Capacity(body, mesh)
    capacity2 = Capacity(body_c, mesh)
    operator1 = DiffusionOps(capacity1)
    operator2 = DiffusionOps(capacity2)

    bc_b = BorderConditions(Dict(
        :bottom  => Dirichlet(0.0),
        :top => Dirichlet(2.0)
    ))

    ic = InterfaceConditions(
        ScalarJump(1.0, 1.0, -1.0),
        FluxJump(1.0, 1.0, 0.0)
    )

    f_zero = (x, y, _=0) -> 0.0
    D_one = (x, y, _=0) -> 1.0

    phase1 = Phase(capacity1, operator1, f_zero, D_one)
    phase2 = Phase(capacity2, operator2, f_zero, D_one)

    solver = DiffusionSteadyDiph(phase1, phase2, bc_b, ic)
    return solver, capacity1, capacity2
end

function run_liufedkiw_case1(nx_list; lx=1.0, x0=0.0)
    h_vals = Float64[]
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

    for nx in nx_list
        mesh = Penguin.Mesh((nx,), (lx,), (x0,))
        solver, capacity1, capacity2 = build_solver_case1(mesh)
        solve_DiffusionSteadyDiph!(solver; method=Base.:\)
        push!(solver.states, solver.x)


        _, _, global_errs, full_errs, cut_errs, empty_errs =
            check_convergence_diph(u_left_case1, u_right_case1, solver, capacity1, capacity2, 2, false)

        push!(h_vals, lx / nx)
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
        norm = 2
    )
end

function write_convergence_csv(method_name, data; csv_path=nothing)
    df = make_diphasic_convergence_dataframe(method_name, data)
    results_dir = isnothing(csv_path) ?
        joinpath(BENCH_ROOT, "results", "scalar", "diphasic", "LiuFedkiw") :
        dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ?
        joinpath(results_dir, "$(method_name)_Convergence.csv") :
        csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing)
    nx_vals = isnothing(nx_list) ? [8, 16, 32, 64, 128, 256] : nx_list
    data = run_liufedkiw_case1(nx_vals)
    csv_info = write_convergence_csv("LiuFedkiw_Case1", data; csv_path=csv_path)
    return (data = data, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()

@testset "Liu-Fedkiw Diphasic Case 1" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
