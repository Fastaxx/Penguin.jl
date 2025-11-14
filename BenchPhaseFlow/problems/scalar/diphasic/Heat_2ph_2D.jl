using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using SpecialFunctions
using QuadGK
using Test

"""
Diphasic 2D heat diffusion benchmark reproduced from `benchmark/Heat_2ph_2D.jl`.
Performs a mesh-convergence study with a circular interface and writes CSV data
only (no plots).
"""
const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

struct Heat2Ph2DParams
    lx::Float64
    ly::Float64
    x0::Float64
    y0::Float64
    center::Tuple{Float64,Float64}
    radius::Float64
    Tend::Float64
    Dg::Float64
    Dl::Float64
    He::Float64
    cg0::Float64
    cl0::Float64
end

Heat2Ph2DParams(; lx=8.0, ly=8.0, x0=0.0, y0=0.0, center=(4.0, 4.0),
                radius=2.0, Tend=0.1, Dg=1.0, Dl=1.0, He=1.0,
                cg0=1.0, cl0=0.0) =
    Heat2Ph2DParams(lx, ly, x0, y0, center, radius, Tend, Dg, Dl, He, cg0, cl0)

phi_val(u, params::Heat2Ph2DParams) = begin
    D = sqrt(params.Dg / params.Dl)
    term1 = params.Dg * sqrt(params.Dl) * besselj1(u * params.radius) * bessely0(D * u * params.radius)
    term2 = params.He * params.Dl * sqrt(params.Dg) * besselj0(u * params.radius) * bessely1(D * u * params.radius)
    term1 - term2
end

psi_val(u, params::Heat2Ph2DParams) = begin
    D = sqrt(params.Dg / params.Dl)
    term1 = params.Dg * sqrt(params.Dl) * besselj1(u * params.radius) * besselj0(D * u * params.radius)
    term2 = params.He * params.Dl * sqrt(params.Dg) * besselj0(u * params.radius) * besselj1(D * u * params.radius)
    term1 - term2
end

function cg_integrand(u, x, y, params::Heat2Ph2DParams)
    r = hypot(x - params.center[1], y - params.center[2])
    Φu = phi_val(u, params)
    Ψu = psi_val(u, params)
    denom = u^2 * (Φu^2 + Ψu^2)
    numer = exp(-params.Dg * u^2 * params.Tend) * besselj0(u * r) * besselj1(u * params.radius)
    iszero(denom) ? 0.0 : numer / denom
end

function cl_integrand(u, x, y, params::Heat2Ph2DParams)
    r = hypot(x - params.center[1], y - params.center[2])
    Φu = phi_val(u, params)
    Ψu = psi_val(u, params)
    D = sqrt(params.Dg / params.Dl)
    denom = u * (Φu^2 + Ψu^2)
    contrib = besselj0(D * u * r) * Φu - bessely0(D * u * r) * Ψu
    numer = exp(-params.Dg * u^2 * params.Tend) * besselj1(u * params.radius) * contrib
    iszero(denom) ? 0.0 : numer / denom
end

function heat2d_phase1_solution(params::Heat2Ph2DParams)
    prefactor = (4 * params.cg0 * params.Dg * params.Dl^2 * params.He) / (π^2 * params.radius)
    Umax = 5.0 / sqrt(params.Dg * params.Tend)
    return (x, y) -> begin
        r = hypot(x - params.center[1], y - params.center[2])
        r >= params.radius && return 0.0
        val, _ = quadgk(u -> cg_integrand(u, x, y, params), 0, Umax; atol=1e-6, rtol=1e-6)
        prefactor * val
    end
end

function heat2d_phase2_solution(params::Heat2Ph2DParams)
    prefactor = (2 * params.cg0 * params.Dg * sqrt(params.Dl) * params.He) / π
    Umax = 5.0 / sqrt(params.Dg * params.Tend)
    return (x, y) -> begin
        r = hypot(x - params.center[1], y - params.center[2])
        r < params.radius && return 0.0
        val, _ = quadgk(u -> cl_integrand(u, x, y, params), 0, Umax; atol=1e-6, rtol=1e-6)
        prefactor * val
    end
end

function run_heat_2ph_2d(
    nx_list::Vector{Int};
    params::Heat2Ph2DParams=Heat2Ph2DParams(),
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

    u1_exact = heat2d_phase1_solution(params)
    u2_exact = heat2d_phase2_solution(params)

    for nx in nx_list
        ny = nx
        mesh = Penguin.Mesh((nx, ny), (params.lx, params.ly), (params.x0, params.y0))
        circle = (x, y, _=0) -> hypot(x - params.center[1], y - params.center[2]) - params.radius
        circle_c = (x, y, _=0) -> params.radius - hypot(x - params.center[1], y - params.center[2])
        capacity1 = Capacity(circle, mesh)
        capacity2 = Capacity(circle_c, mesh)
        operator1 = DiffusionOps(capacity1)
        operator2 = DiffusionOps(capacity2)

        bc_b = BorderConditions(Dict{Symbol,AbstractBoundary}())
        ic = InterfaceConditions(
            ScalarJump(1.0, params.He, 0.0),
            FluxJump(1.0, 1.0, 0.0)
        )

        f_zero = (x, y, z, t) -> 0.0
        D1_func = (x, y, z) -> params.Dg
        D2_func = (x, y, z) -> params.Dl

        phase1 = Phase(capacity1, operator1, f_zero, D1_func)
        phase2 = Phase(capacity2, operator2, f_zero, D2_func)

        ndofs = (nx + 1) * (ny + 1)
        u0 = vcat(ones(ndofs), ones(ndofs), zeros(ndofs), zeros(ndofs))
        Δt = 0.5 * (params.lx / nx)^2
        push!(dt_vals, Δt)

        solver = DiffusionUnsteadyDiph(phase1, phase2, bc_b, ic, Δt, u0, "BE")
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

function main(; csv_path=nothing, nx_list=nothing, params::Heat2Ph2DParams=Heat2Ph2DParams())
    nx_vals = isnothing(nx_list) ? [4, 8, 16, 32, 64, 128] : nx_list
    data = run_heat_2ph_2d(nx_vals; params=params)
    csv_info = write_convergence_csv("Heat_2ph_2D_He$(params.He)_Dl$(params.Dl)", data; csv_path=csv_path)
    return (data = data, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()

@testset "Diphasic Heat 2D convergence" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
