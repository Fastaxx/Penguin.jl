using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test
using CairoMakie

"""
Transient diffusion benchmark on two disconnected circular domains (inner disk
and exterior of an outer disk inside a square box), with a void annulus in
between. Uses the manufactured harmonic solution φ⋆(x,y,t) = exp(-2π²κ t)
sin(πx) sin(πy) restricted to the active regions.
"""
const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

struct TwoRingParams
    L::Float64
    R₁::Float64
    R₂::Float64
    κ::Float64
    C::Float64
    K::Float64
    Tend::Float64
end

TwoRingParams(; L=1.0, R₁=0.31, R₂=0.6, κ=1.0, C=1.0, K=1.0, Tend=0.1) =
    TwoRingParams(L, R₁, R₂, κ, C, K, Tend)

φ_exact(x, y, t, params::TwoRingParams) =
    exp(-2π^2 * params.κ * t) * sin(π * x) * sin(π * y)

interval_min(a, b) = 0.5 * (a + b - abs(a - b))

function diffusion_levelset(params::TwoRingParams)
    return (x, y, _=0.0) -> begin
        r = sqrt(x^2 + y^2)
        φ_inner = r - params.R₁          # negative inside small disk
        φ_outer_ext = params.R₂ - r      # negative outside the large circle
        interval_min(φ_inner, φ_outer_ext)
    end
end

function run_two_ring_diffusion(nx_list::Vector{Int};
    params::TwoRingParams=TwoRingParams(),
    norm::Real=2,
    relative::Bool=false
)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    inside_cells = Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    for nx in nx_list
        mesh = Penguin.Mesh((nx, nx), (2*params.L, 2*params.L), (-params.L, -params.L))
        body = diffusion_levelset(params)
        capacity = Capacity(body, mesh)
        operator = DiffusionOps(capacity)

        bc = Dirichlet((x, y, t=0.0) -> φ_exact(x, y, t, params))
        bc_b = BorderConditions(Dict(
            :left => bc,
            :right => bc,
            :top => bc,
            :bottom => bc
        ))

        source = (x, y, z, t) -> 0.0
        Dcoeff = (x, y, z) -> params.K / params.C
        phase = Phase(capacity, operator, source, Dcoeff)

        ndofs = (nx + 1)^2
        u0φ = [φ_exact(mesh.nodes[1][i], mesh.nodes[2][j], 0.0, params)
               for j in 1:nx+1 for i in 1:nx+1]
        u0ᵧ = zeros(ndofs)
        u0 = vcat(u0φ, u0ᵧ)

        Δt = 0.5 * (2 * params.L / nx)^2
        solver = DiffusionUnsteadyMono(phase, bc_b, bc, Δt, u0, "CN")
        solve_DiffusionUnsteadyMono!(solver, phase, Δt, params.Tend, bc_b, bc, "CN"; method=Base.:\)

        # Debut plot
        fig = Figure(resolution = (800, 400))
        ax1 = Axis(fig[1, 1], title = "Numerical Solution at t=$(params.Tend)", xlabel = "x", ylabel = "y")
        unum = reshape(solver.x[1:ndofs], (nx+1, nx+1))
        heatmap!(ax1, mesh.nodes[1], mesh.nodes[2], unum'; colormap = :viridis)
        ax2 = Axis(fig[1, 2], title = "Exact Solution at t=$(params.Tend)", xlabel = "x", ylabel = "y")
        uexact = [φ_exact(mesh.nodes[1][i], mesh.nodes[2][j], params.Tend, params)
                  for j in 1:nx+1 for i in 1:nx+1]
        uexact_mat = reshape(uexact, (nx+1, nx+1))
        heatmap!(ax2, mesh.nodes[1], mesh.nodes[2], uexact_mat'; colormap = :viridis)
        
        display(fig)
        readline()

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence((x, y) -> φ_exact(x, y, params.Tend, params), solver, capacity, norm, relative)

        push!(h_vals, 2 * params.L / nx)
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity))
        push!(inside_cells_by_dim, [count_inside_cells(capacity)])
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
    results_dir = isnothing(csv_path) ?
        joinpath(BENCH_ROOT, "results", "scalar", "ConnectivityTwoCircles") :
        dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ?
        joinpath(results_dir, "$(method_name)_Convergence.csv") :
        csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing, params::TwoRingParams=TwoRingParams())
    nx_vals = isnothing(nx_list) ?  [64] : nx_list
    data = run_two_ring_diffusion(nx_vals; params=params)
    csv_info = write_convergence_csv("TwoRingDiffusion", data; csv_path=csv_path)
    return (data = data, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()

@testset "Two-ring diffusion convergence" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
