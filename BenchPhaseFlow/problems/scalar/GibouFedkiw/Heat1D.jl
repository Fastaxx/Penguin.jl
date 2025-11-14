using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

φ_level(x) = abs(x) - 0.313
T_exact(x,t) = exp(-π^2 * t) * cos(π * x)
function run_heat1d(nx_list; lx=2.0, x0=-1.0, Tend=0.1)
    h_vals=Float64[]; err_vals=Float64[]; err_full_vals=Float64[]
    err_cut_vals=Float64[]; err_empty_vals=Float64[]; inside_cells=Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    body = (x,_=0)->φ_level(x)

    for nx in nx_list
        mesh = Penguin.Mesh((nx,), (lx,), (x0,))
        capacity = Capacity(body, mesh; method="ImplicitIntegration")
        operator = DiffusionOps(capacity)

        bc = Dirichlet((x,y,z,t)->T_exact(x,t))
        bc_b = BorderConditions(Dict(:left=>bc,:right=>bc))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        ndofs = nx + 1
        u0ₒ = [T_exact(mesh.nodes[1][i], 0.0) for i in 1:nx+1]
        u0ᵧ = zeros(ndofs)
        u0 = vcat(u0ₒ, u0ᵧ)

        Δt = 0.25 * (lx / nx)^2
        solver = DiffusionUnsteadyMono(phase, bc_b, bc, Δt, u0, "BE")
        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc, "CN"; method=Base.:\)

        capacity_tend = Capacity(body, mesh; compute_centroids=false)
        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(x->T_exact(x,Tend), solver, capacity_tend, 2)

        push!(h_vals, lx / nx)
        push!(err_vals, global_err); push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err); push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity))
        coverage = ceil(Int, 2 * 0.313 / (lx / nx))
        push!(inside_cells_by_dim, [coverage])
    end

    return (
        h_vals=h_vals, err_vals=err_vals, err_full_vals=err_full_vals,
        err_cut_vals=err_cut_vals, err_empty_vals=err_empty_vals,
        inside_cells=inside_cells, inside_cells_by_dim=inside_cells_by_dim,
        orders=compute_orders(h_vals, err_vals, err_full_vals, err_cut_vals),
        norm=2
    )
end

function write_convergence_csv(method_name, data; csv_path=nothing)
    df = make_convergence_dataframe(method_name, data)
    results_dir = isnothing(csv_path) ? joinpath(BENCH_ROOT, "results", "scalar", "GibouFedkiw") : dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "$(method_name)_Convergence.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path=csv_out, table=df)
end

function main(; csv_path=nothing, nx_list=nothing)
    nx_vals = isnothing(nx_list) ? [32, 64, 128, 256] : nx_list
    data = run_heat1d(nx_vals)
    csv_info = write_convergence_csv("GibouFedkiw_Heat1D", data; csv_path=csv_path)
    return (data=data, csv_path=csv_info.csv_path, table=csv_info.table)
end

results = main()

@testset "Gibou-Fedkiw Heat 1D" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
