using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))
include(joinpath(@__DIR__, "common.jl"))
using .GibouFedkiwCommon

"""
3D Poisson problem on Ω = [0,1]^3 with interior sphere φ = √((x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2) - 0.3.
Analytical solution: u = exp(-x^2 - y^2 - z^2).
"""

u_exact(x,y,z) = exp(-x^2 - y^2 - z^2)
function f_source(x,y,z)
    u = u_exact(x,y,z)
    return u * (4x^2 + 4y^2 + 4z^2 - 6)
end

function run_poisson3d(nx_list, ny_list, nz_list)
    h_vals=Float64[]; err_vals=Float64[]; err_full_vals=Float64[]
    err_cut_vals=Float64[]; err_empty_vals=Float64[]; inside_cells=Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    body = (x,y,z,_=0)->GibouFedkiwCommon.sphere_levelset(x,y,z,(0.5,0.5,0.5),0.3)

    for (nx,ny,nz) in zip(nx_list, ny_list, nz_list)
        mesh = Penguin.Mesh((nx,ny,nz),(1.0,1.0,1.0),(0.0,0.0,0.0))
        capacity = Capacity(body, mesh; method="VOFI")
        operator = DiffusionOps(capacity)

        bc = Dirichlet((x,y,z)->u_exact(x,y,z))
        bc_b = BorderConditions(Dict(:left=>bc,:right=>bc,
                                     :top=>bc,:bottom=>bc,
                                     :front=>bc,:back=>bc))
        phase = Phase(capacity, operator, (x,y,z)->-f_source(x,y,z), (x,y,z)->1.0)

        ndofs = (nx+1)*(ny+1)*(nz+1)
        u0ₒ = [u_exact(mesh.nodes[1][i], mesh.nodes[2][j], mesh.nodes[3][k])
               for k in 1:nz+1, j in 1:ny+1, i in 1:nx+1]
        u0ₒ = reshape(u0ₒ,:)
        u0ᵧ = zeros(ndofs)
        u0 = vcat(u0ₒ, u0ᵧ)

        solver = DiffusionSteadyMono(phase, bc_b, bc)
        solve_DiffusionSteadyMono!(solver; method=Base.:\)

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence(u_exact, solver, capacity, 2)

        push!(h_vals, min(1.0/nx, 1.0/ny, 1.0/nz))
        push!(err_vals, global_err); push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err); push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity))
        coverage = ceil(Int, 0.6 / (1.0/nx))
        push!(inside_cells_by_dim, [coverage, coverage, coverage])
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

function main(; csv_path=nothing, nx_list=nothing, ny_list=nothing, nz_list=nothing)
    nx_vals = isnothing(nx_list) ? [8, 16, 32, 64] : nx_list
    ny_vals = isnothing(ny_list) ? nx_vals : ny_list
    nz_vals = isnothing(nz_list) ? nx_vals : nz_list
    data = run_poisson3d(nx_vals, ny_vals, nz_vals)
    csv_info = write_convergence_csv("GibouFedkiw_Poisson3D", data; csv_path=csv_path)
    return (data=data, csv_path=csv_info.csv_path, table=csv_info.table)
end

results = main()

@testset "Gibou-Fedkiw Poisson 3D" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
