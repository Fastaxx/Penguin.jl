using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

const JOHANSEN_COLELLA_DIR = @__DIR__
include(joinpath(JOHANSEN_COLELLA_DIR, "common.jl"))
using .JohansenColellaCommon

"""
Johansen–Colella moving-ellipse benchmark with Dirichlet data on both the outer
rectangle and all moving elliptical holes. The interface centers translate with
the prescribed velocities from the problem statement.
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

function run_moving_dirichlet_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    Tend::Float64;
    lx::Float64 = 3.0,
    ly::Float64 = 2.0,
    x0::Float64 = -1.5,
    y0::Float64 = -1.0
)
    body = body_function(true)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    inside_cells = Int[]
    inside_cells_by_dim = Vector{Vector{Int}}()

    for (nx, ny) in zip(nx_list, ny_list)
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
        k = 0.75
        Δt = k * (ly / ny)^2
        Tstart = Δt

        st_mesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt])
        capacity = Capacity(body, st_mesh)
        operator = DiffusionOps(capacity)

        bc = Dirichlet((x,y,t)->φ_exact(x,y,t))
        bc_b = BorderConditions(Dict(:left=>bc, :right=>bc, :top=>bc, :bottom=>bc))
        interface_bc = Dirichlet((x,y,t)->φ_exact(x,y,t))
        phase = Phase(capacity, operator, source_term, (x,y,z)->1.0)

        ndofs = (nx + 1)*(ny + 1)
        u0ₒ = [φ_exact(mesh.nodes[1][i], mesh.nodes[2][j], Tstart) for j in 1:ny+1, i in 1:nx+1]
        u0ₒ = reshape(u0ₒ, :)
        u0ᵧ = zeros(ndofs)
        u0 = vcat(u0ₒ, u0ᵧ)

        solver = MovingDiffusionUnsteadyMono(phase, bc_b, interface_bc, Δt, u0, mesh, "BE")
        solve_MovingDiffusionUnsteadyMono!(solver, phase, body, Δt, Tstart, Tend, bc_b, interface_bc, mesh, "BE"; method=Base.:\)

        body_tend = (x,y,_=0)->body(x,y,Tend)
        capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence((x,y)->φ_exact(x,y,Tend), solver, capacity_tend, 2)

        push!(h_vals, min(lx / nx, ly / ny))
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity_tend))
        Δx = lx / nx
        Δy = ly / ny
        coverage_x = ceil(Int, 2 * maximum(a for (_,_,a,_) in ELLIPSES) / Δx)
        coverage_y = ceil(Int, 2 * maximum(b for (_,_,_,b) in ELLIPSES) / Δy)
        push!(inside_cells_by_dim, [coverage_x, coverage_y])
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
    default_dir = joinpath(BENCH_ROOT, "results", "scalar")
    results_dir = isnothing(csv_path) ? default_dir : dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "$(method_name)_Convergence.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; csv_path=nothing, nx_list=nothing, ny_list=nothing)
    nx_vals = isnothing(nx_list) ? [4, 9, 16, 33] : nx_list
    ny_vals = isnothing(ny_list) ? nx_vals : ny_list
    Tend = 0.1

    data = run_moving_dirichlet_convergence(nx_vals, ny_vals, Tend)
    csv_info = write_convergence_csv("JohansenColella_Moving_Dirichlet", data; csv_path=csv_path)
    return (data=data, csv_path=csv_info.csv_path, table=csv_info.table)
end

results = main()

@testset "Johansen-Colella moving Dirichlet" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
