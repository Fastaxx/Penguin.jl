using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using Test

"""
3D Schwartz–Colella heat equation on an expanding sphere Ω₃(t) = {(x,y,z): r < 0.392 + t}.
The analytic solution and source term are identical to the fixed case; the radius
grows with time while Dirichlet data track the exact field on the moving boundary
and the outer box.
NOTE : THIS SCRIPT CURRENTLY NOT WORKING
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

r_initial() = 0.392
center_default() = (0.5, 0.5, 0.5)

expanding_body(center) = (x,y,z,t)->sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - (r_initial() + t)

function a_exact(x,y,z,t)
    r2 = x^2 + y^2 + z^2
    return 4.0 / (5π * (t + 1)) * exp(-r2 / (5 * (t + 1)))
end

function source_term(x,y,z,t)
    r2 = x^2 + y^2 + z^2
    return 4.0 * (r2 + 5 * (t + 1)) / (125π * (t + 1)^3) * exp(-r2 / (5 * (t + 1)))
end

function run_expanding_sphere_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    nz_list::Vector{Int},
    center::Tuple{Float64,Float64,Float64},
    Tend::Float64;
    lx::Float64 = 1.0,
    ly::Float64 = 1.0,
    lz::Float64 = 1.0
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
        Δt = 0.25 * min((lx / nx)^2, (ly / ny)^2, (lz / nz)^2)
        Tstart = Δt

        body = expanding_body(center)
        st_mesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt])
        capacity = Capacity(body, st_mesh)
        operator = DiffusionOps(capacity)

        bc = Dirichlet((x,y,z,t)->a_exact(x,y,z,t))
        bc_b = BorderConditions(Dict(
            :left=>bc, :right=>bc,
            :top=>bc, :bottom=>bc,
            :front=>bc, :back=>bc
        ))
        interface_bc = Dirichlet((x,y,z,t)->a_exact(x,y,z,t))
        phase = Phase(capacity, operator, source_term, (x,y,z)->1.0)

        ndofs = (nx + 1)*(ny + 1)*(nz + 1)
        u0ₒ = [a_exact(mesh.nodes[1][i], mesh.nodes[2][j], mesh.nodes[3][k], Tstart)
            for k in 1:nz+1, j in 1:ny+1, i in 1:nx+1]
        u0ₒ = reshape(u0ₒ, :)
        u0ᵧ = zeros(ndofs)
        u0 = vcat(u0ₒ, u0ᵧ)

        solver = MovingDiffusionUnsteadyMono(phase, bc_b, interface_bc, Δt, u0, mesh, "BE")
        solve_MovingDiffusionUnsteadyMono!(solver, phase, body, Δt, Tstart, Tend, bc_b, interface_bc, mesh, "BE"; method=Base.:\)

        R_tend = r_initial() + Tend
        body_tend = (x,y,z,_=0)->sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - R_tend
        capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)

        _, _, global_err, full_err, cut_err, empty_err =
            check_convergence((x,y,z)->a_exact(x,y,z,Tend), solver, capacity_tend, 2)

        push!(h_vals, min(lx / nx, ly / ny, lz / nz))
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        push!(inside_cells, count_inside_cells(capacity_tend))
        coverage_x = ceil(Int, 2 * R_tend / (lx / nx))
        coverage_y = ceil(Int, 2 * R_tend / (ly / ny))
        coverage_z = ceil(Int, 2 * R_tend / (lz / nz))
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

function main(; csv_path=nothing, nx_list=nothing, ny_list=nothing, nz_list=nothing)
    center = center_default()
    nx_vals = isnothing(nx_list) ? [8, 12, 16] : nx_list
    ny_vals = isnothing(ny_list) ? nx_vals : ny_list
    nz_vals = isnothing(nz_list) ? nx_vals : nz_list
    Tend = 0.1

    data = run_expanding_sphere_convergence(nx_vals, ny_vals, nz_vals, center, Tend)
    csv_info = write_convergence_csv("SchwartzColella3D_ExpandingSphere", data; csv_path=csv_path)
    return (data=data, csv_path=csv_info.csv_path, table=csv_info.table)
end

results = main()

@testset "Schwartz-Colella 3D expanding sphere" begin
    orders = results.data.orders
    @test !isnan(orders.all)
    @test length(results.data.h_vals) == length(results.data.err_vals)
    @test results.data.h_vals[1] > results.data.h_vals[end]
    @test minimum(results.data.err_vals) < maximum(results.data.err_vals)
    @test isfile(results.csv_path)
end
