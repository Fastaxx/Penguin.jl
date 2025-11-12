using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using DataFrames
using Test

"""
2D transient diffusion inside a circle with homogeneous Neumann conditions on
the immersed boundary (cut cells) and on the outer box. The initial field is
identically 1 and the source term is zero, so the total integral of the
solution should remain constant in time. This script marches the solution with
Backward Euler, records the volume integral, and reports the drift together
with the unexpectedly large absolute values produced by the null-space of the
pure Neumann operator.
"""

const BENCH_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(BENCH_ROOT, "utils", "convergence.jl"))

circle_level_set(center::Tuple{Float64,Float64}, radius::Float64) = function (x,y,_=0)
    sqrt((x - center[1])^2 + (y - center[2])^2) - radius
end

function compute_mass_history(solver, capacity::Capacity, Δt::Float64)
    ndofs = size(capacity.V, 1)
    volumes = diag(capacity.V)
    masses = Float64[]
    times = Float64[]
    for (step, state) in enumerate(solver.states)
        u = state[1:ndofs]
        push!(masses, dot(volumes, u))
        push!(times, (step - 1) * Δt)
    end
    return times, masses
end

function run_neumann_mass_conservation(
    nx::Int,
    ny::Int,
    radius::Float64,
    center::Tuple{Float64,Float64};
    lx::Float64 = 1.0,
    ly::Float64 = 1.0,
    Tend::Float64 = 0.1,
    scheme::String = "BE",
    cfl::Float64 = 0.25
)
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))
    body = circle_level_set(center, radius)
    capacity = Capacity(body, mesh; method="ImplicitIntegration")
    operator = DiffusionOps(capacity)

    neumann_zero = Neumann(0.0)
    bc_b = BorderConditions(Dict(
        :left   => neumann_zero,
        :right  => neumann_zero,
        :top    => neumann_zero,
        :bottom => neumann_zero
    ))
    bc_boundary = neumann_zero
    phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

    ndofs = (nx + 1) * (ny + 1)
    u0ₒ = ones(ndofs)
    u0ᵧ = ones(ndofs)
    u0 = vcat(u0ₒ, u0ᵧ)

    Δt = cfl * min((lx / nx)^2, (ly / ny)^2)
    solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, scheme)
    solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, scheme; method=Base.:\)

    times, masses = compute_mass_history(solver, capacity, Δt)
    mass0 = masses[1]
    drift = maximum(abs.(masses .- mass0))
    max_val = maximum(abs.(solver.x[1:ndofs]))

    return (
        times = times,
        masses = masses,
        mass0 = mass0,
        drift = drift,
        max_val = max_val,
        solver = solver,
        capacity = capacity
    )
end

function write_mass_csv(method_name::String, data; csv_path=nothing)
    df = DataFrame(time = data.times, mass = data.masses, drift = abs.(data.masses .- data.mass0))
    results_dir = isnothing(csv_path) ? joinpath(BENCH_ROOT, "results", "scalar") : dirname(csv_path)
    mkpath(results_dir)
    csv_out = isnothing(csv_path) ? joinpath(results_dir, "$(method_name)_Mass.csv") : csv_path
    CSV.write(csv_out, df)
    return (csv_path = csv_out, table = df)
end

function main(; nx=64, ny=64, csv_path=nothing)
    radius = 0.25
    center = (0.5, 0.5)
    Tend = 0.1

    data = run_neumann_mass_conservation(nx, ny, radius, center; Tend=Tend, scheme="CN")
    csv_info = write_mass_csv("Scalar_2D_Diffusion_Heat_NeumannMass", data; csv_path=csv_path)

    println("Initial mass: ", data.mass0)
    println("Maximum drift: ", data.drift)
    println("Peak |u| (null-space indicator): ", data.max_val)

    return (data = data, csv_path = csv_info.csv_path, table = csv_info.table)
end

results = main()

@testset "Neumann mass conservation" begin
    @test results.data.drift < 1e-10
    @test length(results.data.times) == length(results.data.masses)
    @test isfile(results.csv_path)
end
