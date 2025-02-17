using Penguin
using Test

@testset "Mesh Test" begin
    # Write your tests here.
    include("mesh_test.jl")
end

@testset "Capacity Test" begin
    # Write your tests here.
    include("capacity_test.jl")
end

@testset "Operators Test" begin
    # Write your tests here.
    include("operators_test.jl")
end

@testset "Boundary Test" begin
    # Write your tests here.
    include("boundary_test.jl")
end

@testset "Phase Test" begin
    # Write your tests here.
    include("phase_test.jl")
end

@testset "Solver Test" begin
    # Write your tests here.
    include("solver_test.jl")
end

@testset "Convergence Test" begin
    # Write your tests here.
    include("convergence_test.jl")
end