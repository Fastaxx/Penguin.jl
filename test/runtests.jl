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