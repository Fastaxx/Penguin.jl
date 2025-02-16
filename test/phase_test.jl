using Penguin
using Test

@testset "1D Phase" begin
    x = range(-1.0, stop=1.0, length=10)
    mesh = Mesh((x,))
    Φ(X) = sqrt(X[1]^2) - 0.5
    LS(x,_=0) = sqrt(x^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    operators = DiffusionOps(capacity)
    f(x,_=0) = 0.0
    D(x,_=0) = 1.0
    Fluide = Phase(capacity, operators, f, D)
    @test Fluide.capacity == capacity
    @test Fluide.operator == operators
    @test Fluide.source == f
    @test Fluide.Diffusion_coeff == D
end

@testset "2D Phase" begin
    x = range(-1.0, stop=1.0, length=10)
    y = range(-1.0, stop=1.0, length=10)
    mesh = Mesh((x, y))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = sqrt(x^2 + y^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    operators = DiffusionOps(capacity)
    f(x,y,_=0) = 0.0
    D(x,y,_=0) = 1.0
    Fluide = Phase(capacity, operators, f, D)
    @test Fluide.capacity == capacity
    @test Fluide.operator == operators
    @test Fluide.source == f
    @test Fluide.Diffusion_coeff == D
end