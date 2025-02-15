using Penguin
using Test

@testset "1D Capacity" begin
    x = range(-1.0, stop=1.0, length=10)
    mesh = Mesh((x,))
    Φ(X) = sqrt(X[1]^2) - 0.5
    LS(x,_=0) = sqrt(x^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    @test capacity.mesh == mesh
    @test capacity.body == LS
    @test length(capacity.A) == 1
    @test length(capacity.B) == 1
    @test length(capacity.W) == 1
end

@testset "2D Capacity" begin
    x = range(-1.0, stop=1.0, length=10)
    y = range(-1.0, stop=1.0, length=10)
    mesh = Mesh((x, y))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = sqrt(x^2 + y^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    @test capacity.mesh == mesh
    @test capacity.body == LS
    @test length(capacity.A) == 2
    @test length(capacity.B) == 2
    @test length(capacity.W) == 2
end

@testset "3D Capacity" begin
    x = range(-1.0, stop=1.0, length=20)
    y = range(-1.0, stop=1.0, length=20)
    z = range(-1.0, stop=1.0, length=20)
    mesh = Mesh((x, y, z))
    Φ(X) = sqrt(X[1]^2 + X[2]^2 + X[3]^2) - 0.5
    LS(x,y,z) = sqrt(x^2 + y^2 + z^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    @test capacity.mesh == mesh
    @test capacity.body == LS
    @test length(capacity.A) == 3
    @test length(capacity.B) == 3
    @test length(capacity.W) == 3
end