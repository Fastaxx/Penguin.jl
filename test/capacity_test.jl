using Penguin
using Test

@testset "1D Capacity" begin
    nx = 10
    lx = 2.0
    x0 = 0.0
    mesh = Mesh((nx,), (lx,), (x0,))
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
    nx, ny = 10, 10
    lx, ly = 2.0, 2.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
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
    nx, ny, nz = 10, 10, 10
    lx, ly, lz = 2.0, 2.0, 2.0
    x0, y0, z0 = 0.0, 0.0, 0.0
    mesh = Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2 + X[3]^2) - 0.5
    LS(x,y,z) = sqrt(x^2 + y^2 + z^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    @test capacity.mesh == mesh
    @test capacity.body == LS
    @test length(capacity.A) == 3
    @test length(capacity.B) == 3
    @test length(capacity.W) == 3
end