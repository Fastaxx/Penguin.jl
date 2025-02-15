using Penguin
using Test

@testset "Capacity" begin
    x = range(-1.0, stop=1.0, length=50)
    y = range(-1.0, stop=1.0, length=50)
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