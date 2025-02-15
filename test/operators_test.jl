using Penguin
using Test

@testset "Gradient Divergence" begin
    nx, ny = 50, 50
    x = range(-1.0, stop=1.0, length=nx)
    y = range(-1.0, stop=1.0, length=ny)
    mesh = Mesh((x, y))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = sqrt(x^2 + y^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    operators = DiffusionOps(capacity)
    grad = ∇(operators, ones(2*length(mesh.nodes[1])*length(mesh.nodes[2])))
    @test grad == zeros(2*length(mesh.nodes[1])*length(mesh.nodes[2]))
    div = ∇₋(operators, ones(2*length(mesh.nodes[1])*length(mesh.nodes[2])), ones(2*length(mesh.nodes[1])*length(mesh.nodes[2])))
    @test div == zeros(length(mesh.nodes[1])*length(mesh.nodes[2]))
end

@testset "Diffusion Operators" begin
    nx, ny = 50, 50
    x = range(-1.0, stop=1.0, length=nx)
    y = range(-1.0, stop=1.0, length=ny)
    mesh = Mesh((x, y))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = sqrt(x^2 + y^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    operators = DiffusionOps(capacity)
    @test size(operators.G'*operators.Wꜝ*operators.G) == (length(mesh.nodes[1])*length(mesh.nodes[2]), length(mesh.nodes[1])*length(mesh.nodes[2]))
    @test operators.size == (length(mesh.nodes[1]), length(mesh.nodes[2]))
end

@testset "Convection Operators" begin
    nx, ny = 50, 50
    x = range(-1.0, stop=1.0, length=nx)
    y = range(-1.0, stop=1.0, length=ny)
    mesh = Mesh((x, y))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = sqrt(x^2 + y^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    nx, ny = length(mesh.nodes[1]), length(mesh.nodes[2])
    operators = ConvectionOps(capacity, (ones(nx*ny), ones(nx*ny)), zeros(2*nx*ny))
    @test size(operators.G'*operators.Wꜝ*operators.G) == (length(mesh.nodes[1])*length(mesh.nodes[2]), length(mesh.nodes[1])*length(mesh.nodes[2]))
    @test operators.size == (length(mesh.nodes[1]), length(mesh.nodes[2]))
end