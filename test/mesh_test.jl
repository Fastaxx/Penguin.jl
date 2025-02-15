using Penguin
using Test

@testset "1D mesh" begin
    x = collect(range(0.0, stop=1.0, length=5))
    mesh1D = Mesh((x,))
    borders1D = get_border_cells(mesh1D)
    @test mesh1D.centers == ([0.0, 0.25, 0.5, 0.75, 1.0],)
    @test nC(mesh1D) == 5
    @test mesh1D.sizes == ([0.125, 0.25, 0.25, 0.25, 0.125],)
    @test length(mesh1D.sizes[1]) == 5
    @test length(borders1D) == 2
    @test borders1D[1] == (CartesianIndex((1,)), (0.0,))
    @test borders1D[2] == (CartesianIndex((5,)), (1.0,))
end

@testset "2D mesh" begin
    x = collect(range(0.0, stop=1.0, length=5))
    y = collect(range(0.0, stop=1.0, length=5))
    mesh2D = Mesh((x, y))
    borders2D = get_border_cells(mesh2D)
    @test mesh2D.centers == ([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0])
    @test nC(mesh2D) == 25
    @test mesh2D.sizes == ([0.125, 0.25, 0.25, 0.25, 0.125], [0.125, 0.25, 0.25, 0.25, 0.125])
    @test length(mesh2D.sizes[1]) == 5
    @test length(mesh2D.sizes[2]) == 5
    @test length(borders2D) == 16
    @test borders2D[1] == (CartesianIndex((1, 1)), (0.0, 0.0))
    @test borders2D[2] == (CartesianIndex((2, 1)), (0.25, 0.0))
end

@testset "3D mesh" begin
    x = collect(range(0.0, stop=1.0, length=5))
    y = collect(range(0.0, stop=1.0, length=5))
    z = collect(range(0.0, stop=1.0, length=5))
    mesh3D = Mesh((x, y, z))
    borders3D = get_border_cells(mesh3D)
    @test mesh3D.centers == ([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0])
    @test nC(mesh3D) == 125
    @test mesh3D.sizes == ([0.125, 0.25, 0.25, 0.25, 0.125], [0.125, 0.25, 0.25, 0.25, 0.125], [0.125, 0.25, 0.25, 0.25, 0.125])
    @test length(mesh3D.sizes[1]) == 5
    @test length(mesh3D.sizes[2]) == 5
    @test length(mesh3D.sizes[3]) == 5
    @test length(borders3D) == 98
end

@testset "4D mesh" begin
    x = collect(range(0.0, stop=1.0, length=5))
    y = collect(range(0.0, stop=1.0, length=5))
    z = collect(range(0.0, stop=1.0, length=5))
    w = collect(range(0.0, stop=1.0, length=5))
    mesh4D = Mesh((x, y, z, w))
    borders4D = get_border_cells(mesh4D)
    @test mesh4D.centers == ([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0])
    @test nC(mesh4D) == 625
    @test mesh4D.sizes == ([0.125, 0.25, 0.25, 0.25, 0.125], [0.125, 0.25, 0.25, 0.25, 0.125], [0.125, 0.25, 0.25, 0.25, 0.125], [0.125, 0.25, 0.25, 0.25, 0.125])
    @test length(mesh4D.sizes[1]) == 5
    @test length(mesh4D.sizes[2]) == 5
    @test length(mesh4D.sizes[3]) == 5
    @test length(mesh4D.sizes[4]) == 5
    @test length(borders4D) == 544
end