using Penguin
using Test

@testset "Utils Test" begin
    nx = 10
    ny = 10
    x_coords = collect(range(0.0, 1.0, length=nx+1))
    y_coords = collect(range(0.0, 1.0, length=ny+1))
    T0ₒ = zeros((nx+1)*(ny+1))
    T0ᵧ = zeros((nx+1)*(ny+1))

    initialize_temperature_uniform!(T0ₒ, T0ᵧ, 1.0)
    @test all(T0ₒ .== 1.0)
    @test all(T0ᵧ .== 1.0)

    T0ₒ = zeros((nx+1)*(ny+1))
    T0ᵧ = zeros((nx+1)*(ny+1))
    initialize_temperature_square!(T0ₒ, T0ᵧ, x_coords, y_coords, (0.5, 0.5), 2, 1.0, nx, ny)
    @test T0ₒ[1] == 0.0
    @test T0ᵧ[1] == 0.0
    @test T0ₒ[37] == 1.0

    T0ₒ = zeros((nx+1)*(ny+1))
    T0ᵧ = zeros((nx+1)*(ny+1))
    initialize_temperature_circle!(T0ₒ, T0ᵧ, x_coords, y_coords, (0.5, 0.5), 0.5, 1.0, nx, ny)
    @test T0ₒ[1] == 0.0
    @test T0ᵧ[1] == 0.0
    @test T0ₒ[37] == 1.0

    T0ₒ = zeros((nx+1)*(ny+1))
    T0ᵧ = zeros((nx+1)*(ny+1))
    initialize_temperature_function!(T0ₒ, T0ᵧ, x_coords, y_coords, (x, y)->1.0, nx, ny)
    @test T0ₒ[1:10] == ones(10)



end