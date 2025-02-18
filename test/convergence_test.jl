using Penguin
using Test

@testset "Convergence Test 1D" begin
    nx = 40
    lx = 4.0
    x0 = 0.0
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    center = 0.5
    radius = 0.1
    LS(x,_=0) = sqrt((x-center)^2) - radius
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Dirichlet(0.0)
    bc1 = Dirichlet(0.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc1, :bottom => bc1))
    f(x,y,z) = x
    D(x,y,z) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    solve_DiffusionSteadyMono!(solver; method=Base.:\)
    u_analytic(x) = - (x-center)^3/6 - (center*(x-center)^2)/2 + radius^2/6 * (x-center) + center*radius^2/2
    u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytic, solver, capacity, 2, false)
    @test global_err < 1e-2
end

@testset "Convergence Test 2D" begin
    nx, ny = 40, 40
    lx, ly = 4., 4.
    x0, y0 = 0., 0.
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    LS(x,y,_=0) = (sqrt((x-2)^2 + (y-2)^2) - 1.0)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Dirichlet(0.0)
    bc1 = Dirichlet(1.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))
    f(x,y,_=0) = 4.0
    D(x,y,_=0) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    solve_DiffusionSteadyMono!(solver; method=Base.:\)
    u_analytic(x,y) = 1.0 - (x-2)^2 - (y-2)^2
    u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytic, solver, capacity, 2, false)
    @test global_err < 1e-2
end

@testset "Convergence Test 3D" begin
    nx, ny, nz = 40, 40, 40
    lx, ly, lz = 4., 4., 4.
    x0, y0, z0 = 0., 0., 0.
    mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))
    LS(x,y,z) = (sqrt((x-2)^2 + (y-2)^2 + (z-2)^2) - 1.0)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Dirichlet(0.0)
    bc1 = Dirichlet(1.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1, :front => bc1, :back => bc1))
    f(x,y,z) = 6.0
    D(x,y,z) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    solve_DiffusionSteadyMono!(solver; method=Base.:\)
    u_analytic(x,y,z) = 1.0 - (x-2)^2 - (y-2)^2 - (z-2)^2
    u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytic, solver, capacity, 2, false)
    @test global_err < 1e-2
end
