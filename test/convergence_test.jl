using Penguin
using Test

@testset "Convergence Test" begin
    nx, ny = 40, 40
    lx, ly = 4., 4.
    x0, y0 = 0., 0.
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
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