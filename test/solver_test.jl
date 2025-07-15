using Penguin
using Test
using SparseArrays
using IterativeSolvers  # Add this import for bicgstabl

@testset "Solver test" begin
    nx, ny = 20, 20
    lx, ly = 2.0, 2.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = (sqrt((x-0.5)^2 + (y-0.5)^2) - 0.5)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Dirichlet(1.0)
    bc1 = Dirichlet(1.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))
    f(x,y,_=0) = 0.0
    D(x,y,_=0) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    solve_DiffusionSteadyMono!(solver)
    uo = solver.x[1:end÷2]
    ug = solver.x[end÷2+1:end]
    @test maximum(uo) ≈ 1.0 atol=1e-2
    @test maximum(ug) ≈ 1.0 atol=1e-2
    
    # Test with Robin boundary conditions
    @testset "Robin boundary conditions" begin
        # Robin BC with parameters: a*u + b*∇u·n = c
        # Here we use a=1, b=2, c=1 which gives u=1 at the boundary when ∇u·n=0
        robin_bc = Robin(1.0, 2.0, 1.0)
        
        # Apply Robin BC to all boundaries
        bc_robin = BorderConditions(Dict{Symbol, AbstractBoundary}(
            :left => robin_bc, 
            :right => robin_bc, 
            :top => robin_bc, 
            :bottom => robin_bc
        ))
        
        # Create solver with Robin boundary conditions
        solver_robin = DiffusionSteadyMono(Fluide, bc_robin, bc)
        solve_DiffusionSteadyMono!(solver_robin)
        
        # Extract solution
        uo_robin = solver_robin.x[1:end÷2]
        ug_robin = solver_robin.x[end÷2+1:end]
        
        # The solution should be close to 1.0 at the boundary
        @test maximum(uo_robin) ≈ 1.0 atol=1e-1
        @test maximum(ug_robin) ≈ 1.0 atol=1e-1
    end
    
    # Test with bicgstabl solver
    @testset "bicgstabl solver" begin
        # Use the same problem as before but with a different linear solver
        solver_bicgstabl = DiffusionSteadyMono(Fluide, bc_b, bc)
        
        # Solve using bicgstabl (use smaller tolerances for better accuracy)
        solve_DiffusionSteadyMono!(
            solver_bicgstabl, 
            method=IterativeSolvers.bicgstabl,
        )
        
        # Extract solution
        uo_bicgstabl = solver_bicgstabl.x[1:end÷2]
        ug_bicgstabl = solver_bicgstabl.x[end÷2+1:end]
        
        # The solution should match the direct solver result
        @test maximum(uo_bicgstabl) ≈ 1.0 atol=1e-2
        @test maximum(ug_bicgstabl) ≈ 1.0 atol=1e-2
        
        # Compare with the direct solver solution
        @test norm(uo_bicgstabl - uo) / norm(uo) < 1e-2
        @test norm(ug_bicgstabl - ug) / norm(ug) < 1e-2
    end
end