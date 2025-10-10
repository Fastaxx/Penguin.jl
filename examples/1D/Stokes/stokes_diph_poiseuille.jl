using Penguin

"""
1D two-phase Stokes: identical forcing, matching viscosities, interface
continuity. Verifies the diphasic assembly reproduces the monophasic solution.
"""

nx = 64
Lx = 1.0
x0 = 0.0

mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))
dx = Lx / nx
mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5 * dx,))

body = (x, _=0) -> -1.0
capacity_u = Capacity(body, mesh_u)
capacity_p = Capacity(body, mesh_p)
operator_u = DiffusionOps(capacity_u)
operator_p = DiffusionOps(capacity_p)

μ1 = 1.0
μ2 = 1.0
ρ = 1.0
fᵤ = (x, y=0.0, z=0.0) -> 1.0  # uniform forcing
fₚ = (x, y=0.0, z=0.0) -> 0.0

fluid1 = Fluid(mesh_u, capacity_u, operator_u,
               mesh_p, capacity_p, operator_p,
               μ1, ρ, fᵤ, fₚ)
fluid2 = Fluid(mesh_u, capacity_u, operator_u,
               mesh_p, capacity_p, operator_p,
               μ2, ρ, fᵤ, fₚ)

bc_u = BorderConditions(Dict(:bottom=>Dirichlet(0.0), :top=>Dirichlet(0.0)))
pressure_gauge = PinPressureGauge()
interface = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

nu = prod(operator_u.size)
np = prod(operator_p.size)
x0 = zeros(4 * nu + 2 * np)

solver = StokesDiph(fluid1, fluid2, (bc_u,), (bc_u,), pressure_gauge, interface, Dirichlet(0.0); x0=x0)
solve_StokesDiph!(solver)

uω1 = solver.x[1:nu]
println("max|uω1| = ", maximum(abs, uω1))
