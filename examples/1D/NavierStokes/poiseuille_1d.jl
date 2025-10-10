using Penguin
using LinearAlgebra

# Simple 1D Poiseuille-like flow solved with the steady Navier–Stokes solver.
# A constant streamwise body force (equivalent to a pressure gradient) drives the flow; convection is trivial in 1D but
# the example exercises the new 1D assembly path.

nx = 128
Lx = 1.0
x0 = 0.0

mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5 * dx,))

body = (x, _=0.0) -> -1.0
capacity_u = Capacity(body, mesh_u; compute_centroids=false)
capacity_p = Capacity(body, mesh_p; compute_centroids=false)

operator_u = DiffusionOps(capacity_u)
operator_p = DiffusionOps(capacity_p)

μ = 1.0
ρ = 1.0

Δp = 1.0
dpdx = -Δp / Lx
fᵤ = (x, y=0.0, z=0.0) -> -dpdx
fₚ = (x, y=0.0, z=0.0) -> 0.0

bc_u = BorderConditions(Dict(:left => Dirichlet(0.0), :right => Dirichlet(0.0)))
pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)

fluid = Fluid(mesh_u, capacity_u, operator_u, mesh_p, capacity_p, operator_p,
              μ, ρ, fᵤ, fₚ)

solver = NavierStokesMono(fluid, bc_u, pressure_gauge, bc_cut)

_, iters, res = solve_NavierStokesMono_steady!(solver; tol=1e-10, maxiter=50, relaxation=0.7)

println("Steady solve converged in $(iters) Picard iterations (residual=$(res))")

nu = prod(operator_u.size)
xs = mesh_u.nodes[1]
uω = solver.x[1:nu]

x_left = mesh_p.nodes[1][1]
x_right = mesh_p.nodes[1][end]
L = x_right - x_left
x_phys = clamp.(xs, x_left, x_right)
ξ = x_phys .- x_left
u_exact = -dpdx / (2μ) .* ξ .* (L .- ξ)
err = norm(uω - u_exact, Inf)
println("Infinity-norm error vs analytic parabola: $(err)")
