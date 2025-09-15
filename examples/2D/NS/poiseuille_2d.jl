using Penguin
using CairoMakie
using SparseArrays
using LinearAlgebra
using IterativeSolvers
"""
2D Stokes Poiseuille flow (steady): u = (Ux(y), 0), p = const or linear.
We enforce a parabolic x-velocity profile via Dirichlet on left/right and no-slip on
top/bottom. This serves as a structural test of the 2D Stokes assembly and BCs.
Analytical profile (channel of height Ly): U(y) = Umax * 4 * (y/Ly) * (1 - y/Ly).
"""
###########
# Grids
###########
nx, ny = 64, 64
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, 0.0
# Pressure grid (cell-centered)
mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
# Component-wise staggered velocity grids
dx, dy = Lx/nx, Ly/ny
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))
###########
# Capacities and operators (per component)
###########
body = (x, y, _=0) -> -1.0  # Full domain
capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)
###########
# BCs (parabolic profile on left/right, no-slip on top/bottom)
###########
Umax = 1.0
Uparab = (x, y) -> Umax * 4 * (y/Ly) * (1 - y/Ly)
u_bot  = Dirichlet((x, y)-> 0.0)
u_top = Dirichlet((x, y)-> 0.0)
u_left   = Dirichlet((x, y)-> 0.0)
u_right  = Dirichlet((x, y)-> 0.0)
bc_u = BorderConditions(Dict(
    :left=>u_left, :right=>u_right, :bottom=>u_bot, :top=>u_top
))
# Pressure: gauge only (no explicit gradient)
bc_p = BorderConditions(Dict{Symbol,AbstractBoundary}(:left=>Dirichlet(1.0), :right=>Dirichlet(0.0)))
# Cut-cell / interface BC for uγ (not used here)
u_bc = Dirichlet(0.0)
###########
# Sources and material
###########
fᵤ = (x, y, z=0.0) -> 0.0  # No body force; profile driven by BCs
fₚ = (x, y, z=0.0) -> 0.0
μ = 1.0
ρ = 1.0
# Build fluid with per-component operators via tuple constructor
fluid = Fluid(capacity_ux, operator_ux, capacity_p, operator_p, μ, ρ, fᵤ, fₚ)
###########
# Initial guess
###########
nu = prod(operator_ux.size)
np = prod(operator_p.size)
x0v = zeros(4*nu + np)
###########
# Solver and solve
###########
solver = StokesMono(fluid, mesh_ux, mesh_p, bc_u, bc_p, u_bc; mesh_ux=mesh_ux, mesh_uy=mesh_uy, x0=x0v)
solve_StokesMono!(solver; method=bicgstabl)
println("2D Poiseuille solved. Unknowns = ", length(solver.x))
# Extract components
uωx = solver.x[1:nu]; uγx = solver.x[nu+1:2nu]
uωy = solver.x[2nu+1:3nu]; uγy = solver.x[3nu+1:4nu]
pω  = solver.x[4nu+1:end]
# Plot profile u_x at mid x vs y and compare to analytical parabola
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
LI = LinearIndices((length(xs), length(ys)))
icol = Int(cld(length(xs), 2))
ux_num = [uωx[LI[icol, j]] for j in 1:length(ys)]
ux_ana = [Uparab(0.0, ys[j]) for j in 1:length(ys)]
fig = Figure(resolution=(900,400))
ax1 = Axis(fig[1,1], xlabel="u_x", ylabel="y", title="Poiseuille profile at mid x")
lines!(ax1, ux_num, ys, color=:blue, label="numeric")
lines!(ax1, ux_ana, ys, color=:red, linestyle=:dash, label="analytic")
axislegend(ax1, position=:rb)
display(fig)
save("stokes2d_poiseuille_profile.png", fig)
# Simple L2 error estimate
err = sqrt(sum((ux_num .- ux_ana).^2) / length(ux_num))
println("L2 error vs analytic parabola (midline): ", err)