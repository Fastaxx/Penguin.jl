using Penguin
using CairoMakie
using SparseArrays
using LinearAlgebra
using IterativeSolvers
"""
2D Stokes Couette flow (steady): u = (Ux(y), 0), p = const.

Domain: [0, Lx] × [0, Ly]
BCs: u_x(x, 0) = 0, u_x(x, Ly) = U0; u_y = 0 on all walls.
Left/Right: set u_x consistent with linear profile in y to avoid incompatibility.
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
# BCs
###########
U0 = 1.0
u_left  = Dirichlet((x, y)-> 0.0)
u_right = Dirichlet((x, y)-> 0.0)
u_bot   = Dirichlet((x, y)-> 1.0)
u_top   = Dirichlet((x, y)-> 0.0)
bc_u = BorderConditions(Dict(
    :left=>u_left, :right=>u_right, :bottom=>u_bot, :top=>u_top
))

# Pressure: gauge only (or set fixed value on one boundary if desired)
bc_p = BorderConditions(Dict{Symbol,AbstractBoundary}())

# Cut-cell / interface BC for uγ (not used here)
u_bc = Dirichlet(0.0)

###########
# Sources and material
###########
fᵤ = (x, y, z=0.0) -> 0.0
fₚ = (x, y, z=0.0) -> 0.0
μ = 1.0
ρ = 1.0

# Temporary: pass x-staggered operator until solver supports split ux/uy
fluid = Fluid(capacity_ux, operator_ux, capacity_p, operator_p, μ, ρ, fᵤ, fₚ)

###########
# Initial guess
###########
nu = prod(operator_ux.size)
np = prod(operator_p.size)
x0 = zeros(4*nu + np)

###########
# Solver and solve
###########
solver = StokesMono(fluid, mesh_ux, mesh_p, bc_u, bc_p, u_bc; x0=x0)
solve_StokesMono!(solver; method=bicgstabl)

println("2D Couette solved. Unknowns = ", length(solver.x))

# Extract components
uωx = solver.x[1:nu]; uγx = solver.x[nu+1:2nu]
uωy = solver.x[2nu+1:3nu]; uγy = solver.x[3nu+1:4nu]
pω  = solver.x[4nu+1:end]

# Plot profile u_x at mid x vs y
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
LI = LinearIndices((length(xs), length(ys)))
icol = Int(cld(length(xs), 2))
ux_profile = [uωx[LI[icol, j]] for j in 1:length(ys)]

fig = Figure(resolution=(900,400))
ax1 = Axis(fig[1,1], xlabel="u_x", ylabel="y", title="Couette profile at mid x")
lines!(ax1, ux_profile, ys)
display(fig)
save("stokes2d_couette_profile.png", fig)
