using Penguin
using CairoMakie
using LinearAlgebra

"""
Two-layer 2D Stokes Poiseuille (steady): top/bottom no-slip, pressure-driven in x.
Interface at y = Ly/2: enforce u-continuity and μ-weighted flux continuity.
"""

nx, ny = 96, 64
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, 0.0

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx, dy = Lx/nx, Ly/ny
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

# Phase indicator: top/bottom layers
y_mid = y0 + Ly/2
body_top = (x, y, _=0.0) -> y - y_mid    # >0 above midline
body_bot = (x, y, _=0.0) -> y_mid - y    # >0 below midline

cap_ux_top = Capacity(body_top, mesh_ux)
cap_uy_top = Capacity(body_top, mesh_uy;compute_centroids=false)
cap_p_top  = Capacity(body_top, mesh_p)

cap_ux_bot = Capacity(body_bot, mesh_ux)
cap_uy_bot = Capacity(body_bot, mesh_uy;compute_centroids=false)
cap_p_bot  = Capacity(body_bot, mesh_p)

op_ux_top = DiffusionOps(cap_ux_top)
op_uy_top = DiffusionOps(cap_uy_top)
op_p_top  = DiffusionOps(cap_p_top)

op_ux_bot = DiffusionOps(cap_ux_bot)
op_uy_bot = DiffusionOps(cap_uy_bot)
op_p_bot  = DiffusionOps(cap_p_bot)

# Viscosities
μ_top = 2.0
μ_bot = 1.0
ρ = 1.0

fᵤ = (x, y, z=0.0) -> 0.0
fₚ = (x, y, z=0.0) -> 0.0

fluid_top = Fluid((mesh_ux, mesh_uy), (cap_ux_top, cap_uy_top), (op_ux_top, op_uy_top),
                  mesh_p, cap_p_top, op_p_top, μ_top, ρ, fᵤ, fₚ)
fluid_bot = Fluid((mesh_ux, mesh_uy), (cap_ux_bot, cap_uy_bot), (op_ux_bot, op_uy_bot),
                  mesh_p, cap_p_bot, op_p_bot, μ_bot, ρ, fᵤ, fₚ)

# BCs: top/bottom no-slip; pressure drop left->right
ux_wall = Dirichlet((x, y) -> 0.0)
uy_wall = Dirichlet((x, y) -> 0.0)

bc_ux_top = BorderConditions(Dict(:bottom=>ux_wall, :top=>ux_wall))
bc_uy_top = BorderConditions(Dict(:bottom=>uy_wall, :top=>uy_wall))

bc_ux_bot = BorderConditions(Dict(:bottom=>ux_wall, :top=>ux_wall))
bc_uy_bot = BorderConditions(Dict(:bottom=>uy_wall, :top=>uy_wall))

Δp = 1.0
p_in = Δp
p_out = 0.0
bc_p = BorderConditions(Dict(:left=>Dirichlet(p_in), :right=>Dirichlet(p_out)))

# Interface conditions (continuity of velocity and flux)
interface = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Initial vector
nu = prod(op_ux_top.size)
np = prod(op_p_top.size)
x0 = zeros(2 * (2*nu + np))

solver = StokesDiph(fluid_top, fluid_bot, (bc_ux_top, bc_uy_top), (bc_ux_bot, bc_uy_bot), bc_p, interface, Dirichlet(0.0); x0=x0)
solve_StokesDiph!(solver; method=Base.:\)

nu_x = prod(op_ux_top.size)
nu_y = prod(op_uy_top.size)

uωx1 = solver.x[1:nu_x]
ux_nodes = mesh_ux.nodes[1]; uy_nodes = mesh_ux.nodes[2]
LIux = LinearIndices((length(ux_nodes), length(uy_nodes)))
icol = Int(cld(length(ux_nodes), 2))
ux_profile = [uωx1[LIux[icol, j]] for j in 1:length(uy_nodes)]

fig = Figure(resolution=(900, 400))
ax = Axis(fig[1,1], xlabel="u_x", ylabel="y", title="Two-layer Poiseuille: u_x mid-column")
lines!(ax, ux_profile, uy_nodes)

Ux1 = reshape(uωx1, (length(ux_nodes), length(uy_nodes)))
fig2 = Figure(resolution=(800, 360))
ax2 = Axis(fig2[1,1], xlabel="x", ylabel="y", title="u_x (phase 1)")
heatmap!(ax2, ux_nodes, uy_nodes, Ux1'; colormap=:viridis)

save("stokes2d_diph_poiseuille_profile.png", fig)
save("stokes2d_diph_poiseuille_ux1.png", fig2)
