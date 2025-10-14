using Penguin
using CairoMakie

"""
2D linear shallow water travelling wave on a rectangular periodic-like channel.
A cosine free-surface perturbation is paired with the corresponding velocity
field such that the wave propagates predominantly in the positive x-direction.
"""

# Domain ----------------------------------------------------------------------
nx, ny = 80, 40
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, -Ly / 2

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

# Capacities & operators ------------------------------------------------------
full_domain = (x, y, _=0.0) -> -1.0
capacity_ux = Capacity(full_domain, mesh_ux)
capacity_uy = Capacity(full_domain, mesh_uy)
capacity_p  = Capacity(full_domain, mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# Boundary conditions ---------------------------------------------------------
bc_ux = BorderConditions(Dict(
    :left => Periodic(),
    :right => Periodic(),
    :bottom => Dirichlet(0.0),
    :top => Dirichlet(0.0),
))
bc_uy = BorderConditions(Dict(
    :left => Periodic(),
    :right => Periodic(),
    :bottom => Dirichlet(0.0),
    :top => Dirichlet(0.0),
))
height_gauge = PinPressureGauge()
interface_bc = Dirichlet(0.0)

# Physics ---------------------------------------------------------------------
depth = 1.0
grav = 9.81
μ = 1.0e-6  # Viscosity is unused by the shallow water solver but required by Fluid
ρ = 1.0
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

# Initial conditions ----------------------------------------------------------
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)

x0_vec = zeros(2 * (nu_x + nu_y) + np)

Xp = mesh_p.nodes[1]
Yp = mesh_p.nodes[2]
Ux_nodes = mesh_ux.nodes[1]
Uy_nodes = mesh_ux.nodes[2]

amplitude = 5e-3
wavenumber = 2π / Lx
c = sqrt(grav * depth)

η_init = [amplitude * cos(wavenumber * x) for y in Yp, x in Xp]
η_vec = vec(Float64.(η_init))

u_amp = (c / depth) * amplitude
ux_init = [u_amp * cos(wavenumber * x) for y in Uy_nodes, x in Ux_nodes]
ux_vec = vec(Float64.(ux_init))

x0_vec[1:nu_x] .= ux_vec
x0_vec[nu_x+1:2nu_x] .= 0.0
x0_vec[2nu_x+1:2nu_x+nu_y] .= 0.0
x0_vec[2nu_x+nu_y+1:2*(nu_x+nu_y)] .= 0.0
x0_vec[2*(nu_x+nu_y)+1:end] .= η_vec

# Solver ----------------------------------------------------------------------
solver = ShallowWater2D(fluid,
                        bc_ux,
                        bc_uy,
                        height_gauge,
                        interface_bc;
                        depth=depth,
                        gravity=grav,
                        x0=x0_vec)

Δt = 0.25 * dx / c
T_end = 0.01 * Lx / c

println("Running shallow water travelling wave example...")
times, histories = solve_ShallowWater2D_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=:CN)
println("Simulation finished. Stored states = ", length(histories))

# Post-processing -------------------------------------------------------------
height_final = interpolate_height(solver)
height_initial = reshape(η_vec, operator_p.size)

xs = mesh_p.nodes[1]
ys = mesh_p.nodes[2]
mid_idx = Int(cld(length(ys), 2))

fig = Figure(resolution=(900, 500))
ax = Axis(fig[1, 1], xlabel="x", ylabel="η(x, y_mid)", title="Shallow water travelling wave")
lines!(ax, xs, height_initial[:, mid_idx], color=:gray, label="Initial")
lines!(ax, xs, height_final[:, mid_idx], color=:blue, label="Final")
axislegend(ax, position=:lt)

ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", title="Final free-surface elevation")
hm = heatmap!(ax2, xs, ys, height_final'; colormap=:viridis)
Colorbar(fig[1, 3], hm, label="η")

fig
