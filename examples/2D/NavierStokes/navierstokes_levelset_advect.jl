using Penguin
using LevelSetMethods
using CairoMakie

"""
Unsteady 2D Navier–Stokes flow that pushes a circular inclusion downstream in
a rectangular channel. The moving boundary is tracked with LevelSetMethods,
and the Navier–Stokes operators are rebuilt after each time step to follow the
interface.
"""

###########
# Geometry
###########
nx, ny = 120, 80
Lx, Ly = 2.0, 1.0
x0, y0 = -1.0, -0.5

###########
# Level-set definition
###########
grid = CartesianGrid((x0, y0), (x0 + Lx, y0 + Ly), (240, 160))
radius = 0.18
center = (x0 + 0.4, y0 + Ly / 2)
ϕ₀ = LevelSet(x -> hypot(x[1] - center[1], x[2] - center[2]) - radius, grid)
body_fun = body_function_from_levelset(ϕ₀; invert_sign=true)

###########
# Meshes
###########
mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

###########
# Capacities & operators
###########
capacity_ux = Capacity(body_fun, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(body_fun, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(body_fun, mesh_p; compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
U_in = 1.0

ux_inlet = Dirichlet((x, y, t=0.0) -> U_in)
ux_wall  = Dirichlet((x, y, t=0.0) -> 0.0)

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_inlet,
    :right=>Outflow(),
    :bottom=>ux_wall,
    :top=>ux_wall
))
bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero
))
pressure_gauge = PinPressureGauge()

interface_bc = Dirichlet(0.0)

###########
# Physics
###########
μ = 0.002
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

###########
# Solver setup
###########
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)

x0_vec = zeros(2 * (nu_x + nu_y) + np)

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)

###########
# Coupled solve
###########
Δt = 0.0025
T_end = 0.03

println("Running coupled Navier–Stokes + LevelSet simulation...")
result = solve_NavierStokesLevelSet_unsteady!(solver, ϕ₀;
                                              Δt=Δt,
                                              T_end=T_end,
                                              scheme=:CN,
                                              bc=NeumannBC(),  # zero-gradient at walls
                                              store_levelsets=true)
println("Simulation finished. Stored frames = ", length(result.levelsets))

###########
# Post-processing helpers
###########
function velocity_components(state, solver)
    size_x = solver.fluid.operator_u[1].size
    size_y = solver.fluid.operator_u[2].size
    nu_x = prod(size_x)
    nu_y = prod(size_y)

    Ux = reshape(@view(state[1:nu_x]), size_x...)
    Uy = reshape(@view(state[2 * nu_x + 1:2 * nu_x + nu_y]), size_y...)
    return Ux, Uy
end

function levelset_axes(ϕ::LevelSet)
    mesh = ϕ.mesh
    xs = range(mesh.lc[1], mesh.hc[1], length=mesh.n[1])
    ys = range(mesh.lc[2], mesh.hc[2], length=mesh.n[2])
    return xs, ys
end

function contour_levelset!(ax, ϕ; color=:white, linewidth=2)
    xs, ys = levelset_axes(ϕ)
    vals = LevelSetMethods.values(ϕ)
    contour!(ax, xs, ys, vals; levels=[0.0], color=color, linewidth=linewidth)
end

###########
# Visualisation
###########
final_state = result.states[end]
Ux, Uy = velocity_components(final_state, solver)
speed = sqrt.(Ux.^2 .+ Uy.^2)
xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]

snap_times = (0.0, 0.1, 0.25, T_end)
function nearest_index(times, t)
    clamp(argmin(abs.(times .- t)), 1, length(times))
end
snap_indices = map(t -> nearest_index(result.times, t), snap_times)

fig = Figure(resolution=(1200, 600))

ax_speed = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Speed at t = $(round(T_end; digits=3))")
hm = heatmap!(ax_speed, xs, ys, speed; colormap=:plasma)
contour_levelset!(ax_speed, result.levelsets[snap_indices[end]])
Colorbar(fig[1, 2], hm)

for (col, idx) in enumerate(snap_indices)
    ax = Axis(fig[2, col], xlabel="x", ylabel="y",
              title="t = $(round(result.times[idx]; digits=3))")
    contour_levelset!(ax, result.levelsets[idx]; color=:dodgerblue, linewidth=3)
    xlims!(ax, x0, x0 + Lx)
    ylims!(ax, y0, y0 + Ly)
    hidedecorations!(ax, ticks=false, ticklabels=false)
end

display(fig)

result.levelsets[1].vals != result.levelsets[2].vals