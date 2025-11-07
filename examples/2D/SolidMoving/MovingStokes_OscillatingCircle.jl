using Penguin
using CairoMakie

# Domain and discretization ----------------------------------------------------
nx, ny = 96, 64
Lx, Ly = 4.0, 2.0
x0, y0 = -Lx/2, -Ly/2

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

# Moving circular inclusion ----------------------------------------------------
radius = 0.3
x_center0 = -0.5
y_center = 0.0
amplitude = 0.4
ω = 2π

body(x, y, t) = radius - sqrt((x - (x_center0 + amplitude * sin(ω * t)))^2 +
                              (y - y_center)^2)

interface_velocity(x, y, t) = (amplitude * ω * cos(ω * t), 0.0)

# Material properties and body forces -----------------------------------------
μ = 1.0
ρ = 1.0
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

# Boundary conditions: quiescent walls ----------------------------------------
zero_dirichlet = Dirichlet(0.0)
bc_ux = BorderConditions(Dict(
    :left => zero_dirichlet,
    :right => zero_dirichlet,
    :bottom => zero_dirichlet,
    :top => zero_dirichlet,
))
bc_uy = BorderConditions(Dict(
    :left => zero_dirichlet,
    :right => zero_dirichlet,
    :bottom => zero_dirichlet,
    :top => zero_dirichlet,
))

# Solver creation --------------------------------------------------------------
moving_solver = MovingStokesSolver(body, interface_velocity,
                                   mesh_ux, mesh_uy, mesh_p,
                                   bc_ux, bc_uy;
                                   μ=μ, ρ=ρ, fᵤ=fᵤ, fₚ=fₚ,
                                   pressure_gauge=PinPressureGauge(),
                                   t0=0.0)

Δt = 0.05
T_end = 0.3
times, states = solve_MovingStokesSolver!(moving_solver; Δt=Δt, T_end=T_end, method=Base.:\)

println("Solved ", length(states), " Stokes snapshots between t = ",
        times[1], " and t = ", times[end])

# Extract final state ----------------------------------------------------------
last_state = states[end]
op_x = moving_solver.solver.fluid.operator_u[1]
op_y = moving_solver.solver.fluid.operator_u[2]
nu_x = prod(op_x.size)
nu_y = prod(op_y.size)

uωx = last_state[1:nu_x]
uωy = last_state[2nu_x+1:2nu_x+nu_y]

xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]
Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))
speed = sqrt.(Ux.^2 .+ Uy.^2)

circle_x(t) = x_center0 + amplitude * sin(ω * t)
circle_y(_t) = y_center

# Visualization ----------------------------------------------------------------
fig = Figure(resolution=(900, 400))

ax_speed = Axis(fig[1, 1], xlabel="x", ylabel="y",
                title="Speed magnitude at t = $(round(T_end, digits=2))")
hm = heatmap!(ax_speed, xs, ys, speed; colormap=:inferno)
θ = range(0, 2π, length=200)
xc = circle_x(T_end)
yc = circle_y(T_end)
lines!(ax_speed, xc .+ radius .* cos.(θ), yc .+ radius .* sin.(θ), color=:white, linewidth=2)
Colorbar(fig[1, 2], hm, label="‖u‖")

ax_stream = Axis(fig[1, 3], xlabel="x", ylabel="y", title="Velocity field")
nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))
velocity_field(x, y) = Point2f(Ux[nearest_index(xs, x), nearest_index(ys, y)],
                               Uy[nearest_index(xs, x), nearest_index(ys, y)])
streamplot!(ax_stream, velocity_field, xs[1]..xs[end], ys[1]..ys[end]; colormap=:plasma)
lines!(ax_stream, xc .+ radius .* cos.(θ), yc .+ radius .* sin.(θ), color=:black, linewidth=2)

save("moving_stokes_oscillating_circle.png", fig)
display(fig)
