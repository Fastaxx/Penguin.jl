using Penguin
using LinearAlgebra
using Statistics
try
    using CairoMakie
catch
    @warn "CairoMakie not available; visualization disabled."
end

# ---------------------------------------------------------------------------
# Monolithic coupling demo: differentially heated cavity (Ra = 1e3, Pr = 0.71)
# ---------------------------------------------------------------------------
# This example revisits the classic cavity but solves each time step with the
# fully coupled Newton strategy (MonolithicCoupling). The coupling drives a
# single Newton solve for velocity, pressure, and temperature simultaneously.
# ---------------------------------------------------------------------------

# Geometry and mesh ---------------------------------------------------------
nx, ny = 48, 48
L = 1.0
origin = (0.0, 0.0)

mesh_p = Penguin.Mesh((nx, ny), (L, L), origin)
dx = L / nx
dy = L / ny
mesh_ux = Penguin.Mesh((nx, ny), (L, L), (origin[1] - 0.5 * dx, origin[2]))
mesh_uy = Penguin.Mesh((nx, ny), (L, L), (origin[1], origin[2] - 0.5 * dy))
mesh_T = mesh_p

body = (x, y, _=0.0) -> -1.0

capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
capacity_T  = Capacity(body, mesh_T)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# Physical parameters -------------------------------------------------------
Ra = 1.0e5
Pr = 0.71
ΔT = 1.0
T_hot = 0.5
T_cold = -0.5

ν = sqrt(Pr / Ra)
α = ν / Pr
β = 1.0
gravity = (0.0, -1.0)

# Velocity boundary conditions ----------------------------------------------
zero = Dirichlet((x, y, t=0.0) -> 0.0)
bc_ux = BorderConditions(Dict(
    :left=>zero, :right=>zero,
    :bottom=>zero, :top=>zero
))
bc_uy = BorderConditions(Dict(
    :left=>zero, :right=>zero,
    :bottom=>zero, :top=>zero
))
pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              ν, 1.0,
              (x,y,z=0,t=0)->0.0,
              (x,y,z=0,t=0)->0.0)

ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut)

# Temperature boundary conditions -------------------------------------------
bc_T = BorderConditions(Dict(
    :left=>Dirichlet(T_hot),
    :right=>Dirichlet(T_cold),
    :top=>Neumann(0.0),
    :bottom=>Neumann(0.0)
))
bc_T_cut = Dirichlet(0.0)

# Initial temperature: linear gradation ------------------------------------
nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_scalar = Nx_T * Ny_T

T_init_center = Vector{Float64}(undef, N_scalar)
for j in 1:Ny_T
    y = nodes_Ty[j]
    for i in 1:Nx_T
        x = nodes_Tx[i]
        frac = (x - first(nodes_Tx)) / (last(nodes_Tx) - first(nodes_Tx))
        T_val = T_hot + (T_cold - T_hot) * frac
        idx = i + (j - 1) * Nx_T
        T_init_center[idx] = T_val
    end
end
T_init_interface = copy(T_init_center)
T_init = vcat(T_init_center, T_init_interface)

# Coupler with monolithic strategy ------------------------------------------
coupler = NavierStokesScalarCoupler(ns_solver,
                                    capacity_T,
                                    α,
                                    (x, y, z=0.0, t=0.0) -> 0.0,
                                    bc_T,
                                    bc_T_cut;
                                    strategy=MonolithicCoupling(tol=1e-6, maxiter=12, damping=1.0, verbose=false),
                                    β=β,
                                    gravity=gravity,
                                    T_ref=0.0,
                                    T0=T_init,
                                    store_states=true)

# Time stepping --------------------------------------------------------------
Δt = 2.5e-3
T_end = 0.25

println("=== Monolithic coupling: differentially heated cavity ===")
println("Grid: $nx × $ny, Ra = $Ra, Pr = $Pr, Δt = $Δt, T_end = $T_end")

times, velocity_hist, scalar_hist = solve_NavierStokesScalarCoupling!(coupler;
                                                                      Δt=Δt,
                                                                      T_end=T_end,
                                                                      scheme=:CN)

# Extract final fields -------------------------------------------------------
u_state = coupler.velocity_state
T_state = coupler.scalar_state

data = Penguin.navierstokes2D_blocks(coupler.momentum)
nu_x = data.nu_x
nu_y = data.nu_y

Ux = reshape(view(u_state, 1:nu_x), length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2]))
Uy = reshape(view(u_state, 2 * nu_x + 1:2 * nu_x + nu_y), length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2]))
T_field = reshape(view(T_state, 1:N_scalar), Nx_T, Ny_T)

speed = sqrt.(Ux.^2 .+ Uy.^2)
println("Final max velocity magnitude ≈ ", maximum(speed))

# Hot-wall Nusselt number ----------------------------------------------------
Δx = nodes_Tx[2] - nodes_Tx[1]
Nu_profile = zeros(Float64, Ny_T)
for i in 1:Nx_T
    T1 = T_field[i, 1]
    T2 = T_field[i, 2]
    grad = (T2 - T1) / Δx
    Nu_profile[i] = -(L) * grad / (T_hot - T_cold)
end
println(Nu_profile)
Nu_mean = mean(Nu_profile[1:end-1])
println("Mean hot-wall Nusselt ≈ ", Nu_mean)

# Visualization --------------------------------------------------------------
if @isdefined CairoMakie
    xs = nodes_Tx
    ys = nodes_Ty

    fig = Figure(resolution=(960, 480))
    ax_T = Axis(fig[1, 1], xlabel="x", ylabel="y",
                title="Temperature (t = $(round(times[end]; digits=3)))",
                aspect=DataAspect())
    hm = heatmap!(ax_T, xs, ys, T_field'; colormap=:thermal)
    Colorbar(fig[1, 2], hm; label="T")

    ax_u = Axis(fig[2, 1], xlabel="x", ylabel="y",
                title="Velocity magnitude",
                aspect=DataAspect())
    hm_u = heatmap!(ax_u, mesh_ux.nodes[1], mesh_ux.nodes[2], speed'; colormap=:viridis)
    Colorbar(fig[2, 2], hm_u; label="|u|")

    display(fig)

    # Animation --------------------------------------------------------------
    temp_snapshots = map(state -> reshape(view(state, 1:N_scalar), Nx_T, Ny_T), scalar_hist)
    tmin = minimum(minimum.(temp_snapshots))
    tmax = maximum(maximum.(temp_snapshots))

    anim_fig = Figure(resolution=(640, 360))
    anim_ax = Axis(anim_fig[1, 1], xlabel="x", ylabel="y",
                   title="Temperature evolution",
                   aspect=DataAspect())
    temp_obs = Observable(temp_snapshots[1]')
    heatmap!(anim_ax, xs, ys, temp_obs; colormap=:thermal, colorrange=(tmin, tmax))

    output_path = joinpath(@__DIR__, "monolithic_differential_cavity.mp4")
    println("Recording monolithic coupling animation → $(output_path)")
    record(anim_fig, output_path, eachindex(times)) do idx
        temp_obs[] = temp_snapshots[idx]'
        anim_ax.title = "Temperature (t = $(round(times[idx]; digits=3)))"
    end
    println("Animation saved to $(output_path)")
end
