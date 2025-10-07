using Penguin
using CairoMakie
using LinearAlgebra

nx, ny = 64, 32
width, height = 1.0, 1.0
x0, y0 = 0.0, 0.0

mesh_p = Penguin.Mesh((nx, ny), (width, height), (x0, y0))
dx = width / nx
dy = height / ny
mesh_ux = Penguin.Mesh((nx, ny), (width, height), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (width, height), (x0, y0 - 0.5 * dy))
mesh_T = mesh_p

body = (x, y, _=0.0) -> -1.0

capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
capacity_T  = Capacity(body, mesh_T)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

zero_dirichlet = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>zero_dirichlet,
    :right=>zero_dirichlet,
    :bottom=>zero_dirichlet,
    :top=>zero_dirichlet
))
bc_uy = BorderConditions(Dict(
    :left=>zero_dirichlet,
    :right=>zero_dirichlet,
    :bottom=>zero_dirichlet,
    :top=>zero_dirichlet
))
bc_p = BorderConditions(Dict{Symbol,AbstractBoundary}())
interface_bc = Dirichlet(0.0)

μ = 1.0e-3
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

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
x0_vec = zeros(2 * (nu_x + nu_y) + np)
ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), bc_p, interface_bc; x0=x0_vec)

T_hot = 0.5
T_cold = -0.5
bc_T = BorderConditions(Dict(
    :bottom=>Dirichlet(T_hot),
    :top=>Dirichlet(T_cold)
))
bc_T_cut = Dirichlet(0.0)

nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_temp = Nx_T * Ny_T

T0ω = zeros(Float64, N_temp)
y_min = nodes_Ty[1]
y_max = nodes_Ty[end]
span_y = y_max - y_min
for j in 1:Ny_T
    y = nodes_Ty[j]
    frac = span_y ≈ 0 ? 0.0 : (y - y_min) / span_y
    val = T_hot + (T_cold - T_hot) * frac
    for i in 1:Nx_T
        idx = i + (j - 1) * Nx_T
        T0ω[idx] = val
    end
end
T0γ = copy(T0ω)
T0 = vcat(T0ω, T0γ)

κ = 1.0e-3
heat_source = (x, y, z=0.0, t=0.0) -> 0.0

coupled = NavierStokesHeat2D(ns_solver, capacity_T, κ, heat_source,
                             bc_T, bc_T_cut;
                             β=1.0,
                             gravity=(0.0, -1.0),
                             T_ref=0.0,
                             T0=T0)

Δt = 0.005
T_end = 0.5

println("Running coupled Navier–Stokes / heat simulation...")
times, velocity_hist, temperature_hist = solve_NavierStokesHeat2D_unsteady!(coupled;
                                                                            Δt=Δt,
                                                                            T_end=T_end,
                                                                            scheme=:CN)
println("Done. Stored snapshots: ", length(temperature_hist))

uωx = coupled.momentum.x[1:nu_x]
uωy = coupled.momentum.x[2nu_x+1:2nu_x+nu_y]

xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]
Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))
speed = sqrt.(Ux.^2 .+ Uy.^2)

Tω_final = coupled.temperature[1:N_temp]
Temperature = reshape(Tω_final, (Nx_T, Ny_T))

nearest_index(vec::AbstractVector{<:Real}, val::Real) = begin
    idx = searchsortedfirst(vec, val)
    if idx <= 1
        return 1
    elseif idx > length(vec)
        return length(vec)
    else
        prev_val = vec[idx - 1]
        curr_val = vec[idx]
        return abs(val - prev_val) <= abs(curr_val - val) ? idx - 1 : idx
    end
end

velocity_field(x, y) = Point2f(Ux[nearest_index(xs, x), nearest_index(ys, y)],
                               Uy[nearest_index(xs, x), nearest_index(ys, y)])

fig = Figure(resolution=(1200, 600))
ax_temp = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Temperature")
hm_temp = heatmap!(ax_temp, nodes_Tx, nodes_Ty, Temperature'; colormap=:thermal)
Colorbar(fig[1, 2], hm_temp)

ax_speed = Axis(fig[1, 3], xlabel="x", ylabel="y", title="Velocity magnitude")
hm_speed = heatmap!(ax_speed, xs, ys, speed; colormap=:viridis)
Colorbar(fig[1, 4], hm_speed)

ax_stream = Axis(fig[2, 1], xlabel="x", ylabel="y", title="Velocity streamlines")
streamplot!(ax_stream, velocity_field, xs[1]..xs[end], ys[1]..ys[end]; colormap=:plasma)

save("rayleigh_benard_snapshot.png", fig)
display(fig)

println("Creating temperature animation...")
n_frames = min(80, length(temperature_hist))
frame_indices = collect(unique(round.(Int, range(1, length(temperature_hist), length=n_frames))))

fig_anim = Figure(resolution=(800, 600))
ax_anim = Axis(fig_anim[1, 1], xlabel="x", ylabel="y", title="Temperature evolution")
temp_obs = Observable(Temperature')
hm_anim = heatmap!(ax_anim, nodes_Tx, nodes_Ty, temp_obs; colormap=:thermal)
Colorbar(fig_anim[1, 2], hm_anim)

record(fig_anim, "rayleigh_benard_temperature.gif", 1:length(frame_indices); framerate=10) do frame
    hist = temperature_hist[frame_indices[frame]]
    Tω_hist = hist[1:N_temp]
    temp_obs[] = reshape(Tω_hist, (Nx_T, Ny_T))'
    ax_anim.title = "Temperature at t = $(round(times[frame_indices[frame]], digits=3))"
end

println("Animation saved as rayleigh_benard_temperature.gif")
