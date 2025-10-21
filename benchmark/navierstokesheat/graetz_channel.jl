using Penguin
using LinearAlgebra
using Printf

const T_IN = 0.0
const T_WALL = 1.0
const U_MEAN = 1.0
const MU = 1.0
const RHO = 1.0
const KAPPA = 1.0e-2
const HEIGHT = 1.0
const LENGTH = 1.0

poiseuille_velocity(y; height=HEIGHT, umean=U_MEAN, y0=0.0) = begin
    y_frac = (y - y0) / height
    return 6.0 * umean * y_frac * (1.0 - y_frac)
end

function build_channel_solver(nx::Int, ny::Int;
                              length::Float64=LENGTH,
                              height::Float64=HEIGHT)
    mesh_p = Penguin.Mesh((nx, ny), (length, height), (0.0, 0.0))
    dx = length / nx
    dy = height / ny
    mesh_ux = Penguin.Mesh((nx, ny), (length, height), (-0.5 * dx, 0.0))
    mesh_uy = Penguin.Mesh((nx, ny), (length, height), (0.0, -0.5 * dy))
    mesh_T = mesh_p

    body = (x, y, _=0.0) -> -1.0

    capacity_ux = Capacity(body, mesh_ux; compute_centroids=false)
    capacity_uy = Capacity(body, mesh_uy; compute_centroids=false)
    capacity_p  = Capacity(body, mesh_p;  compute_centroids=false)
    capacity_T  = Capacity(body, mesh_T;  compute_centroids=false)

    operator_ux = DiffusionOps(capacity_ux)
    operator_uy = DiffusionOps(capacity_uy)
    operator_p  = DiffusionOps(capacity_p)

    inlet_profile = Dirichlet((x, y, t=0.0) -> begin
        if y < 0 || y > height
            return 0.0
        end
        poiseuille_velocity(y; height=height, umean=U_MEAN, y0=0.0)
    end)
    zero_dirichlet = Dirichlet((x, y, t=0.0) -> 0.0)

    bc_ux = BorderConditions(Dict(
        :left=>inlet_profile,
        :right=>Outflow(),
        :top=>zero_dirichlet,
        :bottom=>zero_dirichlet
    ))
    bc_uy = BorderConditions(Dict(
        :left=>zero_dirichlet,
        :right=>Outflow(),
        :top=>zero_dirichlet,
        :bottom=>zero_dirichlet
    ))

    pressure_gauge = PinPressureGauge()
    interface_bc = Dirichlet(0.0)

    f_u = (x, y, z=0.0, t=0.0) -> 0.0
    f_p = (x, y, z=0.0, t=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (capacity_ux, capacity_uy),
                  (operator_ux, operator_uy),
                  mesh_p,
                  capacity_p,
                  operator_p,
                  MU, RHO, f_u, f_p)

    ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc)

    bc_T = BorderConditions(Dict(
        :left=>Dirichlet(T_IN),
        :bottom=>Dirichlet(T_WALL),
        :top=>Neumann(0.0),
        :right=>Neumann(0.0)
    ))
    bc_T_cut = Dirichlet(0.0)

    nodes_Tx = mesh_T.nodes[1]
    nodes_Ty = mesh_T.nodes[2]
    Nx_T = size(nodes_Tx, 1)
    Ny_T = size(nodes_Ty, 1)
    N_temp = Nx_T * Ny_T

    T0_center = fill(T_IN, N_temp)
    T0_interface = copy(T0_center)
    T0 = vcat(T0_center, T0_interface)

    coupled = NavierStokesHeat2D(ns_solver,
                                 capacity_T,
                                 KAPPA,
                                 (x, y, z=0.0, t=0.0) -> 0.0,
                                 bc_T,
                                 bc_T_cut;
                                 β=0.0,
                                 gravity=(0.0, 0.0),
                                 T_ref=T_IN,
                                 T0=T0)

    return coupled, mesh_T
end

function run_graetz_case(nx::Int, ny::Int;
                         length::Float64=LENGTH,
                         height::Float64=HEIGHT,
                         dt::Float64=1.0e-3,
                         T_end::Float64=0.4)
    coupled, mesh_T = build_channel_solver(nx, ny; length=length, height=height)

    println(@sprintf("  -> Running Navier-Stokes/heat: %d x %d grid, dt=%.3e, T_end=%.3f",
                     nx, ny, dt, T_end))
    solve_NavierStokesHeat2D_unsteady!(coupled; Δt=dt, T_end=T_end, scheme=:CN, store_states=false)

    nodes_x = mesh_T.nodes[1]
    nodes_y = mesh_T.nodes[2]
    Nx = length(nodes_x)
    Ny = length(nodes_y)
    N = Nx * Ny

    data = Penguin.navierstokes2D_blocks(coupled.momentum)
    nu_x = data.nu_x

    mesh_ux = coupled.momentum.fluid.mesh_u[1]
    u_center = Penguin.project_field_between_nodes(
        coupled.momentum.x[1:nu_x],
        (mesh_ux.nodes[1], mesh_ux.nodes[2]),
        (nodes_x, nodes_y))

    return nodes_x, nodes_y, copy(coupled.temperature), u_center
end

function compute_nusselt(nodes_x::Vector{Float64},
                         nodes_y::Vector{Float64},
                         temperature_state::Vector{Float64},
                         u_bulk_x::Vector{Float64};
                         height::Float64=HEIGHT)
    Nx = length(nodes_x)
    Ny = length(nodes_y)
    N = Nx * Ny
    T_center = view(temperature_state, 1:N)
    dy = Ny > 1 ? nodes_y[2] - nodes_y[1] : height

    Nu = zeros(Float64, Nx)
    u_means = zeros(Float64, Nx)

    for i in 1:Nx
        idxs = (i - 1) * Ny .+ (1:Ny)
        column_T = T_center[idxs]
        column_u = u_bulk_x[idxs]
        mass_flow = sum(column_u) * dy
        T_bulk = abs(mass_flow) < 1.0e-12 ? T_IN : sum(column_u .* column_T) * dy / mass_flow
        grad_bottom = (column_T[1] - T_WALL) / (0.5 * dy)
        Nu[i] = -height * grad_bottom / max(1.0e-12, T_WALL - T_bulk)
        u_means[i] = (sum(column_u) * dy) / height
    end

    return Nu, u_means
end

function interp_linear(xs_ref::Vector{Float64},
                       ys_ref::Vector{Float64},
                       xs_target::Vector{Float64})
    out = similar(xs_target)
    n_ref = length(xs_ref)
    for (k, x) in enumerate(xs_target)
        if x <= xs_ref[1]
            out[k] = ys_ref[1]
        elseif x >= xs_ref[end]
            out[k] = ys_ref[end]
        else
            idx = searchsortedfirst(xs_ref, x)
            x0 = xs_ref[idx - 1]
            x1 = xs_ref[idx]
            y0 = ys_ref[idx - 1]
            y1 = ys_ref[idx]
            alpha = (x - x0) / (x1 - x0)
            out[k] = (1 - alpha) * y0 + alpha * y1
        end
    end
    return out
end

println("="^72)
println("Forced-convection Graetz benchmark (Navier-Stokes/heat)")
println("="^72)

T_end = 0.4

println("Generating high-resolution reference solution...")
xs_ref, ys_ref, T_ref, u_ref = run_graetz_case(256, 64; dt=5.0e-6, T_end=T_end)
Nu_ref, u_means_ref = compute_nusselt(xs_ref, ys_ref, T_ref, u_ref)

println("Running benchmark resolution case...")
xs_num, ys_num, T_num, u_num = run_graetz_case(128, 32; dt=1.0e-3, T_end=T_end)
Nu_num, u_means_num = compute_nusselt(xs_num, ys_num, T_num, u_num)

Nu_ref_interp = interp_linear(xs_ref, Nu_ref, xs_num)

tol_region = 0.05 * LENGTH
valid_idx = findall(x -> x >= tol_region, xs_num)
isempty(valid_idx) && (valid_idx = collect(1:length(xs_num)))

rel_err = maximum(abs.(Nu_num[valid_idx] .- Nu_ref_interp[valid_idx]) ./
                  max.(1.0e-12, abs.(Nu_ref_interp[valid_idx])))
println(@sprintf("Max relative deviation vs reference (x >= %.3f L): %.3e",
                 tol_region / LENGTH, rel_err))
@assert rel_err <= 5.0e-2 "Graetz temperature field deviated more than 5% from reference"

Nu_asymp = Nu_num[end]
println(@sprintf("Asymptotic Nu (numerical): %.3f (target 7.541)", Nu_asymp))
@assert abs(Nu_asymp - 7.541) <= 0.2 "Asymptotic Nusselt number departed from Graetz limit"

u_mean_channel = mean(u_means_num)
Re = RHO * u_mean_channel * HEIGHT / MU
Pr = MU / (RHO * KAPPA)
Dh = 2.0 * HEIGHT
Gz_vals = Re * Pr * Dh ./ max.(1.0e-6, xs_num)

println(@sprintf("Channel Re (from simulation): %.3f, Pr: %.3f", Re, Pr))

sample_idx = round.(Int, range(1, length(xs_num), length=6))
println("x/L   Gz      Nu_num   Nu_ref")
for idx in sample_idx
    println(@sprintf("%4.2f  %7.1f  %7.3f  %7.3f",
                     xs_num[idx] / LENGTH,
                     Gz_vals[idx],
                     Nu_num[idx],
                     Nu_ref_interp[idx]))
end

println("Graetz forced-convection verification passed.")
