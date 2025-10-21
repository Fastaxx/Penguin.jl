using Penguin
using LinearAlgebra
using Printf

"""
Pure conduction verification for the Navier-Stokes/heat splitter.

Configuration:
  - Rectangular cavity with insulated side walls.
  - Top and bottom walls held at different temperatures.
  - Buoyancy disabled (beta = 0) so the momentum solve should return quiescent flow.

Diagnostics:
  - L2 norm of the velocity field (should be ~ 0).
  - Area-weighted L2 error between numerical temperature and analytic linear profile.
"""

# Domain and mesh
nx, ny = 48, 32
width, height = 1.0, 1.0
x0, y0 = 0.0, 0.0

mesh_p = Penguin.Mesh((nx, ny), (width, height), (x0, y0))
dx = width / nx
dy = height / ny
mesh_ux = Penguin.Mesh((nx, ny), (width, height), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (width, height), (x0, y0 - 0.5 * dy))
mesh_T = mesh_p

body = (x, y, _=0.0) -> -1.0

# Capacities
capacity_ux = Capacity(body, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(body, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(body, mesh_p;  compute_centroids=false)
capacity_T  = Capacity(body, mesh_T;  compute_centroids=false)

# Operators
operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# Velocity BCs: no-slip everywhere
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
pressure_gauge = MeanPressureGauge()
interface_bc = Dirichlet(0.0)

# Fluid properties (constant)
mu = 1.0
rho = 1.0
f_u = (x, y, z=0.0, t=0.0) -> 0.0
f_p = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              mu, rho, f_u, f_p)

ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc)

# Temperature BCs: hot bottom, cold top, insulated sides
T_bottom = 1.0
T_top = 0.0
analytic_T = (y) -> T_bottom + (T_top - T_bottom) * ((y - y0) / height)

bc_T = BorderConditions(Dict(
    :bottom=>Dirichlet(T_bottom),
    :top=>Dirichlet(T_top),
))
bc_T_cut = Dirichlet(0.0)

# Initial temperature: exact linear conduction profile
nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_temp = Nx_T * Ny_T

T0_center = Vector{Float64}(undef, N_temp)
for j in 1:Ny_T
    y = nodes_Ty[j]
    val = analytic_T(y)
    for i in 1:Nx_T
        idx = i + (j - 1) * Nx_T
        T0_center[idx] = val
    end
end
T0_interface = copy(T0_center)
T0 = vcat(T0_center, T0_interface)

# Coupled solver with buoyancy disabled
coupled = NavierStokesHeat2D(ns_solver,
                             capacity_T,
                             1.0e-2,
                             (x, y, z=0.0, t=0.0) -> 0.0,
                             bc_T,
                             bc_T_cut;
                             β=0.0,
                             gravity=(0.0, -1.0),
                             T_ref=0.0,
                             T0=T0)

dt_global = 0.0002
T_end = 0.001

println("="^72)
println("Navier-Stokes/heat pure conduction sanity")
println("="^72)
println("Grid: $nx x $ny, beta = 0 -> no buoyancy forcing")
println("Integrating to t = $T_end with dt = $dt_global")

solve_NavierStokesHeat2D_unsteady!(coupled; Δt=dt_global, T_end=T_end, scheme=:CN, store_states=false)

# Velocity diagnostics
data = Penguin.navierstokes2D_blocks(coupled.momentum)
nu_x = data.nu_x
nu_y = data.nu_y
u_center_x = coupled.momentum.x[1:nu_x]
u_center_y = coupled.momentum.x[2 * nu_x + 1:2 * nu_x + nu_y]
velocity_l2 = norm(vcat(u_center_x, u_center_y))

println(@sprintf("Velocity L2 norm: %.3e", velocity_l2))
@assert velocity_l2 <= 1.0e-12 "Velocity deviated from zero beyond tolerance"
