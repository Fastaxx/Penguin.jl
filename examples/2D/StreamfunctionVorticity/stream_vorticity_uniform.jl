using Penguin
using IterativeSolvers
using LinearAlgebra

# Domain and mesh definition
nx, ny = 64, 64
Lx, Ly = 1.0, 1.0
mesh = Mesh((nx, ny), (Lx, Ly), (0.0, 0.0))

# Trivial embedded geometry: the whole domain is fluid
body = (x, y, _=0.0) -> -1.0
capacity = Capacity(body, mesh)

# Pre-compute operators for initial conditions
operator = DiffusionOps(capacity)
n = prod(operator.size)

# Initial vorticity: Gaussian spot centred in the domain
centre = (Lx / 2, Ly / 2)
σ² = 0.02
ω_bulk = zeros(n)
for (i, coord) in enumerate(capacity.C_ω)
    x, y = coord[1], coord[2]
    if body(x, y) < 0
        r2 = (x - centre[1])^2 + (y - centre[2])^2
        ω_bulk[i] = exp(-r2 / σ²)
    end
end
ω_interface = zeros(n)
ω0 = vcat(ω_bulk, ω_interface)

# Boundary conditions
dirichlet_zero = Dirichlet(0.0)
border_bc = BorderConditions(Dict(
    :left => dirichlet_zero,
    :right => dirichlet_zero,
    :bottom => dirichlet_zero,
    :top => dirichlet_zero,
))

ν = 0.01
Δt = 1.0e-3

solver = StreamVorticitySolver(capacity, ν, Δt;
    bc_stream = dirichlet_zero,
    bc_vorticity = dirichlet_zero,
    bc_stream_border = border_bc,
    bc_vorticity_border = border_bc,
    ω0 = ω0,
)

# Run a few implicit steps to diffuse the vorticity blob
run!(solver, 20; method = gmres)

bulk_norm = norm(solver.ω[1:n])
println("Final vorticity L2 norm (uniform domain): $(bulk_norm)")
