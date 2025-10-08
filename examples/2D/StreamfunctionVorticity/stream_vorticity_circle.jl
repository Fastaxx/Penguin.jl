using Penguin
using IterativeSolvers
using LinearAlgebra

# Square box with an immersed circular boundary
nx, ny = 96, 96
Lx, Ly = 1.0, 1.0
mesh = Mesh((nx, ny), (Lx, Ly), (0.0, 0.0))

radius = 0.2
centre = (0.5 * Lx, 0.5 * Ly)
circle = (x, y, _=0.0) -> sqrt((x - centre[1])^2 + (y - centre[2])^2) - radius
capacity = Capacity(circle, mesh)

operator = DiffusionOps(capacity)
n = prod(operator.size)

# Localised vorticity ring hugging the circular interface
ω_bulk = zeros(n)
for (i, coord) in enumerate(capacity.C_ω)
    x, y = coord[1], coord[2]
    if circle(x, y) < 0
        r = sqrt((x - centre[1])^2 + (y - centre[2])^2)
        ω_bulk[i] = cospi(clamp((r / radius), 0.0, 1.0))
    end
end
ω_interface = zeros(n)
ω0 = vcat(ω_bulk, ω_interface)

dirichlet_zero = Dirichlet(0.0)
border_bc = BorderConditions(Dict(
    :left => dirichlet_zero,
    :right => dirichlet_zero,
    :bottom => dirichlet_zero,
    :top => dirichlet_zero,
))

ν = 0.005
Δt = 5.0e-4

solver = StreamVorticitySolver(capacity, ν, Δt;
    bc_stream = dirichlet_zero,
    bc_vorticity = dirichlet_zero,
    bc_stream_border = border_bc,
    bc_vorticity_border = border_bc,
    ω0 = ω0,
)

run!(solver, 40; method = gmres)

u, v = solver.velocity
println("Velocity extrema with circular cut cells: |u|ₘₐₓ=$(maximum(abs.(u))) |v|ₘₐₓ=$(maximum(abs.(v)))")
