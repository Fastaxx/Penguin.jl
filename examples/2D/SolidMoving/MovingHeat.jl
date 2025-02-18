using Penguin
using IterativeSolvers, SpecialFunctions
using Roots


### 2D Test Case : Monophasic Unsteady Diffusion Equation inside a moving Disk
# Define the mesh
nx, ny = 80, 80
lx, ly = 4., 4.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/4, (lx/2, ly/2) .+ (0.01, 0.01)
c = 0.0
body = (x,y,t)->(sqrt((x - center[1])^2 + (y - center[2])^2) - radius + c*t)

# Define the Space-Time mesh
Δt = 0.01
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(1.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc))

# Define the source term
f = (x,y,z,t)-> 0.0 #sin(x)*cos(10*y)
K = (x,y,z)-> 1.0

Fluide = Phase(capacity, operator, f, K)

# Initial condition
u0ₒ = zeros((nx+1)*(ny+1))
u0ᵧ = ones((nx+1)*(ny+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
solver = MovingDiffusionUnsteadyMono(Fluide, bc_b, bc1, Δt, u0, mesh, "BE")

# Solve the problem
solve_MovingDiffusionUnsteadyMono!(solver, Fluide, body, Δt, Tend, bc_b, bc1, mesh, "CN"; method=Base.:\)

# Plot the solution
#plot_solution(solver, mesh, body, capacity; state_i=1)

# Animation
animate_solution(solver, mesh, body)