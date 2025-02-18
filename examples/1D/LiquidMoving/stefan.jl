using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit

### 1D Test Case : One-phase Stefan Problem
# Define the spatial mesh
nx = 40
lx = 1.
x0 = 0.
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
xf = 0.5*lx   # Interface position
body = (x,t, _=0)->(x - xf)

# Define the Space-Time mesh
Δt = 0.001
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))

# Define the source term
f = (x,y,z,t)-> 0.0 #sin(x)*cos(10*y)
K = (x,y,z)-> 1.0

# Define the phase
Fluide = Phase(capacity, operator, f, K)

# Initial condition
u0ₒ = zeros((nx+1))
u0ᵧ = zeros((nx+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solve_MovingLiquidDiffusionUnsteadyMono!(solver, Fluide, xf, Δt, Tend, bc_b, bc, mesh, "BE"; method=Base.:\)