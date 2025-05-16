using Penguin
using IterativeSolvers

### 1D Test Case : Monophasic Unsteady Diffusion Equation 
# Define the mesh
nx = 160
lx = 4.0
x0 = 0.0
domain=((x0,lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
xint = 2.0 + 0.1
body = (x, _=0) -> (x - xint)


# Define the capacity
capacity = Capacity(body, mesh)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc0 = Dirichlet(0.0)
bc1 = Dirichlet(1.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc0, :bottom => bc1))

# Define the source term
f = (x,y,z,t)->0.0
D = (x,y,z)->10.0

# Define the phase
Fluide = Phase(capacity, operator, f, D)

# Initial condition
u0ₒ = zeros(nx+1)
u0ᵧ = zeros(nx+1)
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
Δt = 0.5*(lx/nx)^2
Tend = 1.0
solver = DiffusionUnsteadyMono(Fluide, bc_b, bc0, Δt, u0, "CN")

# Solve the problem
solve_DiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, bc0, "CN"; method=Base.:\)

# Write the solution to a VTK file
#write_vtk("heat_1d", mesh, solver)

# Plot the solution
plot_solution(solver, mesh, body, capacity; state_i=10)

# Animation
animate_solution(solver, mesh, body)

