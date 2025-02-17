using Penguin

# Poisson equation inside a disk

# Define the mesh
nx, ny = 40, 40
lx, ly = 4., 4.
x0, y0 = 0., 0.
mesh = Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body with a signed distance function
# Two ways to define the same function (vectorized or not) and convert one to the other
# VOFI uses the non-vectorized version (LS) and ImplicitIntegration uses the vectorized version (Φ)
Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5 # ϕ(x, y) = Φ([x, y]) to switch from vectorized to non-vectorized

LS(x,y,_=0) = (sqrt((x-2)^2 + (y-2)^2) - 1.0) # ls(X) = LS(X[1], X[2]) to switch from non-vectorized to vectorized

# Define the capacity
capacity = Capacity(LS, mesh, method="VOFI") # or capacity = Capacity(Φ, mesh, method="ImplicitIntegration")

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(1.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))

# Define the source term and coefficients
f(x,y,_=0) = 1.0
D(x,y,_=0) = 1.0

# Define the Fluid
Fluide = Phase(capacity, operator, f, D)

# Define the solver
solver = DiffusionSteadyMono(Fluide, bc_b, bc)

# Solve the system
solve_DiffusionSteadyMono!(solver; method=Base.:\)

# Plot the solution
plot_solution(solver, mesh, LS, capacity)