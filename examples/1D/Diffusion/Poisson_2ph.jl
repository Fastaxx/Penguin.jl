using Penguin
using IterativeSolvers

### 1D Test Case : Diphasic Steady Diffusion Equation
# Define the mesh
nx = 40
lx = 1.0
x0 = 0.
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
pos = 0.5
body = (x, _=0) -> x - pos
body_c = (x, _=0) -> pos - x

# Define the capacity
capacity = Capacity(body, mesh)
capacity_c = Capacity(body_c, mesh)

# Define the operators
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc = Dirichlet(2.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc, :bottom => bc1))

ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Fedkiw test case : 1) ScalarJump(1.0, 1.0, -1.0), FluxJump(1.0, 1.0, 0.0), f=0, u(0)=0, u(1)=2
#                     2) ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, -1.0), f=0, u(0)=0, u(1)=2

# Define the source term
f1 = (x, y, _=0) -> 0.0
f2 = (x, y, _=0) -> 0.0

D1 = (x, y, _=0) -> 1.0
D2 = (x, y, _=0) -> 1.0

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, D1)
Fluide_2 = Phase(capacity_c, operator_c, f2, D2)

# Define the solver 
solver = DiffusionSteadyDiph(Fluide_1, Fluide_2, bc_b, ic)

# Solve the problem
solve_DiffusionSteadyDiph!(solver; method=Base.:\)

# Plot the solution
plot_solution(solver, mesh, body, capacity)
