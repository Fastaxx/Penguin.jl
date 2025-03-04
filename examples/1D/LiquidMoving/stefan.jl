using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie

### 1D Test Case : One-phase Stefan Problem
# Define the spatial mesh
nx = 40
lx = 1.
x0 = 0.
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
xf = 0.05*lx   # Interface position
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
ρ, L = 1.0, 1.0
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))

# Define the source term
f = (x,y,z,t)-> 0.0 #sin(x)*cos(10*y)
K = (x,y,z)-> 1.0

# Define the phase
Fluide = Phase(capacity, operator, f, K)

# Initial condition
u0ₒ = zeros((nx+1))
u0ᵧ = zeros((nx+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Newton parameters
max_iter = 1000
tol = 1e-6
reltol = 1e-10
Newton_params = (max_iter, tol, reltol)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log = solve_MovingLiquidDiffusionUnsteadyMono!(solver, Fluide, xf, Δt, Tend, bc_b, bc, stef_cond, mesh, "BE"; Newton_params=Newton_params, method=Base.:\)

# Animation
animate_solution(solver, mesh, body)

# Plot residuals   
#residuals[i] might be empty, remove them
residuals = filter(x -> !isempty(x), residuals)

figure = Figure()
ax = Axis(figure[1,1], xlabel = "Newton Iterations", ylabel = "Residuals", title = "Residuals")
for i in 1:length(residuals)
    lines!(ax, log10.(residuals[i]), label = "Time = $(i*Δt)")
end
#axislegend(ax)
display(figure)

# Plot the position
figure = Figure()
ax = Axis(figure[1,1], xlabel = "Time", ylabel = "Interface position", title = "Interface position")
lines!(ax, 0.0:Δt:Tend, xf_log, label = "Interface position")
display(figure)

# save xf_log
open("xf_log_$nx.txt", "w") do io
    for i in 1:length(xf_log)
        println(io, xf_log[i])
    end
end

# Plot the solution
plot_solution(solver, mesh, body, capacity; state_i=10)

# create a directory to save solver.states[i]
if !isdir("solver_states_$nx")
    mkdir("solver_states_$nx")
end
# save solver.states[i]
for i in 1:length(solver.states)
    open("solver_states_$nx/solver_states_$i.txt", "w") do io
        for j in 1:length(solver.states[i])
            println(io, solver.states[i][j])
        end
    end
end