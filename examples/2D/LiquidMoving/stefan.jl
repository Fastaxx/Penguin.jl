using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie

### 2D Test Case : One-phase Stefan Problem : Growing Planar Interface
# Define the spatial mesh
nx, ny = 40, 40
lx, ly = 1., 1.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body : Small wave sinusoidal perturbation
yf = 0.5*ly   # Interface position
body = (x,y,t)-> (y - yf + 0.1*sin(2π*x))
# Define the Space-Time mesh
Δt = 0.001
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)

Vn_1 = capacity.A[3][1:end÷2, 1:end÷2]
Vn   = capacity.A[3][end÷2+1:end, end÷2+1:end]

Vn = diag(Vn)
Vn_1 = diag(Vn_1)

Vn = reshape(Vn, (nx+1, ny+1))
Vn_1 = reshape(Vn_1, (nx+1, ny+1))
# Compute the height of each column by summing over the y-direction (columns)
Hₙ = collect(vec(sum(Vn, dims=2)))
Hₙ₊₁ = collect(vec(sum(Vn_1, dims=2)))
println("Heights for each column at n: ", Hₙ)
println("Heights for each column at n+1: ", Hₙ₊₁)

# Plot the column heights as a bar plot
cols = collect(1:length(Hₙ))
bar_width = 0.4

fig2 = Figure()
ax2 = Axis(fig2[1,1], xlabel="Column Index (x)", ylabel="Height", 
    title="Column Heights (Dodged Bar Plot)")

# Plot Hₙ (blue), shifted left
barplot!(ax2, cols .- bar_width/2, Hₙ, width=bar_width, color=:blue, label="Hₙ")
# Plot Hₙ₊₁ (red), shifted right
barplot!(ax2, cols .+ bar_width/2, Hₙ₊₁, width=bar_width, color=:red, label="Hₙ₊₁")

axislegend(ax2, position = :rb)
display(fig2)

# Interpolations to reconstruct the interface position (heights)
using Interpolations

x = mesh.centers[1]
Hₙ = Hₙ[1:end-1]  # Remove the last element to match the length of x
Hₙ₊₁ = Hₙ₊₁[1:end-1]  # Remove the last element to match the length of x
# Interpolations for Hₙ and Hₙ₊₁
itp_Hₙ = linear_interpolation(x, Hₙ)
itp_Hₙ₊₁ = linear_interpolation(x, Hₙ₊₁)

# Plot the interpolations
fig3 = Figure()
ax3 = Axis(fig3[1,1], xlabel="Column Index (x)", ylabel="Height", 
    title="Interpolated Column Heights")
# Plot Hₙ (blue)
lines!(ax3, x, itp_Hₙ(x), color=:blue, label="Hₙ")
# Plot Hₙ₊₁ (red)
lines!(ax3, x, itp_Hₙ₊₁(x), color=:red, label="Hₙ₊₁")

axislegend(ax3, position = :rb)
display(fig3)
readline()

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(1.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))
ρ, L = 1.0, 1.0
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))

# Define the source term
f = (x,y,z,t)-> 0.0 #sin(x)*cos(10*y)
K = (x,y,z)-> 1.0

Fluide = Phase(capacity, operator, f, K)

# Initial condition
u0ₒ = zeros((nx+1)*(ny+1))
u0ᵧ = zeros((nx+1)*(ny+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Newton parameters
max_iter = 1000
tol = 1e-6
reltol = 1e-10
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

function solve_MovingLiquidDiffusionUnsteadyMono2!(s::Solver, phase::Phase, Hₙ, Hₙ₊₁, Δt::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh, scheme::String; Newton_params=(1000, 1e-10, 1e-10, 1.0), method=IterativeSolvers.gmres, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the problem:")
    println("- Moving problem")
    println("- Non prescibed motion")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Solve system for the initial condition
    t=0.0
    println("Time : $(t)")

    # Params
    ρL = ic.flux.value
    max_iter = Newton_params[1]
    tol      = Newton_params[2]
    reltol   = Newton_params[3]
    α        = Newton_params[4]

    # Log residuals and interface positions for each time step:
    nt = Int(Tₑ/Δt)
    residuals = [Float64[] for _ in 1:2nt]
    xf_log = Float64[]

    # Determine how many dimensions
    dims = phase.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Create the 1D or 2D indices
    if len_dims == 2
        # 1D case
        nx, nt = dims
        n = nx
    elseif len_dims == 3
        # 2D case
        nx, ny, nt = dims
        n = nx*ny
    else
        error("Only 1D and 2D problems are supported.")
    end

    err = Inf
    iter = 0
    current_xf = Hₙ
    new_xf = current_xf
    xf = current_xf
    # First time step : Newton to compute the interface position xf1
    while (iter < max_iter) && (err > tol) 
        iter += 1

        # 1) Solve the linear system
        solve_system!(s; method=method, kwargs...)
        Tᵢ = s.x

        # Extract volume matrices (assumed stored diagonally in capacity.A[cap_index])
        Vn_1_full = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
        Vn_full   = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]

        Vn_full = [Vn_full[i,i] for i in 1:size(Vn_full, 1)]
        Vn_1_full = [Vn_1_full[i,i] for i in 1:size(Vn_1_full, 1)]

        # Reshape them into 2D grids: (nx+1) rows, (ny+1) columns.
        Vn = reshape(Vn_full, (nx, ny))
        Vn_1 = reshape(Vn_1_full, (nx, ny))

        # Compute the height of each column (summing over y)
        Hₙ   = vec(sum(Vn, dims=2))     # Hₙ is a vector of length (nx+1)
        Hₙ₊₁ = vec(sum(Vn_1, dims=2))   # Hₙ₊₁ is a vector of length (nx+1)

        # Compute the interface term
        W! = phase.operator.Wꜝ[1:n, 1:n]  # n = nx*ny (full 2D system)
        G  = phase.operator.G[1:n, 1:n]
        H  = phase.operator.H[1:n, 1:n]
        V  = phase.operator.V[1:n, 1:n]
        Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)[1:n, 1:n]
        Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
        # Here sum columnwise over blocks corresponding to each x; for example you can reshape the flux contribution:
        flux_full = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        flux_full = reshape(flux_full, (nx, ny))
        Interface_term = 1/(ρL) * vec(sum(flux_full, dims=2))
        println("Interface term: ", Interface_term)
        # New interface position
        res = Hₙ₊₁ - Hₙ - Interface_term
        new_xf = current_xf .+ α .* res            # Elementwise update for each column
        err = maximum(abs.(new_xf .- current_xf))
        println("Iteration $iter | xf (max) = $(maximum(new_xf)) | err = $err")

        # Store residuals (if desired, you could store the full vector or simply the norm)
        push!(residuals[1], err)

        # 3) Update geometry if not converged
        if (err <= tol) || (err <= reltol * maximum(abs.(current_xf)))
            push!(xf_log, new_xf)
            break
        end

    end

    if (err <= tol) || (err <= reltol * abs(current_xf))
        println("Converged after $iter iterations with xf = $new_xf, error = $err")
    else
        println("Reached max_iter = $max_iter with xf = $new_xf, error = $err")
    end
    
    Tᵢ = s.x
    push!(s.states, s.x)
    println("Time : $(t[1])")
    println("Max value : $(maximum(abs.(s.x)))")
end

# Solve the problem
#solver, residuals, xf_log = solve_MovingLiquidDiffusionUnsteadyMono2!(solver, Fluide, Hₙ, Hₙ₊₁, Δt, Tend, bc_b, bc, stef_cond, mesh, "BE"; Newton_params=Newton_params, method=Base.:\)