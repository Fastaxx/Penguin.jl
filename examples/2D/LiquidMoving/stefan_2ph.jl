using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations
using Statistics

### 2D Test Case : Two-phase Stefan Problem : Growing Interface
# Define the spatial mesh
nx, ny = 64, 64
lx, ly = 1.0, 1.0
x0, y0 = 0.0, 0.0
Δx, Δy = lx/(nx), ly/(ny)
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the initial interface shape
sₙ(y) = 0.5 * ly + 0.05 * ly * sin(2π*y)

# Define the body for each phase
body1 = (x,y,t,_=0) -> (x - sₙ(y))          # Phase 1 (left)
body2 = (x,y,t,_=0) -> -(x - sₙ(y))         # Phase 2 (right)

# Define the Space-Time mesh
Δt = 0.005
Tend = 0.08
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity for both phases
capacity1 = Capacity(body1, STmesh)
capacity2 = Capacity(body2, STmesh)

# Initial Height Vₙ₊₁ and Vₙ for phase 1
Vₙ₊₁_1 = capacity1.A[3][1:end÷2, 1:end÷2]
Vₙ_1 = capacity1.A[3][end÷2+1:end, end÷2+1:end]
Vₙ_1 = diag(Vₙ_1)
Vₙ₊₁_1 = diag(Vₙ₊₁_1)
Vₙ_1 = reshape(Vₙ_1, (nx+1, ny+1))
Vₙ₊₁_1 = reshape(Vₙ₊₁_1, (nx+1, ny+1))

# Get the column-wise height for the interface (initial position)
Hₙ⁰ = collect(vec(sum(Vₙ_1, dims=1)))
Hₙ₊₁⁰ = collect(vec(sum(Vₙ₊₁_1, dims=1)))

# Initial interface position
Interface_position = x0 .+ Hₙ⁰./Δy
println("Initial interface position: $(Interface_position)")

# Define the diffusion operators
operator1 = DiffusionOps(capacity1)
operator2 = DiffusionOps(capacity2)

# Define the boundary conditions
bc_hot = Dirichlet(1.0)    # bottom boundary (hot)
bc_cold = Dirichlet(0.0)   # top boundary (cold)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :bottom => bc_cold,
    :top => bc_hot
))

# Phase properties
ρ, L = 1.0, 1.0               # Density and latent heat
D1, D2 = 1.0, 1.0             # Diffusion coefficients
Tm = 0.3                      # Melting temperature

# Stefan condition
stef_cond = InterfaceConditions(
    ScalarJump(1.0, 1.0, Tm),  # Temperature jump condition (T₁ = T₂ = Tm at interface)
    FluxJump(1.0, 1.0, ρ*L)    # Flux jump condition (latent heat release)
)

# Define the source terms (no internal heat generation)
f1 = (x,y,z,t) -> 0.0
f2 = (x,y,z,t) -> 0.0

# Define diffusion coefficients
K1 = (x,y,z) -> D1
K2 = (x,y,z) -> D2

# Define the phases
Fluide1 = Phase(capacity1, operator1, f1, K1)
Fluide2 = Phase(capacity2, operator2, f2, K2)

# Initial condition
# Phase 1 (left) - hot side
u1ₒ = ones((nx+1)*(ny+1))    # Bulk initial temperature = 1.0
u1ᵧ = Tm*ones((nx+1)*(ny+1)) # Interface temperature = Tm (melting temperature)

# Phase 2 (right) - cold side
u2ₒ = Tm*ones((nx+1)*(ny+1))   # Bulk initial temperature = 0.0
u2ᵧ = Tm*ones((nx+1)*(ny+1)) # Interface temperature = Tm (melting temperature)

# Combine all initial values
u0 = vcat(u1ₒ, u1ᵧ, u2ₒ, u2ᵧ)

# Newton parameters
max_iter = 100
tol = 1e-5
reltol = 1e-5
α = 1.0  # Relaxation factor
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyDiph(Fluide1, Fluide2, bc_b, stef_cond, Δt, u0, mesh, "BE")

function psip_be(args::Vararg{T,2}) where {T<:Real}
    if all(iszero, args)
        0.0
    elseif all(!iszero, args)
        1.0
    else
        1.0
    end
end

function psim_be(args::Vararg{T,2}) where {T<:Real}
    0.0
end

# Build the matrix operators for the diphasic Stefan problem
function A_diph_unstead_diff_moving_stef2(operator1::DiffusionOps, operator2::DiffusionOps, capacity1::Capacity, capacity2::Capacity, D1, D2, ic::InterfaceConditions, scheme::String)
    # Determine dimensionality from operator1
    dims1 = operator1.size
    dims2 = operator2.size
    len_dims1 = length(dims1)
    len_dims2 = length(dims2)

    # For both phases, define n1 and n2 as total dof
    n1 = prod(dims1)
    n2 = prod(dims2)

    # If 1D => n = nx; if 2D => n = nx*ny
    # (We use the same dimension logic for each operator.)
    if len_dims1 == 2
        # 1D problem
        nx1, _ = dims1
        nx2, _ = dims2
        n = nx1  # used for sub-block sizing
    elseif len_dims1 == 3
        # 2D problem
        nx1, ny1, _ = dims1
        nx2, ny2, _ = dims2
        n = nx1 * ny1
    else
        error("Only 1D or 2D supported, got dimension: $len_dims1")
    end

    # Retrieve jump & flux from the interface conditions
    jump, flux = ic.scalar, ic.flux

    Iₐ1, Iₐ2 = jump.α₁ * Penguin.I(n), jump.α₂ * Penguin.I(n)
    Iᵦ1, Iᵦ2 = flux.β₁ * Penguin.I(n), flux.β₂ * Penguin.I(n)

    # Build diffusion operators
    Id1 = build_I_D(operator1, D1, capacity1)
    Id2 = build_I_D(operator2, D2, capacity2)

    # Capacity indexing (2 for 1D, 3 for 2D)
    cap_index1 = len_dims1
    cap_index2 = len_dims2

    # Extract Vr−1 and Vr
    Vn1_1 = capacity1.A[cap_index1][1:end÷2, 1:end÷2]
    Vn1   = capacity1.A[cap_index1][end÷2+1:end, end÷2+1:end]
    Vn2_1 = capacity2.A[cap_index2][1:end÷2, 1:end÷2]
    Vn2   = capacity2.A[cap_index2][end÷2+1:end, end÷2+1:end]

    # Time integration weighting
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else
        psip, psim = psip_be, psim_be
    end

    Ψn1 = Diagonal(psip.(Vn1, Vn1_1))
    Ψn2 = Diagonal(psip.(Vn2, Vn2_1))

    # Operator sub-blocks for each phase
    W!1 = operator1.Wꜝ[1:n, 1:n]
    G1  = operator1.G[1:n, 1:n]
    H1  = operator1.H[1:n, 1:n]

    W!2 = operator2.Wꜝ[1:n, 1:n]
    G2  = operator2.G[1:n, 1:n]
    H2  = operator2.H[1:n, 1:n]

    Iᵦ1 = Iᵦ1[1:n, 1:n]
    Iᵦ2 = Iᵦ2[1:n, 1:n]
    Iₐ1 = Iₐ1[1:n, 1:n]
    Iₐ2 = Iₐ2[1:n, 1:n]
    Id1 = Id1[1:n, 1:n]
    Id2 = Id2[1:n, 1:n]

    # Construct blocks
    block1 = Vn1_1 + Id1 * G1' * W!1 * G1 * Ψn1
    block2 = -(Vn1_1 - Vn1) + Id1 * G1' * W!1 * H1 * Ψn1
    block3 = Vn2_1 + Id2 * G2' * W!2 * G2 * Ψn2
    block4 = -(Vn2_1 - Vn2) + Id2 * G2' * W!2 * H2 * Ψn2

    block5 = Iᵦ1 * H1' * W!1 * G1 * Ψn1
    block6 = Iᵦ1 * H1' * W!1 * H1 * Ψn1
    block7 = Iᵦ2 * H2' * W!2 * G2 * Ψn2
    block8 = Iᵦ2 * H2' * W!2 * H2 * Ψn2

    # Build the 4n×4n matrix
    A = spzeros(Float64, 4n, 4n)

    # Assign sub-blocks
    A[1:n, 1:n]         = block1
    A[1:n, n+1:2n]      = block2
    A[1:n, 2n+1:3n]     = spzeros(n, n)
    A[1:n, 3n+1:4n]     = spzeros(n, n)

    A[n+1:2n, 1:n]      = spzeros(n, n)
    A[n+1:2n, n+1:2n]   = Iₐ1
    A[n+1:2n, 2n+1:3n]  = spzeros(n, n)
    A[n+1:2n, 3n+1:4n]  = spzeros(n, n) #-Iₐ2

    A[2n+1:3n, 1:n]     = spzeros(n, n)
    A[2n+1:3n, n+1:2n]  = spzeros(n, n)
    A[2n+1:3n, 2n+1:3n] = block3
    A[2n+1:3n, 3n+1:4n] = block4

    A[3n+1:4n, 1:n]     = spzeros(n, n)
    A[3n+1:4n, n+1:2n]  = spzeros(n, n)
    A[3n+1:4n, 2n+1:3n] = spzeros(n, n)
    A[3n+1:4n, 3n+1:4n] = Iₐ2

    return A
end

function b_diph_unstead_diff_moving_stef2(operator1::DiffusionOps, operator2::DiffusionOps, capacity1::Capacity, capacity2::Capacity, D1, D2, f1::Function, f2::Function, ic::InterfaceConditions, Tᵢ::Vector{Float64}, Δt::Float64, t::Float64, scheme::String)
    # 1) Determine total degrees of freedom for each operator
    dims1 = operator1.size
    dims2 = operator2.size
    len_dims1 = length(dims1)
    len_dims2 = length(dims2)

    n1 = prod(dims1)  # total cells in phase 1
    n2 = prod(dims2)  # total cells in phase 2

    # 2) Identify which capacity index to read (2 for 1D, 3 for 2D)
    cap_index1 = len_dims1
    cap_index2 = len_dims2

    # 3) Build the source terms
    f1ₒn  = build_source(operator1, f1, t,      capacity1)
    f1ₒn1 = build_source(operator1, f1, t+Δt,  capacity1)
    f2ₒn  = build_source(operator2, f2, t,      capacity2)
    f2ₒn1 = build_source(operator2, f2, t+Δt,  capacity2)

    # 4) Build interface data
    jump, flux = ic.scalar, ic.flux
    Iᵧ1, Iᵧ2   = capacity1.Γ, capacity2.Γ
    gᵧ  = build_g_g(operator1, jump, capacity1)
    hᵧ  = build_g_g(operator2, flux, capacity2)
    Id1, Id2 = build_I_D(operator1, D1, capacity1), build_I_D(operator2, D2, capacity2)

    # 5) Extract Vr (current) & Vr−1 (previous) from each capacity
    Vn1_1 = capacity1.A[cap_index1][1:end÷2, 1:end÷2]
    Vn1   = capacity1.A[cap_index1][end÷2+1:end, end÷2+1:end]
    Vn2_1 = capacity2.A[cap_index2][1:end÷2, 1:end÷2]
    Vn2   = capacity2.A[cap_index2][end÷2+1:end, end÷2+1:end]

    # 6) Time-integration weighting
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else
        psip, psim = psip_be, psim_be
    end
    Ψn1 = Diagonal(psim.(Vn1, Vn1_1))
    Ψn2 = Diagonal(psim.(Vn2, Vn2_1))

    # 7) Determine whether 1D or 2D from dims1, and form local n for sub-blocks
    if len_dims1 == 2
        # 1D
        nx1, _ = dims1
        nx2, _ = dims2
        n = nx1
    else
        # 2D
        nx1, ny1, _ = dims1
        nx2, ny2, _ = dims2
        n = nx1 * ny1   # local block size for each operator
    end

    # 8) Build the bulk terms for each phase
    Tₒ1 = Tᵢ[1:n]
    Tᵧ1 = Tᵢ[n+1:2n]

    Tₒ2 = Tᵢ[2n + 1 : 3n]
    Tᵧ2 = Tᵢ[3n + 1 : end]

    f1ₒn  = f1ₒn[1:n]
    f1ₒn1 = f1ₒn1[1:n]
    f2ₒn  = f2ₒn[1:n]
    f2ₒn1 = f2ₒn1[1:n]

    gᵧ = gᵧ[1:n]
    hᵧ = hᵧ[1:n]
    Iᵧ1 = Iᵧ1[1:n, 1:n]
    Iᵧ2 = Iᵧ2[1:n, 1:n]
    Id1 = Id1[1:n, 1:n]
    Id2 = Id2[1:n, 1:n]

    W!1 = operator1.Wꜝ[1:n, 1:n]
    G1  = operator1.G[1:n, 1:n]
    H1  = operator1.H[1:n, 1:n]
    V1  = operator1.V[1:n, 1:n]

    W!2 = operator2.Wꜝ[1:n, 1:n]
    G2  = operator2.G[1:n, 1:n]
    H2  = operator2.H[1:n, 1:n]
    V2  = operator2.V[1:n, 1:n]

    # 9) Build the right-hand side
    if scheme == "CN"
        b1 = (Vn1 - Id1 * G1' * W!1 * G1 * Ψn1) * Tₒ1 - 0.5 * Id1 * G1' * W!1 * H1 * Tᵧ1 + 0.5 * V1 * (f1ₒn + f1ₒn1)
        b3 = (Vn2 - Id2 * G2' * W!2 * G2 * Ψn2) * Tₒ2 - 0.5 * Id2 * G2' * W!2 * H2 * Tᵧ2 + 0.5 * V2 * (f2ₒn + f2ₒn1)
    else
        b1 = Vn1 * Tₒ1 + V1 * f1ₒn1
        b3 = Vn2 * Tₒ2 + V2 * f2ₒn1
    end

    # 10) Build boundary terms
    b2 = gᵧ
    b4 = gᵧ

    # Final right-hand side
    return vcat(b1, b2, b3, b4)
end

# Create a linear interpolant function from centroids to values.
function create_linear_interpolant(centroids::AbstractVector{T}, values::AbstractVector{T}) where T<:Real
    N = length(centroids)
    @assert N == length(values) "Centroids and values must have the same length."
    # Assumes centroids are sorted in increasing order.
    return function (x::Real)
        if x <= centroids[1]
            # Extrapolate to the left
            x0, x1 = centroids[1], centroids[2]
            y0, y1 = values[1], values[2]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        elseif x >= centroids[end]
            # Extrapolate to the right
            x0, x1 = centroids[end-1], centroids[end]
            y0, y1 = values[end-1], values[end]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        else
            # Find the interval containing x using searchsortedlast
            i = searchsortedlast(centroids, x)
            # Ensure we have a valid index i (should be between 1 and N-1)
            x0, x1 = centroids[i], centroids[i+1]
            y0, y1 = values[i], values[i+1]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        end
    end
end

# Create a constant (upwind) interpolant function from centroids to values.
function create_constant_interpolant(centroids::AbstractVector{T}, values::AbstractVector{T}) where T<:Real
    N = length(centroids)
    @assert N == length(values) "Centroids and values must have the same length."
    return function (x::Real)
        if x < centroids[1]
            # Extrapolate (use the first constant value)
            return values[1]
        elseif x >= centroids[end]
            # Extrapolate to the right (use the last constant value)
            return values[end]
        else
            # Find the interval: return the left (upwind) value in the interval
            i = searchsortedlast(centroids, x)
            return values[i]
        end
    end
end

# Create a full constant interpolant function that return the mean value of all the values
function create_full_constant_interpolant(values::AbstractVector{T}) where T<:Real
    N = length(values)
    # Compute the mean value of the values. Remove the last value
    val = values[1:end-1]
    m̅ =  sum(val)/length(val)
    return function (x::Real)
        return m̅
    end
end
# Main solver function for the diphasic Stefan problem in 2D
function solve_MovingLiquidDiffusionUnsteadyDiph2!(s::Solver, phase1::Phase, phase2::Phase, Interface_position, Hₙ⁰, Δt::Float64, Tₑ::Float64, bc_b::BorderConditions, ic::InterfaceConditions, mesh, scheme::String; interpo="quad", Newton_params=(1000, 1e-10, 1e-10, 1.0), method=IterativeSolvers.gmres, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the problem:")
    println("- Moving problem")
    println("- Non prescibed motion")
    println("- Diphasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Solve system for the initial condition
    t = 0.0
    println("Time : $(t)")

    # Params
    ρL = ic.flux.value
    max_iter = Newton_params[1]
    tol      = Newton_params[2]
    reltol   = Newton_params[3]
    α        = Newton_params[4]

    # Log residuals and interface positions for each time step:
    nt = Int(Tₑ/Δt)
    residuals = [[] for _ in 1:2nt]
    xf_log = []
    reconstruct = []

    # Determine how many dimensions
    dims = phase1.operator.size
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

    # Initialize newton variables
    err = Inf
    err_rel = Inf
    iter = 0

    # Initialize newton height variables
    current_Hₙ = Hₙ⁰
    new_Hₙ = current_Hₙ

    # Initialize newton interface position variables
    current_xf = Interface_position
    new_xf = current_xf
    xf = current_xf

    # First time step : Newton to compute the interface position xf1
    while (iter < max_iter) && (err > tol) && (err_rel > reltol)
        iter += 1

        # 1) Solve the linear system
        solve_system!(s; method=method, kwargs...)
        Tᵢ = s.x

        # 2) Recompute heights for phase 1
        Vₙ₊₁_1 = phase1.capacity.A[cap_index][1:end÷2, 1:end÷2]
        Vₙ_1 = phase1.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
        Vₙ_1 = diag(Vₙ_1)
        Vₙ₊₁_1 = diag(Vₙ₊₁_1)
        Vₙ_1 = reshape(Vₙ_1, (nx, ny))
        Vₙ₊₁_1 = reshape(Vₙ₊₁_1, (nx, ny))
        Hₙ_1 = collect(vec(sum(Vₙ_1, dims=1)))
        Hₙ₊₁_1 = collect(vec(sum(Vₙ₊₁_1, dims=1)))

        # 3) Compute the interface flux term for phase 1
        W!1 = phase1.operator.Wꜝ[1:n, 1:n]  # n = nx*ny (full 2D system)
        G1  = phase1.operator.G[1:n, 1:n]
        H1  = phase1.operator.H[1:n, 1:n]
        V1  = phase1.operator.V[1:n, 1:n]
        Id1 = build_I_D(phase1.operator, phase1.Diffusion_coeff, phase1.capacity)
        Id1 = Id1[1:n, 1:n]
        Tₒ1, Tᵧ1 = Tᵢ[1:n], Tᵢ[n+1:2n]
        Interface_term_1 = Id1 * H1' * W!1 * G1 * Tₒ1 + Id1 * H1' * W!1 * H1 * Tᵧ1

        # 4) Compute flux term for phase 2
        W!2 = phase2.operator.Wꜝ[1:n, 1:n]
        G2  = phase2.operator.G[1:n, 1:n]
        H2  = phase2.operator.H[1:n, 1:n]
        V2  = phase2.operator.V[1:n, 1:n]
        Id2 = build_I_D(phase2.operator, phase2.Diffusion_coeff, phase2.capacity)
        Id2 = Id2[1:n, 1:n]
        Tₒ2, Tᵧ2 = Tᵢ[2*n+1:3*n], Tᵢ[3*n+1:4*n]
        Interface_term_2 = Id2 * H2' * W!2 * G2 * Tₒ2 + Id2 * H2' * W!2 * H2 * Tᵧ2

        # Combine interface terms and reshape to match the columns
        Interface_term = 1/(ρL) * (Interface_term_1 + Interface_term_2)
        Interface_term = reshape(Interface_term, (nx, ny))
        Interface_term = vec(sum(Interface_term, dims=1))
        
        # 4) Update the height function
        res = Hₙ₊₁_1 - Hₙ_1 - Interface_term
        new_Hₙ = current_Hₙ .+ α .* res            # Elementwise update for each column
        
        # Calculate error
        err = maximum(abs.(new_Hₙ .- current_Hₙ))
        err_rel = err / maximum(abs.(current_Hₙ))
        println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

        # Store residuals
        push!(residuals[1], err)

        # 5) Update geometry if not converged
        if (err <= tol) || (err_rel <= reltol)
            push!(xf_log, new_xf)
            break
        end

        # 6) Compute the new interface position table
        new_xf = x0 .+ new_Hₙ./Δy
        new_xf[end] = new_xf[1]  # Ensure periodic BC in y-direction

        # 7) Construct interpolation functions for new interface position
        centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        if interpo == "linear"
            sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
        elseif interpo == "quad"
            sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Line())
        elseif interpo == "cubic"
            sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
        else
            println("Interpolation method not supported")
        end

        # 8) Rebuild the domains with linear time interpolation
        tₙ₊₁ = t + Δt
        tₙ = t
        body1 = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return xx - x_interp
        end
        body2 = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return -(xx - x_interp)
        end        
        STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
        capacity1 = Capacity(body1, STmesh; compute_centroids=false)
        capacity2 = Capacity(body2, STmesh; compute_centroids=false)
        operator1 = DiffusionOps(capacity1)
        operator2 = DiffusionOps(capacity2)
        phase1 = Phase(capacity1, operator1, phase1.source, phase1.Diffusion_coeff)
        phase2 = Phase(capacity2, operator2, phase2.source, phase2.Diffusion_coeff)

        # 9) Rebuild the matrix A and the vector b
        s.A = A_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, scheme)
        s.b = b_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, t, scheme)

        BC_border_diph!(s.A, s.b, bc_b, mesh)

        # 10) Update variables
        current_Hₙ = new_Hₙ
        current_xf = new_xf
    end

    if (err <= tol) || (err_rel <= reltol)
        println("Converged after $iter iterations with Hₙ = $new_Hₙ, error = $err")
    else
        println("Reached max_iter = $max_iter with Hₙ = $new_Hₙ, error = $err")
    end

    # Save state after first time step
    Tᵢ = s.x
    push!(s.states, s.x)
    println("Time : $(t)")
    println("Max value : $(maximum(abs.(s.x)))")

    # Time loop for remaining steps
    k = 2
    while t < Tₑ
        t += Δt
        tₙ = t
        tₙ₊₁ = t + Δt
        println("Time : $(t)")

        # 1) Construct interpolation functions for interface position
        centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        if interpo == "linear"
            sₙ = linear_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Line())
            sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
        elseif interpo == "quad"
            sₙ = extrapolate(scale(interpolate(current_xf, BSpline(Quadratic())), centroids), Interpolations.Line())
            sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Line())
        elseif interpo == "cubic"
            sₙ = cubic_spline_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Line())# filepath: /home/libat/github/Penguin.jl/examples/2D/LiquidMoving/stefan_2d_2ph.jl
            sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
        else
            println("Interpolation method not supported")
        end

        # 1) Reconstruct
        STmesh = SpaceTimeMesh(mesh, [t-Δt, t], tag=mesh.tag)
        body1 = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return xx - x_interp
        end
        body2 = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return -(xx - x_interp)
        end
        capacity1 = Capacity(body1, STmesh; compute_centroids=false)
        capacity2 = Capacity(body2, STmesh; compute_centroids=false)
        operator1 = DiffusionOps(capacity1)
        operator2 = DiffusionOps(capacity2)
        phase1 = Phase(capacity1, operator1, phase1.source, phase1.Diffusion_coeff)
        phase2 = Phase(capacity2, operator2, phase2.source, phase2.Diffusion_coeff)

        s.A = A_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, scheme)
        s.b = b_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, t, scheme)

        BC_border_diph!(s.A, s.b, bc_b, mesh)

        push!(reconstruct, sₙ₊₁)

        # Initialize newton variables
        err = Inf
        err_rel = Inf
        iter = 0

        # Initialize newton height variables
        current_Hₙ = new_Hₙ
        new_Hₙ = current_Hₙ

        # Initialize newton interface position variables
        current_xf = new_xf
        new_xf = current_xf
        xf = current_xf

        # Newton to compute the interface position xf1
        while (iter < max_iter) && (err > tol) && (err_rel > reltol)
            iter += 1

            # 1) Solve the linear system
            solve_system!(s; method=method, kwargs...)
            Tᵢ = s.x

            # 2) Recompute heights for phase 1
            Vₙ₊₁_1 = phase1.capacity.A[cap_index][1:end÷2, 1:end÷2]
            Vₙ_1 = phase1.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
            Vₙ_1 = diag(Vₙ_1)
            Vₙ₊₁_1 = diag(Vₙ₊₁_1)
            Vₙ_1 = reshape(Vₙ_1, (nx, ny))
            Vₙ₊₁_1 = reshape(Vₙ₊₁_1, (nx, ny))
            Hₙ_1 = collect(vec(sum(Vₙ_1, dims=1)))
            Hₙ₊₁_1 = collect(vec(sum(Vₙ₊₁_1, dims=1)))

            # 3) Compute the interface flux term for phase 1
            W!1 = phase1.operator.Wꜝ[1:n, 1:n]  # n = nx*ny (full 2D system)
            G1  = phase1.operator.G[1:n, 1:n]
            H1  = phase1.operator.H[1:n, 1:n]
            V1  = phase1.operator.V[1:n, 1:n]
            Id1 = build_I_D(phase1.operator, phase1.Diffusion_coeff, phase1.capacity)
            Id1 = Id1[1:n, 1:n]
            Tₒ1, Tᵧ1 = Tᵢ[1:n], Tᵢ[n+1:2n]
            Interface_term_1 = Id1 * H1' * W!1 * G1 * Tₒ1 + Id1 * H1' * W!1 * H1 * Tᵧ1

            # 4) Compute flux term for phase 2
            W!2 = phase2.operator.Wꜝ[1:n, 1:n]
            G2  = phase2.operator.G[1:n, 1:n]
            H2  = phase2.operator.H[1:n, 1:n]
            V2  = phase2.operator.V[1:n, 1:n]
            Id2 = build_I_D(phase2.operator, phase2.Diffusion_coeff, phase2.capacity)
            Id2 = Id2[1:n, 1:n]
            Tₒ2, Tᵧ2 = Tᵢ[2*n+1:3*n], Tᵢ[3*n+1:4*n]
            Interface_term_2 = Id2 * H2' * W!2 * G2 * Tₒ2 + Id2 * H2' * W!2 * H2 * Tᵧ2

            # Combine interface terms and reshape to match the columns
            Interface_term = 1/(ρL) * (Interface_term_1 + Interface_term_2)
            Interface_term = reshape(Interface_term, (nx, ny))
            Interface_term = vec(sum(Interface_term, dims=1))

            # 4) Update the height function
            res = Hₙ₊₁_1 - Hₙ_1 - Interface_term
            new_Hₙ = current_Hₙ .+ α .* res            # Elementwise update for each column

            # Calculate error
            err = maximum(abs.(new_Hₙ .- current_Hₙ))
            err_rel = err / maximum(abs.(current_Hₙ))
            println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

            # Store residuals
            push!(residuals[k], err)

            # 5) Update geometry if not converged
            if (err <= tol) || (err_rel <= reltol)
                push!(xf_log, new_xf)
                break
            end

            # 6) Compute the new interface position table
            new_xf = x0 .+ new_Hₙ./Δy
            new_xf[end] = new_xf[1]  # Ensure Line BC in y-direction

            # 7) Construct interpolation functions for new interface position
            centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
            if interpo == "linear"
                sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
            elseif interpo == "quad"
                sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Line())
            elseif interpo == "cubic"
                sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
            else
                println("Interpolation method not supported")
            end

            # 8) Rebuild the domains with linear time interpolation
            tₙ₊₁ = t + Δt
            tₙ = t

            body1 = (xx, yy, tt, _=0) -> begin
                # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
                t_norm = (tt - tₙ) / Δt
                
                # Quadratic interpolation coefficients
                a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
                b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
                c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
                
                # Position at start, middle, and end points
                pos_start = sₙ(yy)
                pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
                pos_end = sₙ₊₁(yy)
                
                # Compute interpolated position
                x_interp = a * pos_start + b * pos_mid + c * pos_end
                
                # Return signed distance
                return xx - x_interp
            end
            body2 = (xx, yy, tt, _=0) -> begin
                # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
                t_norm = (tt - tₙ) / Δt
                
                # Quadratic interpolation coefficients
                a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
                b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
                c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
                
                # Position at start, middle, and end points
                pos_start = sₙ(yy)
                pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
                pos_end = sₙ₊₁(yy)
                
                # Compute interpolated position
                x_interp = a * pos_start + b * pos_mid + c * pos_end
                
                # Return signed distance
                return -(xx - x_interp)
            end      
            STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
            capacity1 = Capacity(body1, STmesh; compute_centroids=false)
            capacity2 = Capacity(body2, STmesh; compute_centroids=false)
            operator1 = DiffusionOps(capacity1)
            operator2 = DiffusionOps(capacity2)
            phase1 = Phase(capacity1, operator1, phase1.source, phase1.Diffusion_coeff)
            phase2 = Phase(capacity2, operator2, phase2.source, phase2.Diffusion_coeff)

            s.A = A_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, scheme)
            s.b = b_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, t, scheme)

            BC_border_diph!(s.A, s.b, bc_b, mesh)

            # 9) Update variables
            current_Hₙ = new_Hₙ
            current_xf = new_xf
        end

        if (err <= tol) || (err_rel <= reltol)
            println("Converged after $iter iterations with Hₙ = $new_Hₙ, error = $err")
        else
            println("Reached max_iter = $max_iter with Hₙ = $new_Hₙ, error = $err")
        end

        # Save state after first time step
        Tᵢ = s.x
        push!(s.states, s.x)
        println("Max value : $(maximum(abs.(s.x)))")
        
        # Update variables
        k += 1
    end

    return s, residuals, xf_log, reconstruct
end

# Solve the problem
solver, residuals, xf_log, reconstruct= solve_MovingLiquidDiffusionUnsteadyDiph2!(solver, Fluide1, Fluide2, Interface_position, Hₙ⁰, Δt, Tend, bc_b, stef_cond, mesh, "BE"; interpo="linear", Newton_params=Newton_params, method=Base.:\)

# Plot the position of the interface
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Interface position")
for i in 1:length(xf_log)
    lines!(ax, xf_log[i][1:end-1])
end
display(fig)

# Plot the residuals for one column
residuals = filter(x -> !isempty(x), residuals)

figure = Figure()
ax = Axis(figure[1,1], xlabel = "Newton Iterations", ylabel = "Residuals", title = "Residuals")
for i in 1:length(residuals)
    lines!(ax, log10.(residuals[i]), label = "Time = $(i*Δt)")
end
#axislegend(ax)
display(figure)

# Plot the position of one column
# Collect the interface position for column 5 from each time step in xf_log
column_vals = [xf[5] for xf in xf_log]

# save xf_log
open("xf_log_$nx.txt", "w") do io
    for i in 1:length(column_vals)
        println(io, column_vals[i])
    end
end


# Create a time axis (assuming each entry in xf_log corresponds to a time step; adjust if needed)
time_axis = Δt * collect(1:length(xf_log))

# Plot the time series
fig = Figure()
ax = Axis(fig[1,1], xlabel = "Time", ylabel = "Interface position", title = "Interface position (Column 5)")
lines!(ax, time_axis, column_vals, color=:blue)
display(fig)

# Animation
animate_solution(solver, mesh, body1)


function animate_stefan_diphasic(
    solver, 
    mesh,
    xf_log,
    Δt,
    nx, ny, lx, ly, x0, y0;
    filename="stefan_diphasic_animation.mp4",
    fps=10,
    title="Stefan Problem - Diphasic Heat Transfer",
    colorrange_bulk1=(0, 1),
    colorrange_interface1=(0, 1),
    colorrange_bulk2=(0, 1),
    colorrange_interface2=(0, 1),
    colormap1=:thermal,
    colormap2=:viridis,
    interpo="linear"
)
    # Create meshgrid for plotting
    xrange = range(x0, stop=x0+lx, length=nx+1)
    yrange = range(y0, stop=y0+ly, length=ny+1)
    
    # Create grid for visualization
    xs = range(x0, stop=x0+lx, length=5*nx)
    ys = range(y0, stop=y0+ly, length=5*ny)
    X = repeat(reshape(xs, 1, :), length(ys), 1)
    Y = repeat(ys, 1, length(xs))
    
    # Number of frames = number of saved states
    num_frames = length(solver.states)
    
    # Create a figure with 2x2 layout
    fig = Figure(resolution=(1200, 900))
    
    # Create titles for each subplot
    titles = [
        "Bulk Field - Phase 1", 
        "Interface Field - Phase 1",
        "Bulk Field - Phase 2", 
        "Interface Field - Phase 2"
    ]
    
    # Create axes for each subplot
    ax_bulk1 = Axis3(fig[1, 1], 
                    title=titles[1],
                    xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_interface1 = Axis3(fig[1, 2], 
                        title=titles[2], 
                        xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_bulk2 = Axis3(fig[2, 1], 
                    title=titles[3],
                    xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_interface2 = Axis3(fig[2, 2], 
                        title=titles[4],
                        xlabel="x", ylabel="y", zlabel="Temperature")
    
    # Add a main title
    Label(fig[0, :], title, fontsize=20)
    
    # Create colorbar for each phase
    Colorbar(fig[1, 3], colormap=colormap1, limits=colorrange_bulk1, label="Temperature (Phase 1)")
    Colorbar(fig[2, 3], colormap=colormap2, limits=colorrange_bulk2, label="Temperature (Phase 2)")
    
    # Set common view angles for 3D plots
    
    viewangle = (0.4pi, pi/8)
    for ax in [ax_bulk1, ax_interface1, ax_bulk2, ax_interface2]
        ax.azimuth = viewangle[1]
        ax.elevation = viewangle[2]
    end
    
    
    # Create time label and interface position indicator
    time_label = Label(fig[3, 1:2], "t = 0.00", fontsize=16)
    interface_label = Label(fig[3, 3], "Interface at x = 0.00", fontsize=16)
    
    # Create initial surface plots - will be updated in the animation
    bulk1_surface = surface!(ax_bulk1, xrange, yrange, zeros(ny+1, nx+1), 
                          colormap=colormap1, colorrange=colorrange_bulk1)
    
    interface1_surface = surface!(ax_interface1, xrange, yrange, zeros(ny+1, nx+1), 
                               colormap=colormap1, colorrange=colorrange_interface1)
    
    bulk2_surface = surface!(ax_bulk2, xrange, yrange, zeros(ny+1, nx+1), 
                          colormap=colormap2, colorrange=colorrange_bulk2)
    
    interface2_surface = surface!(ax_interface2, xrange, yrange, zeros(ny+1, nx+1), 
                               colormap=colormap2, colorrange=colorrange_interface2)
    
    # Create record of the animation
    println("Creating animation with $num_frames frames...")
    record(fig, filename, 1:num_frames; framerate=fps) do frame_idx
        # Extract the state at the current frame
        state = solver.states[frame_idx]
        
        # Get interface position for the current frame
        if frame_idx <= length(xf_log)
            current_xf = xf_log[frame_idx]
        else
            current_xf = xf_log[end]  # Use last position if we have more states than interface positions
        end
        
        # Determine interface position from height function
        centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        if interpo == "linear"
            sₙ = linear_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
        elseif interpo == "quad"
            sₙ = extrapolate(scale(interpolate(current_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
        elseif interpo == "cubic"
            sₙ = cubic_spline_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
        else
            println("Interpolation method not supported")
            sₙ = y -> x0[1] + current_xf[1]  # Fallback to constant interface
        end
        
        # Create body function for domain splitting
        body1 = (x,y) -> (x - sₙ(y))
        body2 = (x,y) -> -(x - sₙ(y))
        
        # Extract solutions for each field
        u1_bulk = reshape(state[1:(nx+1)*(ny+1)], (ny+1, nx+1))
        u1_interface = reshape(state[(nx+1)*(ny+1)+1:2*(nx+1)*(ny+1)], (ny+1, nx+1))
        u2_bulk = reshape(state[2*(nx+1)*(ny+1)+1:3*(nx+1)*(ny+1)], (ny+1, nx+1))
        u2_interface = reshape(state[3*(nx+1)*(ny+1)+1:end], (ny+1, nx+1))
        
        # Compute phase indicators for masking
        phase1_indicator = zeros(ny+1, nx+1)
        phase2_indicator = zeros(ny+1, nx+1)
        
        # Compute mask for each phase
        for i in 1:nx+1
            for j in 1:ny+1
                phase1_indicator[j,i] = body1(xrange[i], yrange[j]) <= 0 ? 1.0 : NaN
                phase2_indicator[j,i] = body2(xrange[i], yrange[j]) <= 0 ? 1.0 : NaN
            end
        end
        
        # Apply masks to the solutions
        u1_bulk_masked = u1_bulk #.* phase1_indicator
        u1_interface_masked = u1_interface #.* phase1_indicator
        u2_bulk_masked = u2_bulk #.* phase2_indicator
        u2_interface_masked = u2_interface #.* phase2_indicator
        
        # Update surface plots with current data
        bulk1_surface[3] = u1_bulk_masked
        interface1_surface[3] = u1_interface_masked
        bulk2_surface[3] = u2_bulk_masked
        interface2_surface[3] = u2_interface_masked
        
        # Plot interface curve as a line on each surface plot
        interface_y = collect(ys)
        interface_x = sₙ.(interface_y)
        
        # Update time and interface labels
        time_t = round((frame_idx-1)*(Δt), digits=3)
        time_label.text = "t = $time_t"
        avg_pos = round(mean(current_xf), digits=3)
        interface_label.text = "Interface at x ≈ $avg_pos"
        
        # Progress indicator
        if frame_idx % 10 == 0
            println("Processing frame $frame_idx / $num_frames")
        end
    end
    
    println("Animation saved to $filename")
end

# Function to add interface contour to existing plots
function add_interface_contour!(
    ax, 
    xf, 
    mesh; 
    interpo="linear", 
    color=:white, 
    linewidth=2, 
    linestyle=:solid,
    z_height=nothing
)
    # Get y coordinates
    centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
    
    # Create interpolation function for interface position
    if interpo == "linear"
        sₙ = linear_interpolation(centroids, xf, extrapolation_bc=Interpolations.Periodic())
    elseif interpo == "quad"
        sₙ = extrapolate(scale(interpolate(xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
    elseif interpo == "cubic"
        sₙ = cubic_spline_interpolation(centroids, xf, extrapolation_bc=Interpolations.Periodic())
    else
        println("Interpolation method not supported")
        return
    end
    
    # Create array of interface points
    y_points = range(mesh.nodes[2][1], mesh.nodes[2][end], length=100)
    x_points = sₙ.(y_points)
    
    # Determine appropriate z-value
    if isnothing(z_height)
        # Default to 1.0 if no specific height provided
        z_values = ones(length(y_points)) * 1.0
    else
        z_values = ones(length(y_points)) * z_height
    end
    
    # Plot interface contour
    lines!(ax, x_points, y_points, z_values, 
           color=color, linewidth=linewidth, linestyle=linestyle)
end

# Function to visualize a single frame with interface
function plot_stefan_diphasic_frame(
    solver,
    mesh,
    xf_log,
    frame_idx,
    nx, ny, lx, ly, x0, y0;
    interpo="linear",
    colorrange_bulk1=(0, 1),
    colorrange_interface1=(0, 1),
    colorrange_bulk2=(0, 1),
    colorrange_interface2=(0, 1),
    colormap1=:thermal,
    colormap2=:viridis
)

    # Create meshgrid for plotting
    xrange = range(x0, stop=x0+lx, length=nx+1)
    yrange = range(y0, stop=y0+ly, length=ny+1)

    # Get current state
    state = solver.states[frame_idx]
    
    # Get interface position
    if frame_idx <= length(xf_log)
        current_xf = xf_log[frame_idx]
    else
        current_xf = xf_log[end]
    end
    
    # Create interpolation function for interface
    centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
    if interpo == "linear"
        sₙ = linear_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
    elseif interpo == "quad"
        sₙ = extrapolate(scale(interpolate(current_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
    elseif interpo == "cubic"
        sₙ = cubic_spline_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
    else
        println("Interpolation method not supported")
        sₙ = y -> x0[1] + current_xf[1]
    end
    
    # Create body functions
    body1 = (x,y) -> (x - sₙ(y))
    body2 = (x,y) -> -(x - sₙ(y))
    
    # Extract solutions for each field
    u1_bulk = reshape(state[1:(nx+1)*(ny+1)], (ny+1, nx+1))
    u1_interface = reshape(state[(nx+1)*(ny+1)+1:2*(nx+1)*(ny+1)], (ny+1, nx+1))
    u2_bulk = reshape(state[2*(nx+1)*(ny+1)+1:3*(nx+1)*(ny+1)], (ny+1, nx+1))
    u2_interface = reshape(state[3*(nx+1)*(ny+1)+1:end], (ny+1, nx+1))
    
    # Compute phase masks
    phase1_indicator = zeros(ny+1, nx+1)
    phase2_indicator = zeros(ny+1, nx+1)
    
    for i in 1:nx+1
        for j in 1:ny+1
            phase1_indicator[j,i] = body1(xrange[i], yrange[j]) <= 0 ? 1.0 : NaN
            phase2_indicator[j,i] = body2(xrange[i], yrange[j]) <= 0 ? 1.0 : NaN
        end
    end
    
    # Apply masks
    u1_bulk_masked = u1_bulk .* phase1_indicator
    u1_interface_masked = u1_interface .* phase1_indicator
    u2_bulk_masked = u2_bulk .* phase2_indicator
    u2_interface_masked = u2_interface .* phase2_indicator
    
    # Create figure with 2x2 layout
    fig = Figure(resolution=(1200, 900))
    
    # Create titles for each subplot
    titles = [
        "Bulk Field - Phase 1", 
        "Interface Field - Phase 1",
        "Bulk Field - Phase 2", 
        "Interface Field - Phase 2"
    ]
    
    # Create axes for each subplot
    ax_bulk1 = Axis3(fig[1, 1], 
                    title=titles[1],
                    xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_interface1 = Axis3(fig[1, 2], 
                        title=titles[2], 
                        xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_bulk2 = Axis3(fig[2, 1], 
                    title=titles[3],
                    xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_interface2 = Axis3(fig[2, 2], 
                        title=titles[4],
                        xlabel="x", ylabel="y", zlabel="Temperature")
    
    # Add a main title with frame info
    Label(fig[0, :], "Stefan Problem - Frame $frame_idx", fontsize=20)
    
    # Create surface plots
    surface!(ax_bulk1, xrange, yrange, u1_bulk_masked, 
             colormap=colormap1, colorrange=colorrange_bulk1)
    
    surface!(ax_interface1, xrange, yrange, u1_interface_masked, 
             colormap=colormap1, colorrange=colorrange_interface1)
    
    surface!(ax_bulk2, xrange, yrange, u2_bulk_masked, 
             colormap=colormap2, colorrange=colorrange_bulk2)
    
    surface!(ax_interface2, xrange, yrange, u2_interface_masked, 
             colormap=colormap2, colorrange=colorrange_interface2)
    
    # Add interface contours with specific z-height
    add_interface_contour!(ax_bulk1, current_xf, mesh, interpo=interpo, z_height=1.0)
    add_interface_contour!(ax_interface1, current_xf, mesh, interpo=interpo, z_height=1.0)
    add_interface_contour!(ax_bulk2, current_xf, mesh, interpo=interpo, z_height=1.0)
    add_interface_contour!(ax_interface2, current_xf, mesh, interpo=interpo, z_height=1.0)
    
    # Add colorbars
    Colorbar(fig[1, 3], colormap=colormap1, limits=colorrange_bulk1, label="Temperature (Phase 1)")
    Colorbar(fig[2, 3], colormap=colormap2, limits=colorrange_bulk2, label="Temperature (Phase 2)")
    
    # Set common view angles
    viewangle = (0.4pi, pi/8)
    for ax in [ax_bulk1, ax_interface1, ax_bulk2, ax_interface2]
        ax.azimuth = viewangle[1]
        ax.elevation = viewangle[2]
    end

    return fig
end

# Example use at the end of the stefan_2d_2ph.jl file:
# Create animation
animate_stefan_diphasic(
    solver, 
    mesh, 
    xf_log, 
    Δt,
    nx, ny, lx, ly, x0, y0;
    filename="stefan_diphasic_animation.mp4",
    fps=15,
    title="Stefan Problem - Diphasic Heat Transfer",
    colorrange_bulk1=(0, 1.0),
    colorrange_interface1=(0.4, 0.6),
    colorrange_bulk2=(0, 1.0),
    colorrange_interface2=(0.4, 0.6),
    colormap1=:viridis,
    colormap2=:viridis,
    interpo="linear"
)

# Plot last frame for detailed visualization
last_frame_fig = plot_stefan_diphasic_frame(
    solver,
    mesh,
    xf_log,
    length(solver.states),
    nx, ny, lx, ly, x0, y0,
    interpo="linear",
    colorrange_bulk1=(0, 1.0),
    colorrange_interface1=(0.4, 0.6),
    colorrange_bulk2=(0, 1.0),
    colorrange_interface2=(0.4, 0.6),
    colormap1=:viridis,
    colormap2=:viridis
)
display(last_frame_fig)