using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations

### 2D Test Case : One-phase Stefan Problem : Growing Planar Interface
# Define the spatial mesh
nx, ny = 80, 80
lx, ly = 1., 1.
x0, y0 = 0., 0.
Δx, Δy = lx/(nx), ly/(ny)
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
sₙ(y) =  0.75*lx + 0.025*sin(4π*y+π/2)
body = (x,y,t,_=0)->(x - sₙ(y))

# Define the Space-Time mesh
Δt = 0.01
Tend = 0.25
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)

# Initial Height Vₙ₊₁ and Vₙ
Vₙ₊₁ = capacity.A[3][1:end÷2, 1:end÷2]
Vₙ = capacity.A[3][end÷2+1:end, end÷2+1:end]
Vₙ = diag(Vₙ)
Vₙ₊₁ = diag(Vₙ₊₁)
Vₙ = reshape(Vₙ, (nx+1, ny+1))
Vₙ₊₁ = reshape(Vₙ₊₁, (nx+1, ny+1))

Hₙ⁰ = collect(vec(sum(Vₙ, dims=1)))
Hₙ₊₁⁰ = collect(vec(sum(Vₙ₊₁, dims=1)))

Interface_position = x0 .+ Hₙ⁰./Δy
println("Interface position : $(Interface_position)")
# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(1.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}( :bottom => bc1))
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
max_iter = 10000
tol = 1e-7
reltol = 1e-7
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

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

function A_mono_unstead_diff_moving2(operator::DiffusionOps, capacity::Capacity, D, bc::AbstractBoundary, scheme::String)
    # Determine dimension (1D vs 2D) from operator.size
    dims = operator.size
    len_dims = length(dims)
    
    # Pick capacity.A index based on dimension
    #  => 2 for 1D: capacity.A[2]
    #  => 3 for 2D: capacity.A[3]
    cap_index = len_dims
    
    # Extract Vr (current) & Vr-1 (previous) from capacity
    Vn_1 = capacity.A[cap_index][1:end÷2, 1:end÷2]
    Vn   = capacity.A[cap_index][end÷2+1:end, end÷2+1:end]

    # Select time integration weighting
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else
        psip, psim = psip_be, psim_be
    end
    Ψn1 = Diagonal(psip.(Vn, Vn_1))

    # Build boundary blocks
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Iᵧ     = capacity.Γ
    Id     = build_I_D(operator, D, capacity)

    # Adjust for dimension
    if len_dims == 2
        # 1D problem
        nx, nt = dims
        n = nx
    elseif len_dims == 3
        # 2D problem
        nx, ny, nt = dims
        n = nx*ny
    else
        error("A_mono_unstead_diff_moving_generic not supported for dimension $(len_dims).")
    end

    W! = operator.Wꜝ[1:n, 1:n]
    G  = operator.G[1:n, 1:n]
    H  = operator.H[1:n, 1:n]
    Iᵦ = Iᵦ[1:n, 1:n]
    Iₐ = Iₐ[1:n, 1:n]
    Iᵧ = Iᵧ[1:n, 1:n]
    Id  = Id[1:n, 1:n]

    # Construct subblocks
    block1 = Vn_1 + Id * G' * W! * G * Ψn1
    block2 = -(Vn_1 - Vn) + Id * G' * W! * H * Ψn1
    block3 = Iᵦ * H' * W! * G
    block4 = Iᵦ * H' * W! * H + (Iₐ * Iᵧ)

    return [block1 block2; block3 block4]
end

function b_mono_unstead_diff_moving2(operator::DiffusionOps, capacity::Capacity, D, f::Function, bc::AbstractBoundary, Tᵢ::Vector{Float64}, Δt::Float64, t::Float64, scheme::String)
    # Determine how many dimensions
    dims = operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Build common data
    fₒn  = build_source(operator, f, t,      capacity)
    fₒn1 = build_source(operator, f, t+Δt,  capacity)
    gᵧ   = build_g_g(operator, bc, capacity)
    Id   = build_I_D(operator, D, capacity)

    # Select the portion of capacity.A to handle Vr−1 (above half) and Vr (below half)
    Vn_1 = capacity.A[cap_index][1:end÷2, 1:end÷2]
    Vn   = capacity.A[cap_index][end÷2+1:end, end÷2+1:end]

    # Time integration weighting
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else
        psip, psim = psip_be, psim_be
    end
    Ψn = Diagonal(psim.(Vn, Vn_1))

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
        error("b_mono_unstead_diff_moving not supported for dimension $len_dims")
    end

    # Extract operator sub-blocks
    W! = operator.Wꜝ[1:n, 1:n]
    G  = operator.G[1:n, 1:n]
    H  = operator.H[1:n, 1:n]
    V  = operator.V[1:n, 1:n]
    Iᵧ_mat = capacity.Γ[1:n, 1:n]
    Tₒ = Tᵢ[1:n]
    Tᵧ = Tᵢ[n+1:end]
    fₒn, fₒn1 = fₒn[1:n], fₒn1[1:n]
    gᵧ = gᵧ[1:n]
    Id = Id[1:n, 1:n]

    # Construct the right-hand side
    if scheme == "CN"
        b1 = (Vn - Id * G' * W! * G * Ψn)*Tₒ - 0.5 * Id * G' * W! * H * Tᵧ + 0.5 * V * (fₒn + fₒn1)
    else
        b1 = Vn * Tₒ + V * fₒn1
    end
    b2 = Iᵧ_mat * gᵧ

    return [b1; b2]
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

function solve_MovingLiquidDiffusionUnsteadyMono2!(s::Solver, phase::Phase, Interface_position, Hₙ⁰, Δt::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh, scheme::String; interpo="quad", Newton_params=(1000, 1e-10, 1e-10, 1.0), method=IterativeSolvers.gmres, kwargs...)
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
    residuals = [[] for _ in 1:2nt]
    xf_log = []
    reconstruct = []

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

        # 2) Recompute heights
        Vₙ₊₁ = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
        Vₙ = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
        Vₙ = diag(Vₙ)
        Vₙ₊₁ = diag(Vₙ₊₁)
        Vₙ = reshape(Vₙ, (nx, ny))
        Vₙ₊₁ = reshape(Vₙ₊₁, (nx, ny))
        Hₙ = collect(vec(sum(Vₙ, dims=1)))
        Hₙ₊₁ = collect(vec(sum(Vₙ₊₁, dims=1)))

        # 3) Compute the interface flux term
        W! = phase.operator.Wꜝ[1:n, 1:n]  # n = nx*ny (full 2D system)
        G  = phase.operator.G[1:n, 1:n]
        H  = phase.operator.H[1:n, 1:n]
        V  = phase.operator.V[1:n, 1:n]
        Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id = Id[1:n, 1:n]
        Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
        Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        
        # Check if bc is a Gibbs-Thomson condition
        if bc isa GibbsThomson
            velocity = 1/(ρL) * abs.(Interface_term) / Δt # Still need to find the interface velocity. Right now i've got the lower veloc
            @show velocity
            np = prod(phase.operator.size)
            # Complete the velocity vector with zeros
            velocity = vcat(velocity, zeros(np-length(velocity)))
            bc.vᵞ = velocity
        end

        Interface_term = reshape(Interface_term, (nx, ny))
        Interface_term = 1/(ρL) * vec(sum(Interface_term, dims=1))
        println("Interface term: ", Interface_term)

        # 4) Update the height function
        res = Hₙ₊₁ - Hₙ - Interface_term
        println("res: ", res)
        new_Hₙ = current_Hₙ .+ α .* res            # Elementwise update for each column
        println("new_Hₙ: ", new_Hₙ)
        err = abs.(new_Hₙ[5] .- current_Hₙ[5])
        err_rel = err/maximum(abs.(current_xf[5]))
        println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

        # Store residuals (if desired, you could store the full vector or simply the norm)
        push!(residuals[1], err)

        # 5) Update geometry if not converged
        if (err <= tol) || (err_rel <= reltol)
            push!(xf_log, new_Hₙ)
            break
        end

        # Store tn+1 and tn
        tₙ₊₁ = t + Δt
        tₙ  = t

        # 6) Compute the new interface position table
        new_xf = x0 .+ new_Hₙ./Δy
        new_xf[end] = new_xf[1]

        # 7) Construct a interpolation function for the new interface position : sn and sn+1
        centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        if interpo == "linear"
            sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
        elseif interpo == "quad"
            sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
        elseif interpo == "cubic"
            sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
        else
            println("Interpolation method not supported")
        end

        # 8) Rebuild the domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
        body = (xx,yy,tt,_=0)->(xx - (sₙ(yy)*(tₙ₊₁ - tt)/Δt + sₙ₊₁(yy)*(tt - tₙ)/Δt))
        STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
        capacity = Capacity(body, STmesh; compute_centroids=false)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        # 9) Rebuild the matrix A and the vector b
        s.A = A_mono_unstead_diff_moving2(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving2(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, t, scheme)

        BC_border_mono!(s.A, s.b, bc_b, mesh)

        # 10) Update variables
        current_Hₙ = new_Hₙ
        current_xf = new_xf
    end

    if (err <= tol) || (err_rel <= reltol)
        println("Converged after $iter iterations with Hₙ = $new_Hₙ, error = $err")
    else
        println("Reached max_iter = $max_iter with Hₙ = $new_Hₙ, error = $err")
    end

    Tᵢ = s.x
    push!(s.states, s.x)
    println("Time : $(t[1])")
    println("Max value : $(maximum(abs.(s.x)))")

    # Time loop
    k=2
    while t<Tₑ
        t+=Δt
        println("Time : $(t)")

        # 1) Construct an interpolation function for the interface position
        centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        #centroids = collect(centroids)
        if interpo == "linear"
            sₙ = linear_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
            sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
        elseif interpo == "quad"
            sₙ = extrapolate(scale(interpolate(current_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
            sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
        elseif interpo == "cubic"
            sₙ = cubic_spline_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
            sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
        else
            println("Interpolation method not supported")
        end

        # 1) Reconstruct
        STmesh = SpaceTimeMesh(mesh, [Δt, 2Δt], tag=mesh.tag)
        body = (xx,yy,tt,_=0)->(xx - (sₙ(yy)*(2Δt - tt)/Δt + sₙ₊₁(yy)*(tt - Δt)/Δt))
        capacity = Capacity(body, STmesh; compute_centroids=false)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        s.A = A_mono_unstead_diff_moving2(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving2(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, 0.0, scheme)

        BC_border_mono!(s.A, s.b, bc_b, mesh)

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

            # 2) Recompute heights
            Vₙ₊₁ = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
            Vₙ = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
            Vₙ = diag(Vₙ)
            Vₙ₊₁ = diag(Vₙ₊₁)
            Vₙ = reshape(Vₙ, (nx, ny))
            Vₙ₊₁ = reshape(Vₙ₊₁, (nx, ny))
            Hₙ = collect(vec(sum(Vₙ, dims=1)))
            Hₙ₊₁ = collect(vec(sum(Vₙ₊₁, dims=1)))

            # 3) Compute the interface flux term
            W! = phase.operator.Wꜝ[1:n, 1:n]  # n = nx*ny (full 2D system)
            G  = phase.operator.G[1:n, 1:n]
            H  = phase.operator.H[1:n, 1:n]
            V  = phase.operator.V[1:n, 1:n]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:n, 1:n]
            Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
            Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
                    
            # Check if bc is a Gibbs-Thomson condition
            if bc isa GibbsThomson
                velocity = 1/(ρL) * abs.(Interface_term)/ Δt # Still need to find the interface velocity
                @show velocity
                np = prod(phase.operator.size)
                # Complete the velocity vector with zeros
                velocity = vcat(velocity, zeros(np-length(velocity)))
                bc.vᵞ = velocity
            end

            Interface_term = reshape(Interface_term, (nx, ny))
            Interface_term = 1/(ρL) * vec(sum(Interface_term, dims=1))
            println("Interface term: ", Interface_term)

            # 4) Update the height function
            res = Hₙ₊₁ - Hₙ - Interface_term
            println("res: ", res)
            new_Hₙ = current_Hₙ .+ α .* res            # Elementwise update for each column
            println("new_Hₙ: ", new_Hₙ)
            err = abs.(new_Hₙ[5] .- current_Hₙ[5])
            err_rel = err/maximum(abs.(current_xf[5]))
            println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

            # Store residuals (if desired, you could store the full vector or simply the norm)
            push!(residuals[k], err)

            # 5) Update geometry if not converged
            if (err <= tol) || (err_rel <= reltol)
                push!(xf_log, new_Hₙ)
                break
            end

            # Store tn+1 and tn
            tₙ₊₁ = t + Δt
            tₙ  = t

            # 6) Compute the new interface position table
            new_xf = x0 .+ new_Hₙ./Δy
            new_xf[end] = new_xf[1]

            # 7) Construct a interpolation function for the new interface position :
            centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
            if interpo == "linear"
                sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
            elseif interpo == "quad"
                sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
            elseif interpo == "cubic"
                sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
            else
                println("Interpolation method not supported")
            end

            # 8) Rebuild the domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
            body = (xx,yy,tt,_=0)->(xx - (sₙ(yy)*(tₙ₊₁ - tt)/Δt + sₙ₊₁(yy)*(tt - tₙ)/Δt))
            STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
            capacity = Capacity(body, STmesh; compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

            # 9) Rebuild the matrix A and the vector b
            s.A = A_mono_unstead_diff_moving2(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving2(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, 0.0, scheme)

            BC_border_mono!(s.A, s.b, bc_b, mesh)

            # 10) Update variables
            current_Hₙ = new_Hₙ
            current_xf = new_xf

        end

        if (err <= tol) || (err_rel <= reltol)
            println("Converged after $iter iterations with xf = $new_Hₙ, error = $err")
        else
            println("Reached max_iter = $max_iter with xf = $new_Hₙ, error = $err")
        end

        push!(s.states, s.x)
        println("Time : $(t[1])")
        println("Max value : $(maximum(abs.(s.x)))")
        k+=1
    end
return s, residuals, xf_log, reconstruct
end 

# Solve the problem
solver, residuals, xf_log, reconstruct= solve_MovingLiquidDiffusionUnsteadyMono2!(solver, Fluide, Interface_position, Hₙ⁰, Δt, Tend, bc_b, bc, stef_cond, mesh, "BE"; Newton_params=Newton_params, method=Base.:\)

# Plot the position of the interface

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Interface position")
for i in 1:length(xf_log)
    lines!(ax, xf_log[i][1:end-1])
end
display(fig)

# write it to a file : each column is a time step
open("interface_position_cub.txt", "w") do io
    for i in 1:length(xf_log)
        println(io, xf_log[i][1:end-1])
    end
end

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

# Create a time axis (assuming each entry in xf_log corresponds to a time step; adjust if needed)
time_axis = Δt * collect(1:length(xf_log))

# Plot the time series
fig = Figure()
ax = Axis(fig[1,1], xlabel = "Time", ylabel = "Interface position", title = "Interface position (Column 5)")
lines!(ax, time_axis, column_vals, color=:blue)
display(fig)

# Animation
animate_solution(solver, mesh, body)

# Choose the time step (e.g., 1)
i = 2

# Extract the temperature field stored in solver.states[i].
# Here we assume the temperature field is the first half of s.x and that
# the grid has (nx+1)×(ny+1) nodes.
T_vec = solver.states[i][1:((nx+1)*(ny+1))]
T_field = reshape(T_vec, (nx+1, ny+1))
# For contour plotting, we want a matrix of size (ny+1, nx+1) so that
# each row corresponds to a y value and each column to an x value.
T_field = T_field'  # now size is (ny+1, nx+1)
# Convert T_field to Float64 if necessary.
T_field = Float64.(T_field)

# Remove the last row and column of T_field to match the mesh nodes.
T_field = T_field[1:end-1, 1:end-1]

# Create a figure and axis using mesh.nodes[1] for x (length = nx+1) and mesh.nodes[2] for y (length = ny+1).
fig = Figure()
ax = Axis(fig[1,1], xlabel="x", ylabel="y", title="Isotherms")

# Define several isotherm levels, for example:
levels = [0.01, 0.05, 0.1, 0.2, 0.5]

# Plot contours (isotherms)
contour!(ax, T_field, levels=levels, linewidth=2)

display(fig)

# Evaluate the amplitude of reconstructed interface
# reconstruct is a vector of functions that interpolate the interface position at each time step.
# For example, to evaluate the amplitude of the interface at the first time step:

centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
amplitude = Float64[]
for i in 1:length(reconstruct)
    s = reconstruct[i]
    # Evaluate the function at the centroids
    values = s.(centroids)
    # Compute the amplitude
    push!(amplitude, (maximum(values) - minimum(values)) / 2)
end

# Plot the amplitude of the interface over time
fig = Figure()
ax = Axis(fig[1,1], xlabel="Time", ylabel="Amplitude", title="Interface Amplitude")
lines!(ax, Δt * collect(1:length(amplitude)), amplitude, color=:blue)
display(fig)

# write the amplitude to a file
interpo = "linear"  # define default interpolation method
open("amplitude_$(ϵᵥ)_$(ny).txt", "w") do io
    for i in 1:length(amplitude)
        println(io, i, "\t", amplitude[i])
    end
end