# Full Moving - Diffusion - Unsteady - Monophasic
"""
    MovingLiquidDiffusionUnsteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)

Create a solver for the unsteady diffusion problem with moving interface in a monophasic problem.

# Arguments
- phase::Phase: The phase object containing the capacity and the operator.
- bc_b::BorderConditions: The border conditions.
- bc_i::AbstractBoundary: The interface condition.
- Δt::Float64: The time step.
- Tᵢ::Vector{Float64}: The initial temperature field.
- mesh::AbstractMesh: The mesh.
- scheme::String: The scheme to use for the time discretization ("BE" or "CN").
"""
function MovingLiquidDiffusionUnsteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Moving problem")
    println("- Non prescibed motion")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, ConvergenceHistory(), [])
    
    if scheme == "CN"
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "CN")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "CN")
    else # BE
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "BE")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "BE")
    end
    BC_border_mono!(s.A, s.b, bc_b, mesh)
    return s
end

function solve_MovingLiquidDiffusionUnsteadyMono!(s::Solver, phase::Phase, xf, Δt::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh::AbstractMesh, scheme::String; Newton_params=(1000, 1e-10, 1e-10, 1.0), method=IterativeSolvers.gmres, kwargs...)
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
    current_xf = xf
    new_xf = current_xf
    xf = current_xf
    # First time step : Newton to compute the interface position xf1
    while (iter < max_iter) && (err > tol) && (err > reltol * abs(current_xf))
        iter += 1

        # 1) Solve the linear system
        solve_system!(s; method=method, kwargs...)
        Tᵢ = s.x

        # 2) Update volumes / compute new interface
        Vn_1 = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
        Vn   = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
        Hₙ   = sum(diag(Vn))
        Hₙ₊₁ = sum(diag(Vn_1))

        # Compute flux
        W! = phase.operator.Wꜝ[1:n, 1:n]
        G = phase.operator.G[1:n, 1:n]
        H = phase.operator.H[1:n, 1:n]
        V = phase.operator.V[1:n, 1:n]
        Id   = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id  = Id[1:n, 1:n]
        Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
        Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        Interface_term = 1/(ρL) * sum(Interface_term)

        # New interface position
        res = Hₙ₊₁ - Hₙ - Interface_term
        new_xf = current_xf + α * res
        err = abs(new_xf - current_xf)
        println("Iteration $iter | xf = $new_xf | error = $err | res = $res")
        # Store residuals
        push!(residuals[1], err)

        # 3) Update geometry if not converged
        if (err <= tol) || (err <= reltol * abs(current_xf))
            push!(xf_log, new_xf)
            break
        end

        # Store tn+1 and tn
        tn1 = t + Δt
        tn  = t

        # 4) Rebuild domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
        body = (xx,tt, _=0)->(xx - (xf*(tn1 - tt)/Δt + new_xf*(tt - tn)/Δt))
        STmesh = SpaceTimeMesh(mesh, [tn, tn1], tag=mesh.tag)
        capacity = Capacity(body, STmesh)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, t, scheme)

        BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh)

        # 5) Update variables
        current_xf = new_xf
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

    # Time loop
    k=2
    while t < Tₑ
        t += Δt
        println("Time : $(t)")

        # 1) Reconstruct
        STmesh = SpaceTimeMesh(mesh, [Δt, 2Δt], tag=mesh.tag)
        #v_guess = (new_xf - xf)/Δt
        #body = (xx, tt, _=0) -> xx - ( new_xf - v_guess * (tt - t) )
        body = (xx,tt, _=0)->(xx - new_xf) 
        capacity = Capacity(body, STmesh)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, 0.0, scheme)

        BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh)

        err = Inf
        iter = 0
        current_xf = new_xf
        new_xf = current_xf
        xf = current_xf
        # Newton to compute the interface position xf1
        while (iter < max_iter) && (err > tol) && (err > reltol * abs(current_xf))
            iter += 1

            # 1) Solve the linear system
            solve_system!(s; method=method, kwargs...)
            Tᵢ = s.x

            # 2) Update volumes / compute new interface
            Vn_1 = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
            Vn   = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
            Hₙ   = sum(diag(Vn))
            Hₙ₊₁ = sum(diag(Vn_1))
            
            # Compute flux
            W! = phase.operator.Wꜝ[1:n, 1:n]
            G = phase.operator.G[1:n, 1:n]
            H = phase.operator.H[1:n, 1:n]
            V = phase.operator.V[1:n, 1:n]
            Id   = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id  = Id[1:n, 1:n]
            Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
            Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            Interface_term = 1/(ρL) * sum(Interface_term)

            # New interface position
            res = Hₙ₊₁ - Hₙ - Interface_term
            new_xf = current_xf + α * res
            err = abs(new_xf - current_xf)
            println("Iteration $iter | xf = $new_xf | error = $err | res = $res")
            # Store residuals
            push!(residuals[k], err)

            # 3) Update geometry if not converged
            if (err <= tol) || (err <= reltol * abs(current_xf))
                push!(xf_log, new_xf)
                break
            end

            # Store tn+1 and tn
            tn1 = t + Δt
            tn  = t

            # 4) Rebuild domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
            body = (xx,tt, _=0)->(xx - (xf*(tn1 - tt)/Δt + new_xf*(tt - tn)/Δt))
            STmesh = SpaceTimeMesh(mesh, [tn, tn1], tag=mesh.tag)
            capacity = Capacity(body, STmesh)
            operator = DiffusionOps(capacity)
            phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

            s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, t, scheme)

            BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh)

            # 5) Update variables
            current_xf = new_xf

        end

        if (err <= tol) || (err <= reltol * abs(current_xf))
            println("Converged after $iter iterations with xf = $new_xf, error = $err")
        else
            println("Reached max_iter = $max_iter with xf = $new_xf, error = $err")
        end

        push!(s.states, s.x)
        println("Time : $(t[1])")
        println("Max value : $(maximum(abs.(s.x)))")
        k += 1
    end

    return s, residuals, xf_log
end 