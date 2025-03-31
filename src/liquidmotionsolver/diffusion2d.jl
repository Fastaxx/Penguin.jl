"""
    adapt_timestep(velocity_field, mesh, cfl_target, Δt_current, Δt_min, Δt_max, growth_factor=1.1)

Adapte le pas de temps en fonction du critère CFL basé sur la vitesse de l'interface.

Paramètres:
- `velocity_field`: Vitesses à l'interface [m/s]
- `mesh`: Maillage de calcul
- `cfl_target`: Nombre CFL cible (typiquement entre 0.1 et 1.0)
- `Δt_current`: Pas de temps actuel [s]
- `Δt_min`: Pas de temps minimum autorisé [s]
- `Δt_max`: Pas de temps maximum autorisé [s]
- `growth_factor`: Facteur limitant l'augmentation du pas de temps (par défaut 1.1)

Retourne:
- `Δt_new`: Nouveau pas de temps [s]
- `cfl_actual`: Nombre CFL qui sera obtenu avec le nouveau pas de temps
"""
function adapt_timestep(velocity_field, mesh, cfl_target, Δt_current, Δt_min, Δt_max, growth_factor=0.9)
    # 1. Calcul de la vitesse maximale de l'interface
    v_max = maximum(abs.(velocity_field))
    
    # Éviter la division par zéro si l'interface est statique
    if v_max < 1e-10
        return min(Δt_current * growth_factor, Δt_max), 0.0
    end
    
    # 2. Calcul de la taille de maille minimale dans chaque direction
    Δx = minimum(diff(mesh.nodes[1]))
    Δy = minimum(diff(mesh.nodes[2]))
    Δh_min = min(Δx, Δy)
    
    # 3. Calcul du pas de temps optimal pour le CFL cible
    Δt_optimal = cfl_target * Δh_min / v_max
    
    # 4. Limitation de la croissance du pas de temps pour éviter les oscillations
    Δt_limited = min(Δt_optimal, Δt_current * growth_factor)
    
    # 5. Application des contraintes min/max
    Δt_new = clamp(Δt_limited, Δt_min, Δt_max)
    
    # 6. Calcul du CFL effectif avec le nouveau pas de temps
    cfl_actual = v_max * Δt_new / Δh_min
    
    return Δt_new, cfl_actual
end


# Full Moving 2D - Diffusion - Unsteady - Monophasic
function MovingLiquidDiffusionUnsteadyMono2D(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
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

function solve_MovingLiquidDiffusionUnsteadyMono2D!(s::Solver, phase::Phase, Interface_position, Hₙ⁰, sₙ, Δt::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh, scheme::String; interpo="linear", Newton_params=(1000, 1e-10, 1e-10, 1.0), cfl_target=0.5,
    Δt_min=1e-4,
    Δt_max=1.0,
    adaptive_timestep=true, method=IterativeSolvers.gmres, kwargs...)
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
    nt = Int(round(Tₑ/Δt))
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = []
    reconstruct = []
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))

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
        #println("Interface term: ", Interface_term)

        # 4) Update the height function
        res = Hₙ₊₁ - Hₙ - Interface_term
        #println("res: ", res)
        new_Hₙ = current_Hₙ .+ α .* res            # Elementwise update for each column
        #println("new_Hₙ: ", new_Hₙ)
        err = abs.(new_Hₙ[5] .- current_Hₙ[5])
        err_rel = err/maximum(abs.(current_xf[5]))
        println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

        # Store residuals (if desired, you could store the full vector or simply the norm)
        if !haskey(residuals, 1)
            residuals[1] = Float64[]
        end
        push!(residuals[1], err)

        # 5) Update geometry if not converged
        if (err <= tol) || (err_rel <= reltol)
            push!(xf_log, new_xf)
            break
        end

        # Store tn+1 and tn
        tₙ₊₁ = t + Δt
        tₙ  = t

        # 6) Compute the new interface position table
        Δy = mesh.nodes[2][2] - mesh.nodes[2][1]
        new_xf = mesh.nodes[1][1] .+ new_Hₙ./Δy
        new_xf = new_xf[1:end-1]
        # 7) Construct a interpolation function for the new interface position : sn and sn+1
        centroids = mesh.centers[1]
        if interpo == "linear"
            #sₙ₊₁ = height_interpolation_linear(centroids, new_xf)
            #sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
            sₙ₊₁ = lin_interpol(centroids, new_xf)
        elseif interpo == "quad"
            #sₙ₊₁ = height_interpolation_quadratic(centroids, new_xf)
            #sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
            sₙ₊₁ = quad_interpol(centroids, new_xf)
        elseif interpo == "cubic"
            #sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
            sₙ₊₁ = cubic_interpol(centroids, new_xf)
        else
            println("Interpolation method not supported")
        end

        # 8) Rebuild the domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
        body = (xx, yy, tt, _=0) -> begin
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
        STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
        capacity = Capacity(body, STmesh; compute_centroids=false)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        # 9) Rebuild the matrix A and the vector b
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, t, scheme)

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
        # Calcul de la vitesse d'interface à partir des flux
        W! = phase.operator.Wꜝ[1:n, 1:n]  # n = nx*ny (full 2D system)
        G  = phase.operator.G[1:n, 1:n]
        H  = phase.operator.H[1:n, 1:n]
        V  = phase.operator.V[1:n, 1:n]
        Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id = Id[1:n, 1:n]
        Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
        Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        velocity_field = 1/(ρL) * abs.(Interface_term)
        
        # Adaptation du pas de temps si demandée
        if adaptive_timestep
            # Limiter le temps restant pour ne pas dépasser Tₑ
            time_left = Tₑ - t
            Δt_max_current = min(Δt_max, time_left)
            
            Δt, cfl = adapt_timestep(velocity_field, mesh, cfl_target, Δt, Δt_min, Δt_max_current)
            push!(timestep_history, (Δt, cfl))
            println("Adaptive timestep: Δt = $(round(Δt, digits=6)), CFL = $(round(cfl, digits=3))")
        end

        # Update the time
        t+=Δt
        tₙ = t
        tₙ₊₁ = t + Δt
        println("Time : $(round(t, digits=6))")

        # 1) Construct an interpolation function for the interface position
        centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        if interpo == "linear"
            #sₙ = linear_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
            #sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
            sₙ = lin_interpol(centroids, current_xf)
            sₙ₊₁ = lin_interpol(centroids, new_xf)
        elseif interpo == "quad"
            #sₙ = extrapolate(scale(interpolate(current_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
            #sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
            sₙ = quad_interpol(centroids, current_xf)
            sₙ₊₁ = quad_interpol(centroids, new_xf)
        elseif interpo == "cubic"
            #sₙ = cubic_spline_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
            #sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
            sₙ = cubic_interpol(centroids, current_xf)
            sₙ₊₁ = cubic_interpol(centroids, new_xf)
        else
            println("Interpolation method not supported")
        end

        # 1) Reconstruct
        STmesh = SpaceTimeMesh(mesh, [Δt, 2Δt], tag=mesh.tag)
        body = (xx, yy, tt, _=0) -> begin
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
        capacity = Capacity(body, STmesh; compute_centroids=false)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, 0.0, scheme)

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
            #println("Interface term: ", Interface_term)

            # 4) Update the height function
            res = Hₙ₊₁ - Hₙ - Interface_term
            #println("res: ", res)
            new_Hₙ = current_Hₙ .+ α .* res            # Elementwise update for each column
            #println("new_Hₙ: ", new_Hₙ)
            err = abs.(new_Hₙ[5] .- current_Hₙ[5])
            err_rel = err/maximum(abs.(current_xf[5]))
            println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

            # Store residuals (if desired, you could store the full vector or simply the norm)
            if !haskey(residuals, k)
                residuals[k] = Float64[]
            end
            push!(residuals[k], err)

            # 5) Update geometry if not converged
            if (err <= tol) || (err_rel <= reltol)
                push!(xf_log, new_xf)
                break
            end

            # Store tn+1 and tn
            tₙ₊₁ = t + Δt
            tₙ  = t

            # 6) Compute the new interface position table
            Δy = mesh.nodes[2][2] - mesh.nodes[2][1]
            new_xf = mesh.nodes[1][1] .+ new_Hₙ./Δy
            new_xf[end] = new_xf[1]

            # 7) Construct a interpolation function for the new interface position :
            centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
            if interpo == "linear"
                #sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
                sₙ₊₁ = lin_interpol(centroids, new_xf)
            elseif interpo == "quad"
                #sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
                sₙ₊₁ = quad_interpol(centroids, new_xf)
            elseif interpo == "cubic"
                #sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
                sₙ₊₁ = cubic_interpol(centroids, new_xf)
            else
                println("Interpolation method not supported")
            end

            # 8) Rebuild the domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
            body = (xx, yy, tt, _=0) -> begin
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
            STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
            capacity = Capacity(body, STmesh; compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

            # 9) Rebuild the matrix A and the vector b
            s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, 0.0, scheme)

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

        # Afficher les informations du pas de temps
        if adaptive_timestep
            println("Time step info: Δt = $(round(Δt, digits=6)), CFL = $(round(timestep_history[end][2], digits=3))")
        end

        push!(s.states, s.x)
        println("Time : $(t[1])")
        println("Max value : $(maximum(abs.(s.x)))")
        k+=1
    end
return s, residuals, xf_log, reconstruct, timestep_history
end 
