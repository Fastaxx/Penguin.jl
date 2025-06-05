function compute_volume_jacobian(mesh::Penguin.Mesh{2}, front::FrontTracker, epsilon::Float64=1e-6)
    # Extract mesh data
    x_faces = vcat(mesh.nodes[1][1], mesh.nodes[1][2:end])
    y_faces = vcat(mesh.nodes[2][1], mesh.nodes[2][2:end])
    
    # Call Julia function directly
    return compute_volume_jacobian(front, x_faces, y_faces, epsilon)
end

function StefanMono2D(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Stefan problem")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, ConvergenceHistory(), [])    
    s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, scheme)
    s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, scheme)
    
    BC_border_mono!(s.A, s.b, bc_b, mesh)
    
    s.x = Tᵢ
    return s
end

"""
Crée une fonction body pour représenter l'interface en SDF
- front: l'interface actuelle
- updated_front: l'interface mise à jour (optionnel)
- tₙ: temps au début du pas de temps
- Δt: pas de temps
"""
function create_interface_body_function(front::FrontTracker, updated_front::Union{FrontTracker,Nothing}=nothing, 
                                       tₙ::Float64=0.0, Δt::Float64=1.0)
    if updated_front === nothing
        # Pas d'interpolation, utiliser uniquement front actuel
        return function(x, y, t_local, _=0)
            return -sdf(front, x, y)
        end
    else
        # Interpolation entre deux interfaces
        return function(x, y, t_local, _=0)
            # Temps normalisé entre 0 et 1
            τ = (t_local - tₙ) / Δt
            
            # Interpolation linéaire entre les SDFs
            sdf1 = -sdf(front, x, y)
            sdf2 = -sdf(updated_front, x, y)
            return (1-τ) * sdf1 + τ * sdf2
        end
    end
end

function solve_StefanMono2D!(s::Solver, phase::Phase, front::FrontTracker, Δt::Float64, 
                            Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, 
                            bc::AbstractBoundary, ic::InterfaceConditions, 
                            mesh::Penguin.Mesh{2}, scheme::String; 
                            adaptive_timestep=true, method=Base.:\, 
                            Newton_params=(100, 1e-6, 1e-6, 1.0),
                            cfl_target=0.5, Δt_min=1e-4, Δt_max=1.0, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving Stefan problem with Front Tracking (Julia implementation):")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Unpack parameters
    max_iter, tol, reltol, α = Newton_params
    ρL = ic.flux.value
    
    # Initialize tracking variables
    t = Tₛ
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))
    
    # Mesh dimensions
    dims = phase.operator.size
    cap_index = length(dims)
    nx, ny, _ = dims
    n = nx*ny

    # Store initial state
    Tᵢ = s.x
    push!(s.states, s.x)
    xf_log[1] = get_markers(front)
    
    # Main time-stepping loop
    timestep = 1
    while t < Tₑ
        println("\n--- Timestep $timestep, t = $(round(t, digits=6)) ---")
        
        # Create time interval for space-time mesh
        tₙ = t
        t += Δt
        tₙ₊₁ = t
        time_interval = [tₙ, tₙ₊₁]
        
        # Get current markers and normals
        markers = get_markers(front)
        normals = compute_marker_normals(front, markers)
        
        # Calculate total number of markers (excluding duplicated closing point if closed)
        n_markers = length(markers) - (front.is_closed ? 1 : 0)
        
        # Initialize displacement vector and residual history
        displacements = zeros(n_markers)
        residual_norm_history = Float64[]
        
        # Create a temporary updated front for first iteration (identical to current front)
        updated_front = FrontTracker(markers, front.is_closed)
        phase_updated = phase  # Start with the same phase
        
        # Gauss-Newton iterations to solve least squares problem
        for iter in 1:max_iter
            # 0. Create space-time body function interpolating between fronts
            body_func = create_interface_body_function(front, updated_front, tₙ, Δt)
            
            # Update space-time mesh and capacity
            STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
            capacity = Capacity(body_func, STmesh; compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase_updated = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)
            
            # Rebuild the matrix system
            s.A = A_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, phase_updated.source, 
                                            bc, Tᵢ, Δt, tₙ, scheme)
            BC_border_mono!(s.A, s.b, bc_b, mesh)
            
            # 1. Solve temperature field with current interface position
            solve_system!(s; method=method, kwargs...)
            Tᵢ = s.x
            
            # Get capacity matrices
            V_matrices = phase_updated.capacity.A[cap_index]
            Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]
            Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]
            
            # 2. Calculate the interface flux
            W! = phase_updated.operator.Wꜝ[1:n, 1:n]
            G = phase_updated.operator.G[1:n, 1:n]
            H = phase_updated.operator.H[1:n, 1:n]
            Id = build_I_D(phase_updated.operator, phase_updated.Diffusion_coeff, phase_updated.capacity)
            Id = Id[1:n, 1:n]
            
            Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]

            interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            
            # Reshape to get flux per cell (with symmetric part)
            interface_flux_2d = reshape(interface_flux, (nx, ny)) 

            """
            # Plot temperature field for debugging
            fig = Figure()
            ax = Axis(fig[1, 1], title="Temperature Field", xlabel="x", ylabel="y")
            hm = heatmap!(ax, reshape(Tₒ, (nx, ny)), colormap=:viridis)
            Colorbar(fig[1, 2], hm, label="Temperature")
            display(fig)

            # Plot interface flux for debugging
            fig = Figure()
            ax = Axis(fig[1, 1], title="Interface Flux", xlabel="x", ylabel="y")
            hm = heatmap!(ax, interface_flux_2d, colormap=:viridis)
            Colorbar(fig[1, 2], hm, label="Interface Flux")
            display(fig)
            """

            # 3. Compute volume Jacobian for the current front
            volume_jacobian = compute_volume_jacobian(mesh, updated_front)
            
            # Build least squares system
            row_indices = Int[]
            col_indices = Int[]
            values = Float64[]
            cells_idx = []
            
            # Precompute affected cells and their indices for residual vector
            for i in 1:nx, j in 1:ny
                if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                    push!(cells_idx, (i, j))
                end
            end
            
            # Number of equations (cells) in our system
            m = length(cells_idx)
            
            if timestep == 1
                println("Iteration $iter: $m equations for $n_markers unknowns")
            else
                println("  Iteration $iter: $m equations for $n_markers unknowns")
            end
            
            # Build the Jacobian matrix for volume changes
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                for (marker_idx, jac_value) in volume_jacobian[(i,j)]
                    if 0 <= marker_idx < n_markers
                        push!(row_indices, eq_idx)
                        push!(col_indices, marker_idx + 1)  # 1-based indexing
                        push!(values, ρL * jac_value)
                    end
                end
            end
            
            # Create Jacobian matrix J for the system
            J = sparse(row_indices, col_indices, values, m, n_markers)
            
            # Calculate current residual vector F
            F = zeros(m)
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                # Get the desired volume change based on interface flux
                volume_change = Vₙ₊₁_matrix[i,j] - Vₙ_matrix[i,j]
                
                # F_i = ρL * volume_change - interface_flux
                F[eq_idx] = ρL * volume_change - interface_flux_2d[i,j]
            end
            
            # 4. Implement the Gauss-Newton formula: X^{n+1} = X^n - (J^T J)^{-1} J^T F
            JTJ = J' * J
            
            # Diagnose the system
            used_columns = unique(col_indices)
            
            if timestep == 1
                println("Matrix info: size(J)=$(size(J)), markers used: $(length(used_columns))/$n_markers")
            else
                println("    Matrix info: size(J)=$(size(J)), markers used: $(length(used_columns))/$n_markers")
            end
            
            # Check if JTJ is singular and handle appropriately
            newton_step = zeros(n_markers)
            try
                # Add regularization for numerical stability
                reg_JTJ = JTJ + 1e-1 * I(size(JTJ, 1))
                newton_step = reg_JTJ \ (J' * F)
            catch e
                if isa(e, SingularException)
                    println("JTJ is singular, using SVD solver")
                    
                    # Use SVD-based pseudoinverse for robust solving
                    F_svd = svd(Matrix(JTJ))
                    tol_svd = eps(Float64) * max(size(JTJ)...) * maximum(F_svd.S)
                    S_inv = [s > tol_svd ? 1/s : 0.0 for s in F_svd.S]
                    
                    # Compute pseudoinverse solution
                    JTF = J' * F
                    newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
                else
                    rethrow(e)
                end
            end
            
            # Apply the step with relaxation
            displacements -= α * newton_step
            
            # For closed curves, match first and last displacement to ensure continuity
            if front.is_closed
                displacements[1] = displacements[end]
            end

            # Smooth the displacements for stability
            smooth_factor = timestep == 1 ? 1.0 : 0.5  # First timestep uses stronger smoothing
            smooth_displacements!(displacements, markers, front.is_closed, smooth_factor, 4)

            # Print maximum displacement for diagnostics
            max_disp = maximum(abs.(displacements))
            if timestep == 1
                println("Maximum displacement (after smoothing): $max_disp")
            else
                println("    Maximum displacement (after smoothing): $max_disp")
            end

            # Optional: Limit maximum displacement if needed
            max_allowed = min(mesh.nodes[1][2] - mesh.nodes[1][1], mesh.nodes[2][2] - mesh.nodes[2][1]) * 0.2
            if max_disp > max_allowed
                scale = max_allowed / max_disp
                displacements .*= scale
                println("    Scaling displacements by $scale to limit maximum displacement")
            end
            
            # Calculate residual norm for convergence check
            residual_norm = norm(F)
            push!(residual_norm_history, residual_norm)
            
            # Report progress
            if timestep == 1
                println("Iteration $iter | Residual = $residual_norm")
            else
                println("    Iteration $iter | Residual = $residual_norm | Max disp = $max_disp | Min disp = $(minimum(abs.(displacements)))")
            end

            # Check convergence
            if residual_norm < tol #|| (iter > 1 && abs(residual_norm_history[end] - residual_norm_history[end-1]) < reltol)
                println(timestep == 1 ? 
                    "Converged after $iter iterations with residual $residual_norm" :
                    "    Converged after $iter iterations with residual $residual_norm")
                xf_log[timestep+1] = copy(markers)  # Store final marker positions
                break
            end
            
            # 5. Update marker positions
            new_markers = copy(markers)
            for i in 1:n_markers
                normal = normals[i]
                new_markers[i] = (
                    markers[i][1] + displacements[i] * normal[1],
                    markers[i][2] + displacements[i] * normal[2]
                )
            end
            
            # print mean radius
            if front.is_closed
                # Calculer le centre approximatif (centre de masse des marqueurs)
                center_x = sum(m[1] for m in new_markers) / length(new_markers)
                center_y = sum(m[2] for m in new_markers) / length(new_markers)
                
                # Calculer le rayon moyen
                mean_radius = mean([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in new_markers])
                
                # Afficher le rayon moyen
                if timestep == 1
                    println("Mean radius: $(round(mean_radius, digits=6))")
                else
                    println("    Mean radius: $(round(mean_radius, digits=6))")
                end
                
                # Calculer aussi l'écart-type du rayon (pour vérifier la régularité)
                std_radius = std([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in new_markers])
                if std_radius / mean_radius > 0.05  # Plus de 5% de variation
                    println("    ⚠️ Warning: Interface irregularity detected ($(round(100*std_radius/mean_radius, digits=2))% variation)")
                end
            end
            
            # If interface is closed, update the duplicated last marker
            if front.is_closed
                new_markers[end] = new_markers[1]
            end
            
            # Create updated front tracking object for next iteration
            updated_front = FrontTracker(new_markers, front.is_closed)
        end
        
        # Store residuals from this time step
        residuals[timestep] = residual_norm_history
        
        # Update front with final marker positions
        """
        new_markers = copy(markers)
        for i in 1:n_markers
            normal = normals[i]
            new_markers[i] = (
                markers[i][1] + displacements[i] * normal[1],
                markers[i][2] + displacements[i] * normal[2]
            )
        end
        """

        new_markers = get_markers(updated_front)  # Use final updated front markers
        #set_markers!(front, new_markers)
        
        # If interface is closed, ensure first and last markers match
        if front.is_closed
            new_markers[end] = new_markers[1]
        end
        
        # Update front with new markers
        set_markers!(front, new_markers)
        
        # Store updated interface position
        xf_log[timestep+1] = new_markers
        
        # Store solution
        push!(s.states, s.x)
        phase = phase_updated  # Update phase for next timestep
        
        # Print status
        if timestep == 1
            println("Time: $(round(t, digits=6))")
            println("Max temperature: $(maximum(abs.(s.x[1:n])))")
        else
            println("  Time: $(round(t, digits=6)), Max temperature: $(maximum(abs.(s.x[1:n])))")
        end
        
        # Prepare for next time step
        Tᵢ = s.x
        
        # Adapt timestep if needed based on CFL condition
        if adaptive_timestep && timestep > 1
            # Calculate interface velocities to estimate CFL
            # We've already computed the interface flux earlier
            interface_velocities = abs.(interface_flux) ./ ρL
            max_velocity = maximum(interface_velocities)
            
            if max_velocity > 0
                # Calculate cell size
                Δx = mesh.nodes[1][2] - mesh.nodes[1][1]
                Δy = mesh.nodes[2][2] - mesh.nodes[2][1]
                min_cell_size = min(Δx, Δy)
                
                # CFL-based timestep
                cfl = max_velocity * Δt / min_cell_size
                
                time_left = Tₑ - t
                Δt_max_current = min(Δt_max, time_left)
                
                if cfl > cfl_target
                    # Reduce timestep if CFL exceeds target
                    Δt = cfl_target * min_cell_size / max_velocity * 0.9
                    Δt = max(Δt, Δt_min)
                else
                    # Grow timestep if CFL is small
                    Δt = min(Δt * 1.1, Δt_max_current)
                end
                
                println("  Adaptive timestep: Δt = $(round(Δt, digits=6))")
            end
        end
        
        push!(timestep_history, (t, Δt))
        timestep += 1
    end
    
    return s, residuals, xf_log, timestep_history
end

"""
Smooth marker displacements using weighted averaging of neighbors
This helps maintain interface regularity and stability

Parameters:
- displacements: Vector of displacement values to smooth
- markers: Vector of marker positions (Tuple{Float64,Float64})
- is_closed: Boolean indicating if the interface is a closed curve
- smoothing_factor: Weight given to neighbor values (0.0-1.0)
- window_size: Number of neighbors to consider on each side
"""
function smooth_displacements!(displacements::Vector{Float64}, 
                              markers::Vector{Tuple{Float64,Float64}}, 
                              is_closed::Bool=true,
                              smoothing_factor::Float64=0.5,
                              window_size::Int=2)
    n = length(displacements)
    if n <= 1
        return displacements  # Nothing to smooth with single marker
    end
    
    # Create a copy of original displacements
    original_displacements = copy(displacements)
    
    for i in 1:n
        # Calculate weighted sum of neighbors
        neighbor_sum = 0.0
        weight_sum = 0.0
        
        for j in -window_size:window_size
            if j == 0
                continue  # Skip the marker itself
            end
            
            # Handle wrapping for closed curves
            idx = i + j
            if is_closed
                # Apply modulo to wrap around for closed curves
                idx = mod1(idx, n)
            else
                # Skip out of bounds indices for open curves
                if idx < 1 || idx > n
                    continue
                end
            end
            
            # Calculate weight based on distance (closer markers have higher weight)
            distance = sqrt((markers[i][1] - markers[idx][1])^2 + 
                           (markers[i][2] - markers[idx][2])^2)
            weight = 1.0 / (distance + 1e-10)  # Avoid division by zero
            
            # Add weighted contribution
            neighbor_sum += weight * original_displacements[idx]
            weight_sum += weight
        end
        
        # Calculate weighted average
        if weight_sum > 0
            neighbor_avg = neighbor_sum / weight_sum
            
            # Apply smoothing with blend factor
            displacements[i] = (1.0 - smoothing_factor) * original_displacements[i] + 
                               smoothing_factor * neighbor_avg
        end
    end
    
    return displacements
end