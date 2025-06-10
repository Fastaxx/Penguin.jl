function compute_volume_jacobian(mesh::Penguin.Mesh{2}, front::FrontTracker, epsilon::Float64=1e-6)
    # Extract mesh data
    x_faces = vcat(mesh.nodes[1][1], mesh.nodes[1][2:end])
    y_faces = vcat(mesh.nodes[2][1], mesh.nodes[2][2:end])
    
    # Call Julia function directly
    return compute_volume_jacobian(front, x_faces, y_faces, epsilon)
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

function StefanMono2D(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Stefan problem")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, ConvergenceHistory(), [])    
    if scheme == "CN"
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "CN")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "CN")
    else # BE
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "BE")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "BE")
    end
    
    BC_border_mono!(s.A, s.b, bc_b, mesh)
    
    s.x = Tᵢ
    return s
end

function solve_StefanMono2D!(s::Solver, phase::Phase, front::FrontTracker, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh::Penguin.Mesh{2}, scheme::String; 
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
    max_iter = Newton_params[1]
    tol = Newton_params[2]
    reltol = Newton_params[3]
    α = Newton_params[4]
    
    # Extract interface flux parameter
    ρL = ic.flux.value
    
    # Initialize tracking variables
    t = Tₛ
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))
    position_increments = Dict{Int, Vector{Float64}}()  # Nouveau dictionnaire pour les incréments de position

    
    # Determine how many dimensions
    dims = phase.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Create the 1D or 2D indices
    nx, ny, _ = dims
    n = nx * ny  # Total number of nodes in the mesh
    
    # Store initial state
    Tᵢ = s.x
    push!(s.states, s.x)
    
    # Get initial interface markers
    markers = get_markers(front)
    
    # Store initial interface position
    xf_log[1] = markers
    
    # Boucle principale pour tous les pas de temps
    timestep = 1
    phase_2d = phase  # Initialize phase_2d for later use
    while t < Tₑ
        # Affichage du pas de temps actuel
        if timestep == 1
            println("\nFirst time step: t = $(round(t, digits=6))")
        else
            println("\nTime step $(timestep), t = $(round(t, digits=6))")
        end

        # Get current markers and calculate normals
        markers = get_markers(front)
        normals = compute_marker_normals(front, markers)
        
        # Update time for this step
        t += Δt
        tₙ = t - Δt
        tₙ₊₁ = t
        time_interval = [tₙ, tₙ₊₁]
        
        # Calculate total number of markers (excluding duplicated closing point if interface is closed)
        n_markers = length(markers) - (front.is_closed ? 1 : 0)
        
        # Initialize displacement vector and residual vector
        displacements = zeros(n_markers)
        residual_norm_history = Float64[]
        position_increment_history = Float64[]  # Nouveau vecteur pour les incréments de position

        # Gauss-Newton iterations
        for iter in 1:max_iter
            # 1. Solve temperature field with current interface position
            solve_system!(s; method=method, kwargs...)
            Tᵢ = s.x
            
            # Get capacity matrices
            V_matrices = phase.capacity.A[cap_index]
            Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]
            Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]
            
            # 2. Calculate the interface flux
            W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
            G = phase.operator.G[1:end÷2, 1:end÷2]
            H = phase.operator.H[1:end÷2, 1:end÷2]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:end÷2, 1:end÷2]  # Adjust for 2D case
            
            Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
            interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            
            # Reshape to get flux per cell - IMPORTANT: use the same reshape consistently
            interface_flux_2d = reshape(interface_flux, (nx, ny)) #+ reshape(interface_flux, (nx, ny))'
            
            # Debug visualizations
            if timestep <= 2 || mod(timestep, 5) == 0  # Limit visualizations
                # Plot temperature field
                fig = Figure()
                ax = Axis(fig[1, 1], title="Temperature Field", xlabel="x", ylabel="y", aspect=DataAspect())
                hm = heatmap!(ax, reshape(Tₒ, (nx, ny)), colormap=:viridis)
                Colorbar(fig[1, 2], hm, label="Temperature")
                display(fig)
                
                # Plot interface flux
                fig = Figure()
                ax = Axis(fig[1, 1], title="Interface Flux", xlabel="x", ylabel="y", aspect=DataAspect())
                hm = heatmap!(ax, interface_flux_2d, colormap=:viridis)
                Colorbar(fig[1, 2], hm, label="Interface Flux")
                display(fig)
                
                if iter > 1  # Only when we have updated markers
                    # Calculate updated marker positions
                    updated_markers = Vector{Tuple{Float64, Float64}}(undef, length(markers))
                    
                    # Use eachindex instead of 1:length()
                    for (i, marker) in enumerate(markers)
                        if i <= length(displacements) && i <= length(normals)  # Check bounds for both arrays
                            updated_markers[i] = (
                                marker[1] + displacements[i] * normals[i][1],
                                marker[2] + displacements[i] * normals[i][2]
                            )
                        else
                            updated_markers[i] = marker  # Keep the same if no normal or displacement
                        end

                        if front.is_closed && length(updated_markers) > 1
                            # Ensure the last marker matches the first for closed curves
                            updated_markers[end] = updated_markers[1]
                        end
                    end
                    
                    # Create the comparison plot
                    fig = Figure()
                    ax = Axis(fig[1, 1], title="Interface Evolution (Iteration $iter)", 
                              xlabel="x", ylabel="y", aspect=DataAspect())
                    
                    # Plot original markers
                    orig_x = [m[1] for m in markers]
                    orig_y = [m[2] for m in markers]
                    lines!(ax, orig_x, orig_y, color=:blue, linewidth=1.5, 
                           label="Original", linestyle=:dash)
                    scatter!(ax, orig_x, orig_y, color=:blue, markersize=3)
                    
                    # Plot updated markers
                    new_x = [m[1] for m in updated_markers]
                    new_y = [m[2] for m in updated_markers]
                    lines!(ax, new_x, new_y, color=:red, linewidth=1.5, 
                           label="Updated")
                    scatter!(ax, new_x, new_y, color=:red, markersize=3)
                    
                    # Add legend
                    axislegend(ax, position=:rt)
                    
                    display(fig)
                end
            end
            
            # Compute volume Jacobian for the mesh
            volume_jacobian = compute_volume_jacobian(mesh, front, 1e-6)  # Use smaller epsilon
            
            # 3. Build least squares system
            row_indices = Int[]
            col_indices = Int[]
            values = Float64[]
            cells_idx = []
            
            # Precompute affected cells and their indices for residual vector
            for i in 1:nx
                for j in 1:ny
                    if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                        push!(cells_idx, (i, j))
                    end
                end
            end
            
            # Number of equations (cells) in our system
            m = length(cells_idx)
            
            # Now build the Jacobian matrix for volume changes
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                # Handle each marker affecting this cell
                for (marker_idx, jac_value) in volume_jacobian[(i,j)]
                    if 0 <= marker_idx < n_markers
                        push!(row_indices, eq_idx)
                        push!(col_indices, marker_idx + 1)  # 1-based indexing
                        # Volume Jacobian is multiplied by ρL to match the Stefan condition
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

            
            # Visualiser le résidual F sur la grille
            fig_residual = Figure()
            ax_residual = Axis(fig_residual[1, 1], 
                              title="Residual F (Iteration $iter)", 
                              xlabel="x", ylabel="y",
                              aspect=DataAspect())
            
            # Créer une matrice 2D pour représenter le résidual sur la grille
            residual_2d = zeros(nx, ny)
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                residual_2d[i, j] = F[eq_idx]
            end
            
            # Tracer le heatmap du résidual
            hm_residual = heatmap!(ax_residual, residual_2d, colormap=:balance)
            Colorbar(fig_residual[1, 2], hm_residual, label="Residual value")
            
            # Ajouter les contours de l'interface pour référence
            markers_x = [m[1] for m in markers]
            markers_y = [m[2] for m in markers]
            lines!(ax_residual, markers_x, markers_y, color=:black, linewidth=1.5)
            
            display(fig_residual)
            
            # 4. Implement the Gauss-Newton formula: X^{n+1} = X^n - (J^T J)^{-1} J^T F
            JTJ = J' * J
            
            # Diagnose the system
            used_columns = unique(col_indices)
            println("Matrix info: size(J)=$(size(J)), n_markers=$n_markers")
            println("Used marker indices: $(length(used_columns)) of $n_markers")
            
            # Add regularization using JTJ diagonal (Levenberg-Marquardt style)
            reg_param = 1e-6
            diag_JTJ = diag(JTJ)  # Extract the diagonal of JTJ
            
            # Ensure diagonal elements aren't too small (protect against near-zero entries)
            min_diag = 1e-10 * maximum(diag_JTJ)
            for i in 1:length(diag_JTJ)
                if diag_JTJ[i] < min_diag
                    diag_JTJ[i] = min_diag
                end
            end
            
            # Apply Levenberg-Marquardt style regularization
            reg_JTJ = JTJ + reg_param * Diagonal(diag_JTJ)
            
            # Solve the system using robust method
            newton_step = zeros(n_markers)
            try
                newton_step = reg_JTJ \ (J' * F)
            catch e
                println("Matrix solver failed, using SVD as backup")
                # SVD-based pseudoinverse for robust solving
                F_svd = svd(Matrix(reg_JTJ))
                svd_tol = eps(Float64) * max(size(reg_JTJ)...) * maximum(F_svd.S)
                S_inv = [s > svd_tol ? 1/s : 0.0 for s in F_svd.S]
                
                # Compute pseudoinverse solution
                JTF = J' * F
                newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
            end
            # Calculer la norme de l'incrément de position avant de l'appliquer
            position_increment_norm = α * norm(newton_step)
            push!(position_increment_history, position_increment_norm)
            
            # Apply the step with adjustment factor
            displacements -= α * newton_step
            
            # For closed curves, match first and last displacement to ensure continuity
            if front.is_closed
                displacements[end] = displacements[1]
            end
            
            # Smooth the displacements for stability
            smooth_factor = 0.5
            window_size = 2
            smooth_displacements!(displacements, markers, front.is_closed, smooth_factor, window_size)
            
            # Print maximum displacement for diagnostics
            max_disp = maximum(abs.(displacements))
            println("Maximum displacement (after smoothing): $max_disp")
            
            # Optional: Limit maximum displacement if needed
            max_allowed = min(mesh.nodes[1][2] - mesh.nodes[1][1], mesh.nodes[2][2] - mesh.nodes[2][1]) * 0.1
            if max_disp > max_allowed
                scale = max_allowed / max_disp
                displacements .*= scale
                println("Scaling displacements by $scale to limit maximum displacement")
            end
            
            # Calculate residual norm for convergence check
            residual_norm = norm(F)
            push!(residual_norm_history, residual_norm)
            
            # Report progress
            println("Iteration $iter | Residual = $residual_norm")
            
            # Check convergence
            if residual_norm < tol || (iter > 1 && abs(residual_norm_history[end] - residual_norm_history[end-1]) < reltol)
                println("Converged after $iter iterations with residual $residual_norm and position increment $position_increment_norm")
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
            
            # If interface is closed, update the duplicated last marker
            if front.is_closed
                new_markers[end] = new_markers[1]
            end
            
            # Print mean radius for diagnostic
            if front.is_closed
                # Calculer le centre approximatif
                center_x = sum(m[1] for m in new_markers) / length(new_markers)
                center_y = sum(m[2] for m in new_markers) / length(new_markers)
                
                # Calculer le rayon moyen
                mean_radius = mean([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in new_markers])
                
                # Afficher le rayon moyen
                println("Mean radius: $(round(mean_radius, digits=6))")
                
                # Vérifier la régularité
                std_radius = std([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in new_markers])
                if std_radius / mean_radius > 0.05
                    println("⚠️ Warning: Interface irregularity detected ($(round(100*std_radius/mean_radius, digits=2))% variation)")
                end
            end
            
            # 6. Create updated front tracking object
            updated_front = FrontTracker(new_markers, front.is_closed)
            
            # 7. Create space-time level set for capacity calculation
            function body(x, y, t_local, _=0)
                # Normalized time in [0,1]
                τ = (t_local - tₙ) / Δt

                # Linear interpolation between SDFs
                sdf1 = -sdf(front, x, y)
                sdf2 = -sdf(updated_front, x, y)
                return (1-τ) * sdf1 + τ * sdf2
            end
            
            # 8. Update space-time mesh and capacity
            STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
            capacity = Capacity(body, STmesh; compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase_updated = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)
            
            # 9. Rebuild the matrix system
            s.A = A_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, phase_updated.source, 
                                            bc, Tᵢ, Δt, tₙ, scheme)
            
            BC_border_mono!(s.A, s.b, bc_b, mesh)
            
            # 10. Update phase for next iteration
            phase = phase_updated

            body_2d(x,y,_=0) = body(x, y, tₙ₊₁)
            capacity_2d = Capacity(body_2d, mesh; compute_centroids=false)
            operator_2d = DiffusionOps(capacity_2d)
            phase_2d = Phase(capacity_2d, operator_2d, phase.source, phase.Diffusion_coeff)
        end
        
        # Store residuals from this time step
        residuals[timestep] = residual_norm_history
        position_increments[timestep] = position_increment_history  # Stocker l'historique des incréments de position

        
        # Update front with new marker positions
        new_markers = copy(markers)
        for i in 1:n_markers
            normal = normals[i]
            new_markers[i] = (
                markers[i][1] + displacements[i] * normal[1],
                markers[i][2] + displacements[i] * normal[2]
            )
        end
        
        # If interface is closed, update the duplicated last marker
        if front.is_closed
            new_markers[end] = new_markers[1]
        end
        
        # Update front with new markers
        set_markers!(front, new_markers)
        
        # Store updated interface position
        xf_log[timestep+1] = new_markers
        
        # Store solution
        push!(s.states, s.x)
        
        println("Time: $(round(t, digits=6))")
        println("Max temperature: $(maximum(abs.(s.x)))")
        
        # Increment timestep counter
        timestep += 1
    end
    
    return s, residuals, xf_log, timestep_history, phase_2d, position_increments
end















function solve_StefanMono2Dunclosed!(s::Solver, phase::Phase, front::FrontTracker, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh::Penguin.Mesh{2}, scheme::String; 
                            adaptive_timestep=true, method=Base.:\, 
                            Newton_params=(100, 1e-6, 1e-6, 1.0),
                            cfl_target=0.5, Δt_min=1e-4, Δt_max=1.0, 
                            fixed_endpoints=true, endpoint_smoothing=0.1, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving Stefan problem with Front Tracking (unclosed interface):")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")
    println("- Non-closed interface handling enabled")

    # Unpack parameters
    max_iter = Newton_params[1]
    tol = Newton_params[2]
    reltol = Newton_params[3]
    α = Newton_params[4]
    
    # Extract interface flux parameter
    ρL = ic.flux.value
    
    # Initialize tracking variables
    t = Tₛ
    residuals = Dict{Int, Vector{Float64}}()
    interface_position_changes = Dict{Int, Vector{Float64}}()
    xf_log = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))
    
    # Determine how many dimensions
    dims = phase.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Create the 1D or 2D indices
    nx, ny, _ = dims
    n = nx * ny  # Total number of nodes in the mesh
    
    # Store initial state
    Tᵢ = s.x
    push!(s.states, s.x)
    
    # Get initial interface markers
    markers = get_markers(front)
    
    # Verify that the interface is indeed non-closed
    is_closed = false
    println("Interface type: $(is_closed ? "Closed" : "Unclosed") with $(length(markers)) markers")
    
    # Store initial interface position
    xf_log[1] = markers
    
    # Main time stepping loop
    timestep = 1
    phase_2d = phase  # Initialize phase_2d for later use
    while t < Tₑ
        # Display current timestep
        if timestep == 1
            println("\nFirst time step: t = $(round(t, digits=6))")
        else
            println("\nTime step $(timestep), t = $(round(t, digits=6))")
        end

        # Get current markers and calculate normals
        markers = get_markers(front)
        normals = compute_marker_normals(front, markers)  # False indicates non-closed
        
        # Update time for this step
        t += Δt
        tₙ = t - Δt
        tₙ₊₁ = t
        time_interval = [tₙ, tₙ₊₁]
        
        # For unclosed interfaces, count all markers
        n_markers = length(markers)
        
        # Initialize displacement vector and residual vector
        displacements = zeros(n_markers)
        residual_norm_history = Float64[]
        position_change_history = Float64[]
        
        # Store previous marker positions to calculate position changes
        previous_markers = copy(markers)

        # Gauss-Newton iterations
        for iter in 1:max_iter
            # 1. Solve temperature field with current interface position
            solve_system!(s; method=method, kwargs...)
            Tᵢ = s.x
            
            # Get capacity matrices
            V_matrices = phase.capacity.A[cap_index]
            Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]
            Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]
            
            # 2. Calculate the interface flux
            W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
            G = phase.operator.G[1:end÷2, 1:end÷2]
            H = phase.operator.H[1:end÷2, 1:end÷2]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:end÷2, 1:end÷2]  # Adjust for 2D case
            
            Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
            interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            
            # Reshape to get flux per cell
            interface_flux_2d = reshape(interface_flux, (nx, ny))
            
            # Debug visualizations (reduced frequency)
            if timestep <= 2 || mod(timestep, 10) == 0  # Reduced visualization frequency
                # Plot temperature field
                fig = Figure()
                ax = Axis(fig[1, 1], title="Temperature Field", xlabel="x", ylabel="y", aspect=DataAspect())
                hm = heatmap!(ax, reshape(Tₒ, (nx, ny)), colormap=:viridis)
                Colorbar(fig[1, 2], hm, label="Temperature")
                display(fig)
                
                # Plot interface flux
                fig = Figure()
                ax = Axis(fig[1, 1], title="Interface Flux", xlabel="x", ylabel="y", aspect=DataAspect())
                hm = heatmap!(ax, interface_flux_2d, colormap=:viridis)
                Colorbar(fig[1, 2], hm, label="Interface Flux")
                display(fig)
            end
            
            # Compute volume Jacobian for the mesh
            volume_jacobian = compute_volume_jacobian(mesh, front, 1e-6)
            
            # 3. Build least squares system
            row_indices = Int[]
            col_indices = Int[]
            values = Float64[]
            cells_idx = []
            
            # Precompute affected cells and their indices for residual vector
            for i in 1:nx
                for j in 1:ny
                    if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                        push!(cells_idx, (i, j))
                    end
                end
            end
            
            # Number of equations (cells) in our system
            m = length(cells_idx)
            
            # Build Jacobian matrix for volume changes
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
            
            # Reduced visualization frequency
            if (timestep <= 2 || mod(timestep, 10) == 0) && iter <= 3
                # Visualize residual
                fig_residual = Figure()
                ax_residual = Axis(fig_residual[1, 1], 
                                title="Residual F (Iteration $iter)", 
                                xlabel="x", ylabel="y",
                                aspect=DataAspect())
                
                residual_2d = zeros(nx, ny)
                for (eq_idx, (i, j)) in enumerate(cells_idx)
                    residual_2d[i, j] = F[eq_idx]
                end
                
                hm_residual = heatmap!(ax_residual, residual_2d, colormap=:balance)
                Colorbar(fig_residual[1, 2], hm_residual, label="Residual value")
                
                markers_x = [m[1] for m in markers]
                markers_y = [m[2] for m in markers]
                lines!(ax_residual, markers_x, markers_y, color=:black, linewidth=1.5)
                
                display(fig_residual)
            end
            
            # 4. Implement the Gauss-Newton step
            JTJ = J' * J
            
            # Diagnose the system
            used_columns = unique(col_indices)
            println("Matrix info: size(J)=$(size(J)), n_markers=$n_markers")
            println("Used marker indices: $(length(used_columns)) of $n_markers")
            
            # Add regularization using JTJ diagonal (Levenberg-Marquardt style)
            reg_param = 1e-6
            diag_JTJ = diag(JTJ)
            
            # Ensure diagonal elements aren't too small
            min_diag = 1e-10 * maximum(diag_JTJ)
            for i in 1:length(diag_JTJ)
                if diag_JTJ[i] < min_diag
                    diag_JTJ[i] = min_diag
                end
            end
            
            # Apply Levenberg-Marquardt style regularization
            reg_JTJ = JTJ + reg_param * Diagonal(diag_JTJ)
            
            # Solve the system using robust method
            newton_step = zeros(n_markers)
            try
                newton_step = reg_JTJ \ (J' * F)
            catch e
                println("Matrix solver failed, using SVD as backup")
                F_svd = svd(Matrix(reg_JTJ))
                svd_tol = eps(Float64) * max(size(reg_JTJ)...) * maximum(F_svd.S)
                S_inv = [s > svd_tol ? 1/s : 0.0 for s in F_svd.S]
                
                JTF = J' * F
                newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
            end
            
            # Store old displacements for position change calculation
            old_displacements = copy(displacements)
            
            # Apply the step with adjustment factor
            displacements -= α * newton_step
            
            # For unclosed interfaces, handle endpoints specially
            if fixed_endpoints
                # Keep endpoints fixed (useful for interfaces that should extend to domain boundaries)
                displacements[1] = 0.0
                displacements[end] = 0.0
            end
            
            # Special smoothing parameters for unclosed interfaces
            smooth_factor = 0.3  # Lower smoothing factor for unclosed interfaces
            window_size = 2
            
            # Smooth displacements for stability, with false indicating non-closed curve
            smooth_displacements!(displacements, markers, false, smooth_factor, window_size)
            
            # Apply special smoothing to endpoints if not fixed
            if !fixed_endpoints && n_markers > 2
                # Endpoints get limited smoothing to prevent instabilities
                displacements[1] = (1.0 - endpoint_smoothing) * displacements[1] + 
                                  endpoint_smoothing * displacements[2]
                displacements[end] = (1.0 - endpoint_smoothing) * displacements[end] + 
                                    endpoint_smoothing * displacements[end-1]
            end
            
            # Calculate maximum displacement for diagnostics
            max_disp = maximum(abs.(displacements))
            println("Maximum displacement (after smoothing): $max_disp")
            
            # Limit maximum displacement if needed
            max_allowed = min(mesh.nodes[1][2] - mesh.nodes[1][1], mesh.nodes[2][2] - mesh.nodes[2][1]) * 0.1
            if max_disp > max_allowed
                scale = max_allowed / max_disp
                displacements .*= scale
                println("Scaling displacements by $scale to limit maximum displacement")
            end
            
            # Calculate position changes for convergence monitoring
            # Calculate proposed new marker positions
            new_markers = Vector{Tuple{Float64, Float64}}(undef, n_markers)
            for i in 1:n_markers
                normal = normals[i]
                new_markers[i] = (
                    markers[i][1] + displacements[i] * normal[1],
                    markers[i][2] + displacements[i] * normal[2]
                )
            end
            
            # Calculate position change between iterations (Xᵏ⁺¹ - Xᵏ)
            position_diff_squared = 0.0
            for i in 1:n_markers
                prev_pos = markers[i]
                new_pos = new_markers[i]
                dx = new_pos[1] - prev_pos[1]
                dy = new_pos[2] - prev_pos[2]
                position_diff_squared += dx^2 + dy^2
            end
            position_change = sqrt(position_diff_squared / n_markers)
            push!(position_change_history, position_change)
            
            # Calculate residual norm for convergence check
            residual_norm = norm(F)
            push!(residual_norm_history, residual_norm)
            
            # Report progress
            println("Iteration $iter | Residual = $residual_norm | Position change = $position_change")
            
            # Check convergence - use both residual and position change
            if (residual_norm < tol && position_change < tol) || 
               (iter > 1 && abs(residual_norm_history[end] - residual_norm_history[end-1]) < reltol &&
                position_change < 10*tol)
                println("Converged after $iter iterations with residual $residual_norm and position change $position_change")
                break
            end
            
            # Update marker positions based on displacements
            for i in 1:n_markers
                normal = normals[i]
                markers[i] = (
                    markers[i][1] + displacements[i] * normal[1],
                    markers[i][2] + displacements[i] * normal[2]
                )
            end
            
            # Create updated front tracking object
            updated_front = FrontTracker(markers, false)  # False indicates non-closed
            
            # Create space-time level set for capacity calculation
            function body(x, y, t_local, _=0)
                # Normalized time in [0,1]
                τ = (t_local - tₙ) / Δt

                # Linear interpolation between SDFs
                sdf1 = -sdf(front, x, y)
                sdf2 = -sdf(updated_front, x, y)
                return (1-τ) * sdf1 + τ * sdf2
            end
            
            # Update space-time mesh and capacity
            STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
            capacity = Capacity(body, STmesh; compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase_updated = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)
            
            # Rebuild the matrix system
            s.A = A_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, phase_updated.source, 
                                            bc, Tᵢ, Δt, tₙ, scheme)
            
            BC_border_mono!(s.A, s.b, bc_b, mesh)
            
            # Update phase for next iteration
            phase = phase_updated
            front = updated_front  # Update front for next iteration

            body_2d(x,y,_=0) = body(x, y, tₙ₊₁)
            capacity_2d = Capacity(body_2d, mesh; compute_centroids=false)
            operator_2d = DiffusionOps(capacity_2d)
            phase_2d = Phase(capacity_2d, operator_2d, phase.source, phase.Diffusion_coeff)
        end
        
        # Store residuals and position changes from this time step
        residuals[timestep] = residual_norm_history
        interface_position_changes[timestep] = position_change_history
        
        # Store updated interface position
        xf_log[timestep+1] = markers
        
        # Store solution
        push!(s.states, s.x)
        
        println("Time: $(round(t, digits=6))")
        println("Max temperature: $(maximum(abs.(s.x)))")
        
        # Increment timestep counter
        timestep += 1
    end
    
    return s, residuals, xf_log, timestep_history, phase_2d, interface_position_changes
end