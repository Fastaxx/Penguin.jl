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
                            method=Base.:\, 
                            Newton_params=(100, 1e-6, 1e-6, 1.0),
                            jacobian_epsilon=1e-6, smooth_factor=0.5, window_size=10, 
                            algorithm="LM", # "GN" pour Gauss-Newton ou "LM" pour Levenberg-Marquardt 
                            lm_init_lambda=1e-4, # Paramètre d'amortissement initial pour LM
                            lm_lambda_factor=10.0, # Facteur de multiplication/division pour lambda
                            lm_min_lambda=1e-10, # Lambda minimum
                            lm_max_lambda=1e6, # Lambda maximum
                            kwargs...)
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

        # Variables pour Levenberg-Marquardt
        lambda = lm_init_lambda
        prev_residual_norm = Inf

        # Gauss-Newton iterations
        for iter in 1:max_iter
            # 1. Solve temperature field with current interface position
            solve_system!(s; method=method, kwargs...)
            Tᵢ = s.x
            
            # Get capacity matrices
            V_matrices = phase.capacity.A[cap_index]
            Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]
            Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]
            Vₙ₊₁_matrix = diag(Vₙ₊₁_matrix)  # Convert to diagonal matrix for easier handling
            Vₙ_matrix = diag(Vₙ_matrix)  # Convert to diagonal matrix for easier handling
            Vₙ₊₁_matrix = reshape(Vₙ₊₁_matrix, (nx, ny))  # Reshape to 2D matrix
            Vₙ_matrix = reshape(Vₙ_matrix, (nx, ny))  # Reshape to 2D matrix
            
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
            volume_jacobian = compute_volume_jacobian(mesh, front, jacobian_epsilon)
            
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
            mismatches = 0
            total_cells = 0
            
            # Diagnostic arrays
            flux_nonzero = zeros(Bool, nx, ny)
            volume_nonzero = zeros(Bool, nx, ny)
            both_nonzero = zeros(Bool, nx, ny)
            
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                # Get the desired volume change based on interface flux
                volume_change = Vₙ₊₁_matrix[i,j] - Vₙ_matrix[i,j]
                flux = interface_flux_2d[i,j]
                
                # Record diagnostics
                total_cells += 1
                volume_nonzero[i,j] = abs(volume_change) > 1e-10
                flux_nonzero[i,j] = abs(flux) > 1e-10
                both_nonzero[i,j] = volume_nonzero[i,j] && flux_nonzero[i,j]
                
                # Detect mismatches (volume change but no flux or vice versa)
                if (volume_nonzero[i,j] && !flux_nonzero[i,j]) || (!volume_nonzero[i,j] && flux_nonzero[i,j])
                    mismatches += 1
                    if mismatches <= 5  # Limiter le nombre de messages
                        println("Mismatch at ($i,$j): volume_change = $volume_change, flux = $flux")
                    end
                end
                
                # F_i = ρL * volume_change - interface_flux
                F[eq_idx] = ρL * volume_change - flux
            end
            
            # Print diagnostic summary
            nonzero_vol = count(volume_nonzero)
            nonzero_flux = count(flux_nonzero)
            nonzero_both = count(both_nonzero)
            println("Cells with nonzero volume change: $nonzero_vol")
            println("Cells with nonzero interface flux: $nonzero_flux")
            println("Cells with both nonzero: $nonzero_both")
            println("Mismatches: $mismatches out of $total_cells cells")
            
            
            # Visualization of matches/mismatches
            if iter == 1 || mismatches > 0
                mismatch_plot = Figure()
                ax1 = Axis(mismatch_plot[1, 1], title="Volume Change", aspect=DataAspect())
                heatmap!(ax1, [abs(Vₙ₊₁_matrix[i,j] - Vₙ_matrix[i,j]) for i=1:nx, j=1:ny], colormap=:viridis)
                
                ax2 = Axis(mismatch_plot[1, 2], title="Interface Flux", aspect=DataAspect()) 
                heatmap!(ax2, interface_flux_2d, colormap=:viridis)
                
                ax3 = Axis(mismatch_plot[2, 1:2], title="Mismatch Map", aspect=DataAspect())
                mismatch_map = zeros(Int, nx, ny)
                for i=1:nx, j=1:ny
                    if volume_nonzero[i,j] && !flux_nonzero[i,j]
                        mismatch_map[i,j] = 1  # Volume sans flux
                    elseif !volume_nonzero[i,j] && flux_nonzero[i,j]
                        mismatch_map[i,j] = 2  # Flux sans volume
                    elseif volume_nonzero[i,j] && flux_nonzero[i,j]
                        mismatch_map[i,j] = 3  # Les deux
                    end
                end
                heatmap!(ax3, mismatch_map, colormap=[:transparent, :red, :blue, :green])
                
                display(mismatch_plot)
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
            
            
        # 4. Implémenter l'algorithme d'optimisation choisi
        JTJ = J' * J
        
        # Diagnostics initiaux
        used_columns = unique(col_indices)
        println("Matrix info: size(J)=$(size(J)), n_markers=$n_markers")
        println("Used marker indices: $(length(used_columns)) of $n_markers")
        
        
        if uppercase(algorithm) == "LM"
            println("Using Levenberg-Marquardt with adaptive damping")
            
            # Extraire la diagonale pour la régularisation LM
            diag_JTJ = diag(JTJ)
            
            # Protéger contre les valeurs trop petites
            min_diag = 1e-10 * maximum(diag_JTJ)
            for i in 1:length(diag_JTJ)
                if diag_JTJ[i] < min_diag
                    diag_JTJ[i] = min_diag
                end
            end
            
            # Appliquer la régularisation adaptative de Levenberg-Marquardt
            reg_JTJ = JTJ + lambda * Diagonal(diag_JTJ)
            
        else
            println("Using Gauss-Newton algorithm with minimal stabilization")
            
            # Pour GN, juste une petite régularisation pour éviter les matrices singulières
            min_reg = 1e-10 * norm(JTJ, Inf)
            reg_JTJ = JTJ + min_reg * I
        end
        
        # Résoudre le système avec méthode robuste
        newton_step = zeros(n_markers)
        try
            newton_step = reg_JTJ \ (J' * F)
        catch e
            println("Matrix solver failed, using SVD as backup")
            # Résolution par SVD comme fallback
            F_svd = svd(Matrix(reg_JTJ))
            svd_tol = eps(Float64) * max(size(reg_JTJ)...) * maximum(F_svd.S)
            S_inv = [s > svd_tol ? 1/s : 0.0 for s in F_svd.S]
            
            # Calculer la solution pseudo-inverse
            JTF = J' * F
            newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
        end
        
        # Calculer la norme de l'incrément de position
        position_increment_norm = α * norm(newton_step)
        push!(position_increment_history, position_increment_norm)
        
        # Pour Levenberg-Marquardt, ajuster lambda en fonction de la convergence
        if uppercase(algorithm) == "LM" && iter > 1
            residual_norm = norm(F)
            if residual_norm < prev_residual_norm
                # Amélioration - réduire lambda
                lambda = max(lambda / lm_lambda_factor, lm_min_lambda)
                println("Residual improved: decreasing lambda to $lambda")
            else
                # Dégradation - augmenter lambda
                lambda = min(lambda * lm_lambda_factor, lm_max_lambda)
                println("Residual worsened: increasing lambda to $lambda")
            end
            prev_residual_norm = residual_norm
        end
        
        # Appliquer le pas avec facteur d'ajustement
        displacements -= α * newton_step
            
            # For closed curves, match first and last displacement to ensure continuity
            if front.is_closed
                displacements[end] = displacements[1]
            end
            
            # Smooth the displacements for stability
            smooth_displacements!(displacements, markers, front.is_closed, smooth_factor, window_size)
            
            # Print maximum displacement for diagnostics
            max_disp = maximum(abs.(displacements))
            println("Maximum displacement (after smoothing): $max_disp")
            
            # Optional: Limit maximum displacement if needed
            #max_allowed = min(mesh.nodes[1][2] - mesh.nodes[1][1], mesh.nodes[2][2] - mesh.nodes[2][1]) * 0.1
           # if max_disp > max_allowed
            #    scale = max_allowed / max_disp
           #     displacements .*= scale
           #     println("Scaling displacements by $scale to limit maximum displacement")
            #end
            
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

            # Plot the updated interface markers
            fig = Figure()
            ax = Axis(fig[1, 1], title="Updated Interface Markers (Iteration $iter)", 
                      xlabel="x", ylabel="y", aspect=DataAspect())
            marker_x = [m[1] for m in new_markers]
            marker_y = [m[2] for m in new_markers]

            # Draw the interface line
            lines!(ax, marker_x, marker_y, color=:blue, linewidth=2,
                  label="Updated Interface")
            scatter!(ax, marker_x, marker_y, color=:red, markersize=6,
                    label="Markers")

            display(fig)

            
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



# Define the Stefan diphasic solver
function StefanDiph2D(phase1::Phase, phase2::Phase, bc_b::BorderConditions, 
                     interface_cond::InterfaceConditions, Δt::Float64, 
                     u0::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Stefan problem with front tracking")
    println("- Diphasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Diphasic, Diffusion, nothing, nothing, nothing, ConvergenceHistory(), [])
    
    # Create initial matrix system based on selected scheme
    if scheme == "CN"
        s.A = A_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity,
                                         phase2.capacity, phase1.Diffusion_coeff, 
                                         phase2.Diffusion_coeff, interface_cond, "CN")
        s.b = b_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity,
                                         phase2.capacity, phase1.Diffusion_coeff, 
                                         phase2.Diffusion_coeff, phase1.source, phase2.source, 
                                         interface_cond, u0, Δt, 0.0, "CN")
    else # "BE"
        s.A = A_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, 
                                        phase2.capacity, phase1.Diffusion_coeff, 
                                        phase2.Diffusion_coeff, interface_cond, "BE")
        s.b = b_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, 
                                        phase2.capacity, phase1.Diffusion_coeff, 
                                        phase2.Diffusion_coeff, phase1.source, phase2.source, 
                                        interface_cond, u0, Δt, 0.0, "BE")
    end
    
    # Apply boundary conditions
    BC_border_diph!(s.A, s.b, bc_b, mesh)
    
    s.x = u0
    return s
end

function solve_StefanDiph2D!(s::Solver, phase1::Phase, phase2::Phase, 
                            front::FrontTracker, Δt::Float64, 
                            Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, 
                            interface_cond::InterfaceConditions, mesh::Penguin.Mesh{2}, 
                            scheme::String;
                            method=Base.:\, 
                            Newton_params=(100, 1e-6, 1e-6, 1.0),
                            jacobian_epsilon=1e-6, smooth_factor=0.5, 
                            window_size=10, kwargs...)
                            
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving Stefan diphasic problem with Front Tracking:")
    println("- Diphasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Unpack parameters
    max_iter = Newton_params[1]
    tol = Newton_params[2]
    reltol = Newton_params[3]
    α = Newton_params[4]
    
    # Extract interface flux parameter
    ρL = interface_cond.flux.value
    
    # Initialize tracking variables
    t = Tₛ
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))
    position_increments = Dict{Int, Vector{Float64}}()
    
    # Determine how many dimensions
    dims = phase1.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Create the 2D grid dimensions
    nx, ny, _ = dims
    n = nx * ny
    
    # Store initial state
    Tᵢ = s.x
    push!(s.states, s.x)
    
    # Get initial interface markers
    markers = get_markers(front)
    
    # Store initial interface position
    xf_log[1] = markers
    
    # Main time stepping loop
    timestep = 1
    phase1_2d = phase1  # Initialize for later use
    phase2_2d = phase2  # Initialize for later use
    while t < Tₑ
        # Display current time step
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
        
        # Calculate total number of markers (excluding duplicated closing point)
        n_markers = length(markers) - (front.is_closed ? 1 : 0)
        
        # Initialize displacement vector and residual vector
        displacements = zeros(n_markers)
        residual_norm_history = Float64[]
        position_increment_history = Float64[]

        # Gauss-Newton iterations
        for iter in 1:max_iter
            # 1. Solve temperature field with current interface position
            solve_system!(s; method=method, kwargs...)
            Tᵢ = s.x
            
            # Separate solution components for each phase
            n_dof = length(Tᵢ) ÷ 4
            T1_bulk = Tᵢ[1:n_dof]                    # Phase 1 bulk
            T1_interface = Tᵢ[n_dof+1:2*n_dof]       # Phase 1 interface
            T2_bulk = Tᵢ[2*n_dof+1:3*n_dof]          # Phase 2 bulk
            T2_interface = Tᵢ[3*n_dof+1:end]         # Phase 2 interface
            
            # Get capacity matrices for Phase 1
            V1_matrices = phase1.capacity.A[cap_index]
            V1ₙ₊₁_matrix = V1_matrices[1:end÷2, 1:end÷2]
            V1ₙ_matrix = V1_matrices[end÷2+1:end, end÷2+1:end]
            V1ₙ₊₁_matrix = diag(V1ₙ₊₁_matrix)
            V1ₙ_matrix = diag(V1ₙ_matrix)
            V1ₙ₊₁_matrix = reshape(V1ₙ₊₁_matrix, (nx, ny))
            V1ₙ_matrix = reshape(V1ₙ_matrix, (nx, ny))
            
            # Get capacity matrices for Phase 2
            V2_matrices = phase2.capacity.A[cap_index]
            V2ₙ₊₁_matrix = V2_matrices[1:end÷2, 1:end÷2]
            V2ₙ_matrix = V2_matrices[end÷2+1:end, end÷2+1:end]
            V2ₙ₊₁_matrix = diag(V2ₙ₊₁_matrix)
            V2ₙ_matrix = diag(V2ₙ_matrix)
            V2ₙ₊₁_matrix = reshape(V2ₙ₊₁_matrix, (nx, ny))
            V2ₙ_matrix = reshape(V2ₙ_matrix, (nx, ny))
            
            # 2. Calculate the interface flux for each phase
            # Phase 1 interface flux
            W1! = phase1.operator.Wꜝ[1:end÷2, 1:end÷2]
            G1 = phase1.operator.G[1:end÷2, 1:end÷2]
            H1 = phase1.operator.H[1:end÷2, 1:end÷2]
            Id1 = build_I_D(phase1.operator, phase1.Diffusion_coeff, phase1.capacity)
            Id1 = Id1[1:end÷2, 1:end÷2]
            
            T1ₒ, T1ᵧ = T1_bulk, T1_interface
            interface_flux1 = Id1 * H1' * W1! * G1 * T1ₒ + Id1 * H1' * W1! * H1 * T1ᵧ
            interface_flux1_2d = reshape(interface_flux1, (nx, ny))
            
            # Phase 2 interface flux
            W2! = phase2.operator.Wꜝ[1:end÷2, 1:end÷2]
            G2 = phase2.operator.G[1:end÷2, 1:end÷2]
            H2 = phase2.operator.H[1:end÷2, 1:end÷2]
            Id2 = build_I_D(phase2.operator, phase2.Diffusion_coeff, phase2.capacity)
            Id2 = Id2[1:end÷2, 1:end÷2]
            
            T2ₒ, T2ᵧ = T2_bulk, T2_interface
            interface_flux2 = Id2 * H2' * W2! * G2 * T2ₒ + Id2 * H2' * W2! * H2 * T2ᵧ
            interface_flux2_2d = reshape(interface_flux2, (nx, ny))
            
            # 3. Compute volume Jacobian for the mesh
            volume_jacobian = compute_volume_jacobian(mesh, front, jacobian_epsilon)
            
            # 4. Build least squares system
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
            
            # Number of equations (cells) in the system
            m = length(cells_idx)
            
            # Build the Jacobian matrix for volume changes
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
            
            # 5. Calculate current residual vector F
            F = zeros(m)
            mismatches = 0
            total_cells = 0
            
            # Diagnostic arrays
            flux_nonzero = zeros(Bool, nx, ny)
            volume_nonzero = zeros(Bool, nx, ny)
            
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                # Calculate volume changes for both phases
                volume_change1 = V1ₙ₊₁_matrix[i,j] - V1ₙ_matrix[i,j]
                volume_change2 = V2ₙ₊₁_matrix[i,j] - V2ₙ_matrix[i,j]
                
                # Calculate net flux (take both phases into account)
                # Note: Interface moves from Phase 1 to Phase 2, so invert sign for Phase 1
                flux1 = -interface_flux1_2d[i,j]  # Negative sign for Phase 1 (inside)
                flux2 = interface_flux2_2d[i,j]   # Phase 2 (outside)
                net_flux = flux1 + flux2
                
                # Calculate total volume change
                # Note: For conservation, volume change in Phase 1 should be negated
                net_volume_change = -volume_change1 + volume_change2
                
                # Record diagnostics
                total_cells += 1
                volume_nonzero[i,j] = abs(net_volume_change) > 1e-10
                flux_nonzero[i,j] = abs(net_flux) > 1e-10
                
                # Detect mismatches (volume change without flux or vice versa)
                if (volume_nonzero[i,j] && !flux_nonzero[i,j]) || (!volume_nonzero[i,j] && flux_nonzero[i,j])
                    mismatches += 1
                    if mismatches <= 5
                        println("Mismatch at ($i,$j): net_volume_change = $net_volume_change, net_flux = $net_flux")
                    end
                end
                
                # F_i = ρL * net_volume_change - net_flux
                F[eq_idx] = ρL * net_volume_change - net_flux
            end
            
            # Print diagnostic summary
            nonzero_vol = count(volume_nonzero)
            nonzero_flux = count(flux_nonzero)
            nonzero_both = count(volume_nonzero .& flux_nonzero)
            println("Cells with nonzero volume change: $nonzero_vol")
            println("Cells with nonzero interface flux: $nonzero_flux")
            println("Cells with both nonzero: $nonzero_both")
            println("Mismatches: $mismatches out of $total_cells cells")
            
            # 6. Implement the Gauss-Newton formula with regularization
            JTJ = J' * J
            
            # Diagnose the system
            used_columns = unique(col_indices)
            println("Matrix info: size(J)=$(size(J)), n_markers=$n_markers")
            println("Used marker indices: $(length(used_columns)) of $n_markers")
            
            # Add Tikhonov regularization
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
                # SVD-based pseudoinverse for robust solving
                F_svd = svd(Matrix(reg_JTJ))
                svd_tol = eps(Float64) * max(size(reg_JTJ)...) * maximum(F_svd.S)
                S_inv = [s > svd_tol ? 1/s : 0.0 for s in F_svd.S]
                
                # Compute pseudoinverse solution
                JTF = J' * F
                newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
            end
            
            # Calculate position increment norm
            position_increment_norm = α * norm(newton_step)
            push!(position_increment_history, position_increment_norm)
            
            # 7. Apply the step with adjustment factor
            displacements -= α * newton_step
            
            # For closed curves, match first and last displacement to ensure continuity
            if front.is_closed
                displacements[end] = displacements[1]
            end
            
            # 8. Smooth the displacements for stability
            smooth_displacements!(displacements, markers, front.is_closed, smooth_factor, window_size)
            
            # Print maximum displacement for diagnostics
            max_disp = maximum(abs.(displacements))
            println("Maximum displacement (after smoothing): $max_disp")
            
            # 9. Calculate residual norm for convergence check
            residual_norm = norm(F)
            push!(residual_norm_history, residual_norm)
            
            # Report progress
            println("Iteration $iter | Residual = $residual_norm | Position increment = $position_increment_norm")
            
            # 10. Check convergence
            if residual_norm < tol || (iter > 1 && abs(residual_norm_history[end] - residual_norm_history[end-1]) < reltol)
                println("Converged after $iter iterations with residual $residual_norm")
                break
            end
            
            # 11. Update marker positions
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
            
            # 12. Create updated front tracking object
            updated_front = FrontTracker(new_markers, front.is_closed)
            
            # 13. Create space-time level set for capacity calculation
            function body1_update(x, y, t_local, _=0)
                # Normalized time in [0,1]
                τ = (t_local - tₙ) / Δt
                
                # Linear interpolation between SDFs
                sdf1 = sdf(front, x, y)
                sdf2 = sdf(updated_front, x, y)
                return (1-τ) * sdf1 + τ * sdf2
            end
            
            function body2_update(x, y, t_local, _=0)
                # Normalized time in [0,1]
                τ = (t_local - tₙ) / Δt
                
                # Linear interpolation between SDFs
                sdf1 = -sdf(front, x, y)
                sdf2 = -sdf(updated_front, x, y)
                return (1-τ) * sdf1 + τ * sdf2
            end
            
            # 14. Update space-time mesh and capacities for both phases
            STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
            
            capacity1_updated = Capacity(body1_update, STmesh; compute_centroids=false)
            operator1_updated = DiffusionOps(capacity1_updated)
            phase1_updated = Phase(capacity1_updated, operator1_updated, phase1.source, phase1.Diffusion_coeff)
            
            capacity2_updated = Capacity(body2_update, STmesh; compute_centroids=false)
            operator2_updated = DiffusionOps(capacity2_updated)
            phase2_updated = Phase(capacity2_updated, operator2_updated, phase2.source, phase2.Diffusion_coeff)
            
            # 15. Rebuild the matrix system
            s.A = A_diph_unstead_diff_moving_stef2(phase1_updated.operator, phase2_updated.operator,
                                            phase1_updated.capacity, phase2_updated.capacity,
                                            phase1_updated.Diffusion_coeff, phase2_updated.Diffusion_coeff, 
                                            interface_cond, scheme)
                                            
            s.b = b_diph_unstead_diff_moving_stef2(phase1_updated.operator, phase2_updated.operator,
                                            phase1_updated.capacity, phase2_updated.capacity,
                                            phase1_updated.Diffusion_coeff, phase2_updated.Diffusion_coeff,
                                            phase1_updated.source, phase2_updated.source,
                                            interface_cond, Tᵢ, Δt, tₙ, scheme)
            
            BC_border_diph!(s.A, s.b, bc_b, mesh)
            
            # 16. Update phases and front for next iteration
            phase1 = phase1_updated
            phase2 = phase2_updated
            front = updated_front

            # Update snapshot phases for visualization
            body1_2d(x,y,_=0) = body1_update(x, y, tₙ₊₁)
            body2_2d(x,y,_=0) = body2_update(x, y, tₙ₊₁)
            
            capacity1_2d = Capacity(body1_2d, mesh; compute_centroids=false)
            operator1_2d = DiffusionOps(capacity1_2d)
            phase1_2d = Phase(capacity1_2d, operator1_2d, phase1.source, phase1.Diffusion_coeff)
            
            capacity2_2d = Capacity(body2_2d, mesh; compute_centroids=false)
            operator2_2d = DiffusionOps(capacity2_2d)
            phase2_2d = Phase(capacity2_2d, operator2_2d, phase2.source, phase2.Diffusion_coeff)
        end
        
        # Store residuals and position increments from this time step
        residuals[timestep] = residual_norm_history
        position_increments[timestep] = position_increment_history
        
        # Store updated interface position
        xf_log[timestep+1] = get_markers(front)
        
        # Store solution
        push!(s.states, s.x)
        
        # Print radius info for a circle
        markers = get_markers(front)
        center_x = sum(m[1] for m in markers) / length(markers)
        center_y = sum(m[2] for m in markers) / length(markers)
        mean_radius = mean([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in markers])
        println("Mean radius: $(round(mean_radius, digits=6))")
        
        println("Time: $(round(t, digits=6))")
        println("Max temperature: $(maximum(abs.(s.x)))")
        
        # Increment timestep counter
        timestep += 1
        
        # Add current time and timestep size to history
        push!(timestep_history, (t, Δt))
    end
    
    return s, residuals, xf_log, timestep_history, phase1_2d, phase2_2d, position_increments
end